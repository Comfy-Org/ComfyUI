import torch
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.utils
from comfy.ldm.flux.layers import EmbedND, LastLayer, MLPEmbedder, Modulation, modulated_norm, timestep_embedding
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class QKNorm(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.key_norm = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)

    def forward(self, q, k, pe):
        if comfy.model_management.in_training:
            return apply_rope1(self.query_norm(q), pe), apply_rope1(self.key_norm(k), pe)
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(self.query_norm, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.key_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.query_norm.eps)
        comfy.ops.uncast_bias_weight(self.query_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(self.key_norm, k_scale, None, k_stream)
        return q, k


class Block(nn.Module):
    # parallel attention + SwiGLU MLP under one adaLN and one gate, used for the per-mode and the joint blocks
    def __init__(self, hidden_size, num_heads, mlp_ratio, fused_qkv=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            # one GEMM and one activation quantization instead of three
            self.qkv = operations.Linear(hidden_size, 3 * hidden_size, bias=False, dtype=dtype, device=device)
        else:
            self.q_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
            self.k_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
            self.v_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.mlp_in = operations.Linear(hidden_size, 2 * mlp_hidden_dim, bias=False, dtype=dtype, device=device)
        self.attn_out = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.mlp_out = operations.Linear(mlp_hidden_dim, hidden_size, bias=False, dtype=dtype, device=device)
        self.norm = QKNorm(hidden_size // num_heads, dtype=dtype, device=device, operations=operations)
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x, spans, pe, transformer_options={}):
        # spans: (start, end, ModulationOut) row ranges; the first modulates every row, the others redo their rows
        B, N, _ = x.shape
        mod = spans[0][2]
        x_mod = modulated_norm(x, self.pre_norm, mod.scale, mod.shift)
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = modulated_norm(x[:, start:end], self.pre_norm, mod.scale, mod.shift)

        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)
        del x_mod
        q, k = self.norm(q.view(B, N, self.num_heads, -1), k.view(B, N, self.num_heads, -1), pe)
        q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
        out = self.attn_out(optimized_attention(q, k, v, self.num_heads, transformer_options=transformer_options, preferred_attention=self.comfy_attention))
        # attention + MLP sum folded into the mlp_out GEMM epilogue
        out = comfy.ops.linear_input_act(self.mlp_out, mlp, "swiglu", residual=out, residual_scale=out.new_ones(out.shape[-1]))
        del mlp

        for start, end, mod in spans:
            x[:, start:end].addcmul_(out[:, start:end], mod.gate)
        return x


class Flux3(nn.Module):
    def __init__(self, hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=28, fused_qkv=False, flow_map=False, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.in_channels = 128
        self.image_rope_scale = 0.5
        self.flow_map = flow_map
        content_modes = ("image", "image_cond")

        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=10000, axes_dim=[32, 32, 32, 32])
        self.emb_in = nn.ModuleDict({m: operations.Linear(self.in_channels, hidden_size, bias=False, dtype=dtype, device=device) for m in content_modes})
        self.txt_in = operations.Linear(20480, hidden_size, bias=False, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(256, hidden_size, bias=False, dtype=dtype, device=device, operations=operations)
        if flow_map:
            # step-distilled flow map: embeds the time each step jumps to
            self.ref_time_in = MLPEmbedder(256, hidden_size, bias=False, dtype=dtype, device=device, operations=operations)

        modulation = lambda: Modulation(hidden_size, double=False, bias=False, dtype=dtype, device=device, operations=operations)
        self.early_stream_modulations = nn.ModuleDict({m: modulation() for m in content_modes + ("txt",)})
        self.single_stream_modulations = nn.ModuleDict({m: modulation() for m in content_modes + ("txt",)})

        block = lambda: Block(hidden_size, num_heads, 3.0, fused_qkv=fused_qkv, dtype=dtype, device=device, operations=operations)
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([block() for _ in range(depth)]) for m in content_modes})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(hidden_size, 1, self.in_channels, bias=False, dtype=dtype, device=device, operations=operations)})

    def image_ids(self, h, w, index, device):
        # (t, row, col, l); references are told apart on the l axis
        ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * self.image_rope_scale
        ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * self.image_rope_scale
        ids[..., 3] = index
        return ids.flatten(0, 1)

    def ref_ids(self, ref_shapes, device):
        return torch.cat([self.image_ids(h, w, -1 - i, device) for i, (h, w) in enumerate(ref_shapes)])

    def text_ids(self, length, device):
        ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
        ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
        return ids

    def rope(self, ids):
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    def preprocess_conds(self, context, ref_latents=None, vec_target=None, transformer_options={}):
        # text and references only attend to their own mode at t = 0 in the early blocks, so these depend on the conditioning alone
        device = context.device
        if ref_latents:
            # a batch of references stays one per sample
            batch = max([context.shape[0]] + [ref.shape[0] for ref in ref_latents])
            context = comfy.utils.repeat_to_batch_size(context, batch)
            ref_latents = [comfy.utils.repeat_to_batch_size(ref, batch) for ref in ref_latents]
        vec = self.time_in(timestep_embedding(context.new_zeros(1), 256))
        if vec_target is not None:
            vec = vec + vec_target
        txt = self.txt_in(context)
        streams = [(txt, "txt", self.text_ids(txt.shape[1], device), self.txt_mode_blocks)]
        refs = None
        if ref_latents:
            refs = self.emb_in["image_cond"](torch.cat([ref.flatten(2).transpose(1, 2) for ref in ref_latents], dim=1))
            streams.append((refs, "image_cond", self.ref_ids([ref.shape[-2:] for ref in ref_latents], device), self.content_mode_blocks["image_cond"]))
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue([block for *_, blocks in streams for block in blocks], device, transformer_options)
        for tokens, mode, ids, blocks in streams:
            mod = self.early_stream_modulations[mode](vec)[0]
            pe = self.rope(ids)
            for block in blocks:
                comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, tokens.dtype)
                block(tokens, [(0, tokens.shape[1], mod)], pe)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        return txt, refs

    def forward(self, x, timestep, context=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, ref_latents, transformer_options, **kwargs)

    def _forward(self, x, timestep, context=None, ref_latents=None, transformer_options={}, txt_states=None, ref_states=None, ref_shapes=None, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        vec_target = None
        if self.flow_map:
            # every mode's modulation also gets the time this step jumps to: the next sigma of the schedule
            sample_sigmas = transformer_options["sample_sigmas"]
            i = int((sample_sigmas - timestep[0]).abs().argmin())
            target = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
            vec_target = self.ref_time_in(timestep_embedding(target.reshape(1).to(device=device, dtype=dtype), 256))

        # txt_states/ref_states are the early-block outputs of preprocess_conds, normally run once from extra_conds
        if txt_states is None:
            txt_states, ref_states = self.preprocess_conds(context, ref_latents, vec_target, transformer_options)
            ref_shapes = [ref.shape[-2:] for ref in ref_latents] if ref_latents else None
        transformer_options = transformer_options.copy()

        # the reference embeds t in the compute dtype; image_cond and text sit at t = 0
        vec = self.time_in(timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
        if vec_target is not None:
            vec = vec + vec_target
        vec_img, vec_zero = vec[:-1], vec[-1:]
        early = self.early_stream_modulations["image"](vec_img)[0]

        # per sample the joint sequence is [image, image_cond, text]
        img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
        parts = [(img, "image", self.image_ids(H, W, 0, device))]
        if ref_states is not None:
            parts.append((ref_states, "image_cond", self.ref_ids(ref_shapes, device)))
        parts.append((txt_states, "txt", self.text_ids(txt_states.shape[1], device)))
        pe = self.rope(torch.cat([ids for _, _, ids in parts]))
        spans, start = [], 0
        for tokens, mode, _ in parts:
            spans.append((start, start + tokens.shape[1], self.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)[0]))
            start += tokens.shape[1]
        # joint sequence allocated outside the memory compiler region, filled after the image's early blocks
        x = torch.empty(B, start, img.shape[-1], device=device, dtype=img.dtype)

        queue = list(self.content_mode_blocks["image"]) + list(self.single_blocks)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(queue, device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for block in self.content_mode_blocks["image"]:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            img = block(img, [(0, H * W, early)], pe[:, :H * W], transformer_options)

        torch.cat([img] + [tokens for tokens, _, _ in parts[1:]], dim=1, out=x)
        del img, parts
        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.single_blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            x = block(x, spans, pe, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.final_layer["image"](x[:, :H * W], vec_img)
        return out.transpose(1, 2).reshape(B, self.in_channels, H, W)
