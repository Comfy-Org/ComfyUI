"""Block-sparse attention on comfy_kitchen's sparse attention kernels (Sol-Attn adaptive
threshold, SLA-style top-k, or FastVideo's VSA). Generic models go through the
attention override; MiniMax-H3 gets the chunked qkv producer through block patches."""
from __future__ import annotations

import logging
import re
import weakref

import comfy_kitchen as ck
import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_api.latest import ComfyExtension, io

HEAD_DIM = 128
BLOCK_SIZE = 64
PRODUCER_CHUNK = 4096
VSA_CUBE = (4, 4, 4)
VSA_PLAN_CACHE = 4


def parse_block_list(text):
    """'0, 1, 47-49' -> {0, 1, 47, 48, 49}."""
    blocks = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            blocks.update(range(min(a, b), max(a, b) + 1))
        else:
            blocks.add(int(part))
    return blocks


class SparseAttnPatch:
    """Options plus runtime state for one patched model; an ON_CLEANUP callback
    resets the state when the sampling run ends."""

    def __init__(self, tau, topk_ratio, vsa, sigma_start, sigma_end, min_tokens,
                 dense_blocks, sink_conditioning, extra_tokens, verbose):
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.extra_tokens = extra_tokens
        self.vsa = vsa
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.dense_blocks = dense_blocks
        self.sink_conditioning = sink_conditioning
        self.verbose = verbose
        self.installed = set()    # the override closures this patch has put on the hook
        self.reset()

    def reset(self):
        self.pooled = {}          # (block, rows) -> (kmean, vscale) from the previous step
        self.vsa_plans = {}       # small LRU of tiling plans
        self.vsa_rope = None
        self._logged = set()

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            logging.info(f"BlockSparseAttention: {message}")

    def dense_reason(self, transformer_options, tokens, block_index):
        """Why this call stays dense regardless of its tensors, or None."""
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return f"sigma {sigma:.3g} outside the start/end window"
        if tokens < self.min_tokens:
            return f"{tokens} tokens < min_tokens {self.min_tokens}"
        if self.dense_blocks:
            if block_index is None:
                self.log_once("no_block_index", "this model does not report block indices; dense_blocks ignored")
            elif block_index in self.dense_blocks:
                return f"block {block_index} in dense_blocks"
        return None

    def sinks(self, transformer_options, tokens):
        """MiniMax-H3 conditioning rows as (exact-KV blocks, dense-query blocks):
        the packed prefix stays exact for every query, and optionally the
        target-audio query rows run dense."""
        layout = transformer_options.get("minimax_h3_layout")
        if self.sink_conditioning == "off" or layout is None or layout.seq_len != tokens:
            return (0, 0), (0, 0)
        video = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
        if video is None or video[0] <= 0:
            return (0, 0), (0, 0)
        blocks = (0, (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if self.sink_conditioning != "exact_kv_and_rows":
            return blocks, (0, 0)
        audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
        if audio is None:
            return blocks, blocks
        return blocks, (audio[0] // BLOCK_SIZE, blocks[1])

    def vsa_plan(self, layout, device):
        """Padded tile order: prefix segments in their own zero-padded 64-row
        tiles, video in 4x4x4 cubes. `src` maps padded row -> source row (-1 =
        pad), `inv` the reverse."""
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        plan = self.vsa_plans.get(key)
        if plan is not None:
            return plan
        _text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
        tiles, n_prefix = [], 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind != "video":
                m = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                seg = torch.full((m * BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
                seg[:n] = torch.arange(a, b, device=device)
                tiles.append(seg.view(m, BLOCK_SIZE))
                n_prefix += m
                continue
            if grid[0] * grid[1] * grid[2] != n:
                raise RuntimeError(f"VSA: video segment of {n} rows does not match the latent grid {grid}")
            ct, ch, cw = VSA_CUBE
            pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, VSA_CUBE))
            padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
            padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b, device=device).view(*grid)
            cubes = (padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw)
                     .permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK_SIZE))
            order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
            tiles.append(torch.gather(cubes, 1, order))
        tiles = torch.cat(tiles)
        src = tiles.reshape(-1)
        live = src >= 0
        inv = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
        inv[src[live]] = torch.nonzero(live).flatten()
        plan = {"n": int(src.numel()), "n_prefix": n_prefix, "src": src, "inv": inv,
                "block_len": (tiles >= 0).sum(1).to(torch.int32)}
        while len(self.vsa_plans) >= VSA_PLAN_CACHE:
            del self.vsa_plans[next(iter(self.vsa_plans))]
        self.vsa_plans[key] = plan
        return plan

    def vsa_rope_freqs(self, rope_freqs, plan):
        hit = self.vsa_rope
        if hit is not None and hit[0]() is rope_freqs and hit[1] is plan:
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
        padded[0, plan["inv"]] = rope_freqs[0]
        self.vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded


def _ineligible(q, k, v, dim_head):
    """Why these tensors can't go through the kernel, or None. q/k/v are BTHD."""
    if q.device.type != "cuda":
        return "not on CUDA"
    if not ck.sol_attn_is_available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {q.dtype} (kernel takes bf16/fp16)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != {HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "cross-attention or GQA (kept dense)"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"mixed dtypes {q.dtype}/{k.dtype}/{v.dtype}"
    return None


def make_attention_override(patch: SparseAttnPatch, previous):
    """Attention override; declined calls run ``previous`` (the override that was
    on the hook before this one) or ``func``. Dense-only in VSA mode: a
    VSA-trained model must never see plain block-sparse attention."""
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}

        def dense():
            args = (q, k, v, heads)
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
            return func(*args, **kw) if previous is None else previous(func, *args, **kw)

        if mask is not None or patch.vsa:
            return dense()
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get("block_index"))
        if reason is not None:
            patch.log_once(("dense", tokens, reason), f"dense ({tokens} tokens): {reason}")
            return dense()
        if skip_reshape:
            b, _, _, dim_head = q.shape                      # BHND
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape                         # B, N, heads*dim_head
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        reason = _ineligible(qs, ks, vs, dim_head)
        if reason is not None:
            patch.log_once(("ineligible", tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
            return dense()
        sink, sink_q = patch.sinks(transformer_options, tokens)
        if q.dtype == torch.float32:   # the kernel quantizes to int8 anyway; bf16 keeps the fp32 range
            qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
        out = ck.sol_attn(qs, ks, vs, tau=patch.tau, scale=kwargs.get("scale"),
                          sink_blocks=list(sink), sink_q=list(sink_q), topk_ratio=patch.topk_ratio,
                          token_aug=patch.extra_tokens).to(q.dtype)
        patch.log_once(("sparse", tuple(qs.shape)), f"sparse {tuple(qs.shape)}, sinks {sink}/{sink_q}")
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)

    return override


def install_override(patch: SparseAttnPatch, transformer_options):
    """Put this patch's override on top of whatever attention override is on the
    hook. Runs at patch time and again from ON_PREPARE_STATE each step, so a node
    applied later cannot silently replace it; idempotent once it is on top."""
    current = transformer_options.get("optimized_attention_override")
    if current in patch.installed:
        return
    override = make_attention_override(patch, current)
    patch.installed.add(override)
    transformer_options["optimized_attention_override"] = override


def h3_eligible(attn, x, rope_freqs, transformer_options, patch: SparseAttnPatch, block_index):
    """Whether this H3 block call takes the sparse producer (decided before any work)."""
    n_tokens = x.shape[0]
    if rope_freqs is None or x.dtype != torch.bfloat16 or x.device.type != "cuda" or attn.head_dim != HEAD_DIM:
        return False
    reason = patch.dense_reason(transformer_options, n_tokens, block_index)
    if reason is None and not ck.sol_attn_is_available(x.device):
        reason = "no compiled sol_attn kernel for this GPU"
    if reason is not None:
        patch.log_once(("dense", n_tokens, reason), f"dense ({n_tokens} tokens): {reason}")
        return False
    if patch.vsa:
        layout = transformer_options.get("minimax_h3_layout")
        if layout is None or layout.seq_len != n_tokens:
            patch.log_once("no_layout", "no H3 layout for this call; running dense")
            return False
    return True


def h3_sparse_attention(attn, x, rope_freqs, transformer_options, patch: SparseAttnPatch, block_index):
    """H3 attention through the chunked producer: qkv projected in 4K-token
    slices straight into the kernel's int8 carriers, full Q/K/V never built."""
    n_tokens = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    extra, plan, gate = {}, None, None
    n, freqs = n_tokens, rope_freqs
    with comfy.model_prefetch.pause_malloc_graph():
        if patch.vsa:
            plan = patch.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
            n = plan["n"]
            freqs = patch.vsa_rope_freqs(rope_freqs, plan)

        key = (block_index, n, tuple(transformer_options.get("uuids", ())))   # statistics per conditioning branch
        pooled = patch.pooled.get(key)
        first = pooled is None
        if first:
            pooled = (
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
            )

    if patch.vsa:
        sink = sink_q = (0, plan["n_prefix"])
        extra = {"tail": False, "block_len": plan["block_len"]}
        gate = attn.to_gate_compress
        if gate is not None:
            extra["coarse_gate"] = x.new_empty(n, heads * head_dim).view(1, n, heads, head_dim)
    else:
        sink, sink_q = patch.sinks(transformer_options, n_tokens)

    def chunks():
        for i in range(0, n, PRODUCER_CHUNK):
            if plan is None:
                yield attn.qkv_proj(x[i:i + PRODUCER_CHUNK])
                continue
            idx = plan["src"][i:i + PRODUCER_CHUNK]
            xc = x[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(x.dtype)   # pad rows zero
            if gate is not None:
                extra["coarse_gate"].view(n, heads * head_dim)[i:i + xc.shape[0]] = gate(xc)
            yield attn.qkv_proj(xc)

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks, n, heads, freqs, (qw, kw),
        kmean=None if first else pooled[0],
        vscale=None if first else pooled[1],
        tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=patch.extra_tokens,
        sink_blocks=list(sink), sink_q=list(sink_q),
        rope_eps=attn.q_norm.eps, **extra)
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch.pooled[key] = pooled
    mode = f"VSA tiles ({n} padded rows, {sink[1]} prefix tiles)" if plan is not None else f"sinks {sink}/{sink_q}"
    patch.log_once(("producer", n), f"sparse producer path: {n_tokens} tokens, {mode}")
    out = out.view(n, heads * head_dim)
    if plan is not None:
        out = out[plan["inv"]]
    return attn.out_proj(out)


def make_h3_block_patch(block, block_index, patch: SparseAttnPatch):
    """Runs the block with its attention swapped for the sparse producer."""
    def attention(h, rope_freqs=None, transformer_options={}):
        return h3_sparse_attention(block.attn, h, rope_freqs, transformer_options, patch, block_index)

    def block_patch(args, extra):
        if h3_eligible(block.attn, args["img"], args["rope_freqs"], args["transformer_options"], patch, block_index):
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch


def apply_block_sparse_attention(model, *, tau, topk_ratio, vsa, start_percent, end_percent, min_tokens,
                                 dense_blocks, sink_conditioning, extra_tokens, verbose):
    model_sampling = model.get_model_object("model_sampling")
    if vsa and extra_tokens:
        # VSA weights were trained against their sparse pattern, don't pull the attention toward dense
        logging.info("VSA: extra_tokens ignored (the trained sparse pattern is the target)")
        extra_tokens = 0
    patch = SparseAttnPatch(tau=tau, topk_ratio=topk_ratio, vsa=vsa,
                            sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
                            sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
                            min_tokens=min_tokens, dense_blocks=dense_blocks,
                            sink_conditioning=sink_conditioning, extra_tokens=extra_tokens, verbose=verbose)
    m = model.clone()
    install_override(patch, m.model_options["transformer_options"])
    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "block_sparse_attention",
                            lambda model_patcher, timestep, model_options: install_override(patch, model_options["transformer_options"]))
    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
                            "block_sparse_attention", lambda model_patcher: patch.reset())

    diffusion_model = model.get_model_object("diffusion_model")
    if isinstance(diffusion_model, MiniMaxH3Model):
        for i, block in enumerate(diffusion_model.blocks):
            m.set_model_patch_replace(make_h3_block_patch(block, i, patch), "dit", "double_block", i)
        if vsa and diffusion_model.blocks[0].attn.to_gate_compress is None:
            logging.warning("VSA: the model has no to_gate_compress layers; running the fine stage without the coarse branch")
    elif vsa:
        raise ValueError("VSA selection needs a MiniMax-H3 model")
    return m


class BlockSparseAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BlockSparseAttention",
            display_name="Model Sparse Attention",
            category="model/patch",
            is_experimental=True,
            search_aliases=["Block Sparse Attention"],
            description="Applies block-sparse attention to eligible model attention layers, reducing compute for long sequences. "
                        "The speed gain grows with sequence length since short sequences are usually faster dense. "
                        "Outside the start/end_percent, dense_blocks and under min_tokens, the model uses the dense model attention backend. "
                        "Use the node Model Attention Backend to select that fallback.",
            inputs=[
                io.Model.Input("model", tooltip="The model to patch."),
                io.DynamicCombo.Input("selection", display_name="method", options=[
                    io.DynamicCombo.Option("sol-attn", [
                        io.Float.Input("tau", default=1.3, min=0.0, max=4.0, step=0.05,
                                       tooltip="Threshold in score-distribution sigmas. Higher is sparser: "
                                               "1.0 keeps ~16% of key blocks exact, 1.5 ~7%, 2.0 ~2.7%."),
                    ]),
                    io.DynamicCombo.Option("sla", [
                        io.Float.Input("keep_percent", default=10.0, min=0.5, max=95.0, step=0.5,
                                       tooltip="Percent of key blocks each query block keeps exactly (sinks and "
                                               "the diagonal ride on top). The selection SLA-style LoRAs are "
                                               "distilled against; without such a LoRA higher is closer to dense."),
                    ]),
                    io.DynamicCombo.Option("vsa", [
                        io.Float.Input("keep_percent", default=10.0, min=0.5, max=95.0, step=0.5,
                                       tooltip="Percent of video cubes each query cube keeps; FastH3-VSA "
                                               "checkpoints are trained at 10. Uses the model's to_gate_compress "
                                               "layers for the coarse branch when present."),
                    ]),
                ], tooltip="Method used to choose key blocks for full token-level attention. "
                           "sol-attn: Sparsifying Online Attention uses a training-free adaptive threshold for each attention head and query block. "
                           "sla: Sparse-Linear Attention keeps a fixed percentage of the highest-scoring key blocks; use only with model weights trained for this pattern. "
                           "vsa: Video Sparse Attention (FastVideo) uses 3D video-cube tiling and a learned coarse attention branch; requires FastH3 model weights."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Percentage point when sparse attention begins. Before this point, attention stays dense."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Percentage point when sparse attention ends. After this point, attention returns to dense."),
                io.String.Input("dense_blocks", default="", advanced=True,
                                tooltip="Transformer blocks that always run dense, e.g. '0, 1, 47-49'."),
                io.Int.Input("min_tokens", default=12288, min=0, max=1 << 20, step=512, advanced=True,
                             tooltip="Sequences shorter than this stay dense."),
                io.Int.Input("extra_tokens", default=256, min=0, max=256, step=64, advanced=True,
                             tooltip="Extra top-scoring tokens each query block attends beyond its selected "
                                     "blocks. Closer to dense for more attention time; 256 recommended, 0 disables. "
                                     "Ignored for VSA."),
                io.Combo.Input("sink_conditioning", options=["exact_kv", "exact_kv_and_rows", "off"],
                               default="exact_kv_and_rows", advanced=True,
                               tooltip="MiniMax-H3 only. exact_kv: every query attends the packed text/audio/"
                                       "reference rows exactly (~3% cost). exact_kv_and_rows: additionally runs "
                                       "the target-audio query rows dense (keeps generated audio intact)."),
                io.Boolean.Input("verbose", default=False, advanced=True,
                                 tooltip="Logs whether each attention shape used sparse attention or why it stayed dense."),
            ],
            outputs=[io.Model.Output(display_name="model", tooltip="The model with block-sparse attention applied.")],
        )

    @classmethod
    def execute(cls, model, selection, start_percent, end_percent, dense_blocks="", min_tokens=12288,
                extra_tokens=0, sink_conditioning="exact_kv_and_rows", verbose=False) -> io.NodeOutput:
        mode = selection["selection"]
        patched_model = apply_block_sparse_attention(
            model,
            tau=selection.get("tau", 1.3),
            topk_ratio=0.0 if mode == "sol-attn" else selection["keep_percent"] / 100.0,
            vsa=mode == "vsa",
            start_percent=start_percent,
            end_percent=end_percent,
            min_tokens=min_tokens,
            dense_blocks=parse_block_list(dense_blocks),
            sink_conditioning=sink_conditioning,
            extra_tokens=extra_tokens,
            verbose=verbose,
        )
        return io.NodeOutput(patched_model)


class BlockSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [BlockSparseAttention]


async def comfy_entrypoint() -> BlockSparseAttentionExtension:
    return BlockSparseAttentionExtension()
