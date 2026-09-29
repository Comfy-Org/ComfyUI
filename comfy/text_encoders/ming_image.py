from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import comfy.ops
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import MLP, RMSNorm, TransformerBlock, moe_experts_forward, precompute_freqs_cis
from comfy.text_encoders.qwen35 import apply_partial_rope
from comfy.text_encoders import qwen_vl

IMAGE_PATCH_TOKEN = 157157
IMAGE_BLOCK = "<image><imagePatch></image>"


@dataclass
class BailingMoeV2Config:
    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    rope_dim: int = 64
    rope_theta: float = 600000.0
    rms_norm_eps: float = 1e-6
    num_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    mlp_activation: str = "silu"


@dataclass
class MingConnectorConfig:
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    qkv_bias: bool = True
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    q_norm = None
    k_norm = None


class BailingAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.query_key_value = ops.Linear(config.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.dense = ops.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, attention_mask, freqs_cis, optimized_attention):
        batch, seq_len, _ = x.shape
        qkv = self.query_key_value(x).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        xq, xk, xv = qkv.split((self.num_heads, self.num_kv_heads, self.num_kv_heads), dim=1)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rope_dim)
        out = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, enable_gqa=True)
        return self.dense(out)


class BailingGate(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.proj = ops.Linear(config.hidden_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.expert_bias = nn.Parameter(torch.empty(config.num_experts, device=device, dtype=torch.float32))

    def forward(self, x):
        scores = torch.sigmoid(self.proj(x.float()))
        routing = scores + comfy.ops.cast_to_input(self.expert_bias, scores, copy=False)
        num_tokens = routing.shape[0]
        grouped = routing.view(num_tokens, self.n_group, -1)
        group_idx = torch.topk(grouped.topk(2, dim=-1)[0].sum(dim=-1), k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros(num_tokens, self.n_group, dtype=torch.bool, device=x.device).scatter_(1, group_idx, True)
        topk_idx = torch.topk(grouped.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).view(num_tokens, -1), k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = torch.gather(scores, dim=1, index=topk_idx)
        return topk_idx, topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor


class BailingExperts(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_proj = ops.MoEExperts(num_experts=config.num_experts, in_features=config.hidden_size, out_features=2 * config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts=config.num_experts, in_features=config.moe_intermediate_size, out_features=config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, topk_idx, topk_weight):
        return moe_experts_forward(x, topk_idx, topk_weight, self.num_experts, self.gate_up_proj, self.down_proj, comfy.ops._swiglu_eager)


class BailingSparseMoe(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate = BailingGate(config, device=device, dtype=dtype, ops=ops)
        self.image_gate = BailingGate(config, device=device, dtype=dtype, ops=ops)
        self.experts = BailingExperts(config, device=device, dtype=dtype, ops=ops)
        self.shared_experts = MLP(config, device=device, dtype=dtype, ops=ops, intermediate_size=config.moe_intermediate_size)

    def forward(self, x, image_mask):
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        text_idx, text_weight = self.gate(flat)
        image_idx, image_weight = self.image_gate(flat)
        mask = image_mask.reshape(-1, 1)
        out = self.experts(flat, torch.where(mask, image_idx, text_idx), torch.where(mask, image_weight, text_weight))
        return out.view(batch, seq_len, hidden) + self.shared_experts(x)


class BailingDecoderLayer(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        self.attention = BailingAttention(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.dense = index == 0
        if self.dense:
            self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        else:
            self.mlp = BailingSparseMoe(config, device=device, dtype=dtype, ops=ops)

    def forward(self, x, attention_mask, freqs_cis, optimized_attention, image_mask):
        x = x + self.attention(self.input_layernorm(x), attention_mask, freqs_cis, optimized_attention)
        h = self.post_attention_layernorm(x)
        if self.dense:
            return x + self.mlp(h)
        return x + self.mlp(h, image_mask)


class BailingMoeV2(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([BailingDecoderLayer(config, i, device=device, dtype=dtype, ops=ops) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def freqs_cis(self, seq_len, blocks, device):
        # video_rope: an image block takes one text position, its tokens form an (h, w) grid centered there
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        for start, gh, gw in blocks:
            t[start + gh * gw:] -= gh * gw - 1
        h = t.clone()
        w = t.clone()
        shift = 0
        for start, gh, gw in blocks:
            end = start + gh * gw
            t[start:end] = start - shift
            h[start:end] = start - shift + (torch.arange(gh, device=device) - (gh - 1) // 2).repeat_interleave(gw)
            w[start:end] = start - shift + (torch.arange(gw, device=device) - (gw - 1) // 2).repeat(gh)
            shift += gh * gw - 1
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.rope_dim, 2, device=device, dtype=torch.float32) / self.config.rope_dim))
        freqs = t[:, None] * inv_freq
        freqs[:, 0:24:2] = h[:, None] * inv_freq[0:24:2]
        freqs[:, 1:24:2] = w[:, None] * inv_freq[1:24:2]
        freqs = freqs.unsqueeze(0)
        return torch.cat((freqs, freqs), dim=-1).cos(), freqs.sin(), -freqs.sin()

    def forward(self, x, attention_mask, blocks, capture_layers):
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis(seq_len, blocks, x.device)
        neg = torch.finfo(x.dtype).min / 4
        mask = torch.full((seq_len, seq_len), neg, dtype=x.dtype, device=x.device).triu_(1)
        mask = mask + (attention_mask == 0).to(x.dtype)[:, None, None, :] * neg
        optimized_attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        image_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for start, gh, gw in blocks:
            image_mask[:, start:start + gh * gw] = True

        captured = []
        for i, layer in enumerate(self.layers):
            if i in capture_layers:
                captured.append(x)
            x = layer(x, mask, freqs_cis, optimized_attention, image_mask)
        x = self.norm(x)
        captured.append(x)
        return x, captured


class MingConnector(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerBlock(config, index=i, device=device, dtype=dtype, ops=ops) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x):
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(self.config.head_dim, position_ids, self.config.rope_theta, device=x.device)
        optimized_attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        for layer in self.layers:
            x, _ = layer(x, attention_mask=None, freqs_cis=freqs_cis, optimized_attention=optimized_attention)
        return self.norm(x)


class MingImageEncoder(nn.Module):
    capture_layers = (5, 12)

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = BailingMoeV2Config(**config_dict)
        connector_config = MingConnectorConfig()
        self.num_layers = config.num_hidden_layers
        self.thinker = BailingMoeV2(config, device=device, dtype=dtype, ops=operations)
        self.vision = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=8192, intermediate_size=3456, num_heads=16, num_layers=32, device=device, dtype=dtype, ops=operations)
        self.linear_proj = nn.Sequential(
            operations.Linear(8192, config.hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(config.hidden_size, config.hidden_size, device=device, dtype=dtype),
        )
        self.connector = MingConnector(connector_config, device=device, dtype=dtype, ops=operations)
        self.query_tokens = nn.Parameter(torch.empty(256, config.hidden_size, device=device, dtype=dtype))
        self.proj_in = operations.Linear(config.hidden_size, connector_config.hidden_size, device=device, dtype=dtype)
        self.proj_out = operations.Linear(connector_config.hidden_size, 2560, device=device, dtype=dtype)
        direct_dim = config.hidden_size * (len(self.capture_layers) + 1)
        self.proj_directvlm = nn.Sequential(
            operations.RMSNorm(direct_dim, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype),
            operations.Linear(direct_dim, 3840, device=device, dtype=dtype),
        )

    def get_input_embeddings(self):
        return self.thinker.embed_tokens

    def preprocess_embed(self, embed, device):
        if embed["type"] == "query":
            return self.query_tokens, None
        if embed["type"] == "image":
            pixels, grid = qwen_vl.process_qwen2vl_images(embed["data"], min_pixels=451584, max_pixels=451584)
            feats = self.vision(pixels.to(device, dtype=torch.float32), grid)
            return F.normalize(self.linear_proj(feats), dim=-1), grid
        return None, None

    def forward(self, embeds, attention_mask, embeds_info):
        blocks = [(e["index"], int(e["extra"][0][1]) // 2, int(e["extra"][0][2]) // 2) if e["type"] == "image" else (e["index"], 1, e["size"]) for e in embeds_info]
        start, _, size = blocks[-1]
        hidden, captured = self.thinker(embeds, attention_mask, blocks, self.capture_layers)
        cap_feats = self.proj_out(self.connector(self.proj_in(hidden[:, start:start + size])))
        direct = torch.cat([c[:, :start - 1] for c in captured], dim=-1)
        return cap_feats, self.proj_directvlm(direct)


class _MingRawTokenizer:
    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            tokenizer_json_bytes = tokenizer_json_bytes.numpy().tobytes()
        if not isinstance(tokenizer_json_bytes, bytes):
            raise ValueError("The Ming-Image text encoder file must contain the tokenizer_json tensor.")
        self.tokenizer = Tokenizer.from_str(tokenizer_json_bytes.decode("utf-8"))

    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        return cls(tokenizer_json_bytes=tokenizer_data, **kwargs)

    def __call__(self, text):
        return {"input_ids": self.tokenizer.encode(text, add_special_tokens=False).ids}

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, skip_special_tokens=kwargs.get("skip_special_tokens", False))


class MingTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer_json_data = tokenizer_data.get("tokenizer_json", None)
        super().__init__(self.tokenizer_json_data, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=2048, embedding_key='ming_image', tokenizer_class=_MingRawTokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=156892, disable_weights=True, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"tokenizer_json": self.tokenizer_json_data}


class MingImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="ming_image", tokenizer=MingTokenizer)
        self.llama_template = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>{}<|role_end|><role>ASSISTANT</role>" + IMAGE_BLOCK

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], **kwargs):
        if llama_template is None:
            llama_template = self.llama_template
            if len(images) > 0:  # reference images lead the user turn, separated by blank lines like the vendor processor
                llama_template = llama_template.replace("{}", "\n\n".join([IMAGE_BLOCK] * len(images)) + "\n{}")
        tokens = super().tokenize_with_weights(llama_template.format(text), return_word_ids=return_word_ids, **kwargs)
        images = iter(images)
        for r in tokens["ming_image"]:
            for i in range(len(r)):
                if r[i][0] == IMAGE_PATCH_TOKEN:
                    image = next(images, None)
                    r[i] = ({"type": "query"} if image is None else {"type": "image", "data": image},) + r[i][1:]
        return tokens


class MingImageClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 156892}, layer_norm_hidden_state=False, model_class=MingImageEncoder, enable_attention_masks=True, return_attention_masks=False, model_options=model_options)

    def forward(self, tokens):
        if self.execution_device is None:
            device = self.transformer.get_input_embeddings().weight.device
        else:
            device = self.execution_device
        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)
        cap_feats, direct_context = self.transformer(embeds, attention_mask, embeds_info)
        return cap_feats.float(), None, {"direct_context": direct_context.float()}


class MingImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="ming_image", clip_model=MingImageClipModel, model_options=model_options)

    def memory_estimation_function(self, tokens, device=None):
        # both expert banks of a MoE layer sit in fp32 while it runs, on top of the weights; images expand to a few hundred tokens each
        config = BailingMoeV2Config()
        banks = 3 * config.hidden_size * config.moe_intermediate_size * config.num_experts * 4
        num_tokens = sum(600 if isinstance(t[0], dict) else 1 for batch in tokens.get("ming_image", []) for t in batch)
        return banks + num_tokens * config.hidden_size * 64


def te(dtype_llama=None, llama_quantization_metadata=None):
    class MingImageTEModel_(MingImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            model_options = model_options.copy()
            if "custom_operations" not in model_options:
                model_options["custom_operations"] = comfy.ops.mixed_precision_ops(llama_quantization_metadata or {}, dtype, full_precision_mm=True)
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MingImageTEModel_
