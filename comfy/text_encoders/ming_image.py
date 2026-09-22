from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import comfy.ops
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import MLP, RMSNorm, TransformerBlock, apply_rope, moe_experts_forward, precompute_freqs_cis

IMAGE_PATCH_TOKEN = 157157


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
        qkv = self.query_key_value(x).view(batch, seq_len, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        xq, xk, xv = qkv.split((self.num_heads, self.num_kv_heads, self.num_kv_heads), dim=2)
        xq = self.q_norm(xq.transpose(1, 2))
        xk = self.k_norm(xk.transpose(1, 2))
        xv = xv.transpose(1, 2)

        q_rot, k_rot = apply_rope(xq[..., :self.rope_dim].contiguous(), xk[..., :self.rope_dim].contiguous(), freqs_cis)
        xq = torch.cat((q_rot, xq[..., self.rope_dim:]), dim=-1)
        xk = torch.cat((k_rot, xk[..., self.rope_dim:]), dim=-1)
        out = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, enable_gqa=True)
        return self.dense(out)


class BailingGate(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.proj = ops.Linear(config.hidden_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.expert_bias = nn.Parameter(torch.empty(config.num_experts, device=device, dtype=torch.float32))

    def forward(self, x):
        scores = torch.sigmoid(self.proj(x.float()))
        routing = scores + comfy.ops.cast_to_input(self.expert_bias, scores, copy=False)

        num_tokens = routing.shape[0]
        group_scores = routing.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, self.n_group, self.num_experts // self.n_group).reshape(num_tokens, -1)
        topk_idx = torch.topk(routing.masked_fill(~score_mask.bool(), float("-inf")), k=self.top_k, dim=-1, sorted=False)[1]

        topk_weight = torch.gather(scores, dim=1, index=topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


def swiglu(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


class BailingExperts(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_proj = ops.MoEExperts(num_experts=config.num_experts, in_features=config.hidden_size, out_features=2 * config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts=config.num_experts, in_features=config.moe_intermediate_size, out_features=config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, topk_idx, topk_weight):
        return moe_experts_forward(x, topk_idx, topk_weight, self.num_experts, self.gate_up_proj, self.down_proj, swiglu)


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
        topk_idx = torch.where(mask, image_idx, text_idx)
        topk_weight = torch.where(mask, image_weight, text_weight)
        out = self.experts(flat, topk_idx, topk_weight).view(batch, seq_len, hidden)
        return out + self.shared_experts(x)


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

    def freqs_cis(self, seq_len, start, size, device):
        # video_rope: the query block is a 1 x size image grid, text before and after is 1-D
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        after = start + 1 + pos[:seq_len - start - size]
        t = torch.cat((pos[:start], torch.full((size,), float(start), device=device), after))
        w = torch.cat((pos[:start], start + pos[:size] - (size - 1) // 2, after))

        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.rope_dim, 2, device=device, dtype=torch.float32) / self.config.rope_dim))
        freqs_t = t[:, None] * inv_freq
        freqs_w = w[:, None] * inv_freq
        freqs = freqs_t.clone()
        freqs[:, 1:24:2] = freqs_w[:, 1:24:2]
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        half = emb.shape[-1] // 2
        sin = emb.sin()
        return emb.cos(), sin[..., :half], -sin[..., half:]

    def forward(self, embeds, attention_mask, start, size, capture_layers):
        x = embeds
        seq_len = x.shape[1]
        device = x.device
        freqs_cis = self.freqs_cis(seq_len, start, size, device)

        mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        if attention_mask is not None:
            pad = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, 1, seq_len)).expand(-1, 1, seq_len, seq_len)
            mask = mask + pad.masked_fill(pad.to(torch.bool), torch.finfo(x.dtype).min / 4)
        optimized_attention = optimized_attention_for_device(device, mask=True, small_input=True)

        image_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)
        image_mask[:, start:start + size] = True

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
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.thinker = BailingMoeV2(config, device=device, dtype=dtype, ops=operations)
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
        return None, None

    def forward(self, embeds, attention_mask, embeds_info):
        query = next(e for e in embeds_info if e["type"] == "query")
        start, size = query["index"], query["size"]
        hidden, captured = self.thinker(embeds, attention_mask, start, size, self.capture_layers)

        cap_feats = self.connector(self.proj_in(hidden[:, start:start + size]))
        cap_feats = F.normalize(self.proj_out(cap_feats), dim=-1)

        direct = torch.cat([c[:, :start - 1] for c in captured], dim=-1)
        return cap_feats, self.proj_directvlm(direct)


class _MingRawTokenizer:
    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            tokenizer_json_bytes = bytes(tokenizer_json_bytes.tolist())
        self.tokenizer = Tokenizer.from_str(tokenizer_json_bytes.decode("utf-8"))

    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        return cls(tokenizer_json_bytes=tokenizer_data, **kwargs)

    def __call__(self, text):
        return {"input_ids": self.tokenizer.encode(text, add_special_tokens=False).ids}

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(t) for t in tokens]

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, skip_special_tokens=kwargs.get("skip_special_tokens", False))


class MingTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json", None)
        self.tokenizer_json_data = tokenizer_json
        super().__init__(tokenizer_json, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=2048, embedding_key='ming_image', tokenizer_class=_MingRawTokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=156892, disable_weights=True, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"tokenizer_json": self.tokenizer_json_data}


class MingImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="ming_image", tokenizer=MingTokenizer)
        self.llama_template = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>{}<|role_end|><role>ASSISTANT</role><image><imagePatch></image>"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if llama_template is None:
            llama_template = self.llama_template
        tokens = super().tokenize_with_weights(llama_template.format(text), return_word_ids=return_word_ids, **kwargs)
        for r in tokens["ming_image"]:
            for i in range(len(r)):
                if r[i][0] == IMAGE_PATCH_TOKEN:
                    r[i] = ({"type": "query"},) + r[i][1:]
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
