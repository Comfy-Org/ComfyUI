import torch
import torch.nn as nn
from comfy import sd1_clip
from comfy.text_encoders.llama import Attention as LlamaAttention, RMSNorm, MLP, precompute_freqs_cis, apply_rope, _make_scaled_embedding
from comfy.text_encoders.spiece_tokenizer import SPieceTokenizer


class T5GemmaEncoderConfig:
    def __init__(self):
        self.vocab_size = 256000
        self.hidden_size = 768
        self.intermediate_size = 2048
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.num_key_value_heads = 12
        self.head_dim = 64
        self.rms_norm_eps = 1e-6
        self.rms_norm_add = False
        self.rope_theta = 10000.0
        self.attn_logit_softcapping = 50.0
        self.query_pre_attn_scalar = 64
        self.sliding_window = 4096
        self.mlp_activation = "gelu_pytorch_tanh"
        self.layer_types = ["sliding_attention", "full_attention"] * 6
        self.qkv_bias = False
        self.q_norm = None
        self.k_norm = None
        self.rms_norm_add = True


class T5GemmaAttention(LlamaAttention):
    """Reuses LlamaAttention projection setup; overrides forward for softcap attention.

    T5Gemma applies tanh(QK^T * scale / cap) * cap between the matmul and softmax.
    This nonlinearity is incompatible with fused SDPA kernels, so attention is
    computed manually. Everything else (projections, RoPE, GQA expansion) is identical
    to LlamaAttention so __init__ is inherited unchanged.
    """

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.scale = config.query_pre_attn_scalar ** -0.5
        self.softcap = config.attn_logit_softcapping

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None, **kwargs):
        B, S, _ = hidden_states.shape
        xq = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xq, xk = apply_rope(xq, xk, freqs_cis)
        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        attn = torch.matmul(xq * self.scale, xk.transpose(-2, -1))
        attn = torch.tanh(attn / self.softcap) * self.softcap
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = torch.nn.functional.softmax(attn.float(), dim=-1).to(xq.dtype)
        out = torch.matmul(attn, xv).transpose(1, 2).reshape(B, S, self.inner_size)
        return self.o_proj(out), None


class T5GemmaBlock(nn.Module):
    def __init__(self, config, layer_type, device=None, dtype=None, ops=None):
        super().__init__()
        self.self_attn = T5GemmaAttention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        # Names match checkpoint keys: model.encoder.layers.X.<name>.weight
        self.pre_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.post_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.is_sliding = (layer_type == "sliding_attention")
        self.sliding_window = config.sliding_window

    def forward(self, x, attention_mask=None, freqs_cis=None):
        attn_mask = attention_mask
        if self.is_sliding and x.shape[1] > self.sliding_window:
            S = x.shape[1]
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            sw_mask = torch.zeros(S, S, dtype=x.dtype, device=x.device)
            sw_mask.masked_fill_(dist > self.sliding_window, -torch.finfo(x.dtype).max)
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = (attention_mask + sw_mask) if attention_mask is not None else sw_mask
        residual = x
        x = self.pre_self_attn_layernorm(x)
        x, _ = self.self_attn(x, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = self.post_self_attn_layernorm(x)
        x = residual + x
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class T5GemmaEncoder(nn.Module):
    """Encoder stack: embed_tokens, layers, norm.
    Keys: embed_tokens.*, layers.X.*, norm.*"""

    def __init__(self, config, device, dtype, ops):
        super().__init__()
        self.config = config
        # Gemma-style scaled embedding: output *= sqrt(hidden_size)
        self.embed_tokens = _make_scaled_embedding(
            ops, config.vocab_size, config.hidden_size, config.hidden_size ** 0.5, device, dtype)
        self.layers = nn.ModuleList([
            T5GemmaBlock(config, config.layer_types[i], device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, device=device, dtype=dtype)

    def forward(self, input_ids, attention_mask=None, embeds=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, num_layers=None):
        x = embeds if embeds is not None else self.embed_tokens(input_ids, out_dtype=dtype or torch.float32)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(self.config.head_dim, position_ids, self.config.rope_theta, device=x.device)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=mask, freqs_cis=freqs_cis)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)
        return x, intermediate


class T5GemmaBody(nn.Module):
    """Provides the 'encoder' sub-module.
    Keys: encoder.*"""

    def __init__(self, config, device, dtype, ops):
        super().__init__()
        self.encoder = T5GemmaEncoder(config, device, dtype, ops)


class T5GemmaModel(nn.Module):
    """Top-level model class passed to SDClipModel as model_class.
    Module layout: self.model.encoder.* → matches checkpoint keys model.encoder.*"""

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = T5GemmaEncoderConfig()
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype
        self.model = T5GemmaBody(config, device, dtype, operations)

    def get_input_embeddings(self):
        return self.model.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.encoder.embed_tokens = embeddings

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, **kwargs):
        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output = self.num_layers + intermediate_output
        return self.model.encoder(
            input_ids, attention_mask=attention_mask, embeds=embeds,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype, num_layers=self.num_layers)


class T5GemmaSDClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx,
                         textmodel_json_config={}, dtype=dtype,
                         special_tokens={"pad": 0},
                         model_class=T5GemmaModel,
                         enable_attention_masks=True, zero_out_masked=True,
                         model_options=model_options)


class T5GemmaSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_model = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer_model, pad_with_end=False, embedding_size=768,
                         embedding_key="t5gemma", tokenizer_class=SPieceTokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False,
                         max_length=99999999, min_length=1, pad_token=0,
                         tokenizer_data=tokenizer_data,
                         tokenizer_args={"add_bos": False, "add_eos": False})

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class SAT5GemmaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory,
                         tokenizer_data=tokenizer_data, clip_name="t5gemma", tokenizer=T5GemmaSDTokenizer)


class SAT5GemmaModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options,
                         name="t5gemma", clip_model=T5GemmaSDClipModel, **kwargs)
