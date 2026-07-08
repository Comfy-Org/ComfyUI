import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
from comfy.ldm.flux.layers import timestep_embedding

def get_silence_latent(length, device):
    head = torch.tensor([[[ 0.5707,  0.0982,  0.6909, -0.5658,  0.6266,  0.6996, -0.1365, -0.1291,
                        -0.0776, -0.1171, -0.2743, -0.8422, -0.1168,  1.5539, -4.6936,  0.7436,
                        -1.1846, -0.2637,  0.6933, -6.7266,  0.0966, -0.1187, -0.3501, -1.1736,
                        0.0587, -2.0517, -1.3651,  0.7508, -0.2490, -1.3548, -0.1290, -0.7261,
                        1.1132, -0.3249,  0.2337,  0.3004,  0.6605, -0.0298, -0.1989, -0.4041,
                        0.2843, -1.0963, -0.5519,  0.2639, -1.0436, -0.1183,  0.0640,  0.4460,
                        -1.1001, -0.6172, -1.3241,  1.1379,  0.5623, -0.1507, -0.1963, -0.4742,
                        -2.4697,  0.5302,  0.5381,  0.4636, -0.1782, -0.0687,  1.0333,  0.4202],
                        [ 0.3040, -0.1367,  0.6200,  0.0665, -0.0642,  0.4655, -0.1187, -0.0440,
                        0.2941, -0.2753,  0.0173, -0.2421, -0.0147,  1.5603, -2.7025,  0.7907,
                        -0.9736, -0.0682,  0.1294, -5.0707, -0.2167,  0.3302, -0.1513, -0.8100,
                        -0.3894, -0.2884, -0.3149,  0.8660, -0.3817, -1.7061,  0.5824, -0.4840,
                        0.6938,  0.1859,  0.1753,  0.3081,  0.0195,  0.1403, -0.0754, -0.2091,
                        0.1251, -0.1578, -0.4968, -0.1052, -0.4554, -0.0320,  0.1284,  0.4974,
                        -1.1889, -0.0344, -0.8313,  0.2953,  0.5445, -0.6249, -0.1595, -0.0682,
                        -3.1412,  0.0484,  0.4153,  0.8260, -0.1526, -0.0625,  0.5366,  0.8473],
                        [ 5.3524e-02, -1.7534e-01,  5.4443e-01, -4.3501e-01, -2.1317e-03,
                        3.7200e-01, -4.0143e-03, -1.5516e-01, -1.2968e-01, -1.5375e-01,
                        -7.7107e-02, -2.0593e-01, -3.2780e-01,  1.5142e+00, -2.6101e+00,
                        5.8698e-01, -1.2716e+00, -2.4773e-01, -2.7933e-02, -5.0799e+00,
                        1.1601e-01,  4.0987e-01, -2.2030e-02, -6.6495e-01, -2.0995e-01,
                        -6.3474e-01, -1.5893e-01,  8.2745e-01, -2.2992e-01, -1.6816e+00,
                        5.4440e-01, -4.9579e-01,  5.5128e-01,  3.0477e-01,  8.3052e-02,
                        -6.1782e-02,  5.9036e-03,  2.9553e-01, -8.0645e-02, -1.0060e-01,
                        1.9144e-01, -3.8124e-01, -7.2949e-01,  2.4520e-02, -5.0814e-01,
                        2.3977e-01,  9.2943e-02,  3.9256e-01, -1.1993e+00, -3.2752e-01,
                        -7.2707e-01,  2.9476e-01,  4.3542e-01, -8.8597e-01, -4.1686e-01,
                        -8.5390e-02, -2.9018e+00,  6.4988e-02,  5.3945e-01,  9.1988e-01,
                        5.8762e-02, -7.0098e-02,  6.4772e-01,  8.9118e-01],
                        [-3.2225e-02, -1.3195e-01,  5.6411e-01, -5.4766e-01, -5.2170e-03,
                        3.1425e-01, -5.4367e-02, -1.9419e-01, -1.3059e-01, -1.3660e-01,
                        -9.0984e-02, -1.9540e-01, -2.5590e-01,  1.5440e+00, -2.6349e+00,
                        6.8273e-01, -1.2532e+00, -1.9810e-01, -2.2793e-02, -5.0506e+00,
                        1.8818e-01,  5.0109e-01,  7.3546e-03, -6.8771e-01, -3.0676e-01,
                        -7.3257e-01, -1.6687e-01,  9.2232e-01, -1.8987e-01, -1.7267e+00,
                        5.3355e-01, -5.3179e-01,  4.4953e-01,  2.8820e-01,  1.3012e-01,
                        -2.0943e-01, -1.1348e-01,  3.3929e-01, -1.5069e-01, -1.2919e-01,
                        1.8929e-01, -3.6166e-01, -8.0756e-01,  6.6387e-02, -5.8867e-01,
                        1.6978e-01,  1.0134e-01,  3.3877e-01, -1.2133e+00, -3.2492e-01,
                        -8.1237e-01,  3.8101e-01,  4.3765e-01, -8.0596e-01, -4.4531e-01,
                        -4.7513e-02, -2.9266e+00,  1.1741e-03,  4.5123e-01,  9.3075e-01,
                        5.3688e-02, -1.9621e-01,  6.4530e-01,  9.3870e-01]]], device=device).movedim(-1, 1)

    silence_latent = torch.tensor([[[-1.3672e-01, -1.5820e-01,  5.8594e-01, -5.7422e-01,  3.0273e-02,
                                2.7930e-01, -2.5940e-03, -2.0703e-01, -1.6113e-01, -1.4746e-01,
                                -2.7710e-02, -1.8066e-01, -2.9688e-01,  1.6016e+00, -2.6719e+00,
                                7.7734e-01, -1.3516e+00, -1.9434e-01, -7.1289e-02, -5.0938e+00,
                                2.4316e-01,  4.7266e-01,  4.6387e-02, -6.6406e-01, -2.1973e-01,
                                -6.7578e-01, -1.5723e-01,  9.5312e-01, -2.0020e-01, -1.7109e+00,
                                5.8984e-01, -5.7422e-01,  5.1562e-01,  2.8320e-01,  1.4551e-01,
                                -1.8750e-01, -5.9814e-02,  3.6719e-01, -1.0059e-01, -1.5723e-01,
                                2.0605e-01, -4.3359e-01, -8.2812e-01,  4.5654e-02, -6.6016e-01,
                                1.4844e-01,  9.4727e-02,  3.8477e-01, -1.2578e+00, -3.3203e-01,
                                -8.5547e-01,  4.3359e-01,  4.2383e-01, -8.9453e-01, -5.0391e-01,
                                -5.6152e-02, -2.9219e+00, -2.4658e-02,  5.0391e-01,  9.8438e-01,
                                7.2754e-02, -2.1582e-01,  6.3672e-01,  1.0000e+00]]], device=device).movedim(-1, 1).repeat(1, 1, length)
    silence_latent[:, :, :head.shape[-1]] = head
    return silence_latent


def get_layer_class(operations, layer_name):
    if operations is not None and hasattr(operations, layer_name):
        return getattr(operations, layer_name)
    return getattr(nn, layer_name)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype() if dtype is None else dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None, operations=None):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, scale: float = 1000, dtype=None, device=None, operations=None):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True, dtype=dtype, device=device)
        self.act1 = nn.SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True, dtype=dtype, device=device)
        self.in_channels = in_channels
        self.act2 = nn.SiLU()
        self.time_proj = Linear(time_embed_dim, time_embed_dim * 6, dtype=dtype, device=device)
        self.scale = scale

    def forward(self, t, dtype=None):
        t_freq = timestep_embedding(t, self.in_channels, time_factor=self.scale)
        temb = self.linear_1(t_freq.to(dtype=dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).view(-1, 6, temb.shape[-1])
        return temb, timestep_proj

class AceStepAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps=1e-6,
        is_cross_attention=False,
        sliding_window=None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window

        Linear = get_layer_class(operations, "Linear")

        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype, device=device)

        self.q_norm = operations.RMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        position_embeddings=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)
        query_states = query_states.transpose(1, 2)

        if self.is_cross_attention and encoder_hidden_states is not None:
            bsz_enc, kv_len, _ = encoder_hidden_states.size()
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)

            key_states = key_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)
            key_states = self.k_norm(key_states)
            value_states = value_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)

            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
        else:
            kv_len = q_len
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
            key_states = self.k_norm(key_states)
            value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        attn_bias = None
        if self.sliding_window is not None and not self.is_cross_attention:
            indices = torch.arange(q_len, device=query_states.device)
            diff = indices.unsqueeze(1) - indices.unsqueeze(0)
            in_window = torch.abs(diff) <= self.sliding_window

            window_bias = torch.zeros((q_len, kv_len), device=query_states.device, dtype=query_states.dtype)
            min_value = torch.finfo(query_states.dtype).min
            window_bias.masked_fill_(~in_window, min_value)

            window_bias = window_bias.unsqueeze(0).unsqueeze(0)

            if attn_bias is not None:
                if attn_bias.dtype == torch.bool:
                    base_bias = torch.zeros_like(window_bias)
                    base_bias.masked_fill_(~attn_bias, min_value)
                    attn_bias = base_bias + window_bias
                else:
                    attn_bias = attn_bias + window_bias
            else:
                attn_bias = window_bias

        attn_output = optimized_attention(query_states, key_states, value_states, self.num_heads, attn_bias, skip_reshape=True, low_precision_attention=False)
        attn_output = self.o_proj(attn_output)

        return attn_output

class AceStepDiTLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        layer_type="full_attention",
        sliding_window=128,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()

        self_attn_window = sliding_window if layer_type == "sliding_attention" else None

        self.self_attn_norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = AceStepAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
            is_cross_attention=False, sliding_window=self_attn_window,
            dtype=dtype, device=device, operations=operations
        )

        self.cross_attn_norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.cross_attn = AceStepAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
            is_cross_attention=True, dtype=dtype, device=device, operations=operations
        )

        self.mlp_norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden_size, intermediate_size, dtype=dtype, device=device, operations=operations)

        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, hidden_size, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states,
        temb,
        encoder_hidden_states,
        position_embeddings,
        attention_mask=None,
        encoder_attention_mask=None
    ):
        modulation = comfy.model_management.cast_to(self.scale_shift_table, dtype=temb.dtype, device=temb.device) + temb
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation.chunk(6, dim=1)

        norm_hidden = self.self_attn_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa

        attn_out = self.self_attn(
            norm_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + attn_out * gate_msa

        norm_hidden = self.cross_attn_norm(hidden_states)
        attn_out = self.cross_attn(
            norm_hidden,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask
        )
        hidden_states = hidden_states + attn_out

        norm_hidden = self.mlp_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + c_scale_msa) + c_shift_msa

        mlp_out = self.mlp(norm_hidden)
        hidden_states = hidden_states + mlp_out * c_gate_msa

        return hidden_states

class AceStepEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.self_attn = AceStepAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps,
            is_cross_attention=False, dtype=dtype, device=device, operations=operations
        )
        self.input_layernorm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden_size, intermediate_size, dtype=dtype, device=device, operations=operations)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class AceStepLyricEncoder(nn.Module):
    def __init__(
        self,
        text_hidden_dim,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.embed_tokens = Linear(text_hidden_dim, hidden_size, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=1000000.0,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            for _ in range(num_layers)
        ])

    def forward(self, inputs_embeds, attention_mask=None):
        hidden_states = self.embed_tokens(inputs_embeds)
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class AceStepTimbreEncoder(nn.Module):
    def __init__(
        self,
        timbre_hidden_dim,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.embed_tokens = Linear(timbre_hidden_dim, hidden_size, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=1000000.0,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            for _ in range(num_layers)
        ])
        self.special_token = nn.Parameter(torch.empty(1, 1, hidden_size, device=device, dtype=dtype))

    def unpack_timbre_embeddings(self, timbre_embs_packed, refer_audio_order_mask):
        N, d = timbre_embs_packed.shape
        device = timbre_embs_packed.device
        B = N
        counts = torch.bincount(refer_audio_order_mask, minlength=B)
        max_count = counts.max().item()

        sorted_indices = torch.argsort(
            refer_audio_order_mask * N + torch.arange(N, device=device),
            stable=True
        )
        sorted_batch_ids = refer_audio_order_mask[sorted_indices]

        positions = torch.arange(N, device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
        positions_in_sorted = positions - batch_starts[sorted_batch_ids]

        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(N, device=device)
        positions_in_batch = positions_in_sorted[inverse_indices]

        indices_2d = refer_audio_order_mask * max_count + positions_in_batch
        one_hot = F.one_hot(indices_2d, num_classes=B * max_count).to(timbre_embs_packed.dtype)

        timbre_embs_flat = one_hot.t() @ timbre_embs_packed
        timbre_embs_unpack = timbre_embs_flat.view(B, max_count, d)

        mask_flat = (one_hot.sum(dim=0) > 0).long()
        new_mask = mask_flat.view(B, max_count)
        return timbre_embs_unpack, new_mask

    def forward(self, refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask, attention_mask=None):
        hidden_states = self.embed_tokens(refer_audio_acoustic_hidden_states_packed)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask
            )
        hidden_states = self.norm(hidden_states)

        flat_states = hidden_states[:, 0, :]
        unpacked_embs, unpacked_mask = self.unpack_timbre_embeddings(flat_states, refer_audio_order_mask)
        return unpacked_embs, unpacked_mask


def pack_sequences(hidden1, hidden2, mask1, mask2):
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)
    B, L, D = hidden_cat.shape

    if mask1 is not None and mask2 is not None:
        mask_cat = torch.cat([mask1, mask2], dim=1)
        sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
        gather_idx = sort_idx.unsqueeze(-1).expand(B, L, D)
        hidden_sorted = torch.gather(hidden_cat, 1, gather_idx)
        lengths = mask_cat.sum(dim=1)
        new_mask = (torch.arange(L, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1))
    else:
        new_mask = None
        hidden_sorted = hidden_cat

    return hidden_sorted, new_mask

class AceStepConditionEncoder(nn.Module):
    def __init__(
        self,
        text_hidden_dim,
        timbre_hidden_dim,
        hidden_size,
        num_lyric_layers,
        num_timbre_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.text_projector = Linear(text_hidden_dim, hidden_size, bias=False, dtype=dtype, device=device)

        self.lyric_encoder = AceStepLyricEncoder(
            text_hidden_dim=text_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_lyric_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.timbre_encoder = AceStepTimbreEncoder(
            timbre_hidden_dim=timbre_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_timbre_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
            device=device,
            operations=operations
        )

    def forward(
        self,
        text_hidden_states=None,
        text_attention_mask=None,
        lyric_hidden_states=None,
        lyric_attention_mask=None,
        refer_audio_acoustic_hidden_states_packed=None,
        refer_audio_order_mask=None
    ):
        text_emb = self.text_projector(text_hidden_states)

        lyric_emb = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask
        )

        timbre_emb, timbre_mask = self.timbre_encoder(
            refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask
        )

        merged_emb, merged_mask = pack_sequences(lyric_emb, timbre_emb, lyric_attention_mask, timbre_mask)
        final_emb, final_mask = pack_sequences(merged_emb, text_emb, merged_mask, text_attention_mask)

        return final_emb, final_mask

# --------------------------------------------------------------------------------
# Main Diffusion Model (DiT)
# --------------------------------------------------------------------------------

class AceStepDiTModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        patch_size,
        audio_acoustic_hidden_dim,
        condition_dim=None,
        layer_types=None,
        sliding_window=128,
        rms_norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=1000000.0,
            dtype=dtype,
            device=device,
            operations=operations
        )

        Conv1d = get_layer_class(operations, "Conv1d")
        ConvTranspose1d = get_layer_class(operations, "ConvTranspose1d")
        Linear = get_layer_class(operations, "Linear")

        self.proj_in = nn.Sequential(
            nn.Identity(),
            Conv1d(
                in_channels, hidden_size, kernel_size=patch_size, stride=patch_size,
                dtype=dtype, device=device))

        self.time_embed = TimestepEmbedding(256, hidden_size, dtype=dtype, device=device, operations=operations)
        self.time_embed_r = TimestepEmbedding(256, hidden_size, dtype=dtype, device=device, operations=operations)
        self.condition_embedder = Linear(condition_dim, hidden_size, dtype=dtype, device=device)

        if layer_types is None:
            layer_types = ["full_attention"] * num_layers

        if len(layer_types) < num_layers:
            layer_types = list(itertools.islice(itertools.cycle(layer_types), num_layers))

        self.layers = nn.ModuleList([
            AceStepDiTLayer(
                hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps,
                layer_type=layer_types[i],
                sliding_window=sliding_window,
                dtype=dtype, device=device, operations=operations
            ) for i in range(num_layers)
        ])

        self.norm_out = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.proj_out = nn.Sequential(
            nn.Identity(),
            ConvTranspose1d(hidden_size, audio_acoustic_hidden_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)
        )

        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, hidden_size, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents
    ):
        temb_t, proj_t = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_r, proj_r = self.time_embed_r(timestep - timestep_r, dtype=hidden_states.dtype)
        temb = temb_t + temb_r
        timestep_proj = proj_t + proj_r

        x = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = x.shape[1]

        pad_length = 0
        if x.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (x.shape[1] % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_length), mode='constant', value=0)

        x = x.transpose(1, 2)
        x = self.proj_in(x)
        x = x.transpose(1, 2)

        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(x, seq_len=seq_len)

        for layer in self.layers:
            x = layer(
                hidden_states=x,
                temb=timestep_proj,
                encoder_hidden_states=encoder_hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=None,
                encoder_attention_mask=None
            )

        shift, scale = (comfy.model_management.cast_to(self.scale_shift_table, dtype=temb.dtype, device=temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale) + shift

        x = x.transpose(1, 2)
        x = self.proj_out(x)
        x = x.transpose(1, 2)

        x = x[:, :original_seq_len, :]
        return x


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size, num_layers, head_dim, rms_norm_eps, dtype=None, device=None, operations=None):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.embed_tokens = Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype, device=device)
        self.rotary_emb = RotaryEmbedding(head_dim, dtype=dtype, device=device, operations=operations)
        self.special_token = nn.Parameter(torch.empty(1, 1, hidden_size, dtype=dtype, device=device))

        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size, 16, 8, head_dim, hidden_size * 3, rms_norm_eps,
                dtype=dtype, device=device, operations=operations
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, T, P, D = x.shape
        x = self.embed_tokens(x)
        special = comfy.model_management.cast_to(self.special_token, device=x.device, dtype=x.dtype).expand(B, T, 1, -1)
        x = torch.cat([special, x], dim=2)
        x = x.view(B * T, P + 1, D)

        cos, sin = self.rotary_emb(x, seq_len=P + 1)
        for layer in self.layers:
            x = layer(x, (cos, sin))

        x = self.norm(x)
        return x[:, 0, :].view(B, T, D)


class FSQ(nn.Module):
    def __init__(
        self,
        levels,
        dim=None,
        device=None,
        dtype=None,
        operations=None
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.int32, device=device)
        self.register_buffer('_levels', _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32, device=device), dim=0)
        self.register_buffer('_basis', _basis, persistent=False)

        self.codebook_dim = len(levels)
        self.dim = dim if dim is not None else self.codebook_dim

        requires_projection = self.dim != self.codebook_dim
        if requires_projection:
            self.project_in = operations.Linear(self.dim, self.codebook_dim, device=device, dtype=dtype)
            self.project_out = operations.Linear(self.codebook_dim, self.dim, device=device, dtype=dtype)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.codebook_size = self._levels.prod().item()

        indices = torch.arange(self.codebook_size, device=device)
        implicit_codebook = self._indices_to_codes(indices)

        if dtype is not None:
            implicit_codebook = implicit_codebook.to(dtype)

        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

    def bound(self, z):
        levels_minus_1 = (comfy.model_management.cast_to(self._levels, device=z.device, dtype=z.dtype) - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.) + 0.5

        zhat = bracket.floor()
        bracket_ste = bracket + (zhat - bracket).detach()

        return scale * bracket_ste - 1.

    def _indices_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered.float() * (2. / (self._levels.float() - 1)) - 1.

    def codes_to_indices(self, zhat):
        zhat_normalized = (zhat + 1.) / (2. / (comfy.model_management.cast_to(self._levels, device=zhat.device, dtype=zhat.dtype) - 1))
        return (zhat_normalized * comfy.model_management.cast_to(self._basis, device=zhat.device, dtype=zhat.dtype)).sum(dim=-1).round().to(torch.int32)

    def forward(self, z):
        orig_dtype = z.dtype
        z = self.project_in(z)

        codes = self.bound(z)
        indices = self.codes_to_indices(codes)

        out = self.project_out(codes)
        return out.to(orig_dtype), indices


class ResidualFSQ(nn.Module):
    def __init__(
        self,
        levels,
        num_quantizers,
        dim=None,
        bound_hard_clamp=True,
        device=None,
        dtype=None,
        operations=None,
        **kwargs
    ):
        super().__init__()

        codebook_dim = len(levels)
        dim = dim if dim is not None else codebook_dim

        requires_projection = codebook_dim != dim
        if requires_projection:
            self.project_in = operations.Linear(dim, codebook_dim, device=device, dtype=dtype)
            self.project_out = operations.Linear(codebook_dim, dim, device=device, dtype=dtype)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.layers = nn.ModuleList()
        levels_tensor = torch.tensor(levels, device=device)
        scales = []

        for ind in range(num_quantizers):
            scale_val = levels_tensor.float() ** -ind
            scales.append(scale_val)

            self.layers.append(FSQ(
                levels=levels,
                dim=codebook_dim,
                device=device,
                dtype=dtype,
                operations=operations
            ))

        scales_tensor = torch.stack(scales)
        if dtype is not None:
            scales_tensor = scales_tensor.to(dtype)
        self.register_buffer('scales', scales_tensor, persistent=False)

        if bound_hard_clamp:
            val = 1 + (1 / (levels_tensor.float() - 1))
            if dtype is not None:
                val = val.to(dtype)
            self.register_buffer('soft_clamp_input_value', val, persistent=False)

    def get_output_from_indices(self, indices, dtype=torch.float32):
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)

        all_codes = []
        for i, layer in enumerate(self.layers):
            idx = indices[..., i].long()
            codes = F.embedding(idx, comfy.model_management.cast_to(layer.implicit_codebook, device=idx.device, dtype=dtype))
            all_codes.append(codes * comfy.model_management.cast_to(self.scales[i], device=idx.device, dtype=dtype))

        codes_summed = torch.stack(all_codes).sum(dim=0)
        return self.project_out(codes_summed)

    def forward(self, x):
        x = self.project_in(x)

        if hasattr(self, 'soft_clamp_input_value'):
            sc_val = comfy.model_management.cast_to(self.soft_clamp_input_value, device=x.device, dtype=x.dtype)
            x = (x / sc_val).tanh() * sc_val

        quantized_out = torch.tensor(0., device=x.device, dtype=x.dtype)
        residual = x
        all_indices = []

        for layer, scale in zip(self.layers, self.scales):
            scale = comfy.model_management.cast_to(scale, device=x.device, dtype=x.dtype)

            quantized, indices = layer(residual / scale)
            quantized = quantized * scale

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)

        quantized_out = self.project_out(quantized_out)
        all_indices = torch.stack(all_indices, dim=-1)

        return quantized_out, all_indices


class AceStepAudioTokenizer(nn.Module):
    def __init__(
        self,
        audio_acoustic_hidden_dim,
        hidden_size,
        pool_window_size,
        fsq_dim,
        fsq_levels,
        fsq_input_num_quantizers,
        num_layers,
        head_dim,
        rms_norm_eps,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.audio_acoustic_proj = Linear(audio_acoustic_hidden_dim, hidden_size, dtype=dtype, device=device)
        self.attention_pooler = AttentionPooler(
            hidden_size, num_layers, head_dim, rms_norm_eps, dtype=dtype, device=device, operations=operations
        )
        self.pool_window_size = pool_window_size
        self.fsq_dim = fsq_dim
        self.quantizer = ResidualFSQ(
            dim=fsq_dim,
            levels=fsq_levels,
            num_quantizers=fsq_input_num_quantizers,
            bound_hard_clamp=True,
            dtype=dtype, device=device, operations=operations
        )

    def forward(self, hidden_states):
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        quantized, indices = self.quantizer(hidden_states)
        return quantized, indices

    def tokenize(self, x):
        B, T, D = x.shape
        P = self.pool_window_size

        if T % P != 0:
            pad = P - (T % P)
            x = F.pad(x, (0, 0, 0, pad))
            T = x.shape[1]

        T_patch = T // P
        x = x.view(B, T_patch, P, D)

        quantized, indices = self.forward(x)
        return quantized, indices


class AudioTokenDetokenizer(nn.Module):
    def __init__(
        self,
        hidden_size,
        pool_window_size,
        audio_acoustic_hidden_dim,
        num_layers,
        head_dim,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        Linear = get_layer_class(operations, "Linear")
        self.pool_window_size = pool_window_size
        self.embed_tokens = Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        self.special_tokens = nn.Parameter(torch.empty(1, pool_window_size, hidden_size, dtype=dtype, device=device))
        self.rotary_emb = RotaryEmbedding(head_dim, dtype=dtype, device=device, operations=operations)
        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size, 16, 8, head_dim, hidden_size * 3, 1e-6,
                dtype=dtype, device=device, operations=operations
            )
            for _ in range(num_layers)
        ])
        self.norm = operations.RMSNorm(hidden_size, dtype=dtype, device=device)
        self.proj_out = Linear(hidden_size, audio_acoustic_hidden_dim, dtype=dtype, device=device)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embed_tokens(x)
        x = x.unsqueeze(2).repeat(1, 1, self.pool_window_size, 1)
        x = x + comfy.model_management.cast_to(self.special_tokens.expand(B, T, -1, -1), device=x.device, dtype=x.dtype)
        x = x.view(B * T, self.pool_window_size, D)

        cos, sin = self.rotary_emb(x, seq_len=self.pool_window_size)
        for layer in self.layers:
            x = layer(x, (cos, sin))

        x = self.norm(x)
        x = self.proj_out(x)
        return x.view(B, T * self.pool_window_size, -1)


class AceStepConditionGenerationModel(nn.Module):
    def __init__(
        self,
        in_channels=192,
        hidden_size=2048,
        text_hidden_dim=1024,
        timbre_hidden_dim=64,
        audio_acoustic_hidden_dim=64,
        num_dit_layers=24,
        num_lyric_layers=8,
        num_timbre_layers=4,
        num_tokenizer_layers=2,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=6144,
        patch_size=2,
        pool_window_size=5,
        rms_norm_eps=1e-06,
        timestep_mu=-0.4,
        timestep_sigma=1.0,
        data_proportion=0.5,
        sliding_window=128,
        layer_types=None,
        fsq_dim=2048,
        fsq_levels=[8, 8, 8, 5, 5, 5],
        fsq_input_num_quantizers=1,
        encoder_hidden_size=2048,
        encoder_intermediate_size=6144,
        encoder_num_heads=16,
        audio_model=None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dtype = dtype
        self.timestep_mu = timestep_mu
        self.timestep_sigma = timestep_sigma
        self.data_proportion = data_proportion
        self.pool_window_size = pool_window_size

        if layer_types is None:
            layer_types = []
            for i in range(num_dit_layers):
                layer_types.append("sliding_attention" if i % 2 == 0 else "full_attention")

        self.decoder = AceStepDiTModel(
            in_channels, hidden_size, num_dit_layers, num_heads, num_kv_heads, head_dim,
            intermediate_size, patch_size, audio_acoustic_hidden_dim, condition_dim=encoder_hidden_size,
            layer_types=layer_types, sliding_window=sliding_window, rms_norm_eps=rms_norm_eps,
            dtype=dtype, device=device, operations=operations
        )
        self.encoder = AceStepConditionEncoder(
            text_hidden_dim, timbre_hidden_dim, encoder_hidden_size, num_lyric_layers, num_timbre_layers,
            encoder_num_heads, num_kv_heads, head_dim, encoder_intermediate_size, rms_norm_eps,
            dtype=dtype, device=device, operations=operations
        )
        self.tokenizer = AceStepAudioTokenizer(
            audio_acoustic_hidden_dim, encoder_hidden_size, pool_window_size, fsq_dim=fsq_dim, fsq_levels=fsq_levels, fsq_input_num_quantizers=fsq_input_num_quantizers, num_layers=num_tokenizer_layers, head_dim=head_dim, rms_norm_eps=rms_norm_eps,
            dtype=dtype, device=device, operations=operations
        )
        self.detokenizer = AudioTokenDetokenizer(
            encoder_hidden_size, pool_window_size, audio_acoustic_hidden_dim, num_layers=2, head_dim=head_dim,
            dtype=dtype, device=device, operations=operations
        )
        self.null_condition_emb = nn.Parameter(torch.empty(1, 1, encoder_hidden_size, dtype=dtype, device=device))

    def prepare_condition(
        self,
        text_hidden_states, text_attention_mask,
        lyric_hidden_states, lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask,
        src_latents, chunk_masks, is_covers,
        precomputed_lm_hints_25Hz=None,
        audio_codes=None
    ):
        encoder_hidden, encoder_mask = self.encoder(
            text_hidden_states, text_attention_mask,
            lyric_hidden_states, lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask
        )

        if precomputed_lm_hints_25Hz is not None:
            lm_hints = precomputed_lm_hints_25Hz
        else:
            if audio_codes is not None:
                if audio_codes.shape[1] * 5 < src_latents.shape[1]:
                    audio_codes = torch.nn.functional.pad(audio_codes, (0, math.ceil(src_latents.shape[1] / 5) - audio_codes.shape[1]), "constant", 35847)
                lm_hints_5Hz = self.tokenizer.quantizer.get_output_from_indices(audio_codes, dtype=text_hidden_states.dtype)
            else:
                lm_hints_5Hz, indices = self.tokenizer.tokenize(refer_audio_acoustic_hidden_states_packed)

            lm_hints = self.detokenizer(lm_hints_5Hz)

        lm_hints = lm_hints[:, :src_latents.shape[1], :]
        if is_covers is None or is_covers is True:
            src_latents = lm_hints
        elif is_covers is False:
            src_latents = refer_audio_acoustic_hidden_states_packed

        context_latents = torch.cat([src_latents, chunk_masks.to(src_latents.dtype)], dim=-1)

        return encoder_hidden, encoder_mask, context_latents

    def forward(self, x, timestep, context, lyric_embed=None, refer_audio=None, audio_codes=None, is_covers=None, replace_with_null_embeds=False, **kwargs):
        text_attention_mask = None
        lyric_attention_mask = None
        refer_audio_order_mask = None
        attention_mask = None
        chunk_masks = None
        src_latents = None
        precomputed_lm_hints_25Hz = None
        lyric_hidden_states = lyric_embed
        text_hidden_states = context
        refer_audio_acoustic_hidden_states_packed = refer_audio.movedim(-1, -2)

        x = x.movedim(-1, -2)

        if refer_audio_order_mask is None:
            refer_audio_order_mask = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)

        if src_latents is None:
            src_latents = x

        if chunk_masks is None:
            chunk_masks = torch.ones_like(x)

        enc_hidden, enc_mask, context_latents = self.prepare_condition(
            text_hidden_states, text_attention_mask,
            lyric_hidden_states, lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask,
            src_latents, chunk_masks, is_covers, precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz, audio_codes=audio_codes
        )

        if replace_with_null_embeds:
            enc_hidden[:] = self.null_condition_emb.to(enc_hidden)

        out = self.decoder(hidden_states=x,
                           timestep=timestep,
                           timestep_r=timestep,
                           attention_mask=attention_mask,
                           encoder_hidden_states=enc_hidden,
                           encoder_attention_mask=enc_mask,
                           context_latents=context_latents
                           )

        return out.movedim(-1, -2)
