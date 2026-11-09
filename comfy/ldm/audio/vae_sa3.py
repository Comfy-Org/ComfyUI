import torch
import torch.nn as nn

import comfy.ops
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.audio.autoencoder import WNConv1d

ops = comfy.ops.disable_weight_init

class Transpose(nn.Module):
    def forward(self, x, **kwargs):
        return x.transpose(-2, -1)


def _zero_pad_modulo_sequence(x, size, dim=-2):
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
    return x


def _sliding_window_mask(seq_len, window, device, dtype):
    """Additive attention mask enforcing a ±window local window (matches flash_attn window_size)."""
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    out_of_window = (j - i).abs() > window
    return torch.where(
        out_of_window,
        torch.full((1,), torch.finfo(dtype).min / 4, device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=dtype),
    )


class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, dtype=None, device=None, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, dtype=dtype, device=device))
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        alpha = comfy.ops.cast_to_input(self.alpha, x)
        gamma = comfy.ops.cast_to_input(self.gamma, x)
        beta = comfy.ops.cast_to_input(self.beta, x)
        return gamma * torch.tanh(alpha * x) + beta


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, base_rescale_factor=1., dtype=None, device=None):
        super().__init__()
        base = base * base_rescale_factor ** (dim / (dim - 2))
        self.register_buffer("inv_freq", torch.empty(dim // 2, dtype=dtype, device=device))

    def forward_from_seq_len(self, seq_len, device, dtype=None):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        return self.forward(t)

    def forward(self, t):
        freqs = torch.outer(t.float(), comfy.model_management.cast_to(self.inv_freq, dtype=torch.float32, device=t.device))
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rotary_pos_emb(t, freqs):
    out_dtype = t.dtype
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * freqs.cos() + _rotate_half(t_rot) * freqs.sin()
    return torch.cat((t_rot.to(out_dtype), t_pass.to(out_dtype)), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, dim_heads=64, qk_norm="none", qk_norm_eps=1e-6,
                 differential=False, zero_init_output=True,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.num_heads = dim // dim_heads
        self.differential = differential
        self.qk_norm = qk_norm

        self.to_qkv = operations.Linear(
            dim, dim * (5 if differential else 3), bias=False, dtype=dtype, device=device)
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        if qk_norm == "dyt":
            self.q_norm = DynamicTanh(dim_heads, dtype=dtype, device=device)
            self.k_norm = DynamicTanh(dim_heads, dtype=dtype, device=device)
        elif qk_norm == "rms":
            self.q_norm = operations.RMSNorm(dim_heads, eps=qk_norm_eps, dtype=dtype, device=device)
            self.k_norm = operations.RMSNorm(dim_heads, eps=qk_norm_eps, dtype=dtype, device=device)

    def forward(self, x, rotary_pos_emb=None, mask=None, **kwargs):
        B, N, _ = x.shape
        h = self.num_heads

        qkv = self.to_qkv(x)
        if self.differential:
            q, k, v, q_diff, k_diff = qkv.chunk(5, dim=-1)
            del qkv
            q = q.view(B, N, h, -1).transpose(1, 2)
            k = k.view(B, N, h, -1).transpose(1, 2)
            v = v.view(B, N, h, -1).transpose(1, 2)
            q_diff = q_diff.view(B, N, h, -1).transpose(1, 2)
            k_diff = k_diff.view(B, N, h, -1).transpose(1, 2)
        else:
            q, k, v = qkv.chunk(3, dim=-1)
            del qkv
            q = q.view(B, N, h, -1).transpose(1, 2)
            k = k.view(B, N, h, -1).transpose(1, 2)
            v = v.view(B, N, h, -1).transpose(1, 2)

        if self.qk_norm != "none":
            q_dtype, k_dtype = q.dtype, k.dtype
            q = self.q_norm(q).to(q_dtype)
            k = self.k_norm(k).to(k_dtype)
            if self.differential:
                q_diff = self.q_norm(q_diff).to(q_dtype)
                k_diff = self.k_norm(k_diff).to(k_dtype)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype, k_dtype = q.dtype, k.dtype
            q = _apply_rotary_pos_emb(q.float(), freqs).to(q_dtype)
            k = _apply_rotary_pos_emb(k.float(), freqs).to(k_dtype)
            if self.differential:
                q_diff = _apply_rotary_pos_emb(q_diff.float(), freqs).to(q_dtype)
                k_diff = _apply_rotary_pos_emb(k_diff.float(), freqs).to(k_dtype)

        if self.differential:
            out = (optimized_attention(q, k, v, h, mask=mask, skip_reshape=True, low_precision_attention=False)
                   - optimized_attention(q_diff, k_diff, v, h, mask=mask, skip_reshape=True, low_precision_attention=False))
            del q, k, v, q_diff, k_diff
        else:
            out = optimized_attention(q, k, v, h, mask=mask, skip_reshape=True, low_precision_attention=False)
            del q, k, v

        return self.to_out(out)


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(3.14159265359 * x)


class _GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation, dtype=None, device=None, operations=None):
        super().__init__()
        self.act = activation
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, no_bias=False, zero_init_output=True,
                 sinusoidal=False, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        act = _Sin() if sinusoidal else nn.SiLU()
        self.ff = nn.Sequential(
            _GLU(dim, inner_dim, act, dtype=dtype, device=device, operations=operations),
            nn.Identity(),
            operations.Linear(inner_dim, dim, bias=not no_bias, dtype=dtype, device=device),
            nn.Identity(),
        )

    def forward(self, x, **kwargs):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_heads=64, causal=False, zero_init_branch_outputs=True,
                 norm_type="dyt", add_rope=False, attn_kwargs=None, ff_kwargs=None,
                 norm_kwargs=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}
        if ff_kwargs is None:
            ff_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        dim_heads = min(dim_heads, dim)

        Norm = DynamicTanh if norm_type == "dyt" else operations.RMSNorm
        norm_kw = {**norm_kwargs, "dtype": dtype, "device": device}

        self.pre_norm = Norm(dim, **norm_kw)
        self.self_attn = Attention(dim, dim_heads=dim_heads,
                                   zero_init_output=zero_init_branch_outputs,
                                   dtype=dtype, device=device, operations=operations,
                                   **attn_kwargs)
        self.ff_norm = Norm(dim, **norm_kw)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs,
                              dtype=dtype, device=device, operations=operations, **ff_kwargs)
        self.rope = RotaryEmbedding(dim_heads // 2, dtype=dtype, device=device) if add_rope else None

    def forward(self, x, mask=None, **kwargs):
        rope = self.rope.forward_from_seq_len(x.shape[-2], device=x.device) \
               if self.rope is not None else None
        x = x + self.self_attn(self.pre_norm(x), rotary_pos_emb=rope, mask=mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerResamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, type="encoder",
                 transformer_depth=3, dim_heads=128, differential=True,
                 sliding_window=None, chunk_size=128, chunk_midpoint_shift=False,
                 dyt=True, ff_mult=3, mapping_bias=True, variable_stride=False,
                 sinusoidal_blocks=0, conv_mapping=False, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        if type not in ("encoder", "decoder"):
            raise ValueError(f"type must be 'encoder' or 'decoder', got {type!r}")

        self.type = type
        self.stride = stride
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.variable_stride = variable_stride
        self.transformer_depth = transformer_depth

        transformer_dim = out_channels if type == "encoder" else in_channels

        self.mapping = (WNConv1d(in_channels, out_channels, 3 if conv_mapping else 1, padding="same", bias=mapping_bias)
                        if in_channels != out_channels else nn.Identity())

        self.sliding_window_latents = sliding_window
        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(stride)

        token_seq = 1 if variable_stride else self.output_seg_size
        self.new_tokens = nn.Parameter(torch.empty(1, token_seq, transformer_dim, dtype=dtype, device=device))

        norm_type = "dyt" if dyt else "rms_norm"
        attn_kwargs = {"qk_norm": "dyt" if dyt else "rms", "qk_norm_eps": 1e-3,
                       "differential": differential}
        norm_kwargs = {"eps": 1e-3}
        transformers = []
        for i in range(transformer_depth):
            sinusoidal = (transformer_depth - i) < sinusoidal_blocks
            transformers.append(TransformerBlock(
                transformer_dim,
                dim_heads=dim_heads,
                causal=False,
                zero_init_branch_outputs=True,
                norm_type=norm_type,
                add_rope=True,
                attn_kwargs=attn_kwargs,
                ff_kwargs={"mult": ff_mult, "no_bias": False, "sinusoidal": sinusoidal},
                norm_kwargs=norm_kwargs,
                dtype=dtype, device=device, operations=operations,
            ))
        self.transformers = nn.ModuleList(transformers)

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [w * (stride + 1 + prepend_cond_length) for w in window]

    def _get_seg_sizes(self, stride, prepend_cond_length=0):
        sub_chunk_size = stride + 1 + prepend_cond_length
        input_seg_size = stride if self.type == "encoder" else 1
        output_seg_size = 1 if self.type == "encoder" else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    def forward(self, x, stride=None, **kwargs):
        B = x.shape[0]

        if stride is None:
            input_seg = self.input_seg_size
            output_seg = self.output_seg_size
            sub_chunk = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            input_seg, output_seg, sub_chunk = self._get_seg_sizes(stride)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride)

        if self.type == "encoder":
            if self.transformer_depth > 0:
                pad_mod = self.chunk_size if sliding_window is None else input_seg
                x = _zero_pad_modulo_sequence(x, pad_mod, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = x.permute(0, 2, 1)

            if self.type != "encoder":
                pad_mod = 1 if sliding_window is not None else (
                    self.chunk_size // (stride if stride is not None else self.stride))
                x = _zero_pad_modulo_sequence(x, pad_mod)

            C = x.shape[2]
            x = x.reshape(-1, input_seg, C)

            new_tokens = self.new_tokens.expand(x.shape[0], output_seg, -1)
            x = torch.cat([x, comfy.ops.cast_to_input(new_tokens, x)], dim=-2)
            del new_tokens

            x = x.reshape(B, -1, C)

            if sliding_window is None:
                eff_chunk = self.chunk_size + self.chunk_size // (stride if stride is not None else self.stride)

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = eff_chunk // 2

                x = x.reshape(-1, eff_chunk, C)
                for layer in self.transformers[:split]:
                    x = layer(x)
                x = x.reshape(B, -1, C)

                shifted = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                del x
                x = shifted.reshape(-1, eff_chunk, C)
                del shifted
                for layer in self.transformers[split:]:
                    x = layer(x)
                x = x.reshape(B, -1, C)
                x = x[:, shift:-shift, :]
            elif sliding_window is None:
                x = x.reshape(-1, eff_chunk, C)
                for layer in self.transformers:
                    x = layer(x)
                x = x.reshape(B, -1, C)
            else:
                attn_mask = _sliding_window_mask(x.shape[1], sliding_window[0], x.device, x.dtype)
                for layer in self.transformers:
                    x = layer(x, mask=attn_mask)

            x = x.reshape(-1, sub_chunk, C)
            x = x[:, -output_seg:, :]
            x = x.reshape(B, -1, C).transpose(1, 2)

        if self.type == "decoder":
            x = self.mapping(x)

        return x


class SAMEEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=128, latent_dim=32,
                 c_mults=(1, 2, 4, 8), strides=(2, 4, 8, 8),
                 transformer_depths=(3, 3, 3, 3),
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        channel_dims = [in_channels] + [channels * c for c in c_mults]
        layers = []
        for i in range(len(c_mults)):
            layers.append(TransformerResamplingBlock(
                in_channels=channel_dims[i], out_channels=channel_dims[i + 1],
                stride=strides[i], type="encoder",
                transformer_depth=transformer_depths[i],
                dtype=dtype, device=device, operations=operations, **kwargs))
        layers += [
            Transpose(),
            operations.Linear(channel_dims[-1], latent_dim, dtype=dtype, device=device),
            Transpose(),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x


class SAMEDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=128, latent_dim=32,
                 c_mults=(1, 2, 4, 8), strides=(2, 4, 8, 8),
                 transformer_depths=(3, 3, 3, 3), sinusoidal_blocks=None,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        if sinusoidal_blocks is None:
            sinusoidal_blocks = [0] * len(c_mults)
        channel_dims = [out_channels] + [channels * c for c in c_mults]
        layers = [
            Transpose(),
            operations.Linear(latent_dim, channel_dims[-1], dtype=dtype, device=device),
            Transpose(),
        ]
        for i in range(len(c_mults) - 1, -1, -1):
            layers.append(TransformerResamplingBlock(
                in_channels=channel_dims[i + 1], out_channels=channel_dims[i],
                stride=strides[i], type="decoder",
                transformer_depth=transformer_depths[i],
                sinusoidal_blocks=sinusoidal_blocks[i],
                dtype=dtype, device=device, operations=operations, **kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x


class SoftNormBottleneck(nn.Module):
    def __init__(self, dim=32, noise_augment_dim=0, noise_regularize=False,
                 auto_scale=False, freeze=False, dtype=None, device=None, **kwargs):
        super().__init__()
        self.noise_augment_dim = noise_augment_dim
        self.noise_regularize = noise_regularize
        self.scaling_factor = nn.Parameter(torch.empty(1, dim, 1, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(1, dim, 1, dtype=dtype, device=device))
        self.noise_scaling_factor = nn.Parameter(torch.empty(1, noise_augment_dim, 1, dtype=dtype, device=device))
        if auto_scale:
            self.register_parameter("running_std", nn.Parameter(
                torch.empty(1, dtype=dtype, device=device), requires_grad=False))
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def encode(self, x, return_info=False, **kwargs):
        x = x * comfy.ops.cast_to_input(self.scaling_factor, x) \
              + comfy.ops.cast_to_input(self.bias, x)
        if hasattr(self, "running_std"):
            x = x / comfy.ops.cast_to_input(self.running_std, x)
        if return_info:
            return x, {}
        return x

    def decode(self, x, **kwargs):
        if hasattr(self, "running_std"):
            x = x * comfy.ops.cast_to_input(self.running_std, x)
        if self.noise_regularize:
            scaling = self.running_std if hasattr(self, "running_std") \
                      else x.std(dim=-1, keepdim=True)
            noise = torch.randn_like(x) * comfy.ops.cast_to_input(scaling, x) * 1e-3
            x = x + noise
        if self.noise_augment_dim > 0:
            noise = comfy.ops.cast_to_input(self.noise_scaling_factor, x) * torch.randn(
                x.shape[0], self.noise_augment_dim, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, noise], dim=1)
        return x


class PatchedPretransform(nn.Module):
    def __init__(self, channels, patch_size, **kwargs):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.enable_grad = False

    def _pad(self, x):
        pad_len = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x

    def encode(self, x):
        x = self._pad(x)
        B, C, T = x.shape
        h = self.patch_size
        L = T // h
        # b c (l h) -> b (c h) l
        return x.reshape(B, C, L, h).permute(0, 1, 3, 2).reshape(B, C * h, L)

    def decode(self, x):
        B, Ch, L = x.shape
        h = self.patch_size
        C = Ch // h
        # b (c h) l -> b c (l h)
        return x.reshape(B, C, h, L).permute(0, 1, 3, 2).reshape(B, C, L * h)


class SA3AudioVAE(nn.Module):
    """SA3 VAE. State dict keys match checkpoint after stripping 'pretransform.model.'"""

    def __init__(self, channels=256, transformer_depths=12, sinusoidal_blocks=8,
                 sliding_window=None, decoder_conv_mapping=False,
                 chunk_size=128, chunk_midpoint_shift=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            operations = ops

        self.pretransform = PatchedPretransform(channels=2, patch_size=256)

        common_kwargs = dict(
            differential=True, dyt=True, dim_heads=64,
            sliding_window=sliding_window, variable_stride=True,
            chunk_size=chunk_size, chunk_midpoint_shift=chunk_midpoint_shift,
            dtype=dtype, device=device, operations=operations,
        )
        self.encoder = SAMEEncoder(
            in_channels=512, channels=channels, c_mults=[6], strides=[16],
            latent_dim=256, transformer_depths=[transformer_depths],
            conv_mapping=False, **common_kwargs,
        )
        self.decoder = SAMEDecoder(
            out_channels=512, channels=channels, c_mults=[6], strides=[16],
            latent_dim=256, transformer_depths=[transformer_depths], sinusoidal_blocks=[sinusoidal_blocks],
            conv_mapping=decoder_conv_mapping, **common_kwargs,
        )
        self.bottleneck = SoftNormBottleneck(
            dim=256, noise_augment_dim=0, noise_regularize=True,
            auto_scale=True, freeze=True,
            dtype=dtype, device=device,
        )

    @torch.no_grad()
    def _pretransform_encode(self, x):
        return self.pretransform.encode(x)

    @torch.no_grad()
    def _pretransform_decode(self, x):
        return self.pretransform.decode(x)

    def encode(self, x):
        x = self._pretransform_encode(x)
        x = self.encoder(x)
        x = self.bottleneck.encode(x)
        return x

    def decode(self, x):
        x = self.bottleneck.decode(x)
        x = self.decoder(x)
        x = self._pretransform_decode(x)
        return x
