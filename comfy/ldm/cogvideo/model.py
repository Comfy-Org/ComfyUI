# CogVideoX 3D Transformer - ported to ComfyUI native ops
# Architecture reference: diffusers CogVideoXTransformer3DModel
# Style reference: comfy/ldm/wan/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
import comfy.patcher_extension
import comfy.ldm.common_dit


def _get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    """Returns (cos, sin) each with shape [seq_len, dim].

    Frequencies are computed at dim//2 resolution then repeat_interleaved
    to full dim, matching CogVideoX's interleaved (real, imag) pair format.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim))
    angles = torch.outer(pos.float(), freqs.float())
    cos = angles.cos().repeat_interleave(2, dim=-1).float()
    sin = angles.sin().repeat_interleave(2, dim=-1).float()
    return (cos, sin)


def apply_rotary_emb(x, freqs_cos_sin):
    """Apply CogVideoX rotary embedding to query or key tensor.

    x: [B, heads, seq_len, head_dim]
    freqs_cos_sin: (cos, sin) each [seq_len, head_dim//2]

    Uses interleaved pair rotation (same as diffusers CogVideoX/Flux).
    head_dim is reshaped to (-1, 2) pairs, rotated, then flattened back.
    """
    cos, sin = freqs_cos_sin
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)

    # Interleaved pairs: [B, H, S, D] -> [B, H, S, D//2, 2] -> (real, imag)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None] * scale
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_3d_sincos_pos_embed(embed_dim, spatial_size, temporal_size, spatial_interpolation_scale=1.0, temporal_interpolation_scale=1.0, device=None):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    grid_w = torch.arange(spatial_size[0], dtype=torch.float32, device=device) / spatial_interpolation_scale
    grid_h = torch.arange(spatial_size[1], dtype=torch.float32, device=device) / spatial_interpolation_scale
    grid_t = torch.arange(temporal_size, dtype=torch.float32, device=device) / temporal_interpolation_scale

    grid_t, grid_h, grid_w = torch.meshgrid(grid_t, grid_h, grid_w, indexing="ij")

    embed_dim_spatial = 2 * (embed_dim // 3)
    embed_dim_temporal = embed_dim // 3

    pos_embed_spatial = _get_2d_sincos_pos_embed(embed_dim_spatial, grid_h, grid_w, device=device)
    pos_embed_temporal = _get_1d_sincos_pos_embed(embed_dim_temporal, grid_t[:, 0, 0], device=device)

    T, H, W = grid_t.shape
    pos_embed_temporal = pos_embed_temporal.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
    pos_embed = torch.cat([pos_embed_temporal, pos_embed_spatial], dim=-1)

    return pos_embed


def _get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device=None):
    T, H, W = grid_h.shape
    half_dim = embed_dim // 2
    pos_h = _get_1d_sincos_pos_embed(half_dim, grid_h.reshape(-1), device=device).reshape(T, H, W, half_dim)
    pos_w = _get_1d_sincos_pos_embed(half_dim, grid_w.reshape(-1), device=device).reshape(T, H, W, half_dim)
    return torch.cat([pos_h, pos_w], dim=-1)


def _get_1d_sincos_pos_embed(embed_dim, pos, device=None):
    half = embed_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half)
    args = pos.float().reshape(-1)[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



class CogVideoXPatchEmbed(nn.Module):
    def __init__(self, patch_size=2, patch_size_t=None, in_channels=16, dim=1920,
                 text_dim=4096, bias=True, sample_width=90, sample_height=60,
                 sample_frames=49, temporal_compression_ratio=4,
                 max_text_seq_length=226, spatial_interpolation_scale=1.875,
                 temporal_interpolation_scale=1.0, use_positional_embeddings=True,
                 use_learned_positional_embeddings=True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.dim = dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if patch_size_t is None:
            self.proj = operations.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=bias, device=device, dtype=dtype)
        else:
            self.proj = operations.Linear(in_channels * patch_size * patch_size * patch_size_t, dim, device=device, dtype=dtype)

        self.text_proj = operations.Linear(text_dim, dim, device=device, dtype=dtype)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(self, sample_height, sample_width, sample_frames, device=None):
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        if self.patch_size_t is not None:
            post_time_compression_frames = post_time_compression_frames // self.patch_size_t
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
        )
        pos_embedding = pos_embedding.reshape(-1, self.dim)
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.max_text_seq_length + num_patches, self.dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.max_text_seq_length:].copy_(pos_embedding)
        return joint_pos_embedding

    def forward(self, text_embeds, image_embeds):
        input_dtype = text_embeds.dtype
        text_embeds = self.text_proj(text_embeds.to(self.text_proj.weight.dtype)).to(input_dtype)
        batch_size, num_frames, channels, height, width = image_embeds.shape

        proj_dtype = self.proj.weight.dtype
        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds.to(proj_dtype)).to(input_dtype)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)
            image_embeds = image_embeds.flatten(1, 2)
        else:
            p = self.patch_size
            p_t = self.patch_size_t
            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds.to(proj_dtype)).to(input_dtype)

        embeds = torch.cat([text_embeds, image_embeds], dim=1).contiguous()

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            text_seq_length = text_embeds.shape[1]
            num_image_patches = image_embeds.shape[1]

            if self.use_learned_positional_embeddings:
                image_pos = self.pos_embedding[
                    :, self.max_text_seq_length:self.max_text_seq_length + num_image_patches
                ].to(device=embeds.device, dtype=embeds.dtype)
            else:
                image_pos = get_3d_sincos_pos_embed(
                    self.dim,
                    (width // self.patch_size, height // self.patch_size),
                    num_image_patches // ((height // self.patch_size) * (width // self.patch_size)),
                    self.spatial_interpolation_scale,
                    self.temporal_interpolation_scale,
                    device=embeds.device,
                ).reshape(1, num_image_patches, self.dim).to(dtype=embeds.dtype)

            # Build joint: zeros for text + sincos for image
            joint_pos = torch.zeros(1, text_seq_length + num_image_patches, self.dim, device=embeds.device, dtype=embeds.dtype)
            joint_pos[:, text_seq_length:] = image_pos
            embeds = embeds + joint_pos

        return embeds


class CogVideoXLayerNormZero(nn.Module):
    def __init__(self, time_dim, dim, elementwise_affine=True, eps=1e-5, bias=True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(time_dim, 6 * dim, bias=bias, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, hidden_states, encoder_hidden_states, temb):
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class CogVideoXAdaLayerNorm(nn.Module):
    def __init__(self, time_dim, dim, elementwise_affine=True, eps=1e-5,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(time_dim, 2 * dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class CogVideoXBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, time_dim,
                 eps=1e-5, ff_inner_dim=None, ff_bias=True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm1 = CogVideoXLayerNormZero(time_dim, dim, eps=eps, device=device, dtype=dtype, operations=operations)

        # Self-attention (joint text + latent)
        self.q = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.k = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.v = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.norm_q = operations.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        self.norm_k = operations.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        self.attn_out = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

        self.norm2 = CogVideoXLayerNormZero(time_dim, dim, eps=eps, device=device, dtype=dtype, operations=operations)

        # Feed-forward (GELU approximate)
        inner_dim = ff_inner_dim or dim * 4
        self.ff_proj = operations.Linear(dim, inner_dim, bias=ff_bias, device=device, dtype=dtype)
        self.ff_out = operations.Linear(inner_dim, dim, bias=ff_bias, device=device, dtype=dtype)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        text_seq_length = encoder_hidden_states.size(1)

        # Norm & modulate
        norm_hidden, norm_encoder, gate_msa, enc_gate_msa = self.norm1(hidden_states, encoder_hidden_states, temb)

        # Joint self-attention
        qkv_input = torch.cat([norm_encoder, norm_hidden], dim=1)
        b, s, _ = qkv_input.shape
        n, d = self.num_heads, self.head_dim

        q = self.q(qkv_input).view(b, s, n, d)
        k = self.k(qkv_input).view(b, s, n, d)
        v = self.v(qkv_input)

        q = self.norm_q(q).view(b, s, n, d)
        k = self.norm_k(k).view(b, s, n, d)

        # Apply rotary embeddings to image tokens only (diffusers format: [B, heads, seq, head_dim])
        if image_rotary_emb is not None:
            q_img = q[:, text_seq_length:].transpose(1, 2)  # [B, heads, img_seq, head_dim]
            k_img = k[:, text_seq_length:].transpose(1, 2)
            q_img = apply_rotary_emb(q_img, image_rotary_emb)
            k_img = apply_rotary_emb(k_img, image_rotary_emb)
            q = torch.cat([q[:, :text_seq_length], q_img.transpose(1, 2)], dim=1)
            k = torch.cat([k[:, :text_seq_length], k_img.transpose(1, 2)], dim=1)

        attn_out = optimized_attention(
            q.reshape(b, s, n * d),
            k.reshape(b, s, n * d),
            v,
            heads=self.num_heads,
            transformer_options=transformer_options,
        )

        attn_out = self.attn_out(attn_out)

        attn_encoder, attn_hidden = attn_out.split([text_seq_length, s - text_seq_length], dim=1)

        hidden_states = hidden_states + gate_msa * attn_hidden
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder

        # Norm & modulate for FF
        norm_hidden, norm_encoder, gate_ff, enc_gate_ff = self.norm2(hidden_states, encoder_hidden_states, temb)

        # Feed-forward (GELU on concatenated text + latent)
        ff_input = torch.cat([norm_encoder, norm_hidden], dim=1)
        ff_output = self.ff_out(F.gelu(self.ff_proj(ff_input), approximate="tanh"))

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(nn.Module):
    def __init__(self,
                 num_attention_heads=30,
                 attention_head_dim=64,
                 in_channels=16,
                 out_channels=16,
                 flip_sin_to_cos=True,
                 freq_shift=0,
                 time_embed_dim=512,
                 ofs_embed_dim=None,
                 text_embed_dim=4096,
                 num_layers=30,
                 dropout=0.0,
                 attention_bias=True,
                 sample_width=90,
                 sample_height=60,
                 sample_frames=49,
                 patch_size=2,
                 patch_size_t=None,
                 temporal_compression_ratio=4,
                 max_text_seq_length=226,
                 spatial_interpolation_scale=1.875,
                 temporal_interpolation_scale=1.0,
                 use_rotary_positional_embeddings=False,
                 use_learned_positional_embeddings=False,
                 patch_bias=True,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        super().__init__()
        self.dtype = dtype
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.max_text_seq_length = max_text_seq_length
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            dim=dim,
            text_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            device=device, dtype=torch.float32, operations=operations,
        )

        # 2. Time embedding
        self.time_proj_dim = dim
        self.time_proj_flip = flip_sin_to_cos
        self.time_proj_shift = freq_shift
        self.time_embedding_linear_1 = operations.Linear(dim, time_embed_dim, device=device, dtype=dtype)
        self.time_embedding_act = nn.SiLU()
        self.time_embedding_linear_2 = operations.Linear(time_embed_dim, time_embed_dim, device=device, dtype=dtype)

        # Optional OFS embedding (CogVideoX 1.5 I2V)
        self.ofs_proj_dim = ofs_embed_dim
        if ofs_embed_dim:
            self.ofs_embedding_linear_1 = operations.Linear(ofs_embed_dim, ofs_embed_dim, device=device, dtype=dtype)
            self.ofs_embedding_act = nn.SiLU()
            self.ofs_embedding_linear_2 = operations.Linear(ofs_embed_dim, ofs_embed_dim, device=device, dtype=dtype)
        else:
            self.ofs_embedding_linear_1 = None

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            CogVideoXBlock(
                dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                time_dim=time_embed_dim,
                eps=1e-5,
                device=device, dtype=dtype, operations=operations,
            )
            for _ in range(num_layers)
        ])

        self.norm_final = operations.LayerNorm(dim, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype)

        # 4. Output
        self.norm_out = CogVideoXAdaLayerNorm(
            time_dim=time_embed_dim, dim=dim, eps=1e-5,
            device=device, dtype=dtype, operations=operations,
        )

        if patch_size_t is None:
            output_dim = patch_size * patch_size * out_channels
        else:
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = operations.Linear(dim, output_dim, device=device, dtype=dtype)

        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.temporal_compression_ratio = temporal_compression_ratio

    def forward(self, x, timestep, context, ofs=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, ofs, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, ofs=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        # ComfyUI passes [B, C, T, H, W]
        batch_size, channels, t, h, w = x.shape

        # Pad to patch size (temporal + spatial), same pattern as WAN
        p_t = self.patch_size_t if self.patch_size_t is not None else 1
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (p_t, self.patch_size, self.patch_size))

        # CogVideoX expects [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        batch_size, num_frames, channels, height, width = x.shape

        # Time embedding
        t_emb = get_timestep_embedding(timestep, self.time_proj_dim, self.time_proj_flip, self.time_proj_shift)
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embedding_linear_2(self.time_embedding_act(self.time_embedding_linear_1(t_emb)))

        if self.ofs_embedding_linear_1 is not None and ofs is not None:
            ofs_emb = get_timestep_embedding(ofs, self.ofs_proj_dim, self.time_proj_flip, self.time_proj_shift)
            ofs_emb = ofs_emb.to(dtype=x.dtype)
            ofs_emb = self.ofs_embedding_linear_2(self.ofs_embedding_act(self.ofs_embedding_linear_1(ofs_emb)))
            emb = emb + ofs_emb

        # Patch embedding
        hidden_states = self.patch_embed(context, x)

        text_seq_length = context.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # Rotary embeddings (if used)
        image_rotary_emb = None
        if self.use_rotary_positional_embeddings:
            post_patch_height = height // self.patch_size
            post_patch_width = width // self.patch_size
            if self.patch_size_t is None:
                post_time = num_frames
            else:
                post_time = num_frames // self.patch_size_t
            image_rotary_emb = self._get_rotary_emb(post_patch_height, post_patch_width, post_time, device=x.device)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                transformer_options=transformer_options,
            )

        hidden_states = self.norm_final(hidden_states)

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        p = self.patch_size
        p_t = self.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        # Back to ComfyUI format [B, C, T, H, W] and crop padding
        output = output.permute(0, 2, 1, 3, 4)[:, :, :t, :h, :w]
        return output

    def _get_rotary_emb(self, h, w, t, device):
        """Compute CogVideoX 3D rotary positional embeddings.

        For CogVideoX 1.5 (patch_size_t != None): uses "slice" mode — grid positions
        are integer arange computed at max_size, then sliced to actual size.
        For CogVideoX 1.0 (patch_size_t == None): uses "linspace" mode with crop coords
        scaled by spatial_interpolation_scale.
        """
        d = self.attention_head_dim
        dim_t = d // 4
        dim_h = d // 8 * 3
        dim_w = d // 8 * 3

        if self.patch_size_t is not None:
            # CogVideoX 1.5: "slice" mode — positions are simple integer indices
            # Compute at max(sample_size, actual_size) then slice to actual
            base_h = self.patch_embed.sample_height // self.patch_size
            base_w = self.patch_embed.sample_width // self.patch_size
            max_h = max(base_h, h)
            max_w = max(base_w, w)

            grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
            grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
            grid_t = torch.arange(t, device=device, dtype=torch.float32)
        else:
            # CogVideoX 1.0: "linspace" mode with interpolation scale
            grid_h = torch.linspace(0, h - 1, h, device=device, dtype=torch.float32) * self.spatial_interpolation_scale
            grid_w = torch.linspace(0, w - 1, w, device=device, dtype=torch.float32) * self.spatial_interpolation_scale
            grid_t = torch.arange(t, device=device, dtype=torch.float32)

        freqs_t = _get_1d_rotary_pos_embed(dim_t, grid_t)
        freqs_h = _get_1d_rotary_pos_embed(dim_h, grid_h)
        freqs_w = _get_1d_rotary_pos_embed(dim_w, grid_w)

        t_cos, t_sin = freqs_t
        h_cos, h_sin = freqs_h
        w_cos, w_sin = freqs_w

        # Slice to actual size (for "slice" mode where grids may be larger)
        t_cos, t_sin = t_cos[:t], t_sin[:t]
        h_cos, h_sin = h_cos[:h], h_sin[:h]
        w_cos, w_sin = w_cos[:w], w_sin[:w]

        # Broadcast and concatenate into [T*H*W, head_dim]
        t_cos = t_cos[:, None, None, :].expand(-1, h, w, -1)
        t_sin = t_sin[:, None, None, :].expand(-1, h, w, -1)
        h_cos = h_cos[None, :, None, :].expand(t, -1, w, -1)
        h_sin = h_sin[None, :, None, :].expand(t, -1, w, -1)
        w_cos = w_cos[None, None, :, :].expand(t, h, -1, -1)
        w_sin = w_sin[None, None, :, :].expand(t, h, -1, -1)

        cos = torch.cat([t_cos, h_cos, w_cos], dim=-1).reshape(t * h * w, -1)
        sin = torch.cat([t_sin, h_sin, w_sin], dim=-1).reshape(t * h * w, -1)
        return (cos, sin)
