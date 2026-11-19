# SAM3 shared components: primitives, ViTDet backbone, FPN neck, position encodings.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import EmbedND
from comfy.ops import cast_to_input


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False, device=None, dtype=None, operations=None):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([operations.Linear(dims[i], dims[i + 1], device=device, dtype=dtype) for i in range(num_layers)])
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class SAMAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, kv_in_dim=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = embedding_dim // downsample_rate
        kv_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.q_proj = operations.Linear(embedding_dim, internal_dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(internal_dim, embedding_dim, device=device, dtype=dtype)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        return self.out_proj(optimized_attention(q, k, v, self.num_heads, low_precision_attention=False))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, attention_downsample_rate=2, skip_first_layer_pe=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = SAMAttention(embedding_dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_token_to_image = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image_to_token = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.mlp = nn.Sequential(operations.Linear(embedding_dim, mlp_dim, device=device, dtype=dtype), nn.ReLU(), operations.Linear(mlp_dim, embedding_dim, device=device, dtype=dtype))
        self.norm1 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm4 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.norm1(self.self_attn(queries, queries, queries))
        else:
            q = queries + query_pe
            queries = self.norm1(queries + self.self_attn(q, q, queries))
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048, attention_downsample_rate=2, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, attention_downsample_rate,
                                 skip_first_layer_pe=(i == 0), device=device, dtype=dtype, operations=operations)
            for i in range(depth)
        ])
        self.final_attn_token_to_image = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.norm_final = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)

    def forward(self, image_embedding, image_pe, point_embedding):
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """Fourier feature positional encoding with random gaussian projection."""
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", (scale or 1.0) * torch.randn(2, num_pos_feats))

    def _encode(self, normalized_coords):
        """Map normalized [0,1] coordinates to fourier features via random projection. Computes in fp32."""
        orig_dtype = normalized_coords.dtype
        proj_matrix = self.positional_encoding_gaussian_matrix.to(device=normalized_coords.device, dtype=torch.float32)
        projected = 2 * math.pi * (2 * normalized_coords.float() - 1) @ proj_matrix
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(orig_dtype)

    def forward(self, size, device=None):
        h, w = size
        dev = device if device is not None else self.positional_encoding_gaussian_matrix.device
        ones = torch.ones((h, w), device=dev, dtype=torch.float32)
        norm_xy = torch.stack([(ones.cumsum(1) - 0.5) / w, (ones.cumsum(0) - 0.5) / h], dim=-1)
        return self._encode(norm_xy).permute(2, 0, 1).unsqueeze(0)

    def forward_with_coords(self, pixel_coords, image_size):
        norm = pixel_coords.clone()
        norm[:, :, 0] /= image_size[1]
        norm[:, :, 1] /= image_size[0]
        return self._encode(norm)


# ViTDet backbone + FPN neck

def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def rope_2d(end_x: int, end_y: int, dim: int, theta: float = 10000.0, scale_pos: float = 1.0):
    """Generate 2D axial RoPE using flux EmbedND. Returns [1, 1, HW, dim//2, 2, 2]."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    ids = torch.stack([(t % end_x) * scale_pos,
                       torch.div(t, end_x, rounding_mode="floor") * scale_pos], dim=-1)
    return EmbedND(dim=dim, theta=theta, axes_dim=[dim // 2, dim // 2])(ids.unsqueeze(0))


class _ViTMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, device=None, dtype=None, operations=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = operations.Linear(dim, hidden, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = operations.Linear(hidden, dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """ViTDet multi-head attention with fused QKV projection."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rope=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)
        return self.proj(optimized_attention(q, k, v, self.num_heads, skip_reshape=True, low_precision_attention=False))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, window_size=0, use_rope=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.window_size = window_size
        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.attn = Attention(dim, num_heads, qkv_bias, use_rope, device=device, dtype=dtype, operations=operations)
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.mlp = _ViTMLP(dim, mlp_ratio, device=device, dtype=dtype, operations=operations)

    def forward(self, x, freqs_cis=None):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(x.shape[0], self.window_size * self.window_size, -1)
            x = self.attn(x, freqs_cis=freqs_cis)
            x = x.view(-1, self.window_size, self.window_size, x.shape[-1])
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.attn(x, freqs_cis=freqs_cis)
            x = x.view(B, H, W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=1024, device=None, dtype=None, operations=None):
        super().__init__()
        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.proj(x)


class ViTDet(nn.Module):
    def __init__(self, img_size=1008, patch_size=14, embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625, qkv_bias=True, window_size=24,
                 global_att_blocks=(7, 15, 23, 31), use_rope=True, pretrain_img_size=336, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_att_blocks = set(global_att_blocks)

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim, device=device, dtype=dtype, operations=operations)

        num_patches = (pretrain_img_size // patch_size) ** 2 + 1  # +1 for cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim, device=device, dtype=dtype))

        self.ln_pre = operations.LayerNorm(embed_dim, device=device, dtype=dtype)

        grid_size = img_size // patch_size
        pretrain_grid = pretrain_img_size // patch_size

        self.blocks = nn.ModuleList()
        for i in range(depth):
            is_global = i in self.global_att_blocks
            self.blocks.append(Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias,
                window_size=0 if is_global else window_size,
                use_rope=use_rope,
                device=device, dtype=dtype, operations=operations,
            ))

        if use_rope:
            rope_scale = pretrain_grid / grid_size
            self.register_buffer("freqs_cis", rope_2d(grid_size, grid_size, embed_dim // num_heads, scale_pos=rope_scale), persistent=False)
            self.register_buffer("freqs_cis_window", rope_2d(window_size, window_size, embed_dim // num_heads), persistent=False)
        else:
            self.freqs_cis = None
            self.freqs_cis_window = None

    def _get_pos_embed(self, num_tokens):
        pos = self.pos_embed
        if pos.shape[1] == num_tokens:
            return pos
        cls_pos = pos[:, :1]
        spatial_pos = pos[:, 1:]
        old_size = int(math.sqrt(spatial_pos.shape[1]))
        new_size = int(math.sqrt(num_tokens - 1)) if num_tokens > 1 else old_size
        spatial_2d = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        tiles_h = new_size // old_size + 1
        tiles_w = new_size // old_size + 1
        tiled = spatial_2d.tile([1, 1, tiles_h, tiles_w])[:, :, :new_size, :new_size]
        tiled = tiled.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return torch.cat([cls_pos, tiled], dim=1)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)

        pos = cast_to_input(self._get_pos_embed(Hp * Wp + 1), x)
        x = x + pos[:, 1:Hp * Wp + 1]

        x = x.view(B, Hp, Wp, C)
        x = self.ln_pre(x)

        freqs_cis_global = self.freqs_cis
        freqs_cis_win = self.freqs_cis_window
        if freqs_cis_global is not None:
            freqs_cis_global = cast_to_input(freqs_cis_global, x)
        if freqs_cis_win is not None:
            freqs_cis_win = cast_to_input(freqs_cis_win, x)

        for block in self.blocks:
            fc = freqs_cis_win if block.window_size > 0 else freqs_cis_global
            x = block(x, freqs_cis=fc)

        return x.permute(0, 3, 1, 2)


class FPNScaleConv(nn.Module):
    def __init__(self, in_dim, out_dim, scale, device=None, dtype=None, operations=None):
        super().__init__()
        if scale == 4.0:
            self.dconv_2x2_0 = operations.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
            self.dconv_2x2_1 = operations.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2, device=device, dtype=dtype)
            proj_in = in_dim // 4
        elif scale == 2.0:
            self.dconv_2x2 = operations.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
            proj_in = in_dim // 2
        elif scale == 1.0:
            proj_in = in_dim
        elif scale == 0.5:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            proj_in = in_dim
        self.scale = scale
        self.conv_1x1 = operations.Conv2d(proj_in, out_dim, kernel_size=1, device=device, dtype=dtype)
        self.conv_3x3 = operations.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, device=device, dtype=dtype)

    def forward(self, x):
        if self.scale == 4.0:
            x = F.gelu(self.dconv_2x2_0(x))
            x = self.dconv_2x2_1(x)
        elif self.scale == 2.0:
            x = self.dconv_2x2(x)
        elif self.scale == 0.5:
            x = self.pool(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal position encoding (DETR-style) with result caching."""
    def __init__(self, num_pos_feats=256, temperature=10000.0, normalize=True, scale=None):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.half_dim = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi
        self._cache = {}

    def _sincos(self, vals):
        """Encode 1D values to interleaved sin/cos features."""
        freqs = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=vals.device) // 2) / self.half_dim)
        raw = vals[..., None] * self.scale / freqs
        return torch.stack((raw[..., 0::2].sin(), raw[..., 1::2].cos()), dim=-1).flatten(-2)

    def _encode_xy(self, x, y):
        """Encode normalized x, y coordinates to sinusoidal features. Returns (pos_x, pos_y) each [N, half_dim]."""
        dim_t = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=x.device) // 2) / self.half_dim)
        pos_x = x[:, None] * self.scale / dim_t
        pos_y = y[:, None] * self.scale / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, cx, cy, w, h):
        """Encode box center + size to [N, d_model+2] features."""
        pos_x, pos_y = self._encode_xy(cx, cy)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        key = (H, W, x.device)
        if key not in self._cache:
            gy = torch.arange(H, dtype=torch.float32, device=x.device)
            gx = torch.arange(W, dtype=torch.float32, device=x.device)
            if self.normalize:
                gy, gx = gy / (H - 1 + 1e-6), gx / (W - 1 + 1e-6)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            self._cache[key] = torch.cat((self._sincos(yy), self._sincos(xx)), dim=-1).permute(2, 0, 1).unsqueeze(0)
        return self._cache[key].expand(B, -1, -1, -1)


class SAM3VisionBackbone(nn.Module):
    def __init__(self, embed_dim=1024, d_model=256, multiplex=False, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.trunk = ViTDet(embed_dim=embed_dim, device=device, dtype=dtype, operations=operations, **kwargs)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model, normalize=True)
        self.multiplex = multiplex

        fpn_args = dict(device=device, dtype=dtype, operations=operations)
        if multiplex:
            scales = [4.0, 2.0, 1.0]
            self.convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.propagation_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.interactive_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
        else:
            scales = [4.0, 2.0, 1.0, 0.5]
            self.convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.sam2_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])

    def forward(self, images, need_tracker=False, tracker_mode=None, cached_trunk=None, tracker_only=False):
        backbone_out = cached_trunk if cached_trunk is not None else self.trunk(images)

        if tracker_only:
            # Skip detector FPN when only tracker features are needed (video tracking)
            if self.multiplex:
                tracker_convs = self.propagation_convs if tracker_mode == "propagation" else self.interactive_convs
            else:
                tracker_convs = self.sam2_convs
            tracker_features = [conv(backbone_out) for conv in tracker_convs]
            tracker_positions = [cast_to_input(self.position_encoding(f), f) for f in tracker_features]
            return None, None, tracker_features, tracker_positions

        features = [conv(backbone_out) for conv in self.convs]
        positions = [cast_to_input(self.position_encoding(f), f) for f in features]

        if self.multiplex:
            if tracker_mode == "propagation":
                tracker_convs = self.propagation_convs
            elif tracker_mode == "interactive":
                tracker_convs = self.interactive_convs
            else:
                return features, positions, None, None
        elif need_tracker:
            tracker_convs = self.sam2_convs
        else:
            return features, positions, None, None

        tracker_features = [conv(backbone_out) for conv in tracker_convs]
        tracker_positions = [cast_to_input(self.position_encoding(f), f) for f in tracker_features]
        return features, positions, tracker_features, tracker_positions
