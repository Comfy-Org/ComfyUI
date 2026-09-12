# MiniMax H3 video VAE: 3D causal CNN encoder + ViT3D decoder.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
from comfy.ldm.modules.attention import optimized_attention

ops = comfy.ops.disable_weight_init

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]


# 3D causal CNN encoder

class CausalConv3d(ops.Conv3d):
    # Reflect spatial padding, causal (zeros, front-only) temporal padding.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if sum(self.causal_padding) == 0:
            return super().forward(x)

        x = F.pad(x, (self.causal_padding[2], self.causal_padding[2], self.causal_padding[1], self.causal_padding[1], 0, 0),  mode="reflect")
        if x.shape[2] == 1:
            # single frame: the causal front padding is all zeros truncate the temporal taps instead of convolving zero frames
            return super().forward(x, autopad="causal_zero")
        x = F.pad(x, (0, 0, 0, 0, self.causal_padding[0] * 2, 0), mode="constant")
        return super().forward(x)


class TemporalIsolatedGroupNorm(ops.GroupNorm):
    # GroupNorm with statistics computed per frame (time merged into batch).
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h.add_(x)


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                    )
                )
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                )
            self.down.append(down)

        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ViT3D decoder

def create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)
        coords = coords / dim_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        # [B, S, n_dim] -> [B, S, 1, pairs, 2, 2] rotation table for the kitchen split-half rope
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None].float()
            * self.inv_freq.to(img_ids.device)[None, None, None, :]
        )
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)
        return table.to(img_ids.dtype)


class FeedForward(nn.Module):
    # Gated SiLU FFN.
    def __init__(self, dim, mult=4, bias=True, operations=ops):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = operations.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = operations.Linear(inner_dim, dim, bias=bias)

    def forward(self, x):
        gate, x = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate).mul_(x))


class Attention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_q = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = operations.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = operations.Linear(inner_dim, inner_dim, bias=bias)

    def forward(self, x, rotary_pos_emb=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        query = comfy.rmsnorm.rms_norm(query, self.norm_q.weight, self.norm_q.eps)
        key = comfy.rmsnorm.rms_norm(key, self.norm_k.weight, self.norm_k.eps)

        if rotary_pos_emb is not None:
            rot = rotary_pos_emb.shape[-3] * 2
            query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
                query[..., :rot], key[..., :rot], rotary_pos_emb)

        out = optimized_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                  self.heads, skip_reshape=True).nan_to_num_(0.0)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias, operations=operations)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rotary_pos_emb=None):
        x = x.addcmul_(self.attn(comfy.rmsnorm.rms_norm(x, self.norm1.weight, self.norm1.eps), rotary_pos_emb), comfy.ops.cast_to_input(self.scale1, x))
        return x.addcmul_(self.ff(comfy.rmsnorm.rms_norm(x, self.norm2.weight, self.norm2.eps)), comfy.ops.cast_to_input(self.scale2, x))


class ViT3DDecoder(nn.Module):
    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36, heads=32, dim_head=64, rope_theta=100.0,
                 rope_dim_ratio=0.75, bias=True, eps=1e-5, num_register_tokens=4, operations=ops):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = ops.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # unused at inference; kept so the checkpoint loads without leftover keys
        self.register_buffer("mask_token", torch.empty(1, 1, dim))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)
             for _ in range(num_layers)]
        )

        self.norm_out = ops.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = ops.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x):
        B, C, latent_T, latent_H, latent_W = x.shape

        h = self.x_embedder(x.flatten(2).transpose(1, 2))  # [B, T*H*W, C]

        num_patches = h.shape[1]
        num_suffix = 1 + self.num_register_tokens

        h = torch.cat([h, comfy.ops.cast_to_input(self.register_tokens, h).expand(B, -1, -1), torch.zeros_like(h[:, 0:1, :])], dim=1)

        img_ids = create_token_ids((latent_T, latent_H, latent_W), x.device, x.dtype).expand(B, -1, -1)
        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)

        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            h = block(h, rotary_pos_emb)

        output = self.proj_out(self.norm_out(h))

        output = output[:, :num_patches, :]

        output = output.view(
            B, latent_T, latent_H, latent_W,
            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.reshape(
            B, self.out_channels,
            latent_T * self.patch_size_t,
            latent_H * self.patch_size,
            latent_W * self.patch_size,
        )
        return output


# Full VAE

class MiniMaxH3VideoVAE(nn.Module):
    comfy_has_chunked_io = True

    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        clip_length=17,
        token_drop=3,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
        operations=ops,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))
        self.vae_ratio_t = int(math.prod(time_down))

        # temporal chunking parameters
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

        # spatial tiling parameters
        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = ops.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = ops.Conv3d(embed_dim, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
            operations=operations,
        )

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    # single-shot forward

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def _normalize_pixels(self, x):
        return x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))

    def _finalize_pixels(self, part):
        # raw decoder output -> float32 pixels in [0, 1] (the VAE wrapper's process_output is identity)
        part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)
        return part.add_(self.pixel_mean.to(device=part.device, dtype=torch.float32)).clamp_(0.0, 1.0)

    def decode_output_shape(self, input_shape):
        b, c, t, h, w = input_shape
        if t == 1:
            frames = 1
        else:
            pad_tokens, num_chunks = self._decode_temporal_chunks(t)
            frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)
        return (b, self.decoder.out_channels, frames, h * self.vae_ratio, w * self.vae_ratio)

    def _adaptive_encode(self, x):
        if self.tiling:
            return self.tiled_encode(x)
        return self._encode_moments(x)

    def _adaptive_decode(self, z):
        if self.tiling:
            return self.tiled_decode(z)
        return self._decode_pixels(z)

    # spatial tiling

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        return tile_start_idx, [tile_size] * N, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)

        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        # Blended tiles are written straight into a pre-allocated canvas.
        canvas = None
        row_tails = []
        out_y = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            new_tails = []
            left_tail = None
            out_x = 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi:zi + zl, zj:zj + zw])
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i]:, :].clone())
                next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas

    # temporal chunking

    def encode_temporal(self, x, device):
        # chunked input io: x may live on the CPU, clips move to the device as they encode
        z_list = []
        for i in range(math.ceil(x.shape[2] / self.clip_length)):
            clip_x = x[:, :, i * self.clip_length:(i + 1) * self.clip_length, :, :].to(device)
            if clip_x.shape[2] < self.clip_length:
                pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)
                clip_x = torch.cat([clip_x, pad_frames], dim=2)
            z_list.append(self._adaptive_encode(self._normalize_pixels(clip_x)))

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, :-self.token_drop]
        return z

    def _decode_temporal_pad_frames(self, z_len, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t

        z_len_before_pad = z_len - pad_tokens
        return sum(
            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0
             else self.vae_ratio_t)
            for k in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

    def _decode_temporal_chunks(self, z_len):
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens

        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            # too few tokens for one chunk (e.g. T_lat == 2): pad one extra chunk
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        return pad_tokens, num_chunks

    def decode_temporal(self, z, output_buffer=None):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1

        if output_buffer is None:
            # finalized chunks stream out of VRAM so the full video never sits on the GPU
            output_buffer = torch.empty(self.decode_output_shape(z.shape), dtype=torch.float32,
                                        device=comfy.model_management.intermediate_device())

        pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
        if pad_tokens > 0:
            pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
            z = torch.cat([z, pad_z], dim=2)

        dec = output_buffer
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            part = self._finalize_pixels(part)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            clip_dec = self._adaptive_decode(clip_z)

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None

            del clip_dec, clip_z

        return dec


    def encode(self, x, device=None):
        # x: [B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 24, T_lat, H/16, W/16]
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if device is None:
            device = x.device

        if x.shape[2] == 1:
            moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))
            moments = moments[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x, device)

        mean = torch.chunk(moments.float(), 2, dim=1)[0]

        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - latents_mean) / latents_std

    def encode_tiled(self, x, **kwargs):
        # tiling is always on internally with the reference's semantic tile sizes, ignore tiling fallbacks
        return self.encode(x)

    def decode_tiled(self, z, **kwargs):
        return self.decode(z)

    def decode(self, z, output_buffer=None):
        # z: [B, 24, T_lat, H_lat, W_lat] normalized latents -> float32 pixels [B, 3, T, H, W] in [0, 1]
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * latents_std + latents_mean

        if z.shape[2] == 1:
            dec = self._finalize_pixels(self._adaptive_decode(z)[:, :, -1:, :, :])
            if output_buffer is None:
                return dec
            output_buffer.copy_(dec)
            return output_buffer
        return self.decode_temporal(z, output_buffer)
