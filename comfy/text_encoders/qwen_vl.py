import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from comfy.ldm.modules.attention import optimized_attention_for_device


def process_qwen2vl_images(
    images: torch.Tensor,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_mean: list = None,
    image_std: list = None,
    interpolation: str = "bilinear",
):
    if image_mean is None:
        image_mean = [0.48145466, 0.4578275, 0.40821073]
    if image_std is None:
        image_std = [0.26862954, 0.26130258, 0.27577711]

    batch_size, height, width, channels = images.shape
    device = images.device
    # dtype = images.dtype

    images = images.permute(0, 3, 1, 2)

    grid_thw_list = []
    img = images[0]

    factor = patch_size * merge_size

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    img_resized = F.interpolate(
        img.unsqueeze(0),
        size=(h_bar, w_bar),
        mode=interpolation,
        align_corners=False
    ).squeeze(0)
    normalized = img_resized.clone()
    for c in range(3):
        normalized[c] = (img_resized[c] - image_mean[c]) / image_std[c]

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    grid_thw = torch.tensor([1, grid_h, grid_w], device=device, dtype=torch.long)

    pixel_values = normalized
    grid_thw_list.append(grid_thw)
    image_grid_thw = torch.stack(grid_thw_list)

    grid_t = 1
    channel = pixel_values.shape[0]
    pixel_values = pixel_values.unsqueeze(0).repeat(2, 1, 1, 1)

    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    )

    return flatten_patches, image_grid_thw


def _qwen_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    frames: int = 1,
    padded_frames: Optional[int] = None,
):
    """Return the closed Qwen processor resize for one media item.

    This intentionally contains only the geometry shared by the official
    Qwen2.5-VL and Qwen3-VL processors.  Conversation construction and frame
    selection remain caller policy.
    """
    if height <= 0 or width <= 0 or frames <= 0:
        raise ValueError("Qwen media dimensions must be positive")
    if padded_frames is None:
        padded_frames = frames
    if padded_frames <= 0:
        raise ValueError("Qwen padded media length must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen media aspect ratio must not exceed 200")
    if factor <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("invalid Qwen resize bounds")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    volume = padded_frames * h_bar * w_bar
    if volume > max_pixels:
        beta = math.sqrt((frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif volume < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def process_qwen_vl_media(
    frames: torch.Tensor,
    *,
    family: str,
):
    """Patch one already-selected Qwen image/video batch.

    ``frames`` is BHWC RGB in [0, 1].  The return value is
    ``(patches, grid_thw, relative_mrope)``.  ``relative_mrope`` describes the
    merged visual tokens only; the language-model wrapper offsets it around
    surrounding text.  No temporal sampling occurs here.
    """
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("Qwen media must be a BHWC tensor")
    if frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError("Qwen media must contain RGB frames")
    if frames.shape[0] > 64:
        raise ValueError("Qwen video is limited to 64 selected frames")

    frames = frames[..., :3]
    count, height, width, _ = map(int, frames.shape)
    is_qwen3 = family.startswith(("qwen3_vl_", "qwen3vl_"))
    is_qwen25 = family.startswith("qwen2_5_vl_")
    if not (is_qwen3 or is_qwen25):
        raise ValueError(f"unsupported Qwen vision family {family!r}")

    if is_qwen3:
        patch_size = 16
        factor = 32
        min_pixels = 4096
        max_pixels = 25_165_824
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        # The video processor pads an odd final frame instead of dropping it,
        # so the spatial budget must use that same padded temporal length.
        resize_frames = count
        padded_resize_frames = max(2, math.ceil(count / 2) * 2)
    else:
        patch_size = 14
        factor = 28
        min_pixels = 3136
        max_pixels = 12_845_056
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        resize_frames = 1
        padded_resize_frames = 1

    h_bar, w_bar = _qwen_smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        frames=resize_frames,
        padded_frames=padded_resize_frames,
    )
    pixels = frames.permute(0, 3, 1, 2)
    pixels = F.interpolate(
        pixels,
        size=(h_bar, w_bar),
        mode="bicubic",
        align_corners=False,
    )
    mean_tensor = torch.tensor(mean, device=pixels.device, dtype=pixels.dtype)[None, :, None, None]
    std_tensor = torch.tensor(std, device=pixels.device, dtype=pixels.dtype)[None, :, None, None]
    pixels = (pixels - mean_tensor) / std_tensor

    if count % 2:
        pixels = torch.cat((pixels, pixels[-1:]), dim=0)
    grid_t = pixels.shape[0] // 2
    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    merge_size = 2
    channels = pixels.shape[1]
    patches = pixels.reshape(
        grid_t,
        2,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * 2 * patch_size * patch_size,
    )
    grid = torch.tensor(
        [[grid_t, grid_h, grid_w]], device=frames.device, dtype=torch.long)

    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size
    spatial = merged_h * merged_w
    if is_qwen25:
        temporal = torch.arange(
            grid_t, device=frames.device, dtype=torch.long) * 2
    else:
        # Qwen3-VL carries time in the timestamp text preceding each visual
        # span.  Each pair therefore starts its visual MRoPE at t=0.
        temporal = torch.zeros(grid_t, device=frames.device, dtype=torch.long)
    t_ids = temporal.repeat_interleave(spatial)
    h_ids = torch.arange(
        merged_h, device=frames.device, dtype=torch.long
    ).repeat_interleave(merged_w).repeat(grid_t)
    w_ids = torch.arange(
        merged_w, device=frames.device, dtype=torch.long
    ).repeat(merged_h).repeat(grid_t)
    relative_mrope = torch.stack((t_ids, h_ids, w_ids), dim=0)
    return patches, grid, relative_mrope


def qwen2vl_mrope_position_ids(embeds_info, seq_len, device):
    # (3, seq_len) T/H/W MRoPE position ids: text runs sequentially, each image span gets its grid positions.
    # Returns None when there are no image embeds. `extra` is the image grid_thw, or a dict carrying it under "grid".
    media = [
        e for e in embeds_info
        if e.get("type") in {"image", "video", "video_segment"}
    ]
    if not media:
        return None

    # New canonical media wrappers provide exact relative positions.  Keep
    # the legacy grid path below for existing image-generation encoders.
    if all(isinstance(e.get("extra"), dict)
           and e["extra"].get("mrope") is not None for e in media):
        position_ids = torch.zeros(
            (3, seq_len), device=device, dtype=torch.long)
        cursor = 0
        next_position = 0
        for e in sorted(media, key=lambda item: item["index"]):
            start = int(e["index"])
            end = start + int(e["size"])
            if start < cursor or end > seq_len:
                raise ValueError("overlapping or invalid Qwen media span")
            text_len = start - cursor
            if text_len:
                text = torch.arange(
                    next_position,
                    next_position + text_len,
                    device=device,
                    dtype=torch.long,
                )
                position_ids[:, cursor:start] = text
                next_position += text_len
            relative = e["extra"]["mrope"].to(
                device=device, dtype=torch.long)
            if relative.shape != (3, end - start):
                raise ValueError("Qwen media MRoPE shape does not match span")
            position_ids[:, start:end] = relative + next_position
            next_position += int(relative.max().item()) + 1
            cursor = end
        if cursor < seq_len:
            text = torch.arange(
                next_position,
                next_position + seq_len - cursor,
                device=device,
                dtype=torch.long,
            )
            position_ids[:, cursor:] = text
        return position_ids

    position_ids = None
    offset = 0
    for e in embeds_info:
        if e.get("type") == "image":
            extra = e.get("extra", None)
            grid = extra["grid"] if isinstance(extra, dict) else extra
            start = e.get("index")
            if position_ids is None:
                position_ids = torch.zeros((3, seq_len), device=device)
                position_ids[:, :start] = torch.arange(0, start, device=device)
            end = e.get("size") + start
            len_max = int(grid.max()) // 2
            start_next = len_max + start
            position_ids[:, end:] = torch.arange(start_next + offset, start_next + (seq_len - end) + offset, device=device)
            position_ids[0, start:end] = start + offset
            max_d = int(grid[0][1]) // 2
            position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
            max_d = int(grid[0][2]) // 2
            position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
            offset += len_max - (end - start)
    return position_ids


class VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 3584,
        device=None,
        dtype=None,
        ops=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = ops.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            device=device,
            dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states)
        return hidden_states.view(-1, self.embed_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int, device) -> torch.Tensor:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, device=None, dtype=None, ops=None):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = ops.RMSNorm(context_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            ops.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            ops.Linear(self.hidden_size, dim, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        x = self.mlp(x)
        return x


class VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, device=None, dtype=None, ops=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv = ops.Linear(hidden_size, hidden_size * 3, bias=True, device=device, dtype=dtype)
        self.proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens=None,
        optimized_attention=None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            seq_length, _ = hidden_states.shape
            batch_size = 1
            hidden_states = hidden_states.unsqueeze(0)
        else:
            batch_size, seq_length, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        return attn_output


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.up_proj = ops.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class VisionBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, device=None, dtype=None, ops=None):
        super().__init__()
        self.norm1 = ops.RMSNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = ops.RMSNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.attn = VisionAttention(hidden_size, num_heads, device=device, dtype=dtype, ops=ops)
        self.mlp = VisionMLP(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens=None,
        optimized_attention=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings, cu_seqlens, optimized_attention)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2VLVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3584,
        output_hidden_size: int = 3584,
        intermediate_size: int = 3420,
        num_heads: int = 16,
        num_layers: int = 32,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        device=None,
        dtype=None,
        ops=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = [7, 15, 23, 31]

        self.patch_embed = VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=3,
            embed_dim=hidden_size,
            device=device,
            dtype=dtype,
            ops=ops,
        )

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, intermediate_size, num_heads, device, dtype, ops)
            for _ in range(num_layers)
        ])

        self.merger = PatchMerger(
            dim=output_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            device=device,
            dtype=dtype,
            ops=ops,
        )

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size

            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )

            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)

            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_size * self.spatial_merge_size + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()

        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def get_position_embeddings(self, grid_thw, device):
        pos_ids = []

        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size, device)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        optimized_attention = optimized_attention_for_device(pixel_values.device, mask=False, small_input=True)

        hidden_states = self.patch_embed(pixel_values)

        window_index, cu_window_seqlens = self.get_window_index(image_grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        position_embeddings = self.get_position_embeddings(image_grid_thw, hidden_states.device)

        seq_len, _ = hidden_states.size()
        spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        position_embeddings = position_embeddings.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        position_embeddings = position_embeddings[window_index, :, :]
        position_embeddings = position_embeddings.reshape(seq_len, -1)
        position_embeddings = torch.cat((position_embeddings, position_embeddings), dim=-1)
        position_embeddings = (position_embeddings.cos(), position_embeddings.sin())

        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for i, block in enumerate(self.blocks):
            if i in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = block(hidden_states, position_embeddings, cu_seqlens_now, optimized_attention=optimized_attention)

        hidden_states = self.merger(hidden_states)
        # Potentially important for spatially precise edits. This is present in the HF implementation.
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states
