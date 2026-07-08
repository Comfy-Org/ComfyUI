# SAM3 video tracker: memory encoder, memory attention, SAM mask decoder/prompt encoder.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    from scipy import ndimage
    _HAS_CV2 = False

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.sam3.sam import rope_2d, PositionEmbeddingSine
from comfy.ops import cast_to_input
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.cascade.common import LayerNorm2d_op
from comfy.ldm.sam3.sam import MLP, PositionEmbeddingRandom
from comfy.ldm.sam3.sam import TwoWayTransformer as SAMTwoWayTransformer

NO_OBJ_SCORE = -1024.0


def to_spatial(x, H, W):
    """Reshape (B, H*W, C) → (B, C, H, W)."""
    return x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2)

class MultiplexState:
    """Tracks object-to-slot assignments for multiplex tracking. Provides mux/demux operations."""

    def __init__(self, num_objects, multiplex_count, device, dtype):
        self.multiplex_count = multiplex_count
        self.device = device
        self.dtype = dtype
        self._build(num_objects)

    def mux(self, x):
        """[N_obj, ...] -> [num_buckets, multiplex_count, ...]"""
        out_shape = (self.num_buckets, self.multiplex_count) + x.shape[1:]
        return (self.mux_matrix.to(device=x.device, dtype=x.dtype) @ x.reshape(self.total_valid_entries, -1)).view(out_shape)

    def demux(self, x):
        """[num_buckets, multiplex_count, ...] -> [N_obj, ...]"""
        out_shape = (self.total_valid_entries,) + x.shape[2:]
        flat = x.reshape(self.num_buckets * self.multiplex_count, -1)
        return (self.demux_matrix.to(device=x.device, dtype=x.dtype) @ flat).view(out_shape)

    def get_valid_object_mask(self):
        """[num_buckets, multiplex_count] bool tensor, True for valid slots."""
        return (self.mux_matrix.sum(dim=1) > 0).reshape(self.num_buckets, self.multiplex_count)

    def _build(self, num_objects):
        M = self.multiplex_count
        self.num_buckets = (num_objects + M - 1) // M
        self.total_valid_entries = num_objects
        total_slots = self.num_buckets * M
        self.mux_matrix = torch.zeros(total_slots, num_objects, device=self.device, dtype=self.dtype)
        self.demux_matrix = torch.zeros(num_objects, total_slots, device=self.device, dtype=self.dtype)
        oids = torch.arange(num_objects, device=self.device)
        slots = (oids // M) * M + (oids % M)
        self.mux_matrix[slots, oids] = 1.0
        self.demux_matrix[oids, slots] = 1.0

    def add_objects(self, n_new):
        """Grow multiplex state for n_new additional objects."""
        self._build(self.total_valid_entries + n_new)

def _compute_mask_overlap(masks_a, masks_b):
    """Max of IoU and IoM (intersection over minimum area). More robust to size differences."""
    a_flat = (masks_a > 0).float().flatten(1)
    b_flat = (masks_b > 0).float().flatten(1)
    intersection = a_flat @ b_flat.T
    area_a = a_flat.sum(1, keepdim=True)
    area_b = b_flat.sum(1, keepdim=True).T
    iou = intersection / (area_a + area_b - intersection).clamp(min=1)
    iom = intersection / torch.min(area_a.expand_as(iou), area_b.expand_as(iou)).clamp(min=1)
    return torch.max(iou, iom)


def _nms_masks(masks, scores, thresh=0.5):
    """Mask-based NMS using IoU+IoM overlap. Returns (filtered_masks, filtered_scores)."""
    order = scores.argsort(descending=True)
    masks, scores = masks[order], scores[order]
    keep = []
    for i in range(masks.shape[0]):
        if keep:
            if _compute_mask_overlap(masks[i:i+1], masks[torch.tensor(keep, device=masks.device)]).max() >= thresh:
                continue
        keep.append(i)
    return masks[keep], scores[keep]


def _get_connected_components(mask_bin):
    """Get connected component labels and areas. mask_bin: [B, 1, H, W] uint8."""
    labels_list, areas_list = [], []
    for i in range(mask_bin.shape[0]):
        m = mask_bin[i, 0].cpu().numpy()
        if _HAS_CV2:
            _, labeled, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = stats[labeled, cv2.CC_STAT_AREA].astype('int32')
        else:
            labeled, num_features = ndimage.label(m)
            areas = np.zeros_like(m, dtype=np.int32)
            for c in range(1, num_features + 1):
                component = labeled == c
                areas[component] = component.sum()
        labels_list.append(torch.from_numpy(labeled).to(mask_bin.device))
        areas_list.append(torch.from_numpy(areas).to(device=mask_bin.device, dtype=torch.int32))
    return torch.stack(labels_list).unsqueeze(1), torch.stack(areas_list).unsqueeze(1)


def fill_holes_in_mask_scores(mask, max_area=0):
    """Remove small foreground sprinkles and fill small background holes using connected components."""
    if max_area <= 0:
        return mask

    # Fill holes: small connected components in background → foreground
    mask_bg = (mask <= 0).to(torch.uint8)
    _, areas_bg = _get_connected_components(mask_bg)
    small_bg = mask_bg.bool() & (areas_bg <= max_area)
    mask = torch.where(small_bg, 0.1, mask)

    # Remove sprinkles: small connected components in foreground → background
    # Only remove if area < min(max_area, half of total foreground area)
    mask_fg = (mask > 0).to(torch.uint8)
    fg_area_thresh = mask_fg.sum(dim=(2, 3), keepdim=True, dtype=torch.int32)
    fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
    _, areas_fg = _get_connected_components(mask_fg)
    small_fg = mask_fg.bool() & (areas_fg <= fg_area_thresh)
    mask = torch.where(small_fg, -0.1, mask)

    return mask


def apply_rope_memory(q, k, freqs, num_heads, num_k_exclude_rope=0):
    """Apply 2D axial RoPE to memory attention using flux rope format.

    Args:
        q: [B, Nq, C] projected queries (current frame features)
        k: [B, Nk, C] projected keys (memory tokens)
        freqs: [1, Nq, dim//2, 2, 2] flux-format rotation matrices for one frame
        num_heads: number of attention heads
        num_k_exclude_rope: number of trailing k tokens to skip RoPE (object pointers)
    """
    B, Nq, C = q.shape
    head_dim = C // num_heads

    # freqs shape: [1, 1, Nq, dim//2, 2, 2] (heads broadcast dim already included)
    q_h = q.view(B, Nq, num_heads, head_dim).transpose(1, 2)
    q_h = apply_rope1(q_h, freqs)
    q = q_h.transpose(1, 2).reshape(B, Nq, C)

    # Apply RoPE to k (excluding last num_k_exclude_rope tokens)
    Nk = k.shape[1]
    num_k_rope = Nk - num_k_exclude_rope
    if num_k_rope > 0:
        # Repeat freqs for multiple frames of spatial memory
        Nf = freqs.shape[2]  # spatial positions in one frame
        if num_k_rope > Nf:
            r = (num_k_rope + Nf - 1) // Nf
            pe_k = freqs.repeat(1, 1, r, 1, 1, 1)[:, :, :num_k_rope]
        else:
            pe_k = freqs[:, :, :num_k_rope]

        k_h = k[:, :num_k_rope].view(B, num_k_rope, num_heads, head_dim).transpose(1, 2)
        k_h = apply_rope1(k_h, pe_k)
        k = k.clone()
        k[:, :num_k_rope] = k_h.transpose(1, 2).reshape(B, num_k_rope, C)

    return q, k


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """1D sinusoidal positional encoding for temporal positions."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def _pad_to_buckets(tensor, target_buckets):
    """Pad a [num_buckets, ...] tensor to target_buckets along dim 0 if needed."""
    if tensor.shape[0] >= target_buckets:
        return tensor
    pad_shape = (target_buckets - tensor.shape[0],) + tensor.shape[1:]
    return torch.cat([tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)], dim=0)


def pack_masks(masks):
    """Pack binary masks [*, H, W] to bit-packed [*, H, W//8] uint8. W must be divisible by 8."""
    binary = masks > 0
    shifts = torch.arange(8, device=masks.device)
    return (binary.view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def unpack_masks(packed):
    """Unpack bit-packed [*, H, W//8] uint8 to bool [*, H, W*8]."""
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    return (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)


def _prep_frame(images, idx, device, dt, size):
    """Slice CPU full-res frames, transfer to GPU in target dtype, and resize to (size, size)."""
    return comfy.utils.common_upscale(images[idx].to(device=device, dtype=dt), size, size, "bicubic", crop="disabled")


def _compute_backbone(backbone_fn, frame, frame_idx=None):
    """Compute backbone features for a single frame. Returns (vision_feats, vision_pos, feat_sizes, features, trunk_out)."""
    features, positions, trunk_out = backbone_fn(frame, frame_idx=frame_idx)
    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in features]
    vision_feats = [x.flatten(2).permute(0, 2, 1) for x in features]
    vision_pos = [x.flatten(2).permute(0, 2, 1) for x in positions]
    return vision_feats, vision_pos, feat_sizes, features, trunk_out


def collect_memory_tokens(output_dict, frame_idx, num_maskmem, maskmem_tpos_enc, device,
                          collect_image_feats=False, tpos_v2=False, num_buckets=None):
    """Collect spatial memory, position encodings, and optionally image features from past frames."""
    to_cat_memory, to_cat_memory_pos = [], []
    to_cat_image_feat, to_cat_image_pos = [], []

    def _append(out, tpos_idx):
        feats = out["maskmem_features"].to(device)
        if num_buckets is not None:
            feats = _pad_to_buckets(feats, num_buckets)
        to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))
        enc = out["maskmem_pos_enc"][-1].to(device).flatten(2).permute(0, 2, 1)
        if num_buckets is not None:
            enc = _pad_to_buckets(enc, num_buckets)
        tpos = cast_to_input(maskmem_tpos_enc[tpos_idx], enc)
        to_cat_memory_pos.append(enc + tpos)
        if collect_image_feats and "image_features" in out:
            to_cat_image_feat.append(out["image_features"].to(device))
            to_cat_image_pos.append(out["image_pos_enc"].to(device) + tpos)

    cond_outputs = output_dict["cond_frame_outputs"]
    for t, out in cond_outputs.items():
        if tpos_v2:
            t_pos = frame_idx - t
            tpos_idx = num_maskmem - t_pos - 1 if 0 < t_pos < num_maskmem else num_maskmem - 1
        else:
            tpos_idx = num_maskmem - 1
        _append(out, tpos_idx)

    for t_pos in range(1, num_maskmem):
        out = output_dict["non_cond_frame_outputs"].get(frame_idx - (num_maskmem - t_pos), None)
        if out is None or out.get("maskmem_features") is None:
            continue
        _append(out, num_maskmem - t_pos - 1)

    return to_cat_memory, to_cat_memory_pos, to_cat_image_feat, to_cat_image_pos, cond_outputs


def compute_tpos_enc(rel_pos_list, device, d_model, proj_layer, dtype=None, max_abs_pos=None):
    """Temporal position encoding for object pointers."""
    pos_enc = torch.tensor(rel_pos_list, dtype=torch.float32, device=device) / max((max_abs_pos or 2) - 1, 1)
    pos_enc = get_1d_sine_pe(pos_enc, dim=d_model)
    if dtype is not None:
        pos_enc = pos_enc.to(dtype)
    return proj_layer(pos_enc)


def forward_sam_heads(backbone_features, prompt_encoder, mask_decoder, obj_ptr_proj, no_obj_fn,
                      image_size, point_inputs=None, mask_inputs=None, box_inputs=None,
                      high_res_features=None, multimask_output=False):
    """Shared SAM prompt encoder + mask decoder forward for both SAM3 and SAM3.1 trackers."""
    device = backbone_features.device
    # Batch size from inputs (mask_inputs may have N_obj > 1 while backbone is batch 1)
    if mask_inputs is not None:
        B = mask_inputs.shape[0]
    elif box_inputs is not None:
        B = box_inputs.shape[0]
    elif point_inputs is not None:
        B = point_inputs["point_coords"].shape[0]
    else:
        B = backbone_features.shape[0]

    if point_inputs is not None:
        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
    else:
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

    if mask_inputs is not None:
        prompt_size = (prompt_encoder.image_embedding_size[0] * 4, prompt_encoder.image_embedding_size[1] * 4)
        if mask_inputs.shape[-2:] != prompt_size:
            sam_mask_prompt = F.interpolate(mask_inputs, size=prompt_size, mode="bilinear", align_corners=False, antialias=True)
        else:
            sam_mask_prompt = mask_inputs
    else:
        sam_mask_prompt = None

    sparse, dense = prompt_encoder(points=(sam_point_coords, sam_point_labels), boxes=box_inputs, masks=sam_mask_prompt)
    sparse = cast_to_input(sparse, backbone_features)
    dense = cast_to_input(dense, backbone_features)
    image_pe = cast_to_input(prompt_encoder.get_dense_pe(), backbone_features)

    low_res_multimasks, ious, sam_output_tokens, object_score_logits = mask_decoder(
        image_embeddings=backbone_features, image_pe=image_pe,
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        high_res_features=high_res_features, multimask_output=multimask_output, return_all=True,
    )

    is_obj_appearing = object_score_logits > 0
    low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks,
                                      torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_multimasks.dtype))
    high_res_multimasks = F.interpolate(low_res_multimasks, size=(image_size, image_size), mode="bilinear", align_corners=False)

    sam_output_token = sam_output_tokens[:, 0]
    if multimask_output:
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(B, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        if sam_output_tokens.size(1) > 1:
            sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
    else:
        low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

    obj_ptr = obj_ptr_proj(sam_output_token)
    obj_ptr = no_obj_fn(obj_ptr, is_obj_appearing)

    return low_res_masks, high_res_masks, obj_ptr, object_score_logits


def use_mask_as_output(backbone_features, high_res_features, mask_inputs, mask_downsample,
                       prompt_encoder, mask_decoder, obj_ptr_proj, no_obj_fn, image_size, backbone_stride):
    """Shared mask-as-output for both SAM3 and SAM3.1 trackers."""
    out_scale, out_bias = 20.0, -10.0
    mask_inputs_float = cast_to_input(mask_inputs, backbone_features)
    high_res_masks = mask_inputs_float * out_scale + out_bias
    low_res_masks = F.interpolate(high_res_masks, size=(image_size // backbone_stride * 4,) * 2,
                                   mode="bilinear", align_corners=False, antialias=True)
    _, _, obj_ptr, _ = forward_sam_heads(
        backbone_features, prompt_encoder, mask_decoder, obj_ptr_proj, no_obj_fn,
        image_size, mask_inputs=mask_downsample(mask_inputs_float), high_res_features=high_res_features,
    )
    is_obj_appearing = torch.any(mask_inputs.flatten(1) > 0.0, dim=1)[..., None]
    alpha = is_obj_appearing.to(obj_ptr.dtype)
    object_score_logits = out_scale * alpha + out_bias
    return low_res_masks, high_res_masks, obj_ptr, object_score_logits


# Split attention with configurable input dims (for asymmetric cross-attention)
class SplitAttn(nn.Module):
    def __init__(self, embed_dim, num_heads=1, kv_dim=None, internal_dim=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        kv_dim = kv_dim or embed_dim
        internal_dim = internal_dim or embed_dim
        self.q_proj = operations.Linear(embed_dim, internal_dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(internal_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, q, k=None, v=None, rope=None, num_k_exclude_rope=0):
        if k is None:
            k = q
        if v is None:
            v = k
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, num_k_exclude_rope)
        out = optimized_attention(q, k, v, self.num_heads, low_precision_attention=False)
        return self.out_proj(out)


class MemoryAttnLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = SplitAttn(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image = SplitAttn(d_model, num_heads, kv_dim=kv_dim, device=device, dtype=dtype, operations=operations)
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, memory, memory_pos=None, rope=None, num_k_exclude_rope=0):
        x = x + self.self_attn(self.norm1(x), rope=rope)
        mem_k = memory + memory_pos if memory_pos is not None else memory
        x = x + self.cross_attn_image(self.norm2(x), mem_k, memory, rope=rope, num_k_exclude_rope=num_k_exclude_rope)
        normed = self.norm3(x)
        x = x + self.linear2(F.relu(self.linear1(normed)))
        return x


class MemoryAttnEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, num_layers=4, image_size=1008, patch_size=14,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttnLayer(d_model, num_heads, kv_dim, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        hw = image_size // patch_size
        self.register_buffer("_rope", rope_2d(hw, hw, d_model // num_heads), persistent=False)

    def forward(self, x, memory, src_pos=None, memory_pos=None, num_k_exclude_rope=0):
        if src_pos is not None:
            x = x + 0.1 * src_pos

        rope = self._rope.to(device=x.device)
        for layer in self.layers:
            x = layer(x, memory, memory_pos=memory_pos, rope=rope, num_k_exclude_rope=num_k_exclude_rope)
        return self.norm(x)


class MemoryTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = MemoryAttnEncoder(d_model, num_heads, kv_dim, dim_ff, num_layers, device=device, dtype=dtype, operations=operations)


def _upscale_masks(output_upscaling, conv_s0, conv_s1, src_out, high_res_features):
    """Shared upscaling for SAM mask decoders: deconv + high-res feature integration."""
    dc1, ln1, act1, dc2, act2 = output_upscaling
    if high_res_features is not None:
        upscaled = act1(ln1(dc1(src_out) + conv_s1(high_res_features[1])))
        upscaled = act2(dc2(upscaled) + conv_s0(high_res_features[0]))
    else:
        upscaled = act2(dc2(act1(ln1(dc1(src_out)))))
    return upscaled


class SAMMaskDecoder(nn.Module):
    def __init__(self, d_model=256, num_multimask_outputs=3, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = SAMTwoWayTransformer(depth=2, embedding_dim=d_model, num_heads=8, mlp_dim=2048, device=device, dtype=dtype, operations=operations)

        self.iou_token = operations.Embedding(1, d_model, device=device, dtype=dtype)
        self.mask_tokens = operations.Embedding(self.num_mask_tokens, d_model, device=device, dtype=dtype)
        self.obj_score_token = operations.Embedding(1, d_model, device=device, dtype=dtype)

        # Output upscaling: d_model -> d_model//4 -> d_model//8 at 4x resolution
        LN2d = LayerNorm2d_op(operations)
        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2, device=device, dtype=dtype), LN2d(d_model // 4, device=device, dtype=dtype), nn.GELU(),
            operations.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2, device=device, dtype=dtype), nn.GELU(),
        )

        # High-res feature integration
        self.conv_s0 = operations.Conv2d(d_model, d_model // 8, kernel_size=1, device=device, dtype=dtype)
        self.conv_s1 = operations.Conv2d(d_model, d_model // 4, kernel_size=1, device=device, dtype=dtype)

        # Per-mask hypernetwork MLPs
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3, device=device, dtype=dtype, operations=operations)
            for _ in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_tokens, 3, device=device, dtype=dtype, operations=operations)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3, device=device, dtype=dtype, operations=operations)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                high_res_features=None, multimask_output=False, return_all=False):
        B = sparse_prompt_embeddings.shape[0]
        ref = sparse_prompt_embeddings
        # Token order: [obj_score(1), iou(1), mask(num_mask_tokens)]
        tokens = torch.cat([cast_to_input(self.obj_score_token.weight, ref),
                            cast_to_input(self.iou_token.weight, ref),
                            cast_to_input(self.mask_tokens.weight, ref)], dim=0)
        tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos_src.flatten(2).permute(0, 2, 1)

        hs, src_out = self.transformer(src_flat, pos_flat, tokens)

        obj_score_token_out = hs[:, 0, :]
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2:2 + self.num_mask_tokens, :]

        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        upscaled = _upscale_masks(self.output_upscaling, self.conv_s0, self.conv_s1, src_out, high_res_features)

        hyper_in = torch.stack([
            mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)
        ], dim=1)

        masks = (hyper_in @ upscaled.flatten(2)).view(B, self.num_mask_tokens, upscaled.shape[2], upscaled.shape[3])
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)

        if multimask_output:
            out_masks = masks[:, 1:]
            out_iou = iou_pred[:, 1:]
            out_tokens = mask_tokens_out[:, 1:]
        else:
            out_masks = masks[:, 0:1]
            out_iou = iou_pred[:, 0:1]
            out_tokens = mask_tokens_out[:, 0:1]

        if return_all:
            return out_masks, out_iou, out_tokens, object_score_logits
        return out_masks, out_iou


class SAMPromptEncoder(nn.Module):
    def __init__(self, d_model=256, image_embedding_size=(72, 72), input_image_size=(1008, 1008), device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = d_model
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)
        self.point_embeddings = nn.ModuleList([
            operations.Embedding(1, d_model, device=device, dtype=dtype) for _ in range(4)
        ])
        self.not_a_point_embed = operations.Embedding(1, d_model, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.mask_downscaling = nn.Sequential(
            operations.Conv2d(1, 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(4, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(4, 16, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(16, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(16, d_model, kernel_size=1, device=device, dtype=dtype),
        )
        self.no_mask_embed = operations.Embedding(1, d_model, device=device, dtype=dtype)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size)

    def forward(self, points=None, boxes=None, masks=None):
        ref = points[0] if points is not None else boxes if boxes is not None else masks
        B = 1
        sparse = torch.empty((B, 0, self.embed_dim), device=ref.device, dtype=ref.dtype)

        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            # Pad with an extra point (label=-1) when no boxes are provided (matching reference)
            if boxes is None:
                coords = torch.cat([coords, torch.zeros(B, 1, 2, device=coords.device, dtype=coords.dtype)], dim=1)
                labels = torch.cat([labels, -torch.ones(B, 1, device=labels.device, dtype=labels.dtype)], dim=1)
            pe = self.pe_layer.forward_with_coords(coords + 0.5, self.input_image_size)
            for i in range(4):
                pe[labels == i] += cast_to_input(self.point_embeddings[i].weight, ref)
            invalid = (labels == -1)
            pe[invalid] = 0.0
            pe[invalid] += cast_to_input(self.not_a_point_embed.weight, ref)
            sparse = torch.cat([sparse.expand(B, -1, -1), pe], dim=1)

        if boxes is not None:
            B = boxes.shape[0]
            corners = self.pe_layer.forward_with_coords((boxes.reshape(-1, 2, 2) + 0.5), self.input_image_size)
            corners[:, 0] += cast_to_input(self.point_embeddings[2].weight, ref)
            corners[:, 1] += cast_to_input(self.point_embeddings[3].weight, ref)
            sparse = torch.cat([sparse.expand(B, -1, -1), corners], dim=1)

        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = cast_to_input(self.no_mask_embed.weight, ref).reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1])

        return sparse, dense


class CXBlock(nn.Module):
    def __init__(self, dim=256, kernel_size=7, device=None, dtype=None, operations=None):
        super().__init__()
        self.dwconv = operations.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.pwconv1 = operations.Linear(dim, 4 * dim, device=device, dtype=dtype)
        self.pwconv2 = operations.Linear(4 * dim, dim, device=device, dtype=dtype)
        self.gamma = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(F.gelu(self.pwconv1(self.norm(x))))
        x.mul_(cast_to_input(self.gamma, x))
        return residual + x.permute(0, 3, 1, 2)


class MaskDownSampler(nn.Module):
    def __init__(self, out_dim=256, in_chans=1, channels=None, interpol_size=(1152, 1152), device=None, dtype=None, operations=None):
        super().__init__()
        self.interpol_size = list(interpol_size) if interpol_size else None
        if channels is None:
            channels = [4, 16, 64, out_dim]  # SAM3 default
        LN2d = LayerNorm2d_op(operations)
        layers = []
        prev = in_chans
        for ch in channels:
            layers += [operations.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
                       LN2d(ch, device=device, dtype=dtype), nn.GELU()]
            prev = ch
        layers.append(operations.Conv2d(prev, out_dim, kernel_size=1, device=device, dtype=dtype))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        if self.interpol_size is not None and list(x.shape[-2:]) != self.interpol_size:
            x = F.interpolate(x, size=self.interpol_size, mode="bilinear", align_corners=False, antialias=True)
        return self.encoder(x)


class Fuser(nn.Module):
    def __init__(self, dim=256, num_layers=2, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.Sequential(*[CXBlock(dim, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


# --- SAM3.1 Multiplex components ---

class DecoupledMemoryAttnLayer(nn.Module):
    """Decoupled cross-attention layer for SAM3.1: fuses image and memory projections."""

    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        # Self-attention projections (flat, not nested)
        self.self_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # Cross-attention projections
        self.cross_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # Image cross-attention (q/k only, fused with cross_attn)
        self.image_cross_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.image_cross_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # FFN
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, image, x, memory_image, memory, memory_image_pos=None,
                rope=None, num_k_exclude_rope=0):
        # Self-attention with RoPE
        normed = self.norm1(x)
        q = self.self_attn_q_proj(normed)
        k = self.self_attn_k_proj(normed)
        v = self.self_attn_v_proj(normed)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, 0)
        x = x + self.self_attn_out_proj(optimized_attention(q, k, v, self.num_heads, low_precision_attention=False))

        # Decoupled cross-attention: fuse image and memory projections
        normed = self.norm2(x)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(normed)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if memory_image_pos is not None:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, num_k_exclude_rope)
        x = x + self.cross_attn_out_proj(optimized_attention(q, k, v, self.num_heads, low_precision_attention=False))

        # FFN
        x = x + self.linear2(F.gelu(self.linear1(self.norm3(x))))
        return image, x


class DecoupledMemoryEncoder(nn.Module):
    """Memory attention encoder for SAM3.1 with decoupled cross-attention."""

    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, num_layers=4, image_size=1008, patch_size=14,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoupledMemoryAttnLayer(d_model, num_heads, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        hw = image_size // patch_size
        self.register_buffer("_rope", rope_2d(hw, hw, d_model // num_heads), persistent=False)

    def forward(self, x, memory, memory_pos=None, src_pos=None, num_k_exclude_rope=0,
                memory_image=None, memory_image_pos=None):
        image = x  # constant residual for decoupled cross-attention
        output = x
        if src_pos is not None:
            output = output + 0.1 * src_pos

        B, _, C = x.shape
        rope = self._rope.to(device=x.device)

        # memory_image: raw backbone features from past frames for decoupled cross-attention
        if memory_image is None:
            # Fallback: use spatial portion of memory (without obj pointers)
            num_spatial = memory.shape[1] - num_k_exclude_rope
            memory_image = memory[:, :num_spatial]
            memory_image_pos = memory_pos[:, :num_spatial] if memory_pos is not None else None
        # Pad memory_image to match memory length (zeros for obj pointer tokens)
        if memory_image.shape[1] < memory.shape[1]:
            pad_len = memory.shape[1] - memory_image.shape[1]
            pad = torch.zeros(B, pad_len, C, device=memory.device, dtype=memory.dtype)
            memory_image = torch.cat([memory_image, pad], dim=1)
            if memory_image_pos is not None:
                ptr_pos = memory_pos[:, -pad_len:] if memory_pos is not None else torch.zeros_like(pad)
                memory_image_pos = torch.cat([memory_image_pos, ptr_pos], dim=1)

        for layer in self.layers:
            image, output = layer(image, output, memory_image, memory,
                                  memory_image_pos=memory_image_pos, rope=rope,
                                  num_k_exclude_rope=num_k_exclude_rope)

        return self.norm(output)


class DecoupledMemoryTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = DecoupledMemoryEncoder(d_model, num_heads, dim_ff, num_layers,
                                              device=device, dtype=dtype, operations=operations)


class MemoryBackbone(nn.Module):
    """Memory encoder: downsamples mask, fuses with pixel features, optionally compresses."""

    def __init__(self, d_model=256, out_dim=None, in_chans=1, channels=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(d_model, in_chans=in_chans, channels=channels, device=device, dtype=dtype, operations=operations)
        self.pix_feat_proj = operations.Conv2d(d_model, d_model, kernel_size=1, device=device, dtype=dtype)
        self.fuser = Fuser(d_model, num_layers=2, device=device, dtype=dtype, operations=operations)
        self.has_out_proj = out_dim is not None and out_dim != d_model
        if self.has_out_proj:
            self.out_proj = operations.Conv2d(d_model, out_dim, kernel_size=1, device=device, dtype=dtype)
            feat_dim = out_dim
        else:
            feat_dim = d_model
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=feat_dim, normalize=True)

    def forward(self, image_features, mask_for_mem, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            mask_for_mem = mask_for_mem.sigmoid()
        mask_features = self.mask_downsampler(cast_to_input(mask_for_mem, image_features))
        if mask_features.shape[-2:] != image_features.shape[-2:]:
            mask_features = F.interpolate(mask_features, size=image_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.pix_feat_proj(image_features) + mask_features
        features = self.fuser(features)
        if self.has_out_proj:
            features = self.out_proj(features)
        pos = cast_to_input(self.position_encoding(features), features)
        return {"vision_features": features, "vision_pos_enc": [pos]}


class MultiplexMaskDecoder(nn.Module):
    """SAM mask decoder for SAM3.1 multiplex: predicts masks for num_multiplex objects simultaneously.

    Uses multimask_outputs_only=True: num_mask_output_per_object = num_multimask_outputs (no +1).
    Hypernetwork MLPs are shared across multiplex objects.
    Token order: [obj_score_token(M), iou_token(M), mask_tokens(M*T)].
    """

    def __init__(self, d_model=256, num_multiplex=16, num_multimask_outputs=3, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_multiplex = num_multiplex
        self.num_mask_output_per_object = num_multimask_outputs  # 3 (multimask_outputs_only)
        total_mask_tokens = num_multiplex * self.num_mask_output_per_object  # 48

        self.transformer = SAMTwoWayTransformer(depth=2, embedding_dim=d_model, num_heads=8, mlp_dim=2048, device=device, dtype=dtype, operations=operations)

        self.obj_score_token = operations.Embedding(num_multiplex, d_model, device=device, dtype=dtype)
        self.iou_token = operations.Embedding(num_multiplex, d_model, device=device, dtype=dtype)
        self.mask_tokens = operations.Embedding(total_mask_tokens, d_model, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(d_model // 4, device=device, dtype=dtype), nn.GELU(),
            operations.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2, device=device, dtype=dtype), nn.GELU(),
        )
        self.conv_s0 = operations.Conv2d(d_model, d_model // 8, kernel_size=1, device=device, dtype=dtype)
        self.conv_s1 = operations.Conv2d(d_model, d_model // 4, kernel_size=1, device=device, dtype=dtype)

        # Shared across all multiplex objects (one per mask output)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3, device=device, dtype=dtype, operations=operations)
            for _ in range(self.num_mask_output_per_object)
        ])
        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_output_per_object, 3, device=device, dtype=dtype, operations=operations)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3, device=device, dtype=dtype, operations=operations)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                high_res_features=None, multimask_output=False, return_all=False, extra_per_object_embeddings=None):
        B = sparse_prompt_embeddings.shape[0]
        M = self.num_multiplex
        T = self.num_mask_output_per_object

        # Token order: [obj_score(M), iou(M), mask(M*T)]
        ref = sparse_prompt_embeddings
        mask_tokens = cast_to_input(self.mask_tokens.weight, ref)
        if extra_per_object_embeddings is not None:
            mask_tokens = mask_tokens.view(1, M, T, -1).expand(B, -1, -1, -1) + extra_per_object_embeddings.unsqueeze(2)
            mask_tokens = mask_tokens.flatten(1, 2)  # [B, M*T, C]
            other_tokens = torch.cat([cast_to_input(self.obj_score_token.weight, ref),
                                      cast_to_input(self.iou_token.weight, ref)], dim=0).unsqueeze(0).expand(B, -1, -1)
            tokens = torch.cat([other_tokens, mask_tokens, sparse_prompt_embeddings], dim=1)
        else:
            tokens = torch.cat([cast_to_input(self.obj_score_token.weight, ref),
                                cast_to_input(self.iou_token.weight, ref), mask_tokens], dim=0)
            tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        hs, src_out = self.transformer(src.flatten(2).permute(0, 2, 1), pos_src.flatten(2).permute(0, 2, 1), tokens)

        # Parse output tokens
        obj_score_token_out = hs[:, :M]
        iou_token_out = hs[:, M:2 * M]
        mask_tokens_out = hs[:, 2 * M:2 * M + M * T]

        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        upscaled = _upscale_masks(self.output_upscaling, self.conv_s0, self.conv_s1, src_out, high_res_features)

        # Reshape mask tokens to [B, M, T, C] and apply shared hypernetwork MLPs per mask output index
        mask_tokens_2d = mask_tokens_out.view(B, M, T, -1)
        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_2d[:, :, i, :])  # [B, M, C//8]
            for i in range(T)
        ], dim=2)  # [B, M, T, C//8]

        # Generate masks: [B, M*T, H*W] -> [B, M, T, H, W]
        masks = torch.bmm(hyper_in.flatten(1, 2), upscaled.flatten(2)).view(b, M, T, upscaled.shape[2], upscaled.shape[3])

        # IoU and object scores
        iou_pred = self.iou_prediction_head(iou_token_out).view(b, M, T)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)  # [B, M, 1]

        # multimask_outputs_only: always output all T masks (no singlemask token)
        sam_tokens_out = mask_tokens_2d[:, :, 0:1]  # [B, M, 1, C]

        if return_all:
            return masks, iou_pred, sam_tokens_out, object_score_logits
        return masks, iou_pred


class SAM3Tracker(nn.Module):
    def __init__(self, d_model=256, mem_dim=64, num_maskmem=7, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()

        # Memory attention transformer
        self.transformer = MemoryTransformer(d_model, num_heads=1, kv_dim=mem_dim, dim_ff=2048, num_layers=4,
                                             device=device, dtype=dtype, operations=operations)
        # SAM components
        self.sam_mask_decoder = SAMMaskDecoder(d_model, device=device, dtype=dtype, operations=operations)
        self.sam_prompt_encoder = SAMPromptEncoder(d_model, device=device, dtype=dtype, operations=operations)

        # Memory backbone
        self.maskmem_backbone = MemoryBackbone(d_model, out_dim=mem_dim, device=device, dtype=dtype, operations=operations)

        # Standalone parameters
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim, device=device, dtype=dtype))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))
        self.register_buffer("no_mem_pos_enc", torch.zeros(1, 1, d_model, device=device, dtype=dtype))  # checkpoint key, unused in forward
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, mem_dim, device=device, dtype=dtype))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, d_model, device=device, dtype=dtype))

        # Object pointer projection
        self.obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)
        self.obj_ptr_tpos_proj = operations.Linear(d_model, mem_dim, device=device, dtype=dtype)

        # Mask downsample: Conv2d stride 4 to reduce GT mask to SAM logit scale
        self.mask_downsample = operations.Conv2d(1, 1, kernel_size=4, stride=4, device=device, dtype=dtype)

        # Config
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.num_maskmem = num_maskmem
        self.image_size = 1008
        self.backbone_stride = 14
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0

    def _no_obj_blend(self, obj_ptr, is_obj):
        alpha = is_obj.to(obj_ptr.dtype)
        return torch.lerp(cast_to_input(self.no_obj_ptr, obj_ptr), obj_ptr, alpha)

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, box_inputs=None,
                           high_res_features=None, multimask_output=False):
        return forward_sam_heads(backbone_features, self.sam_prompt_encoder, self.sam_mask_decoder,
                                 self.obj_ptr_proj, self._no_obj_blend, self.image_size,
                                 point_inputs, mask_inputs, box_inputs, high_res_features, multimask_output)

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        return use_mask_as_output(backbone_features, high_res_features, mask_inputs,
                                  self.mask_downsample, self.sam_prompt_encoder, self.sam_mask_decoder,
                                  self.obj_ptr_proj, self._no_obj_blend, self.image_size, self.backbone_stride)

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames):
        """Fuse current frame features with memory from previous frames."""
        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(0, 2, 1).view(B, C, H, W)

        if is_init_cond_frame:
            # First conditioning frame: no memory yet, add no_mem_embed
            pix_feat = current_vision_feats[-1] + cast_to_input(self.no_mem_embed, current_vision_feats[-1])
            return to_spatial(pix_feat, H, W)

        to_cat_memory, to_cat_memory_pos, _, _, cond_outputs = collect_memory_tokens(
            output_dict, frame_idx, self.num_maskmem, self.maskmem_tpos_enc, device)

        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = []
        for t, out in cond_outputs.items():
            if t <= frame_idx:
                pos_and_ptrs.append(((frame_idx - t), out["obj_ptr"].to(device)))
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = output_dict["non_cond_frame_outputs"].get(t, None)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"].to(device)))

        num_obj_ptr_tokens = 0
        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=1)  # [B, N, C=256]

            # Temporal position encoding for pointers
            obj_pos = compute_tpos_enc(
                list(pos_list), device, self.d_model, self.obj_ptr_tpos_proj,
                max_abs_pos=max_obj_ptrs, dtype=current_vision_feats[-1].dtype
            )  # [N, mem_dim=64]
            obj_pos = obj_pos.unsqueeze(0).expand(B, -1, -1)  # [B, N, 64]

            # Split each 256-dim pointer into 4 x 64-dim tokens
            if self.mem_dim < C:
                N = obj_ptrs.shape[1]
                obj_ptrs = obj_ptrs.view(B, N, C // self.mem_dim, self.mem_dim)  # [B, N, 4, 64]
                obj_ptrs = obj_ptrs.reshape(B, N * (C // self.mem_dim), self.mem_dim)  # [B, N*4, 64]
                obj_pos = obj_pos.unsqueeze(2).expand(-1, -1, C // self.mem_dim, -1)
                obj_pos = obj_pos.reshape(B, N * (C // self.mem_dim), self.mem_dim)  # [B, N*4, 64]

            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[1]

        if len(to_cat_memory) == 0:
            # No memory available yet, add no_mem_embed
            pix_feat = current_vision_feats[-1] + cast_to_input(self.no_mem_embed, current_vision_feats[-1])
            return to_spatial(pix_feat, H, W)

        # Concatenate all memory and position encodings [B, total_mem, mem_dim=64]
        memory = torch.cat(to_cat_memory, dim=1)
        memory_pos = torch.cat(to_cat_memory_pos, dim=1)

        # Run memory attention encoder
        pix_feat = current_vision_feats[-1]  # [B, HW, C]
        src_pos = current_vision_pos_embeds[-1]  # [B, HW, C]

        pix_feat_with_mem = self.transformer.encoder(
            x=pix_feat,
            memory=memory,
            src_pos=src_pos,
            memory_pos=memory_pos,
            num_k_exclude_rope=num_obj_ptr_tokens,
        )
        return to_spatial(pix_feat_with_mem, H, W)

    def _encode_new_memory(self, pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False):
        """Encode predicted mask into memory features."""
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        mask_for_mem.mul_(self.sigmoid_scale_for_mem_enc).add_(self.sigmoid_bias_for_mem_enc)

        maskmem_out = self.maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # Add no_obj_embed for occluded objects
        alpha = (object_score_logits > 0).to(maskmem_features.dtype)[..., None, None]
        no_obj = cast_to_input(self.no_obj_embed_spatial, maskmem_features)[..., None, None].expand_as(maskmem_features)
        return maskmem_features + (1 - alpha) * no_obj, maskmem_pos_enc

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, mask_inputs, output_dict,
                   num_frames, point_inputs=None):
        """Track one frame: fuse with memory, predict mask, encode memory."""
        current_out = {}

        # High-res features for SAM head [stride-8, stride-4]
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.view(x.shape[0], feat_sizes[i][0], feat_sizes[i][1], -1).permute(0, 3, 1, 2)
                for i, x in enumerate(current_vision_feats[:-1])
            ]
        else:
            high_res_features = None

        # Top-level feature for memory
        H, W = feat_sizes[-1]

        if mask_inputs is not None:
            # Conditioning frame: use mask directly
            pix_feat = to_spatial(current_vision_feats[-1], H, W)
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            # Track frame: fuse with memory, then SAM decoder
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Use multimask for point prompts on init frames (picks best of 3 candidates)
            num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
            multimask_output = is_init_cond_frame and 0 < num_pts <= 1
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        (low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs

        # Clean low-res masks: remove sprinkles and fill holes
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, max_area=200)
        high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        # Encode memory
        if self.num_maskmem > 0:
            pix_feat = to_spatial(current_vision_feats[-1], H, W)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=pix_feat,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _compute_backbone_frame(self, backbone_fn, frame, frame_idx=None):
        vision_feats, vision_pos, feat_sizes, _, _ = _compute_backbone(backbone_fn, frame, frame_idx)
        # SAM3: drop last FPN level
        return vision_feats[:-1], vision_pos[:-1], feat_sizes[:-1]

    def _track_single_object(self, backbone_fn, images, initial_mask, pbar=None,
                             target_device=None, target_dtype=None):
        """Track one object, computing backbone per frame to save VRAM."""
        N = images.shape[0]
        device = target_device if target_device is not None else images.device
        dt = target_dtype if target_dtype is not None else images.dtype
        size = self.image_size
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        all_masks = []

        for frame_idx in tqdm(range(N), desc="tracking"):
            vision_feats, vision_pos, feat_sizes = self._compute_backbone_frame(
                backbone_fn, _prep_frame(images, slice(frame_idx, frame_idx + 1), device, dt, size), frame_idx=frame_idx)
            mask_input = None
            if frame_idx == 0:
                mask_input = F.interpolate(initial_mask.to(device=device, dtype=dt),
                    size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                mask_input = (mask_input > 0.5).to(dt)

            current_out = self.track_step(
                frame_idx=frame_idx, is_init_cond_frame=(frame_idx == 0),
                current_vision_feats=vision_feats, current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes, mask_inputs=mask_input, output_dict=output_dict, num_frames=N)

            if frame_idx == 0:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                lookback = max(self.num_maskmem, self.max_obj_ptrs_in_encoder)
                for old_idx in list(output_dict["non_cond_frame_outputs"]):
                    if old_idx < frame_idx - lookback:
                        del output_dict["non_cond_frame_outputs"][old_idx]
            # Move masks to CPU immediately to free VRAM
            all_masks.append(current_out["pred_masks_high_res"].to(comfy.model_management.intermediate_device()))
            if pbar is not None:
                pbar.update(1)

        return torch.cat(all_masks, dim=0)  # [N, 1, H, W]

    def track_video(self, backbone_fn, images, initial_masks, pbar=None,
                    target_device=None, target_dtype=None, **kwargs):
        """Track one or more objects across video frames.

        Args:
            backbone_fn: callable that returns (sam2_features, sam2_positions, trunk_out) for a frame
            images: [N, 3, H, W] CPU full-res video frames (resized per-frame to self.image_size)
            initial_masks: [N_obj, 1, H, W] binary masks for first frame (one per object)
            pbar: optional progress bar

        Returns:
            [N, N_obj, image_size, image_size] predicted mask logits per frame per object
        """
        N_obj = initial_masks.shape[0]
        per_object = []
        for obj_idx in range(N_obj):
            obj_masks = self._track_single_object(
                backbone_fn, images, initial_masks[obj_idx:obj_idx + 1], pbar=pbar,
                target_device=target_device, target_dtype=target_dtype)
            per_object.append(obj_masks)

        return torch.cat(per_object, dim=1)  # [N, N_obj, H, W]


class SAM31Tracker(nn.Module):
    """SAM3.1 multiplex tracker: decoupled memory attention, dual decoder, 16-object multiplex."""

    def __init__(self, d_model=256, mem_dim=256, num_maskmem=7, num_multiplex=16, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.num_maskmem = num_maskmem
        self.num_multiplex = num_multiplex
        self.image_size = 1008
        self.backbone_stride = 14
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 2.0
        self.sigmoid_bias_for_mem_enc = -1.0

        # Memory attention (decoupled cross-attention, 8 heads matching reference)
        self.transformer = DecoupledMemoryTransformer(d_model, num_heads=8, dim_ff=2048, num_layers=4,
                                                      device=device, dtype=dtype, operations=operations)

        # Propagation decoder (multiplex: 16 objects, multimask_outputs_only)
        self.sam_mask_decoder = MultiplexMaskDecoder(d_model, num_multiplex, num_multimask_outputs=3,
                                                     device=device, dtype=dtype, operations=operations)
        # Interactive decoder (single object, same as SAM3)
        self.interactive_sam_mask_decoder = SAMMaskDecoder(d_model, num_multimask_outputs=3,
                                                           device=device, dtype=dtype, operations=operations)
        self.interactive_sam_prompt_encoder = SAMPromptEncoder(d_model, device=device, dtype=dtype, operations=operations)

        # Memory backbone (mem_dim=256, no out_proj compression)
        self.maskmem_backbone = MemoryBackbone(d_model, in_chans=num_multiplex * 2, channels=[16, 64, 256, 1024],
                                                device=device, dtype=dtype, operations=operations)

        # Standalone parameters
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim, device=device, dtype=dtype))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(num_multiplex, mem_dim, device=device, dtype=dtype))
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))

        # Object pointer projection
        self.obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)
        self.obj_ptr_tpos_proj = operations.Linear(d_model, mem_dim, device=device, dtype=dtype)
        self.no_obj_ptr_linear = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.interactive_obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)

        # Interactive mask downsample
        self.interactive_mask_downsample = operations.Conv2d(1, 1, kernel_size=4, stride=4, device=device, dtype=dtype)

        # Multiplex validity embeddings
        self.output_valid_embed = nn.Parameter(torch.zeros(num_multiplex, d_model, device=device, dtype=dtype))
        self.output_invalid_embed = nn.Parameter(torch.zeros(num_multiplex, d_model, device=device, dtype=dtype))

        # Position encoding for image (used by multiplex decoder)
        self.image_pe_layer = PositionEmbeddingRandom(d_model // 2)

    def _no_obj_blend(self, obj_ptr, is_obj):
        alpha = is_obj.to(obj_ptr.dtype)
        return torch.lerp(self.no_obj_ptr_linear(obj_ptr), obj_ptr, alpha)

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, box_inputs=None,
                           high_res_features=None, multimask_output=False):
        return forward_sam_heads(backbone_features, self.interactive_sam_prompt_encoder, self.interactive_sam_mask_decoder,
                                 self.interactive_obj_ptr_proj, self._no_obj_blend, self.image_size,
                                 point_inputs, mask_inputs, box_inputs, high_res_features, multimask_output)

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        return use_mask_as_output(backbone_features, high_res_features, mask_inputs,
                                  self.interactive_mask_downsample, self.interactive_sam_prompt_encoder,
                                  self.interactive_sam_mask_decoder, self.interactive_obj_ptr_proj,
                                  self._no_obj_blend, self.image_size, self.backbone_stride)

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats,
                                              current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
                                              multiplex_state=None):
        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        num_buc = multiplex_state.num_buckets if multiplex_state is not None else None

        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(0, 2, 1).view(B, C, H, W)

        if is_init_cond_frame:
            pix_feat = current_vision_feats[-1] + cast_to_input(self.interactivity_no_mem_embed, current_vision_feats[-1])
            return to_spatial(pix_feat, H, W)

        to_cat_memory, to_cat_memory_pos, to_cat_image_feat, to_cat_image_pos, cond_outputs = collect_memory_tokens(
            output_dict, frame_idx, self.num_maskmem, self.maskmem_tpos_enc, device,
            collect_image_feats=True, tpos_v2=True, num_buckets=num_buc)

        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = []
        for t, out in cond_outputs.items():
            if t <= frame_idx and "obj_ptr" in out:
                ptr = out["obj_ptr"].to(device)
                if num_buc is not None:
                    ptr = _pad_to_buckets(ptr, num_buc)
                pos_and_ptrs.append(((frame_idx - t), ptr))
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = output_dict["non_cond_frame_outputs"].get(t, None)
            if out is not None and "obj_ptr" in out:
                ptr = out["obj_ptr"].to(device)
                if num_buc is not None:
                    ptr = _pad_to_buckets(ptr, num_buc)
                pos_and_ptrs.append((t_diff, ptr))

        num_obj_ptr_tokens = 0
        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=1)  # [num_buckets, N, M, C]
            B_ptr = obj_ptrs.shape[0]
            N_ptrs = obj_ptrs.shape[1]
            M = obj_ptrs.shape[2]
            obj_ptrs = obj_ptrs.reshape(B_ptr, N_ptrs * M, -1)
            obj_pos = compute_tpos_enc(list(pos_list), device, self.d_model, self.obj_ptr_tpos_proj,
                                       max_abs_pos=max_obj_ptrs, dtype=current_vision_feats[-1].dtype)
            obj_pos = obj_pos.unsqueeze(0).expand(B_ptr, -1, -1)
            obj_pos = obj_pos.unsqueeze(2).expand(-1, -1, M, -1).reshape(B_ptr, N_ptrs * M, -1)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[1]

        if len(to_cat_memory) == 0:
            pix_feat = current_vision_feats[-1] + cast_to_input(self.interactivity_no_mem_embed, current_vision_feats[-1])
            return to_spatial(pix_feat, H, W)

        memory = torch.cat(to_cat_memory, dim=1)
        memory_pos = torch.cat(to_cat_memory_pos, dim=1)

        # Expand vision features to num_buckets if memory has more buckets than B
        mem_B = memory.shape[0]
        x = current_vision_feats[-1]
        x_pos = current_vision_pos_embeds[-1]
        if x.shape[0] < mem_B:
            x = x.expand(mem_B, -1, -1)
            x_pos = x_pos.expand(mem_B, -1, -1)

        if len(to_cat_image_feat) > 0:
            # Decoupled cross-attention: separate image features from memory
            memory_image = cast_to_input(torch.cat(to_cat_image_feat, dim=1), x)
            memory_image_pos = cast_to_input(torch.cat(to_cat_image_pos, dim=1), x)
            if memory_image.shape[0] < mem_B:
                memory_image = memory_image.expand(mem_B, -1, -1)
                memory_image_pos = memory_image_pos.expand(mem_B, -1, -1)
            pix_feat_with_mem = self.transformer.encoder(
                x=x,
                memory=cast_to_input(memory, x),
                memory_pos=cast_to_input(memory_pos, x),
                src_pos=cast_to_input(x_pos, x),
                num_k_exclude_rope=num_obj_ptr_tokens,
                memory_image=memory_image,
                memory_image_pos=memory_image_pos,
            )
        else:
            pix_feat_with_mem = self.transformer.encoder(
                x=x,
                memory=memory,
                memory_pos=memory_pos,
                src_pos=x_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )
        return to_spatial(pix_feat_with_mem, H, W)

    def _encode_new_memory(self, pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False,
                           multiplex_state=None, is_conditioning=False, cond_obj_mask=None):
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem.mul_(self.sigmoid_scale_for_mem_enc).add_(self.sigmoid_bias_for_mem_enc)

        # Mux masks: [N_obj, 1, H, W] -> [num_buckets, M, H, W]
        mux_masks = multiplex_state.mux(mask_for_mem[:, 0])

        # Conditioning channel: 1.0 = clean mask (trust it), 0.0 = propagation (noisy)
        N_obj = mask_for_mem.shape[0]
        cond_values = torch.full((N_obj,), 0.0, device=mask_for_mem.device, dtype=mask_for_mem.dtype)
        if is_conditioning:
            cond_values[:] = 1.0
        elif cond_obj_mask is not None:
            cond_values[cond_obj_mask] = 1.0
        cond_spatial = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem[:, 0:1, :, :]).squeeze(1)
        mux_cond = multiplex_state.mux(cond_spatial)  # [num_buckets, M, H, W]
        mux_input = torch.cat([mux_masks, mux_cond], dim=1)  # [num_buckets, 2*M, H, W]

        maskmem_out = self.maskmem_backbone(pix_feat, mux_input, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # Add no_obj_embed_spatial for occluded objects
        is_obj = (object_score_logits > 0).float()  # [N_obj, 1]
        mux_is_obj = multiplex_state.mux(is_obj)  # [num_buckets, M, 1]
        no_obj_embed = cast_to_input(self.no_obj_embed_spatial, maskmem_features)  # [M, C]
        no_obj_spatial = no_obj_embed.unsqueeze(0)[..., None, None]  # [1, M, C, 1, 1]
        # Expand and sum across multiplex slots weighted by (1 - is_obj)
        alpha = mux_is_obj[..., None, None]  # [num_buckets, M, 1, 1, 1]
        per_slot_no_obj = ((1 - alpha) * no_obj_spatial).sum(dim=1)  # [num_buckets, C, 1, 1]
        maskmem_features = maskmem_features + per_slot_no_obj.expand_as(maskmem_features)

        return maskmem_features, maskmem_pos_enc

    def _forward_propagation(self, backbone_features, high_res_features=None, multiplex_state=None):
        """Propagation path using the multiplex SAM decoder (no prompts)."""
        B = backbone_features.shape[0]
        device = backbone_features.device

        # Suppression embeddings from valid object mask
        valid_mask = cast_to_input(multiplex_state.get_valid_object_mask().unsqueeze(-1).float(), backbone_features)
        output_valid = cast_to_input(self.output_valid_embed, backbone_features).unsqueeze(0)
        output_invalid = cast_to_input(self.output_invalid_embed, backbone_features).unsqueeze(0)
        extra_embed = valid_mask * output_valid + (1 - valid_mask) * output_invalid

        image_pe = self.image_pe_layer((backbone_features.shape[-2], backbone_features.shape[-1]), device=backbone_features.device)
        image_pe = cast_to_input(image_pe, backbone_features)

        masks, iou_pred, sam_tokens_out, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features, image_pe=image_pe,
            sparse_prompt_embeddings=torch.empty(B, 0, self.d_model, device=device, dtype=backbone_features.dtype),
            dense_prompt_embeddings=torch.zeros(B, self.d_model, *backbone_features.shape[-2:], device=device, dtype=backbone_features.dtype),
            high_res_features=high_res_features, multimask_output=True, return_all=True,
            extra_per_object_embeddings=extra_embed.expand(B, -1, -1),
        )
        # masks: [B=num_buckets, M, T, H, W]
        # Demux to per-object: [N_obj, T, H, W]
        masks_obj = multiplex_state.demux(masks)
        iou_obj = multiplex_state.demux(iou_pred)
        score_obj = multiplex_state.demux(object_score_logits)
        tokens_obj = multiplex_state.demux(sam_tokens_out)

        # Select best mask by IoU for each object
        best_idx = torch.argmax(iou_obj, dim=-1)  # [N_obj]
        N_obj = masks_obj.shape[0]
        obj_range = torch.arange(N_obj, device=device)
        low_res_masks = masks_obj[obj_range, best_idx].unsqueeze(1)  # [N_obj, 1, H, W]
        # Suppress masks for objects with low confidence
        is_obj = score_obj > 0
        low_res_masks = torch.where(is_obj[:, :, None, None], low_res_masks,
                                     torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_masks.dtype))
        high_res_masks = F.interpolate(low_res_masks.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        # Object pointer: compute per-object, mux for storage as [num_buckets, M, C]
        sam_token = tokens_obj[:, 0]  # [N_obj, C]
        obj_ptr = self.obj_ptr_proj(sam_token)
        is_obj = (score_obj > 0).float()
        no_obj = self.no_obj_ptr_linear(obj_ptr)
        obj_ptr = is_obj * obj_ptr + (1 - is_obj) * no_obj
        obj_ptr_muxed = multiplex_state.mux(obj_ptr)  # [num_buckets, M, C]

        return low_res_masks, high_res_masks, obj_ptr_muxed, score_obj

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                   feat_sizes, mask_inputs, output_dict, num_frames, point_inputs=None,
                   interactive_high_res=None, interactive_backbone=None, propagation_high_res=None,
                   multiplex_state=None, run_mem_encoder=True):
        current_out = {}
        H, W = feat_sizes[-1]

        if mask_inputs is not None:
            # Conditioning frame: use interactive features if available, else propagation
            if interactive_backbone is not None:
                pix_feat = interactive_backbone
                # Add no_mem_embed for interactive path
                pix_flat = pix_feat.flatten(2)
                bf = pix_flat.permute(0, 2, 1) + cast_to_input(self.interactivity_no_mem_embed, pix_flat)
                pix_feat = to_spatial(bf, H, W)
                hi_res = interactive_high_res
            else:
                # Fallback: interactive backbone not available (e.g. called outside track_video).
                # Propagation features work but may produce lower-quality conditioning.
                pix_feat = to_spatial(current_vision_feats[-1], H, W)
                hi_res = propagation_high_res
            sam_outputs = self._use_mask_as_output(pix_feat, hi_res, mask_inputs)
        elif point_inputs is not None:
            # Interactive path: use interactive SAM decoder
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes, output_dict=output_dict, num_frames=num_frames,
                multiplex_state=multiplex_state,
            )
            hi_res = interactive_high_res if interactive_high_res is not None else propagation_high_res
            num_pts = point_inputs["point_labels"].size(1)
            multimask_output = is_init_cond_frame and 0 < num_pts <= 1
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem, point_inputs=point_inputs,
                high_res_features=hi_res, multimask_output=multimask_output,
            )
        else:
            # Propagation path: use multiplex SAM decoder with propagation features
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes, output_dict=output_dict, num_frames=num_frames,
                multiplex_state=multiplex_state,
            )
            sam_outputs = self._forward_propagation(pix_feat_with_mem, propagation_high_res,
                                                     multiplex_state=multiplex_state)

        (low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs

        # Mux obj_ptr if it came from interactive path (shape [B, C]) vs propagation ([num_buckets, M, C])
        if multiplex_state is not None and obj_ptr.dim() == 2:
            obj_ptr = multiplex_state.mux(obj_ptr)  # [N_obj, C] -> [num_buckets, M, C]

        # Encode memory (can be deferred with run_mem_encoder=False)
        if run_mem_encoder and self.num_maskmem > 0:
            pix_feat = to_spatial(current_vision_feats[-1], H, W)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=pix_feat, pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                multiplex_state=multiplex_state,
                is_conditioning=(mask_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        # Store propagation image features for decoupled memory attention
        current_out["image_features"] = current_vision_feats[-1]  # [B, HW, C]
        current_out["image_pos_enc"] = current_vision_pos_embeds[-1]  # [B, HW, C]

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        return current_out

    def _compute_backbone_frame(self, backbone_fn, frame, frame_idx=None):
        vision_feats, vision_pos, feat_sizes, features, trunk_out = _compute_backbone(backbone_fn, frame, frame_idx)
        return vision_feats, vision_pos, feat_sizes, list(features[:-1]), trunk_out

    @staticmethod
    def _suppress_recently_occluded(low_res_masks, last_occluded, frame_idx, threshold=0.3):
        """Suppress overlapping masks for objects that were most recently occluded.
        Prevents corrupted masks from occluded objects from contaminating other objects."""
        N_obj = low_res_masks.shape[0]
        if N_obj <= 1:
            return low_res_masks
        binary = low_res_masks[:, 0] > 0  # [N_obj, H, W]
        iou = _compute_mask_overlap(low_res_masks[:, 0], low_res_masks[:, 0])
        overlapping = torch.triu(iou >= threshold, diagonal=1)  # [N, N] upper triangle
        last_occ_i = last_occluded.unsqueeze(1)  # [N, 1]
        last_occ_j = last_occluded.unsqueeze(0)  # [1, N]
        # Suppress the more recently occluded object in each overlapping pair
        suppress_i = overlapping & (last_occ_i > last_occ_j) & (last_occ_j > -1)
        suppress_j = overlapping & (last_occ_j > last_occ_i) & (last_occ_i > -1)
        to_suppress = suppress_i.any(dim=1) | suppress_j.any(dim=0)
        # Update last_occluded for occluded/suppressed objects
        is_empty = ~binary.any(dim=(-1, -2))
        newly_occluded = is_empty | to_suppress
        last_occluded[newly_occluded] = frame_idx
        # Suppress masks
        low_res_masks[to_suppress] = -10.0
        return low_res_masks

    def _deferred_memory_encode(self, current_out, N_obj, vision_feats, feat_sizes, mux_state, device,
                                cond_obj_mask=None):
        """Deferred memory encoding for propagation frames. cond_obj_mask: per-object bool for conditioning."""
        low_res_masks = current_out["pred_masks"]  # [N_obj, 1, H_low, W_low]

        if N_obj > 1:
            lr = low_res_masks.squeeze(1)  # [N_obj, H, W]
            max_obj = torch.argmax(lr, dim=0, keepdim=True)
            batch_inds = torch.arange(N_obj, device=device)[:, None, None]
            pixel_nol = torch.where(max_obj == batch_inds, lr, torch.clamp(lr, max=-10.0))
            area_before = (lr > 0).sum(dim=(-1, -2)).float().clamp(min=1)
            area_after = (pixel_nol > 0).sum(dim=(-1, -2)).float()
            shrink_ok = (area_after / area_before) >= 0.3
            low_res_masks = torch.where(
                shrink_ok[:, None, None, None].expand_as(low_res_masks),
                low_res_masks, torch.clamp(low_res_masks, max=-10.0))

        interpol_size = self.maskmem_backbone.mask_downsampler.interpol_size
        mem_masks = F.interpolate(low_res_masks, size=interpol_size,
                                  mode="bilinear", align_corners=False)

        obj_scores = torch.where(
            (mem_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        pix_feat = to_spatial(vision_feats[-1], feat_sizes[-1][0], feat_sizes[-1][1])
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            pix_feat=pix_feat, pred_masks_high_res=mem_masks,
            object_score_logits=obj_scores,
            multiplex_state=mux_state, cond_obj_mask=cond_obj_mask)
        current_out["maskmem_features"] = maskmem_features
        current_out["maskmem_pos_enc"] = maskmem_pos_enc

    def _add_detected_objects(self, new_masks, mux_state, vision_feats, feat_sizes, current_out):
        """Grow MultiplexState with new detections, merge masks, re-encode memory. Modifies current_out."""
        n_old = mux_state.total_valid_entries
        mux_state.add_objects(new_masks.shape[0])
        N_obj = mux_state.total_valid_entries
        # Stored memory with old bucket counts is padded at read time by _pad_to_buckets
        for k in ("pred_masks", "pred_masks_high_res"):
            det = F.interpolate(new_masks.unsqueeze(1), size=current_out[k].shape[-2:],
                                mode="bilinear", align_corners=False)
            current_out[k] = torch.cat([current_out[k], det], dim=0)
        if self.num_maskmem > 0:
            # Mark new objects as conditioning (clean detection masks) so model trusts them
            cond_mask = torch.zeros(N_obj, dtype=torch.bool, device=new_masks.device)
            cond_mask[n_old:] = True
            self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes,
                                         mux_state, new_masks.device, cond_obj_mask=cond_mask)

    def _condition_with_masks(self, masks, frame_idx, vision_feats, vision_pos, feat_sizes,
                              high_res_prop, output_dict, N, mux_state, backbone_obj, frame,
                              trunk_out, threshold=0.5):
        """Condition tracker with masks on a frame."""
        mask_input = F.interpolate(masks if masks.dim() == 4 else masks.unsqueeze(1),
            size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mask_input = (mask_input > threshold).to(masks.dtype)
        hi_res = lo_feat = None
        if backbone_obj is not None and backbone_obj.multiplex:
            _, _, itf, _ = backbone_obj(frame, tracker_mode="interactive", cached_trunk=trunk_out, tracker_only=True)
            hi_res, lo_feat = itf[:-1], itf[-1]
        current_out = self.track_step(
            frame_idx=frame_idx, is_init_cond_frame=True, current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=mask_input,
            output_dict=output_dict, num_frames=N, interactive_high_res=hi_res,
            interactive_backbone=lo_feat, propagation_high_res=high_res_prop,
            multiplex_state=mux_state, run_mem_encoder=True)
        output_dict["cond_frame_outputs"][frame_idx] = current_out
        return current_out

    def _match_and_add_detections(self, det_masks, det_scores, current_out, mux_state,
                                  vision_feats, feat_sizes, device, max_objects=0,
                                  keep_alive=None):
        """Match detections against tracked masks, add new objects, recondition degraded tracks.
        Updates keep_alive counters: +1 for matched tracks, -1 for unmatched."""
        N_obj = mux_state.total_valid_entries
        if det_masks.shape[0] == 0:
            if keep_alive is not None:
                for i in range(N_obj):
                    keep_alive[i] = max(-4, keep_alive.get(i, 0) - 1)
            return []

        # Match at low-res (like reference)
        trk_masks = current_out["pred_masks"][:, 0]  # [N_obj, H_low, W_low]
        det_resized = F.interpolate(det_masks.unsqueeze(1), size=trk_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)[:, 0]
        overlap = _compute_mask_overlap(det_resized, trk_masks)

        # Update keep_alive and find matched tracks
        matched = set()
        if overlap.shape[1] > 0:
            matched = set((overlap >= 0.5).any(dim=0).nonzero(as_tuple=True)[0].tolist())
        if keep_alive is not None:
            for i in range(N_obj):
                if i in matched:
                    keep_alive[i] = min(8, keep_alive.get(i, 0) + 1)
                else:
                    keep_alive[i] = max(-4, keep_alive.get(i, 0) - 1)

        # Recondition: high-confidence detections (>=0.8) with high overlap refresh tracked masks
        reconditioned = False
        if det_scores is not None and overlap.shape[1] > 0:
            HIGH_CONF = 0.8
            for det_idx in range(overlap.shape[0]):
                if det_scores[det_idx] < HIGH_CONF:
                    continue
                best_trk = overlap[det_idx].argmax().item()
                if overlap[det_idx, best_trk] >= 0.5:
                    # Replace tracked mask with fresh detection mask
                    current_out["pred_masks"][best_trk] = det_resized[det_idx].unsqueeze(0)
                    det_hr = F.interpolate(det_masks[det_idx:det_idx+1].unsqueeze(1),
                        size=current_out["pred_masks_high_res"].shape[-2:],
                        mode="bilinear", align_corners=False)
                    current_out["pred_masks_high_res"][best_trk] = det_hr[0]
                    reconditioned = True

        # Re-encode memory if any tracks were reconditioned
        if reconditioned and self.num_maskmem > 0:
            self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes, mux_state, device)

        # Add new detections (not matching any track)
        if max_objects > 0 and N_obj >= max_objects:
            return []
        max_overlap = overlap.max(dim=1)[0] if overlap.shape[1] > 0 else torch.zeros(overlap.shape[0], device=device)
        new_dets = max_overlap < 0.5
        if new_dets.any():
            if max_objects > 0:
                slots = max_objects - N_obj
                new_dets = new_dets & (torch.cumsum(new_dets.int(), 0) <= slots)
            self._add_detected_objects(det_masks[new_dets], mux_state,
                                       vision_feats, feat_sizes, current_out)
            if keep_alive is not None:
                for i in range(N_obj, mux_state.total_valid_entries):
                    keep_alive[i] = 1
            return det_scores[new_dets].tolist() if det_scores is not None else [0.0] * new_dets.sum().item()
        return []

    INTERNAL_MAX_OBJECTS = 64  # Hard ceiling on accumulated tracks; max_objects=0 or any value above this is clamped here.

    def track_video_with_detection(self, backbone_fn, images, initial_masks, detect_fn=None,
                                   new_det_thresh=0.5, max_objects=0, detect_interval=1,
                                   backbone_obj=None, pbar=None, target_device=None, target_dtype=None):
        """Track with optional per-frame detection. Returns [N, max_N_obj, H, W] mask logits."""
        if max_objects <= 0 or max_objects > self.INTERNAL_MAX_OBJECTS:
            max_objects = self.INTERNAL_MAX_OBJECTS
        N = images.shape[0]
        device = target_device if target_device is not None else images.device
        dt = target_dtype if target_dtype is not None else images.dtype
        size = self.image_size
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        all_masks = []
        idev = comfy.model_management.intermediate_device()
        mux_state = None
        if initial_masks is not None:
            mux_state = MultiplexState(initial_masks.shape[0], self.num_multiplex, device, dt)
        obj_scores = []  # per-object detection score (1.0 for initial masks)
        keep_alive = {} if detect_fn is not None else None
        last_occluded = torch.empty(0, device=device, dtype=torch.long)  # per-object last occluded frame

        # Prefetch next frame's backbone on a separate CUDA stream
        prefetch = False
        backbone_stream = None
        if comfy.model_management.is_device_cuda(device):
            try:
                backbone_stream = torch.cuda.Stream(device=device)
                prefetch = True
            except RuntimeError:
                pass
        cur_bb = self._compute_backbone_frame(backbone_fn, _prep_frame(images, slice(0, 1), device, dt, size), frame_idx=0)

        for frame_idx in tqdm(range(N), desc="tracking"):
            vision_feats, vision_pos, feat_sizes, high_res_prop, trunk_out = cur_bb

            # Start next frame's backbone on separate stream (overlaps with current frame's work)
            if prefetch and frame_idx + 1 < N:
                backbone_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(backbone_stream):
                    next_bb = self._compute_backbone_frame(
                        backbone_fn, _prep_frame(images, slice(frame_idx + 1, frame_idx + 2), device, dt, size), frame_idx=frame_idx + 1)

            # Per-frame detection with NMS (skip if no detect_fn, or interval/max not met)
            det_masks = torch.empty(0, device=device)
            det_scores = None
            run_det = (detect_fn is not None
                       and frame_idx % max(detect_interval, 1) == 0
                       and not (max_objects > 0 and mux_state is not None
                                and mux_state.total_valid_entries >= max_objects))
            if run_det:
                det_out = detect_fn(trunk_out)
                scores = det_out["scores"][0].sigmoid()
                keep = scores > new_det_thresh
                det_masks, det_scores = det_out["masks"][0][keep], scores[keep]
                if det_masks.shape[0] > 1:
                    det_masks, det_scores = _nms_masks(det_masks, det_scores)

            if frame_idx == 0 and initial_masks is not None:
                current_out = self._condition_with_masks(
                    initial_masks.to(device=device, dtype=dt), frame_idx, vision_feats, vision_pos,
                    feat_sizes, high_res_prop, output_dict, N, mux_state, backbone_obj,
                    _prep_frame(images, slice(frame_idx, frame_idx + 1), device, dt, size), trunk_out)
                last_occluded = torch.full((mux_state.total_valid_entries,), -1, device=device, dtype=torch.long)
                obj_scores = [1.0] * mux_state.total_valid_entries
                if keep_alive is not None:
                    for i in range(mux_state.total_valid_entries):
                        keep_alive[i] = 8
            elif mux_state is None or mux_state.total_valid_entries == 0:
                if det_masks.shape[0] > 0:
                    if max_objects > 0:
                        det_scores = det_scores[:max_objects]
                        det_masks = det_masks[:max_objects]
                    mux_state = MultiplexState(det_masks.shape[0], self.num_multiplex, device, dt)
                    current_out = self._condition_with_masks(
                        det_masks, frame_idx, vision_feats, vision_pos, feat_sizes, high_res_prop,
                        output_dict, N, mux_state, backbone_obj,
                        _prep_frame(images, slice(frame_idx, frame_idx + 1), device, dt, size), trunk_out, threshold=0.0)
                    last_occluded = torch.full((mux_state.total_valid_entries,), -1, device=device, dtype=torch.long)
                    obj_scores = det_scores[:mux_state.total_valid_entries].tolist()
                    if keep_alive is not None:
                        for i in range(mux_state.total_valid_entries):
                            keep_alive[i] = 1
                else:
                    all_masks.append(None)
                    if pbar is not None:
                        pbar.update(1)
                    # Skip to backbone advance at end of loop
                    if frame_idx + 1 < N:
                        if prefetch:
                            torch.cuda.current_stream(device).wait_stream(backbone_stream)
                            cur_bb = next_bb
                        else:
                            cur_bb = self._compute_backbone_frame(backbone_fn, _prep_frame(images, slice(frame_idx + 1, frame_idx + 2), device, dt, size), frame_idx=frame_idx + 1)
                    continue
            else:
                N_obj = mux_state.total_valid_entries
                current_out = self.track_step(
                    frame_idx=frame_idx, is_init_cond_frame=False, current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                    output_dict=output_dict, num_frames=N, propagation_high_res=high_res_prop,
                    multiplex_state=mux_state, run_mem_encoder=False)
                current_out["pred_masks"] = fill_holes_in_mask_scores(
                    current_out["pred_masks"], max_area=16)
                if last_occluded.shape[0] == N_obj and N_obj > 1:
                    self._suppress_recently_occluded(
                        current_out["pred_masks"], last_occluded, frame_idx)
                if self.num_maskmem > 0:
                    self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes, mux_state, device)
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                lookback = max(self.num_maskmem, self.max_obj_ptrs_in_encoder)
                for old_idx in list(output_dict["non_cond_frame_outputs"]):
                    if old_idx < frame_idx - lookback:
                        del output_dict["non_cond_frame_outputs"][old_idx]
                n_before = mux_state.total_valid_entries
                new_obj_scores = self._match_and_add_detections(det_masks, det_scores, current_out, mux_state,
                                               vision_feats, feat_sizes, device, max_objects,
                                               keep_alive if run_det else None)
                n_added = mux_state.total_valid_entries - n_before
                if n_added > 0:
                    last_occluded = torch.cat([last_occluded,
                        torch.full((n_added,), -1, device=device, dtype=torch.long)])
                    obj_scores.extend(new_obj_scores)

            masks_out = current_out["pred_masks_high_res"][:, 0]
            if keep_alive is not None:
                for i in range(masks_out.shape[0]):
                    if keep_alive.get(i, 0) <= 0:
                        masks_out[i] = NO_OBJ_SCORE
            N_obj_now = mux_state.total_valid_entries if mux_state is not None else 0
            if N_obj_now > 0:
                all_masks.append(pack_masks(masks_out).to(idev))
            else:
                all_masks.append(None)
            if pbar is not None:
                pbar.update(1)

            # Next frame's backbone
            if frame_idx + 1 < N:
                if prefetch:
                    torch.cuda.current_stream(device).wait_stream(backbone_stream)
                    cur_bb = next_bb
                else:
                    cur_bb = self._compute_backbone_frame(backbone_fn, _prep_frame(images, slice(frame_idx + 1, frame_idx + 2), device, dt, size), frame_idx=frame_idx + 1)

        if not all_masks or all(m is None for m in all_masks):
            return {"packed_masks": None, "n_frames": N, "scores": []}

        max_obj = max(m.shape[0] for m in all_masks if m is not None)
        sample = next(m for m in all_masks if m is not None)
        empty_packed = torch.zeros(max_obj, *sample.shape[1:], dtype=torch.uint8, device=sample.device)
        for i, m in enumerate(all_masks):
            if m is None:
                all_masks[i] = empty_packed
            elif m.shape[0] < max_obj:
                pad = torch.zeros(max_obj - m.shape[0], *m.shape[1:], dtype=torch.uint8, device=m.device)
                all_masks[i] = torch.cat([m, pad], dim=0)
        return {"packed_masks": torch.stack(all_masks, dim=0), "n_frames": N, "scores": obj_scores}
