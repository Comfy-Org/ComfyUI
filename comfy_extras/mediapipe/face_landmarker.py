"""Pure-PyTorch port of MediaPipe's face_landmarker_v2_with_blendshapes.task:
BlazeFace detector → FaceMesh v2 → ARKit-52 blendshapes."""


import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from torch import Tensor, nn


# Values below must stay verbatim with the published face_landmarker_v2 graph

# face_blendshapes_graph.cc::kLandmarksSubsetIdxs
_BS_INPUT_INDICES: Tuple[int, ...] = (
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54,
    55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95,
    103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152,
    153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178,
    181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
    283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374,
    375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397,
    398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474,
    475, 476, 477,
)

# face_blendshapes_graph.cc::kCategoryNames
BLENDSHAPE_NAMES: Tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)

# face_detection.pbtxt — short-range BlazeFace.
_BF_NUM_LAYERS = 4
_BF_INPUT_SIZE = 128
_BF_STRIDES = (8, 16, 16, 16)
_BF_ANCHOR_OFFSET_X = 0.5
_BF_ANCHOR_OFFSET_Y = 0.5
_BF_ASPECT_RATIOS = (1.0,)
_BF_INTERP_SCALE_AR = 1.0
_BF_BOX_SCALE = 128.0
_BF_KP_OFFSET = 4
_BF_SCORE_CLIP = 100.0
_BF_MIN_SCORE = 0.5

# face_detection_full_range.pbtxt — 48x48 grid at stride 4, 1 anchor/cell.
_BF_FR_INPUT_SIZE = 192
_BF_FR_GRID = 48
_BF_FR_NUM_ANCHORS = _BF_FR_GRID * _BF_FR_GRID
_BF_FR_BOX_SCALE = 192.0
_BF_FR_SCORE_CLIP = 100.0

_FM_INPUT_SIZE = 192

# Face ROI: 1.5xbbox rect warped anisotropically into 192x192.
_FACE_LEFT_EYE_KP = 0
_FACE_RIGHT_EYE_KP = 1
_FACE_ROI_SCALE_X = 1.5
_FACE_ROI_SCALE_Y = 1.5
_FACE_ROI_TARGET_ANGLE = 0.0


def _tf_same_pad(x: Tensor, kernel: int, stride: int) -> Tensor:
    """TF SAME pad (asymmetric on stride-2; PyTorch's symmetric pad undershoots by 1 px)."""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = max(((H + stride - 1) // stride - 1) * stride + kernel - H, 0)
    pad_w = max(((W + stride - 1) // stride - 1) * stride + kernel - W, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))


# BlazeFace short-range: stem 5x5/s2 → 16 BlazeBlocks → parallel heads at
# 16²x88 (2 anchors/cell) and 8²x96 (6/cell) = 896 anchors. (in, out, stride):
_BLAZEFACE_BLOCKS = [
    (24, 24, 1), (24, 28, 1), (28, 32, 2), (32, 36, 1),
    (36, 42, 1), (42, 48, 2), (48, 56, 1), (56, 64, 1),
    (64, 72, 1), (72, 80, 1), (80, 88, 1), (88, 96, 2),
    (96, 96, 1), (96, 96, 1), (96, 96, 1), (96, 96, 1),
]


class BlazeFaceBlock(nn.Module):
    """DW 3x3 + PW + residual. Residual max-pools on stride>1, channel-pads on out_ch>in_ch."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.depthwise = ops.Conv2d(in_ch, in_ch, 3, stride=stride, padding=0, groups=in_ch, bias=True, device=device, dtype=dtype)
        self.pointwise = ops.Conv2d(in_ch, out_ch, 1, padding=0, bias=True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.max_pool2d(x, 2, 2) if self.stride > 1 else x
        if self.out_ch > self.in_ch:
            residual = F.pad(residual, (0, 0, 0, 0, 0, self.out_ch - self.in_ch))
        x = _tf_same_pad(x, 3, self.stride) if self.stride > 1 else F.pad(x, (1, 1, 1, 1))
        return F.relu(self.pointwise(self.depthwise(x)) + residual)


class BlazeFace(nn.Module):
    """Short-range BlazeFace: (B, 3, 128, 128) in [-1, 1] → 896 anchors x 17."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        self.stem = ops.Conv2d(3, 24, 5, stride=2, padding=0, bias=True, **kw)
        self.blocks = nn.ModuleList(BlazeFaceBlock(i, o, s, device=device, dtype=dtype, operations=operations)
                                    for (i, o, s) in _BLAZEFACE_BLOCKS)
        # 16²x2 + 8²x6 = 512 + 384 = 896 anchors.
        self.cls_16 = ops.Conv2d(88, 2, 1, padding=0, bias=True, **kw)
        self.cls_8 = ops.Conv2d(96, 6, 1, padding=0, bias=True, **kw)
        self.reg_16 = ops.Conv2d(88, 32, 1, padding=0, bias=True, **kw)
        self.reg_8 = ops.Conv2d(96, 96, 1, padding=0, bias=True, **kw)

    def forward(self, image_chw_normalized: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.stem(_tf_same_pad(image_chw_normalized, 5, 2)))
        # 16x16 tap is block-10 output (before the 88→96 stride-2 in block 11).
        for i in range(11):
            x = self.blocks[i](x)
        feat_16 = x
        for i in range(11, 16):
            x = self.blocks[i](x)
        feat_8 = x

        def flat(t, a, k):  # NHWC flatten → (B, H*W*A, K)
            B, _, H, W = t.shape
            return t.permute(0, 2, 3, 1).reshape(B, H * W * a, k)

        cls = torch.cat([flat(self.cls_16(feat_16), 2, 1), flat(self.cls_8(feat_8), 6, 1)], dim=1)
        reg = torch.cat([flat(self.reg_16(feat_16), 2, 16), flat(self.reg_8(feat_8), 6, 16)], dim=1)
        return reg, cls


# BlazeFace full-range (face_detection_full_range_sparse.tflite): MobileNetV2-ish
# backbone + top-down FPN, 192² input → 2304 anchors at the 48x48 grid.
class FRBlock(nn.Module):
    """Double inverted residual: DW → PW(mid) → DW → PW(out) [+ residual].

    Per source tflite: dw* have no fused activation, pw1 is always ReLU, pw2
    is ReLU only when no residual (else ReLU fuses into the ADD).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        self.has_residual = (in_ch == out_ch and stride == 1)
        self.dw1 = ops.Conv2d(in_ch, in_ch, 3, stride=stride, padding=0, groups=in_ch, bias=True, **kw)
        self.pw1 = ops.Conv2d(in_ch, mid_ch, 1, padding=0, bias=True, **kw)
        self.dw2 = ops.Conv2d(mid_ch, mid_ch, 3, stride=1, padding=0, groups=mid_ch, bias=True, **kw)
        self.pw2 = ops.Conv2d(mid_ch, out_ch, 1, padding=0, bias=True, **kw)

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.has_residual else None
        x = F.relu(self.pw1(self.dw1(F.pad(x, (1, 1, 1, 1)))))
        x = self.pw2(self.dw2(F.pad(x, (1, 1, 1, 1))))
        return F.relu(x + residual) if residual is not None else F.relu(x)


# (in_ch, mid_ch, out_ch, stride). Stages downsample 96²x32 → 48²x64 → 24²x128
# → 12²x192 → 6²x384. Lateral taps at indices 4, 7, 10 (see _FR_LATERAL_*).
_FR_BACKBONE_BLOCKS = [
    (32, 8, 32, 1),    (32, 8, 32, 1),                                            # 96²x32
    (32, 16, 64, 2),   (64, 16, 64, 1),   (64, 16, 64, 1),                        # 48²x64 — tap[0]
    (64, 32, 128, 2),  (128, 32, 128, 1), (128, 32, 128, 1),                      # 24²x128 — tap[1]
    (128, 48, 192, 2), (192, 48, 192, 1), (192, 48, 192, 1),                      # 12²x192 — tap[2]
    (192, 96, 384, 2), (384, 96, 384, 1), (384, 96, 384, 1), (384, 96, 384, 1),   # 6²x384
]
_FR_LATERAL_TAP_INDICES = (4, 7, 10)
_FR_LATERAL_CHANNELS = ((64, 48), (128, 64), (192, 96))  # (in, out) per side-conv

# Decoder blocks per FPN level (after upsample-and-merge with the lateral).
_FR_DECODER_BLOCKS = [
    [(96, 48, 96, 1), (96, 48, 96, 1)],  # 12²x96
    [(64, 32, 64, 1), (64, 32, 64, 1)],  # 24²x64
    [(48, 24, 48, 1)],                   # 48²x48 — feeds the heads
]


def _dcr_depth_to_space(t: Tensor, r: int, c_out: int) -> Tensor:
    """TF DEPTH_TO_SPACE in DCR layout (input channels = (i, j, c_out)).
    pixel_shuffle uses CRD which permutes output channels for c_out > 1."""
    B_, _, H_, W_ = t.shape
    t = t.reshape(B_, r, r, c_out, H_, W_)
    t = t.permute(0, 3, 4, 1, 5, 2).contiguous()
    return t.reshape(B_, c_out, H_ * r, W_ * r)


class BlazeFaceFullRange(nn.Module):
    """Full-range face detector: (B, 3, 192, 192) in [-1, 1] → 2304 anchors x 17 values."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        mk_block = lambda i, m, o, s: FRBlock(i, m, o, s, device=device, dtype=dtype, operations=operations)
        self.stem = ops.Conv2d(3, 32, 3, stride=2, padding=0, bias=True, **kw)
        self.backbone = nn.ModuleList(mk_block(i, m, o, s) for (i, m, o, s) in _FR_BACKBONE_BLOCKS)
        self.lateral_convs = nn.ModuleList(ops.Conv2d(i, o, 1, padding=0, bias=True, **kw) for (i, o) in _FR_LATERAL_CHANNELS)
        self.top_conv = ops.Conv2d(384, 96, 1, padding=0, bias=True, **kw)
        self.decoder_levels = nn.ModuleList(
            nn.ModuleList(mk_block(i, m, o, s) for (i, m, o, s) in lvl) for lvl in _FR_DECODER_BLOCKS
        )
        # 96→64 before 12→24, 64→48 before 24→48.
        self.decoder_reduce_convs = nn.ModuleList([
            ops.Conv2d(96, 64, 1, padding=0, bias=True, **kw),
            ops.Conv2d(64, 48, 1, padding=0, bias=True, **kw),
        ])
        # Heads mix 2x2-cell info via DW-stride-2 + depth_to_space block_size=2.
        self.cls_conv = ops.Conv2d(48, 4, 1, padding=0, bias=True, **kw)
        self.cls_dw = ops.Conv2d(4, 4, 3, stride=2, padding=0, groups=4, bias=True, **kw)
        self.reg_conv = ops.Conv2d(48, 64, 1, padding=0, bias=True, **kw)
        self.reg_dw = ops.Conv2d(64, 64, 3, stride=2, padding=0, groups=64, bias=True, **kw)

    def forward(self, image_chw_normalized: Tensor) -> tuple[Tensor, Tensor]:
        # Symmetric pad-1 throughout (full-range tflite uses explicit TF PAD, not SAME).
        x = F.relu(self.stem(F.pad(image_chw_normalized, (1, 1, 1, 1))))
        tap_set = set(_FR_LATERAL_TAP_INDICES)
        laterals: list[Tensor] = []
        for i, blk in enumerate(self.backbone):
            x = blk(x)
            if i in tap_set:
                laterals.append(x)

        # top_conv / lateral_convs / decoder_reduce_convs all have fused ReLU in the tflite.
        p = F.relu(self.top_conv(x))
        laterals_rev = list(reversed(laterals))
        lateral_convs_rev = list(reversed(self.lateral_convs))
        for level in range(len(self.decoder_levels)):
            lateral = laterals_rev[level]
            p = F.interpolate(p, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            p = p + F.relu(lateral_convs_rev[level](lateral))
            for blk in self.decoder_levels[level]:
                p = blk(p)
            if level < len(self.decoder_reduce_convs):
                p = F.relu(self.decoder_reduce_convs[level](p))

        c = self.cls_dw(F.pad(self.cls_conv(p), (1, 1, 1, 1)))
        c = _dcr_depth_to_space(c, r=2, c_out=1)
        r = self.reg_dw(F.pad(self.reg_conv(p), (1, 1, 1, 1)))
        r = _dcr_depth_to_space(r, r=2, c_out=16)
        B = c.shape[0]
        cls_out = c.permute(0, 2, 3, 1).reshape(B, _BF_FR_NUM_ANCHORS, 1)
        reg_out = r.permute(0, 2, 3, 1).reshape(B, _BF_FR_NUM_ANCHORS, 16)
        return reg_out, cls_out


@lru_cache(maxsize=1)
def _blazeface_full_range_anchors() -> np.ndarray:
    """2304 anchors over 48x48; anchor_w=anchor_h=1 (fixed_anchor_size)."""
    feat = _BF_FR_GRID
    yy, xx = np.meshgrid(np.arange(feat, dtype=np.float32), np.arange(feat, dtype=np.float32), indexing="ij")
    cx, cy, ones = (xx + 0.5) / feat, (yy + 0.5) / feat, np.ones_like(xx)
    return np.stack([cx, cy, ones, ones], axis=-1).reshape(_BF_FR_NUM_ANCHORS, 4)


def _decode_blazeface_full_range(regressors: np.ndarray, classificators: np.ndarray,
                                 score_thresh: float = _BF_MIN_SCORE) -> np.ndarray:
    """Same decode as short-range with 2304-anchor grid and box_scale=192."""
    scores = expit(np.clip(classificators[:, 0], -_BF_FR_SCORE_CLIP, _BF_FR_SCORE_CLIP))
    keep = scores >= score_thresh
    if not keep.any():
        return np.empty((0, 17), dtype=np.float32)
    r = regressors[keep] / _BF_FR_BOX_SCALE
    a = _blazeface_full_range_anchors()[keep]
    cxs, cys, aws, ahs = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xc, yc = r[:, 0:1] * aws + cxs, r[:, 1:2] * ahs + cys
    w, h = r[:, 2:3] * aws, r[:, 3:4] * ahs
    out = np.empty((r.shape[0], 17), dtype=np.float32)
    out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4] = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    out[:, 4:16:2] = r[:, _BF_KP_OFFSET::2] * aws + cxs
    out[:, 5:16:2] = r[:, _BF_KP_OFFSET + 1::2] * ahs + cys
    out[:, 16] = scores[keep]
    return out


# FaceMesh (face_landmarks_detector.tflite): PReLU variant of BlazeBlock,
# 17 blocks, heads for 478x3 landmarks + presence.
_FACEMESH_BLOCKS = [  # (in_ch, out_ch, stride)
    (16, 16, 1),  (16, 16, 1),  (16, 32, 2),  (32, 32, 1), (32, 32, 1), (32, 64, 2),
    (64, 64, 1),  (64, 64, 1),  (64, 128, 2), (128, 128, 1), (128, 128, 1), (128, 128, 2),
    (128, 128, 1), (128, 128, 1), (128, 128, 2), (128, 128, 1), (128, 128, 1),
]


class FaceMeshBlock(nn.Module):
    """PReLU BlazeBlock: PReLU between DW and PW, and after the residual add."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.depthwise = ops.Conv2d(in_ch, in_ch, 3, stride=stride, padding=0, groups=in_ch, bias=True, **kw)
        self.prelu_dwise = nn.PReLU(num_parameters=in_ch, **kw)
        self.pointwise = ops.Conv2d(in_ch, out_ch, 1, padding=0, bias=True, **kw)
        self.prelu_out = nn.PReLU(num_parameters=out_ch, **kw)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.max_pool2d(x, 2, 2) if self.stride > 1 else x
        if self.out_ch > self.in_ch:
            residual = F.pad(residual, (0, 0, 0, 0, 0, self.out_ch - self.in_ch))
        x = _tf_same_pad(x, 3, self.stride) if self.stride > 1 else F.pad(x, (1, 1, 1, 1))
        return self.prelu_out(self.pointwise(self.prelu_dwise(self.depthwise(x))) + residual)


class FaceMesh(nn.Module):
    NUM_LANDMARKS = 478

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        self.stem = ops.Conv2d(3, 16, 3, stride=2, padding=0, bias=True, **kw)
        self.prelu_stem = nn.PReLU(num_parameters=16, **kw)
        self.blocks = nn.ModuleList(FaceMeshBlock(i, o, s, device=device, dtype=dtype, operations=operations)
                                    for (i, o, s) in _FACEMESH_BLOCKS)
        self.head_reduce = ops.Conv2d(128, 8, 1, padding=0, bias=True, **kw)
        self.prelu_head_reduce = nn.PReLU(num_parameters=8, **kw)
        self.head_block = FaceMeshBlock(8, 8, 1, device=device, dtype=dtype, operations=operations)
        self.head_presence = ops.Conv2d(8, 1, 3, padding=0, bias=True, **kw)
        self.head_landmarks = ops.Conv2d(8, self.NUM_LANDMARKS * 3, 3, padding=0, bias=True, **kw)

    def forward(self, face_chw_normalized: Tensor) -> tuple[Tensor, Tensor]:
        """(B, 3, 192, 192) in [0, 1] → ((B, 478, 3) landmarks in 192-canonical, (B,) presence)."""
        x = self.prelu_stem(self.stem(_tf_same_pad(face_chw_normalized, 3, 2)))
        for blk in self.blocks:
            x = blk(x)
        x = self.prelu_head_reduce(self.head_reduce(x))
        x = self.head_block(x)
        B = x.shape[0]
        presence = self.head_presence(x).reshape(B)
        lmks = self.head_landmarks(x).reshape(B, self.NUM_LANDMARKS, 3)
        return lmks, presence


# FaceBlendshapes (MLP-Mixer "GhumMarkerPoserMlpMixerGeneral"):
# 146x2 → token-reduce 146→96 → embed 2→64 → +cls token → 4x mixer → cls→52.
_BS_NUM_INPUT_LANDMARKS = 146
_BS_NUM_TOKENS_REDUCED = 96
_BS_NUM_TOKENS = 97  # +1 cls
_BS_TOKEN_DIM = 64
_BS_TOKEN_MIX_HIDDEN = 384
_BS_CHANNEL_MIX_HIDDEN = 256
_BS_NUM_BLENDSHAPES = 52
_BS_LN_EPS = 1e-6


class MlpMixerBlock(nn.Module):
    """MLP-Mixer block: token-mixing MLP (over tokens) → channel-mixing MLP (over dim).
    Both pre-LN, both residual. LN has no beta (bias=False) to match MP."""

    def __init__(self, num_tokens: int, token_dim: int, token_hidden: int, channel_hidden: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        # bias=False → no LN beta (matches MP).
        self.ln1 = ops.LayerNorm(token_dim, eps=_BS_LN_EPS, bias=False, **kw)
        self.ln2 = ops.LayerNorm(token_dim, eps=_BS_LN_EPS, bias=False, **kw)
        self.token_mlp1 = ops.Linear(num_tokens, token_hidden, bias=True, **kw)
        self.token_mlp2 = ops.Linear(token_hidden, num_tokens, bias=True, **kw)
        self.channel_mlp1 = ops.Linear(token_dim, channel_hidden, bias=True, **kw)
        self.channel_mlp2 = ops.Linear(channel_hidden, token_dim, bias=True, **kw)

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln1(x).transpose(1, 2)
        x = x + self.token_mlp2(F.relu(self.token_mlp1(y))).transpose(1, 2)
        return x + self.channel_mlp2(F.relu(self.channel_mlp1(self.ln2(x))))


class FaceBlendshapes(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        kw = dict(device=device, dtype=dtype)
        self.token_reduce = ops.Linear(_BS_NUM_INPUT_LANDMARKS, _BS_NUM_TOKENS_REDUCED, bias=True, **kw)
        self.token_embed = ops.Linear(2, _BS_TOKEN_DIM, bias=True, **kw)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, _BS_TOKEN_DIM, **kw))
        self.blocks = nn.ModuleList(
            MlpMixerBlock(_BS_NUM_TOKENS, _BS_TOKEN_DIM, _BS_TOKEN_MIX_HIDDEN, _BS_CHANNEL_MIX_HIDDEN,
                          device=device, dtype=dtype, operations=operations) for _ in range(4)
        )
        self.head = ops.Linear(_BS_TOKEN_DIM, _BS_NUM_BLENDSHAPES, bias=True, **kw)

    @staticmethod
    def _input_normalize(landmarks_2d: Tensor) -> Tensor:
        # Centroid-subtract → L2 scale → x0.5. The 0.5 is baked into training.
        centroid = landmarks_2d.mean(dim=1, keepdim=True)
        x = landmarks_2d - centroid
        mag = torch.sqrt((x * x).sum(dim=-1, keepdim=True))
        scale = mag.mean(dim=1, keepdim=True)
        return (x / scale.clamp(min=1e-12)) * 0.5

    def forward(self, landmarks_2d: Tensor) -> Tensor:
        """(B, 146, 2) → (B, 52) in [0, 1]. Input units don't matter (centroid + L2 normalize)."""
        x = self._input_normalize(landmarks_2d)
        x = self.token_reduce(x.transpose(1, 2)).transpose(1, 2)
        x = self.token_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return torch.sigmoid(self.head(x[:, 0]))


@lru_cache(maxsize=1)
def _blazeface_anchors() -> np.ndarray:
    """896 anchors per SsdAnchorsCalculator (fixed_anchor_size → anchor_w=anchor_h=1)."""
    per_ar = len(_BF_ASPECT_RATIOS) + (1 if _BF_INTERP_SCALE_AR > 0 else 0)
    layer_anchors: List[np.ndarray] = []
    layer = 0
    while layer < _BF_NUM_LAYERS:
        stride = _BF_STRIDES[layer]
        last = layer
        while last < _BF_NUM_LAYERS and _BF_STRIDES[last] == stride:
            last += 1
        per_cell = per_ar * (last - layer)
        feat = (_BF_INPUT_SIZE + stride - 1) // stride
        yy, xx = np.meshgrid(np.arange(feat, dtype=np.float32), np.arange(feat, dtype=np.float32), indexing="ij")
        cx, cy, ones = (xx + _BF_ANCHOR_OFFSET_X) / feat, (yy + _BF_ANCHOR_OFFSET_Y) / feat, np.ones_like(xx)
        cell = np.stack([cx, cy, ones, ones], axis=-1).reshape(-1, 4)
        layer_anchors.append(np.repeat(cell, per_cell, axis=0))
        layer = last
    out = np.concatenate(layer_anchors, axis=0)
    assert out.shape == (896, 4), out.shape
    return out


def _decode_blazeface(regressors: np.ndarray, classificators: np.ndarray,
                      score_thresh: float = _BF_MIN_SCORE) -> np.ndarray:
    """Decode (regs (896,16), cls (896,1)) → (N, 17) = [xyxy, kp0x..kp5y, score] in [0, 1]."""
    scores = expit(np.clip(classificators[:, 0], -_BF_SCORE_CLIP, _BF_SCORE_CLIP))
    keep = scores >= score_thresh
    if not keep.any():
        return np.empty((0, 17), dtype=np.float32)
    r = regressors[keep] / _BF_BOX_SCALE
    a = _blazeface_anchors()[keep]  # (N, 4) cx, cy, 1, 1
    cxs, cys, aws, ahs = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xc, yc = r[:, 0:1] * aws + cxs, r[:, 1:2] * ahs + cys
    w, h = r[:, 2:3] * aws, r[:, 3:4] * ahs
    out = np.empty((r.shape[0], 17), dtype=np.float32)
    out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4] = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    out[:, 4:16:2] = r[:, _BF_KP_OFFSET::2] * aws + cxs
    out[:, 5:16:2] = r[:, _BF_KP_OFFSET + 1::2] * ahs + cys
    out[:, 16] = scores[keep]
    return out


def _weighted_nms(detections: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """MP weighted NMS — kept boxes are score-weighted averages of overlapping detections."""
    if detections.shape[0] == 0:
        return detections
    dets = detections[np.argsort(-detections[:, 16])]
    N = dets.shape[0]
    areas = np.clip(dets[:, 2] - dets[:, 0], 0, None) * np.clip(dets[:, 3] - dets[:, 1], 0, None)
    kept: List[np.ndarray] = []
    used = np.zeros(N, dtype=bool)
    for i in range(N):
        if used[i]:
            continue
        ax1, ay1, ax2, ay2 = dets[i, 0:4]
        merge_idx = [i]
        for j in range(i + 1, N):
            if used[j]:
                continue
            bx1, by1, bx2, by2 = dets[j, 0:4]
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_thresh:  # strict > matches MP
                merge_idx.append(j)
                used[j] = True
        used[i] = True
        cluster = dets[merge_idx]
        ws = cluster[:, 16:17]
        ws_sum = ws.sum()
        merged = np.copy(cluster[0])
        if ws_sum > 0:
            merged[:16] = (cluster[:, :16] * ws).sum(axis=0) / ws_sum
        kept.append(merged)
    return np.stack(kept, axis=0) if kept else np.empty((0, 17), dtype=np.float32)


def _detection_to_face_rect(detection: np.ndarray, image_w: int, image_h: int) -> Tuple[float, float, float, float, float]:
    """Detection (normalized) → rotated 1.5xbbox ROI in image pixels (anisotropic)."""
    xmin, ymin, xmax, ymax = detection[0:4]
    lx = detection[4 + _FACE_LEFT_EYE_KP * 2 + 0] * image_w
    ly = detection[4 + _FACE_LEFT_EYE_KP * 2 + 1] * image_h
    rx = detection[4 + _FACE_RIGHT_EYE_KP * 2 + 0] * image_w
    ry = detection[4 + _FACE_RIGHT_EYE_KP * 2 + 1] * image_h
    # Image-y-down convention: angle = target - atan2(-dy, dx).
    angle = _FACE_ROI_TARGET_ANGLE - math.atan2(ly - ry, rx - lx)
    return (float((xmin + xmax) * 0.5 * image_w),
            float((ymin + ymax) * 0.5 * image_h),
            float((xmax - xmin) * image_w * _FACE_ROI_SCALE_X),
            float((ymax - ymin) * image_h * _FACE_ROI_SCALE_Y),
            float(angle))


def _sample_warp(image_chw: Tensor, src_x: Tensor, src_y: Tensor, padding_mode: str) -> Tensor:
    """Bilinear-sample image_chw at corner-aligned (src_x, src_y)."""
    H, W = int(image_chw.shape[-2]), int(image_chw.shape[-1])
    grid = torch.stack([(2.0 * src_x + 1.0) / W - 1.0,
                        (2.0 * src_y + 1.0) / H - 1.0], dim=-1).unsqueeze(0)
    return F.grid_sample(image_chw.unsqueeze(0), grid, mode="bilinear",
                         align_corners=False, padding_mode=padding_mode).squeeze(0)


def _warp_face_crop(image_chw: Tensor, cx: float, cy: float, width: float, height: float,
                    angle: float, output_size: int = _FM_INPUT_SIZE) -> Tensor:
    """Rotated rect → output_size² with BORDER_REPLICATE. image_chw must be in [0, 1]."""
    s_x, s_y = width / output_size, height / output_size
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    arange = torch.arange(output_size, dtype=image_chw.dtype, device=image_chw.device) - output_size * 0.5
    v_grid, u_grid = torch.meshgrid(arange, arange, indexing="ij")
    src_x = cx + u_grid * s_x * cos_a - v_grid * s_y * sin_a
    src_y = cy + u_grid * s_x * sin_a + v_grid * s_y * cos_a
    return _sample_warp(image_chw, src_x, src_y, "border")


def _blazeface_input_warp(image_chw_raw: Tensor, target: int = _BF_INPUT_SIZE) -> Tuple[Tensor, float, float, float]:
    """Centered max(W,H) square → target² with BORDER_ZERO + [-1, 1] norm.

    Sub-pixel grid_sample matters; integer-pad-then-resize drifts the bbox ~5%.
    Returns (warped, sub_rect_cx, sub_rect_cy, sub_rect_size) — the triplet maps
    tensor-normalized [0,1] detections back to image pixels.
    """
    H, W = int(image_chw_raw.shape[1]), int(image_chw_raw.shape[2])
    sub_rect_size = float(max(W, H))
    sub_rect_cx, sub_rect_cy = W * 0.5, H * 0.5
    s = sub_rect_size / target
    arange = torch.arange(target, dtype=image_chw_raw.dtype, device=image_chw_raw.device) - target * 0.5
    v_grid, u_grid = torch.meshgrid(arange, arange, indexing="ij")
    out = _sample_warp(image_chw_raw, sub_rect_cx + u_grid * s, sub_rect_cy + v_grid * s, "zeros")
    return (out / 127.5) - 1.0, sub_rect_cx, sub_rect_cy, sub_rect_size


class FaceLandmarker(nn.Module):
    """BlazeFace → FaceMesh v2 → blendshapes. `detector_variant` selects 'short'
    (128², ≤2m) or 'full' (192² FPN, ≤5m). State dict uses inner-module prefixes
    `detector.*` / `mesh.*` / `blendshapes.*`; the outer FaceLandmarkerModel
    wrapper rewrites `detector_{variant}.*` keys to `detector.*` before loading.
    """

    def __init__(self, device=None, dtype=None, operations=None, detector_variant: str = "short"):
        super().__init__()
        det_cls = {"short": BlazeFace, "full": BlazeFaceFullRange}.get(detector_variant)

        self.detector_variant = detector_variant
        self.detector = det_cls(device=device, dtype=dtype, operations=operations)
        self.mesh = FaceMesh(device=device, dtype=dtype, operations=operations)
        self.blendshapes = FaceBlendshapes(device=device, dtype=dtype, operations=operations)
        self.register_buffer("_bs_idx", torch.tensor(_BS_INPUT_INDICES, dtype=torch.long), persistent=False)

    def run_detector_batch(self, images_rgb_uint8: List[np.ndarray],
                           score_thresh: float = _BF_MIN_SCORE,
                           iou_thresh: float = 0.5):
        """Batched detector pass. Returns (img_raws, sub_rects, sizes, per_frame_decoded)
        where per_frame_decoded[b] is (N, 17) in tensor-normalized [0,1] coords."""
        if not images_rgb_uint8:
            return [], [], [], []
        device, dtype = self.detector.stem.weight.device, self.detector.stem.weight.dtype
        det_input_size, decode_fn = ((_BF_FR_INPUT_SIZE, _decode_blazeface_full_range)
                                     if self.detector_variant == "full"
                                     else (_BF_INPUT_SIZE, _decode_blazeface))

        # Same-size frames: stack once and transfer once. Variable size falls back
        # to per-image (only triggers for SAM3DBody's head crops).
        sizes = [tuple(img.shape[:2]) for img in images_rgb_uint8]
        if len(set(sizes)) == 1:
            batch_chw = torch.from_numpy(np.stack(images_rgb_uint8, axis=0)).to(device, dtype).movedim(-1, -3).contiguous()
            img_raws = [batch_chw[bi] for bi in range(batch_chw.shape[0])]
        else:
            img_raws = [torch.from_numpy(img).to(device, dtype).movedim(-1, -3).contiguous() for img in images_rgb_uint8]

        warps = [_blazeface_input_warp(img_raw, det_input_size) for img_raw in img_raws]
        det_crops = [w[0] for w in warps]
        sub_rects = [(w[1], w[2], w[3]) for w in warps]

        regs_b, cls_b = self.detector(torch.stack(det_crops, dim=0))
        regs_np, cls_np = regs_b.float().cpu().numpy(), cls_b.float().cpu().numpy()
        per_frame = []
        for b in range(len(images_rgb_uint8)):
            decoded = decode_fn(regs_np[b], cls_np[b], score_thresh=score_thresh)
            per_frame.append(_weighted_nms(decoded, iou_thresh=iou_thresh) if decoded.shape[0] > 0 else decoded)
        return img_raws, sub_rects, sizes, per_frame

    def detect_batch(self, images_rgb_uint8: List[np.ndarray], num_faces: int = 1,
                     score_thresh: float = _BF_MIN_SCORE) -> List[List[dict]]:
        """Full pipeline batched across `images_rgb_uint8`. Returns one face-dict
        list per image (empty if nothing detected). Face dict:
            bbox_xyxy (4,) image pixels, blendshapes {52} ∈ [0,1],
            landmarks_xy (478, 2) image pixels, landmarks_3d (478, 3) in
            192-canonical (pre-transformation) units, presence float (raw logit).
        """
        img_raws, sub_rects, sizes, per_frame_dets = self.run_detector_batch(
            images_rgb_uint8, score_thresh=score_thresh,
        )
        # tensor-normalized → image-normalized [0,1] for _detection_to_face_rect.
        for b, decoded in enumerate(per_frame_dets):
            if decoded.shape[0] == 0:
                continue
            cx, cy, size = sub_rects[b]
            H, W = sizes[b]
            sx0, sy0 = cx - size * 0.5, cy - size * 0.5
            decoded[:, 0:16:2] = (sx0 + size * decoded[:, 0:16:2]) / W
            decoded[:, 1:16:2] = (sy0 + size * decoded[:, 1:16:2]) / H
            if num_faces > 0:
                per_frame_dets[b] = decoded[: int(num_faces)]

        # Collect every detected face across all frames into one mesh input.
        face_params: List[Tuple[int, float, float, float, float, float, float]] = []
        mesh_crops: List[Tensor] = []
        for b, dets in enumerate(per_frame_dets):
            if dets.shape[0] == 0:
                continue
            H, W = sizes[b]
            img_for_mesh = img_raws[b] / 255.0
            for det in dets:
                cx, cy, w, h, angle = _detection_to_face_rect(det, W, H)
                mesh_crops.append(_warp_face_crop(img_for_mesh, cx, cy, w, h, angle, _FM_INPUT_SIZE))
                face_params.append((b, float(det[16]), cx, cy, w, h, angle))

        results: List[List[dict]] = [[] for _ in range(len(images_rgb_uint8))]
        if not mesh_crops:
            return results

        lmks_canon_b, presence_b = self.mesh(torch.stack(mesh_crops, dim=0))
        bs_out_b = self.blendshapes(lmks_canon_b[:, self._bs_idx, :2])

        # Batched canonical→image affine
        params_t = torch.tensor(
            [(cx, cy, w, h, math.cos(a), math.sin(a)) for (_b, _s, cx, cy, w, h, a) in face_params],
            device=lmks_canon_b.device, dtype=lmks_canon_b.dtype,
        )
        cxs, cys, ws, hs, cos_a, sin_a = params_t.unbind(dim=1)
        inv = 1.0 / _FM_INPUT_SIZE
        u = lmks_canon_b[..., 0] - _FM_INPUT_SIZE * 0.5
        v = lmks_canon_b[..., 1] - _FM_INPUT_SIZE * 0.5
        lmks_xy_t = torch.stack([
            cxs[:, None] + u * (ws * inv * cos_a)[:, None] - v * (hs * inv * sin_a)[:, None],
            cys[:, None] + u * (ws * inv * sin_a)[:, None] + v * (hs * inv * cos_a)[:, None],
        ], dim=-1)

        lmks_xy_np = lmks_xy_t.float().cpu().numpy()
        lmks_canon_np = lmks_canon_b.float().cpu().numpy()
        presence_np = presence_b.float().cpu().numpy()
        bs_np = bs_out_b.float().cpu().numpy()

        for i, (b, score, *_) in enumerate(face_params):
            lmks_xy = lmks_xy_np[i]
            mn, mx = lmks_xy.min(0), lmks_xy.max(0)
            results[b].append({
                "bbox_xyxy": np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float32),
                "blendshapes": dict(zip(BLENDSHAPE_NAMES, bs_np[i].tolist())),
                "landmarks_xy": lmks_xy,
                "landmarks_3d": lmks_canon_np[i],
                "presence": float(presence_np[i]),
                "score": score,
            })
        return results
