"""MoGe v1 / v2 inference modules and a state-dict-driven builder.

V1: DINOv2 backbone + multi-output head (points, mask).
V2: DINOv2 encoder + neck + per-output heads (points, mask, normal, optional metric-scale MLP).
"""


from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.model_management
import comfy.model_patcher

from comfy.image_encoders.dino2 import Dinov2Model

from .geometry import depth_map_to_point_map, intrinsics_from_focal_center, recover_focal_shift
from .modules import ConvStack, DINOv2Encoder, HeadV1, MLP, _view_plane_uv_grid


def _remap_points(points: torch.Tensor) -> torch.Tensor:
    """Apply the exp remap: z -> exp(z), xy stays linear and gets scaled by the new z."""
    xy, z = points.split([2, 1], dim=-1)
    z = torch.exp(z)
    return torch.cat([xy * z, z], dim=-1)


def _detect_dinov2(sd: dict, prefix: str) -> Dict[str, Any]:
    # All shipped MoGe checkpoints use plain DINOv2
    hidden = sd[prefix + "embeddings.cls_token"].shape[-1]
    layer_prefix = prefix + "encoder.layer."
    depth = 1 + max(int(k[len(layer_prefix):].split(".")[0]) for k in sd if k.startswith(layer_prefix))
    return {
        "hidden_size": hidden,
        "num_attention_heads": hidden // 64,
        "num_hidden_layers": depth,
        "layer_norm_eps": 1e-6,
        "use_swiglu_ffn": False,
    }


class MoGeModelV1(nn.Module):
    """MoGe v1: DINOv2 backbone + HeadV1 (points, mask)."""

    image_mean: torch.Tensor
    image_std: torch.Tensor

    intermediate_layers = 4
    num_tokens_range: Tuple[Number, Number] = (1200, 2500)
    mask_threshold = 0.5

    def __init__(self, backbone: Dict[str, Any], dim_upsample: List[int] = (256, 128, 128),
                 num_res_blocks: int = 1, dim_times_res_block_hidden: int = 1,
                 dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.backbone = Dinov2Model(backbone, dtype, device, operations)
        self.head = HeadV1(dim_in=backbone["hidden_size"], dim_upsample=list(dim_upsample),
                           num_res_blocks=num_res_blocks, dim_times_res_block_hidden=dim_times_res_block_hidden,
                           dtype=dtype, device=device, operations=operations)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        H, W = image.shape[-2:]
        resize = ((num_tokens * 14 ** 2) / (H * W)) ** 0.5
        rh, rw = int(H * resize), int(W * resize)
        x = F.interpolate(image, (rh, rw), mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.image_mean) / self.image_std
        x14 = F.interpolate(x, (rh // 14 * 14, rw // 14 * 14), mode="bilinear", align_corners=False, antialias=True)

        n_layers = len(self.backbone.encoder.layer)
        indices = list(range(n_layers - self.intermediate_layers, n_layers))
        feats = self.backbone.get_intermediate_layers(x14, indices, apply_norm=True)

        points, mask = self.head(feats, x)
        points = F.interpolate(points.float(), (H, W), mode="bilinear", align_corners=False)
        points = _remap_points(points.permute(0, 2, 3, 1))

        mask = F.interpolate(mask.float(), (H, W), mode="bilinear", align_corners=False).squeeze(1)

        return {"points": points, "mask": mask}

    @classmethod
    def from_state_dict(cls, sd, dtype=None, device=None, operations=comfy.ops.manual_cast):
        """Detect the v1 head config from sd, build a model, and load weights."""
        n_up = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("head.upsample_blocks."))
        dim_upsample = [sd[f"head.upsample_blocks.{i}.0.0.weight"].shape[1] for i in range(n_up)]
        # Each upsample stage is Sequential[upsampler, *res_blocks]; count res blocks at level 0.
        num_res_blocks = max({int(k.split(".")[3]) for k in sd if k.startswith("head.upsample_blocks.0.")})
        hidden_out = sd["head.upsample_blocks.0.1.layers.2.weight"].shape[0]
        dim_times = max(hidden_out // dim_upsample[0], 1)
        model = cls(backbone=_detect_dinov2(sd, prefix="backbone."),
                    dim_upsample=dim_upsample, num_res_blocks=num_res_blocks, dim_times_res_block_hidden=dim_times,
                    dtype=dtype, device=device, operations=operations)
        model.load_state_dict(sd, strict=True)
        return model


class MoGeModelV2(nn.Module):
    """MoGe v2: DINOv2 encoder + neck + per-output heads (points/mask/normal/metric-scale)."""

    intermediate_layers = 4
    num_tokens_range: Tuple[Number, Number] = (1200, 3600)

    def __init__(self,
                 encoder: Dict[str, Any],
                 neck: Dict[str, Any],
                 points_head: Dict[str, Any],
                 mask_head: Dict[str, Any],
                 scale_head: Dict[str, Any],
                 normal_head: Optional[Dict[str, Any]] = None,
                 dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.encoder = DINOv2Encoder(**encoder, dtype=dtype, device=device, operations=operations)
        self.neck = ConvStack(**neck, dtype=dtype, device=device, operations=operations)
        self.points_head = ConvStack(**points_head, dtype=dtype, device=device, operations=operations)
        self.mask_head = ConvStack(**mask_head, dtype=dtype, device=device, operations=operations)
        self.scale_head = MLP(**scale_head, dtype=dtype, device=device, operations=operations)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head, dtype=dtype, device=device, operations=operations)

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        B, _, H, W = image.shape
        device, dtype = image.device, image.dtype
        aspect_ratio = W / H
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        feat_top, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)

        # 5-level pyramid: feat at level 0 concatenated with UV, other levels UV-only.
        levels = [_view_plane_uv_grid(B, base_h * (2 ** L), base_w * (2 ** L), aspect_ratio, dtype, device)
                                    for L in range(5)]
        levels[0] = torch.cat([feat_top, levels[0]], dim=1)

        feats = self.neck(levels)

        def _resize(v):
            return F.interpolate(v, (H, W), mode="bilinear", align_corners=False)

        points = _remap_points(_resize(self.points_head(feats)[-1]).permute(0, 2, 3, 1))
        mask = _resize(self.mask_head(feats)[-1]).squeeze(1).sigmoid()
        metric_scale = self.scale_head(cls_token).squeeze(1).exp()

        result = {"points": points, "mask": mask, "metric_scale": metric_scale}
        if hasattr(self, "normal_head"):
            normal = _resize(self.normal_head(feats)[-1])
            result["normal"] = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        return result

    @classmethod
    def from_state_dict(cls, sd, dtype=None, device=None, operations=comfy.ops.manual_cast):
        """Detect the v2 encoder/neck/heads config from sd, build a model, and load weights."""
        backbone = _detect_dinov2(sd, prefix="encoder.backbone.")
        depth = backbone["num_hidden_layers"]
        n = cls.intermediate_layers
        encoder = {
            "backbone": backbone,
            "intermediate_layers": [(depth // n) * (i + 1) - 1 for i in range(n)],
            "dim_out": sd["encoder.output_projections.0.weight"].shape[0],
        }
        # scale_head is an MLP: Sequential of [Linear, ReLU, ..., Linear]; Linear weight is (out, in).
        scale_idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("scale_head.")})
        scale_first = sd[f"scale_head.{scale_idxs[0]}.weight"]
        cfg: Dict[str, Any] = {
            "encoder": encoder,
            "neck": cls._detect_convstack(sd, "neck."),
            "points_head": cls._detect_convstack(sd, "points_head."),
            "mask_head": cls._detect_convstack(sd, "mask_head."),
            "scale_head": {"dims": [scale_first.shape[1]] + [sd[f"scale_head.{i}.weight"].shape[0] for i in scale_idxs]},
        }
        if any(k.startswith("normal_head.") for k in sd):
            cfg["normal_head"] = cls._detect_convstack(sd, "normal_head.")
        model = cls(**cfg, dtype=dtype, device=device, operations=operations)
        model.load_state_dict(sd, strict=True)
        return model

    @staticmethod
    def _detect_convstack(sd: dict, prefix: str) -> Dict[str, Any]:
        """Reconstruct a ConvStack config from the keys under prefix"""
        in_keys = [k for k in sd if k.startswith(f"{prefix}input_blocks.") and k.endswith(".weight")]
        n = 1 + max(int(k[len(f"{prefix}input_blocks."):].split(".")[0]) for k in in_keys)

        in_shapes = [sd[f"{prefix}input_blocks.{i}.weight"].shape for i in range(n)]
        has_out = lambda i: f"{prefix}output_blocks.{i}.weight" in sd
        has_norm = f"{prefix}res_blocks.0.0.layers.0.weight" in sd

        def num_res_at(i):
            rb_prefix = f"{prefix}res_blocks.{i}."
            return len({int(k[len(rb_prefix):].split(".")[0]) for k in sd if k.startswith(rb_prefix)})

        return {
            "dim_in": [s[1] for s in in_shapes],
            "dim_res_blocks": [s[0] for s in in_shapes],
            "dim_out": [sd[f"{prefix}output_blocks.{i}.weight"].shape[0] if has_out(i) else None for i in range(n)],
            "num_res_blocks": [num_res_at(i) for i in range(n)],
            "resamplers": ["conv_transpose" if f"{prefix}resamplers.{i}.0.weight" in sd else "bilinear"
                           for i in range(n - 1)],
            "res_block_in_norm": "layer_norm" if has_norm else "none",
            "res_block_hidden_norm": "group_norm" if has_norm else "none",
        }


# Translate the Meta-style DINOv2 keys MoGe ships to the naming ComfyUI DINOv2 port expects,
# and split each fused qkv tensor into Q/K/V.
_DINOV2_TOPLEVEL_RENAMES = {
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias":   "embeddings.patch_embeddings.projection.bias",
    "cls_token":               "embeddings.cls_token",
    "pos_embed":               "embeddings.position_embeddings",
    "register_tokens":         "embeddings.register_tokens",
    "mask_token":              "embeddings.mask_token",
    "norm.weight":             "layernorm.weight",
    "norm.bias":               "layernorm.bias",
}
_DINOV2_BLOCK_RENAMES = [
    ("ls1.gamma",  "layer_scale1.lambda1"),
    ("ls2.gamma",  "layer_scale2.lambda1"),
    ("attn.proj.", "attention.output.dense."),
    ("mlp.w12.",   "mlp.weights_in."),
    ("mlp.w3.",    "mlp.weights_out."),
]


def _remap_state_dict(sd: dict) -> dict:
    if "model" in sd and "model_config" in sd:
        sd = sd["model"]
    prefix = "encoder.backbone." if any(k.startswith("encoder.backbone.") for k in sd) else "backbone."
    out: dict = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            out[k] = v
            continue
        rel = k[len(prefix):]
        if rel in _DINOV2_TOPLEVEL_RENAMES:
            out[prefix + _DINOV2_TOPLEVEL_RENAMES[rel]] = v
            continue
        if not rel.startswith("blocks."):
            out[k] = v
            continue
        _, idx, sub = rel.split(".", 2)
        if sub in ("attn.qkv.weight", "attn.qkv.bias"):
            tail = sub.rsplit(".", 1)[1]
            q, kw, vw = v.chunk(3, dim=0)
            base = f"{prefix}encoder.layer.{idx}.attention.attention"
            out[f"{base}.query.{tail}"] = q
            out[f"{base}.key.{tail}"] = kw
            out[f"{base}.value.{tail}"] = vw
            continue
        for old, new in _DINOV2_BLOCK_RENAMES:
            sub = sub.replace(old, new)
        out[f"{prefix}encoder.layer.{idx}.{sub}"] = v
    return out


def build_from_state_dict(sd: dict, dtype=None, device=None, operations=comfy.ops.manual_cast) -> nn.Module:
    """Dispatch to v1 or v2 based on the DINOv2 backbone prefix."""
    sd = _remap_state_dict(sd)
    cls = MoGeModelV2 if any(k.startswith("encoder.backbone.") for k in sd) else MoGeModelV1
    return cls.from_state_dict(sd, dtype=dtype, device=device, operations=operations)


class MoGeModel:
    """Loaded MoGe model + ComfyUI memory management."""

    def __init__(self, state_dict: dict):
        # text encoder dtype closest match
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)

        self.model = build_from_state_dict(state_dict, dtype=self.dtype, device=offload_device, operations=comfy.ops.manual_cast).eval()
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.version = "v2" if hasattr(self.model, "encoder") else "v1"
        self.mask_threshold = float(getattr(self.model, "mask_threshold", 0.5))
        nt = getattr(self.model, "num_tokens_range", (1200, 2500 if self.version == "v1" else 3600))
        self.num_tokens_range = (int(nt[0]), int(nt[1]))

    def infer(self, image: torch.Tensor, num_tokens: Optional[int] = None,
              resolution_level: int = 9, fov_x: Optional[Union[Number, torch.Tensor]] = None,
              force_projection: bool = True, apply_mask: bool = True,
              apply_metric_scale: bool = True
              ) -> Dict[str, torch.Tensor]:
        """Run a single MoGe forward + post-process pass. image is (B, 3, H, W) in [0, 1]."""
        comfy.model_management.load_model_gpu(self.patcher)
        image = image.to(device=self.load_device, dtype=self.dtype)
        H, W = image.shape[-2:]
        aspect_ratio = W / H

        if num_tokens is None:
            lo, hi = self.num_tokens_range
            num_tokens = int(lo + (resolution_level / 9) * (hi - lo))

        out = self.model.forward(image, num_tokens=num_tokens)
        points = out["points"].float()  # recover_focal_shift goes through scipy on CPU; needs fp32.
        mask_binary = out["mask"] > self.mask_threshold
        normal = out.get("normal")
        metric_scale = out.get("metric_scale")

        diag = (1 + aspect_ratio ** 2) ** 0.5

        def focal_from_fov_deg(deg):
            fov = torch.as_tensor(deg, device=points.device, dtype=points.dtype)
            return aspect_ratio / diag / torch.tan(torch.deg2rad(fov / 2))

        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
            # Fall back to 60 deg FoV when the least-squares solver flips the focal sign.
            bad = ~torch.isfinite(focal) | (focal <= 0)
            if bool(bad.any()):
                focal = torch.where(bad, focal_from_fov_deg(60.0), focal)
                _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        else:
            focal = focal_from_fov_deg(fov_x).expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)

        f_diag = focal / 2 * diag
        half = torch.tensor(0.5, device=points.device, dtype=points.dtype)
        intrinsics = intrinsics_from_focal_center(f_diag / aspect_ratio, f_diag, half, half)
        points[..., 2] = points[..., 2] + shift[..., None, None]
        # v2 only: filter mask by depth>0 to drop metric-scale negative-depth artifacts.
        if self.version == "v2":
            mask_binary = mask_binary & (points[..., 2] > 0)
        depth = points[..., 2].clone()

        if force_projection:
            points = depth_map_to_point_map(depth, intrinsics=intrinsics)

        if apply_metric_scale and metric_scale is not None:
            points = points * metric_scale[:, None, None, None]
            depth = depth * metric_scale[:, None, None]

        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.full_like(points, float("inf")))
            depth = torch.where(mask_binary, depth, torch.full_like(depth, float("inf")))
            if normal is not None:
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

        result = {"points": points, "depth": depth, "intrinsics": intrinsics, "mask": mask_binary}
        if normal is not None:
            result["normal"] = normal
        return result
