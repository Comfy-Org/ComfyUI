from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from comfy.image_encoders.dino2 import Dinov2Model

from .camera import CameraDec, CameraEnc
from .dpt import DPT, DualDPT
from .ray_pose import get_extrinsic_from_camray
from .transform import affine_inverse, pose_encoding_to_extri_intri


_HEAD_REGISTRY = {
    "dpt": DPT,
    "dualdpt": DualDPT,
}


# Backbone presets (mirror the upstream DINOv2 ViT variants).
_BACKBONE_PRESETS = {
    "vits": dict(hidden_size=384,  num_hidden_layers=12, num_attention_heads=6,  use_swiglu_ffn=False),
    "vitb": dict(hidden_size=768,  num_hidden_layers=12, num_attention_heads=12, use_swiglu_ffn=False),
    "vitl": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, use_swiglu_ffn=False),
    "vitg": dict(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, use_swiglu_ffn=True),
}


def _build_backbone_config(
    backbone_name: str,
    *,
    alt_start: int,
    qknorm_start: int,
    rope_start: int,
    cat_token: bool,
) -> dict:
    if backbone_name not in _BACKBONE_PRESETS:
        raise ValueError(f"Unknown DINOv2 backbone variant: {backbone_name!r}")
    cfg = dict(_BACKBONE_PRESETS[backbone_name])
    cfg.update(dict(
        layer_norm_eps=1e-6,
        patch_size=14,
        image_size=518,
        # No mask_token in DA3 weights; omit param to avoid load warnings.
        use_mask_token=False,
        alt_start=alt_start,
        qknorm_start=qknorm_start,
        rope_start=rope_start,
        cat_token=cat_token,
        rope_freq=100.0,
    ))
    return cfg


class DepthAnything3Net(nn.Module):

    PATCH_SIZE = 14

    def __init__(
        self,
        # --- Backbone ---
        backbone_name: str = "vitl",
        out_layers: Sequence[int] = (4, 11, 17, 23),
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = False,
        # --- Head ---
        head_type: str = "dpt",  # dpt or dualdpt
        head_dim_in: int = 1024,
        head_output_dim: int = 1,  # 1 = depth only, 2 = depth+conf
        head_features: int = 256,
        head_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        head_use_sky_head: bool = True, # ignored by DualDPT
        head_pos_embed: Optional[bool] = None,  # default: True for DualDPT, False for DPT
        # --- Camera (multi-view) ---
        has_cam_enc: bool = False,
        has_cam_dec: bool = False,
        cam_dim_out: Optional[int] = None,  # CameraEnc dim_out (defaults to embed_dim)
        cam_dec_dim_in: Optional[int] = None,  # CameraDec dim_in  (defaults to 2*embed_dim with cat_token)
        # ComfyUI plumbing
        device=None, dtype=None, operations=None,
        **_ignored,
    ):
        super().__init__()
        head_cls = _HEAD_REGISTRY[head_type.lower()]
        self.head_type = head_type.lower()
        self.has_sky = (self.head_type == "dpt") and head_use_sky_head
        self.has_conf = head_output_dim > 1
        self.out_layers = list(out_layers)

        backbone_cfg = _build_backbone_config(
            backbone_name,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )
        self.backbone = Dinov2Model(backbone_cfg, dtype, device, operations)

        head_kwargs = dict(
            dim_in=head_dim_in,
            patch_size=self.PATCH_SIZE,
            output_dim=head_output_dim,
            features=head_features,
            out_channels=tuple(head_out_channels),
            device=device, dtype=dtype, operations=operations,
        )
        if self.head_type == "dpt":
            head_kwargs.update(
                use_sky_head=head_use_sky_head,
                pos_embed=(False if head_pos_embed is None else head_pos_embed),
            )
        else:  # dualdpt
            head_kwargs.update(
                pos_embed=(True if head_pos_embed is None else head_pos_embed),
            )
        self.head = head_cls(**head_kwargs)

        # Built only if checkpoint has weights; cam_enc output dim == embed_dim.
        embed_dim = backbone_cfg["hidden_size"]
        if has_cam_enc:
            self.cam_enc = CameraEnc(
                dim_out=cam_dim_out if cam_dim_out is not None else embed_dim,
                num_heads=max(1, embed_dim // 64),
                device=device, dtype=dtype, operations=operations,
            )
        else:
            self.cam_enc = None
        if has_cam_dec:
            default_dim = embed_dim * (2 if cat_token else 1)
            self.cam_dec = CameraDec(
                dim_in=cam_dec_dim_in if cam_dec_dim_in is not None else default_dim,
                device=device, dtype=dtype, operations=operations,
            )
        else:
            self.cam_dec = None

        self.dtype = dtype

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        *,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        export_feat_layers: Optional[Sequence[int]] = None,
        **_unused,
    ) -> Dict[str, torch.Tensor]:
        """Run depth and optionally pose prediction."""
        if image.ndim == 4:
            image = image.unsqueeze(1)  # (B, 1, 3, H, W)
        assert image.ndim == 5 and image.shape[2] == 3, \
            f"image must be (B,3,H,W) or (B,S,3,H,W); got {tuple(image.shape)}"

        B, S, _, H, W = image.shape
        assert H % self.PATCH_SIZE == 0 and W % self.PATCH_SIZE == 0, \
            f"image H,W must be multiples of {self.PATCH_SIZE}; got {(H, W)}"

        # Camera-token preparation (multi-view path).
        cam_token = None
        if extrinsics is not None and intrinsics is not None and self.cam_enc is not None:
            cam_token = self.cam_enc(extrinsics, intrinsics, (H, W))

        # Toggle aux ray output on/off depending on what the caller asked for.
        if isinstance(self.head, DualDPT):
            self.head.enable_aux = bool(use_ray_pose)

        feats, aux_feats = self.backbone.get_intermediate_layers_da3(
            image, self.out_layers, cam_token=cam_token,
            ref_view_strategy=ref_view_strategy,
            export_feat_layers=export_feat_layers,
        )
        head_out = self.head(feats, H=H, W=W, patch_start_idx=0)

        # Pose prediction.
        out: Dict[str, torch.Tensor] = {}
        if use_ray_pose and "ray" in head_out and "ray_conf" in head_out:
            ray = head_out["ray"]
            ray_conf = head_out["ray_conf"]
            extr_c2w, focal, pp = get_extrinsic_from_camray(
                ray, ray_conf, ray.shape[-3], ray.shape[-2],
            )
            # Match the upstream output: w2c, drop the homogeneous row.
            extr_w2c = affine_inverse(extr_c2w)[:, :, :3, :]
            # Build pixel-space intrinsics from the normalised focal/pp output.
            intr = torch.eye(3, device=ray.device, dtype=ray.dtype)
            intr = intr[None, None].expand(extr_c2w.shape[0], extr_c2w.shape[1], 3, 3).clone()
            intr[:, :, 0, 0] = focal[:, :, 0] / 2 * W
            intr[:, :, 1, 1] = focal[:, :, 1] / 2 * H
            intr[:, :, 0, 2] = pp[:, :, 0] * W * 0.5
            intr[:, :, 1, 2] = pp[:, :, 1] * H * 0.5
            out["extrinsics"] = extr_w2c
            out["intrinsics"] = intr
        elif self.cam_dec is not None and S > 1:
            # Decode the cam-token of the final out_layer into a pose encoding.
            cam_feat = feats[-1][1]  # (B, S, dim_in_to_cam_dec)
            pose_enc = self.cam_dec(cam_feat)
            c2w_3x4, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
            # Match the upstream output convention: w2c (world->camera), 3x4.
            c2w_4x4 = torch.cat([
                c2w_3x4,
                torch.tensor([0, 0, 0, 1], device=c2w_3x4.device, dtype=c2w_3x4.dtype)
                    .view(1, 1, 1, 4).expand(B, S, 1, 4),
            ], dim=-2)
            out["extrinsics"] = affine_inverse(c2w_4x4)[:, :, :3, :]
            out["intrinsics"] = intr

        # Flatten the views axis for per-pixel outputs (depth/conf/sky) so the
        # per-image consumer keeps its (B*S, H, W) interface.
        for k, v in head_out.items():
            if k in ("ray", "ray_conf"):
                # Keep multi-view shape for downstream pose work.
                out[k] = v
            elif v.ndim >= 3 and v.shape[0] == B and v.shape[1] == S:
                out[k] = v.reshape(B * S, *v.shape[2:])
            else:
                out[k] = v

        if export_feat_layers:
            out["aux_features"] = self._reshape_aux_features(aux_feats, H, W)
        return out

    def _reshape_aux_features(self, aux_feats, H: int, W: int):
        """Reshape (B, S, N, C) aux features into (B, S, h_p, w_p, C)."""
        ph, pw = H // self.PATCH_SIZE, W // self.PATCH_SIZE
        out = []
        for f in aux_feats:
            B, S, N, C = f.shape
            assert N == ph * pw, f"aux feature seq mismatch: {N} != {ph}*{pw}"
            out.append(f.reshape(B, S, ph, pw, C))
        return out
