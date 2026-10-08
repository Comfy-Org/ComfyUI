import math

import torch
import torch.nn.functional as F
from comfy.ldm.cascade.common import LayerNorm2d_op
from torch import nn

from typing import List, Optional, Tuple, Union

from ..utils import perspective_projection
from .transformer import MLP

class CameraEncoder(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 14, device=None, dtype=None, operations=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)

        self.conv = operations.Conv2d(embed_dim + 99, embed_dim, kernel_size=1, bias=False, device=device, dtype=dtype)
        self.norm = LayerNorm2d_op(operations)(embed_dim, device=device, dtype=dtype)

    def forward(self, img_embeddings: torch.Tensor, rays: torch.Tensor):
        B, D, _h, _w = img_embeddings.shape

        scale = 1 / self.patch_size
        rays = F.interpolate(rays, scale_factor=(scale, scale), mode="bilinear", align_corners=False, antialias=True)
        rays = rays.permute(0, 2, 3, 1).contiguous()  # [b, h, w, 2]
        rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
        rays_embeddings = self.camera(pos=rays.reshape(B, -1, 3))  # (bs, N, 99): rays fourier embedding
        rays_embeddings = rays_embeddings.reshape(B, _h, _w, -1).permute(0, 3, 1, 2).contiguous()

        z = torch.cat([img_embeddings, rays_embeddings], dim=1)
        return self.norm(self.conv(z))


class FourierPositionEncoding(nn.Module):
    """Sin/cos Fourier features for ray positions"""

    def __init__(self, n: int, num_bands: int, max_resolution: int):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    def forward(self, pos: torch.Tensor):
        fourier_pos_enc = _generate_fourier_features(pos, num_bands=self.num_bands, max_resolution=self.max_resolution)
        return fourier_pos_enc


def _generate_fourier_features(pos: torch.Tensor, num_bands: int, max_resolution: List[int], min_freq: float = 1.0):
    b, n = pos.shape[:2]

    freq_bands = torch.stack([torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=pos.device, dtype=pos.dtype) for res in max_resolution], dim=0)

    per_pos_features = pos.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
    per_pos_features = per_pos_features.reshape(b, n, -1)

    # Sin-Cos
    per_pos_features = torch.cat([torch.sin(math.pi * per_pos_features), torch.cos(math.pi * per_pos_features)], dim=-1)

    # Concat with initial pos
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features


class PerspectiveHead(nn.Module):
    """
    Predict camera translation (s, tx, ty) and perform full-perspective 2D reprojection (CLIFF/CameraHMR setup).
    """

    def __init__(self, input_dim: int, img_size: Union[int, Tuple[int, int]],  # model input size (W, H)
        mlp_depth: int = 1, mlp_channel_div_factor: int = 8, default_scale_factor: float = 1.0,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        # Metadata to compute 3D skeleton and 2D reprojection
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.ncam = 3  # (s, tx, ty)
        self.default_scale_factor = default_scale_factor

        self.proj = MLP(
            input_dim=input_dim,
            hidden_dim=input_dim // mlp_channel_div_factor,
            output_dim=self.ncam,
            num_layers=mlp_depth,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x: torch.Tensor, init_estimate: Optional[torch.Tensor] = None):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.ncam]
        """
        pred_cam = self.proj(x)
        if init_estimate is not None:
            pred_cam = pred_cam + init_estimate

        return pred_cam

    def perspective_projection(
        self,
        points_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        bbox_center: torch.Tensor, # [N, 2], in original image space (w, h)
        bbox_size: torch.Tensor, # [N,], in original image space
        cam_int: torch.Tensor, # [B, 3, 3]
    ):
        batch_size = points_3d.shape[0]
        pred_cam = pred_cam.clone()
        pred_cam[..., [0, 2]] *= -1  # Camera system difference

        # Compute camera translation: (scale, x, y) --> (x, y, depth)
        # depth ~= f / s, Note that f is in the NDC space
        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bs = bbox_size * s * self.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
        cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        # Compute camera translation
        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)

        # Projection to the image plane, note that the projection output is in original image space now.
        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d.reshape(batch_size, -1, 2),
            "pred_keypoints_2d_depth": j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
            "pred_cam_t": pred_cam_t, "focal_length": focal_length,
        }
