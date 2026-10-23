"""The tonemap Core's previews share: the save node's and the upload decoder's."""

import torch
import torch.nn.functional as F
from PIL import Image

PREVIEW_MAX_PIXELS = 1_000_000


def linear_to_preview(image: torch.Tensor, max_pixels: int = PREVIEW_MAX_PIXELS) -> Image.Image:
    """An 8-bit sRGB image of a scene-linear (H, W, C) image, at most ``max_pixels``; RGBA if C is 4.

    Clamps to [0, 1]. RGBA is straight alpha, averaged weighted by alpha so colour under
    transparent pixels doesn't bleed into the edges.
    """
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    x = torch.nan_to_num(image.float(), nan=0.0, posinf=1.0, neginf=0.0)
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, 3)
    height, width = x.shape[:2]
    x = x.movedim(-1, 0).unsqueeze(0)
    has_alpha = x.shape[1] == 4
    if has_alpha:
        alpha = x[:, 3:].clamp(0.0, 1.0)
        x = torch.cat([x[:, :3] * alpha, alpha], dim=1)
    # WebP's side limit is 16383 px.
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5, 16383 / max(width, height))
    if scale < 1.0:
        x = F.interpolate(x, size=(max(1, int(height * scale)), max(1, int(width * scale))), mode="area")
    if has_alpha:
        alpha = x[:, 3:]
        x = torch.cat([torch.where(alpha > 0, x[:, :3] / alpha.clamp_min(1e-12), 0.0), alpha], dim=1)
    linear = x[0].movedim(0, -1).clamp(0.0, 1.0)
    rgb = linear[..., :3]
    linear[..., :3] = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.pow(1 / 2.4) - 0.055)
    pixels = (linear * 255 + 0.5).to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels, "RGBA" if has_alpha else "RGB")
