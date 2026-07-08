# Live preview for TripoSplat: decode an x0 estimate into a coarse gaussian splat and render it with a perspective orbit camera.
import numpy as np
from PIL import Image

_C0 = 0.28209479177387814
_LATENT_TOKENS = 8192  # q_token_length
_LATENT_CH = 16        # in_channels
_OBJECT_TO_VIEWER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)  # object frame -> viewer Y-up frame


def _view_matrix(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], np.float32)
    return Rx @ Ry


def render_splat(xyz, rgb, scale, opacity=None, yaw=35.0, pitch=30.0, size=320, min_px=2, gain=1.0,
                 max_px=9, min_opacity=0.0, fov=35.0, dist=2.2):
    # Project gaussian centers with a perspective camera and paint each as a filled disk whose screen
    # radius follows the gaussian's world-space scale, composited with a nearest-wins z-buffer.
    # gain scales the footprint (≈ std spanned), `min_px`/`max_px` clamp the on-screen radius.

    pts = xyz.astype(np.float32) @ _OBJECT_TO_VIEWER.T
    v = pts @ _view_matrix(yaw, pitch).T
    zc = v[:, 2] + dist
    keep = zc > 1e-2
    if opacity is not None and min_opacity > 0.0:  # culls gaussians with very low opacity
        keep = keep & (opacity > min_opacity)
    v, zc, scale = v[keep], zc[keep], scale[keep]
    col = (np.clip(rgb, 0, 1)[:, :3] * 255).astype(np.uint8)[keep]
    if v.shape[0] == 0:
        return Image.fromarray(np.zeros((size, size, 3), np.uint8))
    f = (size / 2) / np.tan(np.radians(fov) / 2)
    cx = size / 2 + f * v[:, 0] / zc
    cy = size / 2 + f * v[:, 1] / zc
    radius = np.clip(np.round(f * scale / zc * gain), min_px, max_px).astype(np.int32)

    # Expand each splat to its disk pixels, bucketed by integer radius so it stays vectorized.
    px, py, pz, pc = [], [], [], []
    for r in range(int(radius.min()), int(radius.max()) + 1):
        m = radius == r
        if not m.any():
            continue
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (dx * dx + dy * dy) <= r * r
        ox, oy = dx[disk], dy[disk]
        px.append((cx[m, None] + ox).ravel())
        py.append((cy[m, None] + oy).ravel())
        pz.append(np.repeat(zc[m], ox.size))
        pc.append(np.repeat(col[m], ox.size, axis=0))
    px, py = np.concatenate(px), np.concatenate(py)
    pz, pc = np.concatenate(pz), np.concatenate(pc)
    xi = np.clip(px, 0, size - 1).astype(np.int64)
    yi = np.clip(py, 0, size - 1).astype(np.int64)

    # Nearest-wins z-buffer: pack (quantized depth, source index), per-pixel min picks the closest
    # splat, then decode the winning index back to its color.
    pid = yi * size + xi
    q = np.clip((pz * 1024.0).astype(np.int64), 0, (1 << 20) - 1)  # near = small
    key = (q << 32) | np.arange(pid.size, dtype=np.int64)
    buf = np.full(size * size, 1 << 62, np.int64)
    np.minimum.at(buf, pid, key)
    img = np.zeros((size * size, 3), np.uint8)
    hit = buf < (1 << 62)
    img[hit] = pc[buf[hit] & 0xFFFFFFFF]
    return Image.fromarray(img.reshape(size, size, 3))


def _extract_latent(x0):
    # x0 from the sampler callback is the nested latent packed to (B, 1, TOKENS*CH + 1*5);
    # the plain single-latent case is (B, TOKENS, CH). Return the (B, TOKENS, CH) latent stream.
    if x0.ndim == 3 and x0.shape[1] == _LATENT_TOKENS and x0.shape[2] == _LATENT_CH:
        return x0
    flat = x0.reshape(x0.shape[0], -1)
    return flat[:, :_LATENT_TOKENS * _LATENT_CH].reshape(x0.shape[0], _LATENT_TOKENS, _LATENT_CH)


def decode_x0_to_image(decoder, x0, cfg):
    # Decode x0 at a coarse octree level / few gaussians and render a preview image.
    latent = _extract_latent(x0)
    fsm = decoder.first_stage_model
    gaussian = fsm.decode(latent.to(decoder.device, decoder.vae_dtype),
                          num_gaussians=cfg.get("gaussians", 16384), level=cfg.get("level", 5))[0]
    xyz = gaussian.get_xyz.float().cpu().numpy()
    rgb = gaussian._features_dc.float().cpu().numpy()[:, 0, :] * _C0 + 0.5
    scale = gaussian.get_scaling.float().cpu().numpy().max(axis=1)  # per-splat world radius (largest axis)
    opacity = gaussian.get_opacity.float().cpu().numpy()[:, 0]
    return render_splat(xyz, rgb, scale, opacity=opacity, yaw=cfg.get("yaw", 35.0), pitch=cfg.get("pitch", 30.0),
                        size=cfg.get("size", 320), min_px=1, gain=1.0, max_px=cfg.get("point_size", 3),
                        min_opacity=0.01)
