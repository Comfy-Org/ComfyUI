import numpy as np
import OpenEXR


_CHROMATICITIES = {
    "lin_rec709_scene": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "lin_rec2020_scene": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
    "lin_ap0_scene": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770, 0.32168, 0.33767),
    "lin_ap1_scene": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "lin_p3d65_scene": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290),
    "lin_adobergb_scene": (0.640, 0.330, 0.210, 0.710, 0.150, 0.060, 0.3127, 0.3290),
}


def _rgb_to_xyz(chromaticities):
    xy = np.asarray(chromaticities, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(xy).all() or xy[3, 1] == 0:
        raise ValueError("Invalid EXR chromaticities: expected finite coordinates and a nonzero white-point y.")
    xyz = np.column_stack((xy, 1.0 - xy.sum(axis=1)))
    white = xyz[3] / xy[3, 1]
    primaries = xyz[:3].T
    return primaries * np.linalg.solve(primaries, white), white


def _to_linear(rgb, header):
    color_space = header.get("colorInteropID")
    if header.get("acesImageContainerFlag") == 1:
        color_space = "lin_ap0_scene"
    if color_space == "data":
        return rgb
    if color_space not in (None, "", "unknown"):
        if color_space not in _CHROMATICITIES:
            raise ValueError(f"Unsupported EXR colorInteropID '{color_space}'; convert the file to a supported linear RGB color space before loading.")
        chromaticities = _CHROMATICITIES[color_space]
    else:
        chromaticities = header.get("chromaticities")

    rec709 = _CHROMATICITIES["lin_rec709_scene"]
    if chromaticities is None or np.allclose(chromaticities, rec709, rtol=0, atol=1e-7):
        return rgb

    source, source_white = _rgb_to_xyz(chromaticities)
    target, target_white = _rgb_to_xyz(rec709)
    if not np.allclose(source_white, target_white, rtol=0, atol=1e-7):
        # Bradford adaptation preserves neutral colors when the source white is not D65.
        bradford = np.array(((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)))
        scale = (bradford @ target_white) / (bradford @ source_white)
        source = np.linalg.solve(bradford, scale[:, None] * (bradford @ source))
    matrix = np.linalg.solve(target, source).astype(np.float32)
    return rgb @ matrix.T


def load(path):
    """Return linear Rec.709 RGB (H, W, 3) and optional alpha (H, W), both float32."""
    with OpenEXR.File(path, separate_channels=True) as exr:
        header = exr.header()
        if header["type"] in (OpenEXR.deepscanline, OpenEXR.deeptile):
            raise ValueError("Deep EXR images are not supported.")
        channels = exr.channels()
        if all(c in channels for c in ("R", "G", "B")):
            rgb = np.stack([channels[c].pixels for c in ("R", "G", "B")], axis=-1, dtype=np.float32)
            rgb = _to_linear(rgb, header)
        elif "Y" in channels:
            rgb = np.repeat(channels["Y"].pixels.astype(np.float32, copy=False)[..., None], 3, axis=-1)
        else:
            raise ValueError("EXR images must contain R, G, B channels or a Y channel.")
        alpha = channels["A"].pixels.astype(np.float32, copy=False) if "A" in channels else None
    return rgb, alpha
