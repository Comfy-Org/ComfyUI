"""Format-specific loaders return float32 RGB (H, W, 3) and optional alpha (H, W)."""

import logging

LOADERS = {}

try:
    from .exr import load as load_exr
    LOADERS[".exr"] = load_exr
except ImportError as e:
    logging.warning("OpenEXR import failed: %s. Install OpenEXR for EXR color conversion.", e)
