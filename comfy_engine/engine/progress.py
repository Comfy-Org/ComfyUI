"""Core engine extension: the running node's progress bar."""
from comfy_engine import _runtime

MAX_PREVIEW_RESOLUTION = 512


class EngineOps:
    def update(self, value, total, preview=None, format="JPEG"):
        import comfy.utils

        plan = _runtime.current_runtime().plan
        hook = comfy.utils.PROGRESS_BAR_HOOK
        if hook is None:
            return None
        image = None
        if preview is not None:
            import numpy as np
            from PIL import Image

            if format not in ("JPEG", "PNG"):
                raise ValueError("preview format must be JPEG or PNG")
            pixels = preview[0] if preview.ndim == 4 else preview
            array = (pixels.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            image = (format, Image.fromarray(array), MAX_PREVIEW_RESOLUTION)
        hook(value, total, image, getattr(plan, "prompt_id", None), getattr(plan, "node_id", None))
        return None
