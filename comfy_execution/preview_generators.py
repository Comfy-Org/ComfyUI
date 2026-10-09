"""Preview generators by MIME type, the worker pool that runs them, and the tonemap
Core's own previews share.

Custom nodes register generators through ``comfy_api``; Core's own are fallbacks
that a registration for the same MIME type replaces. Workers are daemon threads, not
a ThreadPoolExecutor, whose workers are joined at exit: a generator that never
returns must not block shutdown.
"""

import logging
import queue
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from comfy_api.latest._previews import PreviewGenerator

PREVIEW_WORKERS = 2
PREVIEW_MAX_PIXELS = 1_000_000

_registered: dict[str, "PreviewGenerator"] = {}
_core: dict[str, "PreviewGenerator"] = {}
_lock = threading.Lock()

_jobs: "queue.SimpleQueue[tuple[Callable[[], Any], Future]]" = queue.SimpleQueue()
_workers_started = False


def register_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in generator.mime_types:
            replaced = _registered.get(mime_type)
            if replaced is not None and replaced is not generator:
                logging.warning("Preview generator for %s replaced by %r", mime_type, generator)
            _registered[mime_type] = generator


def unregister_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in [m for m, g in _registered.items() if g is generator]:
            del _registered[mime_type]


def set_core_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in generator.mime_types:
            _core[mime_type] = generator


def get_preview_generator(mime_type: str | None) -> "PreviewGenerator | None":
    if not mime_type:
        return None
    return _registered.get(mime_type) or _core.get(mime_type)


def preview_deadline_seconds(count: int) -> float:
    """How long a caller waits for ``count`` previews before giving up on the rest."""
    return min(5.0 + 0.5 * count, 30.0)


def _work() -> None:
    while True:
        fn, future = _jobs.get()
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(fn())
        except BaseException as exc:
            future.set_exception(exc)


def submit_preview_job(fn: Callable[[], Any]) -> Future:
    """Run ``fn`` on a preview worker. Cancel the future to drop it if it hasn't started."""
    global _workers_started
    with _lock:
        if not _workers_started:
            for i in range(PREVIEW_WORKERS):
                threading.Thread(target=_work, name=f"preview-worker-{i}", daemon=True).start()
            _workers_started = True
    future: Future = Future()
    _jobs.put((fn, future))
    return future


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
