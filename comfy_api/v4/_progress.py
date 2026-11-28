import math
import threading
import time
import weakref
from dataclasses import dataclass

import comfy.utils
from comfy.cli_args import args
from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import ExecutionContext

from ._execution import _running_node


@dataclass
class _Shown:
    bar: comfy.utils.ProgressBar
    # When the bar was made, and the highest value it has been given.
    made: float
    peak: float


class Progress(ProxiedSingleton):
    def __init__(self):
        super().__init__()
        # ComfyUI's progress bar decides what is sent from what it sent before, so a node's
        # updates share one bar for one run of its prompt.
        self._run: weakref.ref | None = None
        self._shown: dict[ExecutionContext, _Shown] = {}
        self._shown_lock = threading.Lock()

    def _bar(
        self, node: ExecutionContext, value: float, total: float
    ) -> comfy.utils.ProgressBar:
        run = get_progress_state()
        now = time.perf_counter()
        with self._shown_lock:
            if self._run is None or self._run() is not run:
                self._run = weakref.ref(run)
                self._shown = {}
            shown = self._shown.get(node)
            # ComfyUI's bar sends no value below the last it sent. A value below the bar's
            # highest gets a bar of its own, as a node that makes a progress bar for each pass
            # would give it: one in ComfyUI's interval between updates at most, however the
            # values come.
            if shown is None or (
                value < shown.peak
                and now - shown.made >= comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL
            ):
                bar = comfy.utils.ProgressBar(total, node_id=node.node_id)
                shown = self._shown[node] = _Shown(bar, now, value)
            else:
                shown.peak = max(shown.peak, value)
            return shown.bar

    async def update(
        self, value: float, total: float, preview: io.Image.Type | None = None
    ) -> None:
        """Show the running node's progress, and the first image of `preview` if given.

        The update goes through a progress bar of ComfyUI's own, kept for the node while it
        runs, so ComfyUI decides which updates it sends and to which clients: it does not send
        every small step, and shows a `value` above `total` as `total`. A `value` below an
        earlier one starts a new pass, as a new progress bar would, at most once in ComfyUI's
        interval between updates. Raises
        `InterruptProcessingException` if the user cancelled the prompt, and `RuntimeError`
        unless made while the node runs, from its `execute` or a function it calls. Raises
        `ValueError` if `value` or `total` is not a finite number, which a browser cannot read.
        """
        from comfy.model_management import throw_exception_if_processing_interrupted

        node = _running_node()
        throw_exception_if_processing_interrupted()
        if not (math.isfinite(value) and math.isfinite(total)):
            raise ValueError(
                f"progress must be finite numbers, not {value!r} of {total!r}"
            )
        image = None
        if preview is not None:
            from latent_preview import preview_to_image

            # Drop alpha: the server cannot encode an RGBA image as JPEG.
            rgb = preview_to_image(preview[0, ..., :3], do_scale=False)
            image = ("JPEG", rgb, args.preview_size)
        self._bar(node, value, total).update_absolute(value, total, image)
