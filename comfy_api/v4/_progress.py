import math

import comfy.utils
from comfy.cli_args import args
from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io
from comfy_execution.progress import get_progress_state

from ._execution import running_node


class Progress(ProxiedSingleton):
    async def update(
        self, value: float, total: float, preview: io.Image.Type | None = None
    ) -> None:
        """Show the running node's progress, and the first image of `preview` if given.

        Raises `InterruptProcessingException` if the user cancelled the prompt, as ComfyUI's own
        progress bar does, and `RuntimeError` unless made while the node runs, from its `execute`
        or a function it calls. A call is not sent if it has no preview,
        `value` is below `total`, `total` is the one last shown, and `value` has moved forward by
        less than 0.5% of `total`, or not at all. What is sent goes to the clients ComfyUI's own
        progress bar reports to, as it sends it: a `value` above `total` is shown as `total`.
        Raises `ValueError` if `value` or `total` is not a finite number, which a browser cannot
        read.
        """
        from comfy.model_management import throw_exception_if_processing_interrupted

        node_id = running_node().node_id
        throw_exception_if_processing_interrupted()
        if not (math.isfinite(value) and math.isfinite(total)):
            raise ValueError(
                f"progress must be finite numbers, not {value!r} of {total!r}"
            )
        value = min(value, total)
        shown = get_progress_state().nodes[node_id]
        if (
            preview is None
            and value < total == shown["max"]
            and 0 <= value - shown["value"] < total / 200
        ):
            return
        image = None
        if preview is not None:
            from latent_preview import preview_to_image

            # Drop alpha: the server cannot encode an RGBA image as JPEG.
            rgb = preview_to_image(preview[0, ..., :3], do_scale=False)
            image = ("JPEG", rgb, args.preview_size)
        # ComfyUI's server installs this hook; its own progress bar reports through it.
        hook = comfy.utils.PROGRESS_BAR_HOOK
        if hook is None:
            get_progress_state().update_progress(node_id, value, total, image)
        else:
            hook(value, total, image, node_id=node_id)
