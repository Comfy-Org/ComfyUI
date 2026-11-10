"""``io``: node schemas and values, re-exported from ``comfy_api.latest``, plus what engine-API nodes add."""
from comfy_api.latest._io import *  # noqa: F403
from comfy_api.latest._io import __all__ as _REEXPORTED
from comfy_api.latest import _io
from comfy_execution.graph_utils import ExecutionBlocker

MAX_RESOLUTION = 16384


class ModelFolder:
    """Combo options that are the files in one of ComfyUI's model folders.

        io.Combo.Input("lora_name", options=io.ModelFolder("loras"))

    ComfyUI lists the folder each time it builds the node's schema, so new files appear
    without the pack reading folders itself. Where ComfyUI's folders aren't available the
    list is empty.
    """

    def __init__(self, folder: str, extensions: list[str] | None = None):
        self.folder = folder
        self.extensions = [e.lower().lstrip(".") for e in extensions] if extensions else None

    def resolve(self) -> list[str]:
        try:
            import folder_paths
            names = folder_paths.get_filename_list(self.folder)
        except Exception:
            return []
        if self.extensions:
            names = [n for n in names if n.rsplit(".", 1)[-1].lower() in self.extensions]
        return list(names)

    def __repr__(self) -> str:
        return f"ModelFolder({self.folder!r})"


@_io.comfytype(io_type="COMBO")
class Combo(_io.Combo):
    class Input(_io.Combo.Input):
        """Combo input (dropdown); ``options`` may be a list or a ``ModelFolder``."""

        @property
        def options(self):
            options = self.__dict__.get("_options")
            return options.resolve() if isinstance(options, ModelFolder) else options

        @options.setter
        def options(self, value):
            self.__dict__["_options"] = value

    class Output(_io.Combo.Output):
        """A combo output whose declared options are its type, so it links to combo inputs that share one."""

        @property
        def io_type(self):
            return self.options if self.options else super().io_type


__all__ = [*_REEXPORTED, "ExecutionBlocker", "MAX_RESOLUTION", "ModelFolder"]
