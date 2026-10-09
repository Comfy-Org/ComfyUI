from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from PIL import Image

class PreviewGenerator(ABC):
    """Makes a browser-displayable preview of an uploaded file of a type browsers can't show.

    Core looks a generator up by the MIME type ``mimetypes.guess_type`` gives the
    file's path, so a generator for a new extension also registers it with
    ``mimetypes.add_type``. Registering a generator for a MIME type replaces any
    earlier one for it, including Core's own. Generators run for uploads only, not
    for node outputs. An image of a registered type browsers can't display is never
    its own preview: it shows the generated one, or none. A type browsers can show
    falls back to itself when no preview is generated.
    """

    mime_types: ClassVar[tuple[str, ...]]

    @abstractmethod
    def generate(self, source_path: str, max_pixels: int) -> "Image.Image | None":
        """Return an 8-bit image (mode RGB, RGBA, L or P) of the file, or None for no preview.

        Runs on a worker thread, concurrently with other generations, and its result
        is dropped once Core's deadline passes. Core downscales the image to at most
        ``max_pixels`` and encodes it; a generator may use ``max_pixels`` to decode less.
        """
