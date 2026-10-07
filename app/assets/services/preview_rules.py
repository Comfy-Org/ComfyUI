"""Which assets are their own preview."""
import mimetypes
import os

# What a client can render from the bytes themselves; anything else needs a nominated preview.
PREVIEWABLE_MIME_PREFIXES = ("image/", "video/", "audio/", "text/")

# Images browsers can't display. Matched by extension too, since hosts differ on .hdr.
_NEVER_SELF_EXTENSIONS = frozenset({".exr", ".hdr"})
_NEVER_SELF_MIME_TYPES = frozenset({"image/x-exr", "image/vnd.radiance"})


def own_preview_kind(mime_type: str | None, path: str | None) -> str | None:
    """The media kind ("image", "video", ...) when the file previews itself, else None.

    ``path`` is the on-disk path, not the caller-editable name, so a rename cannot
    change what previews.
    """
    mime = (mime_type or mimetypes.guess_type(path or "")[0] or "").split(";", 1)[0].strip().lower()
    if not mime.startswith(PREVIEWABLE_MIME_PREFIXES):
        return None
    if mime in _NEVER_SELF_MIME_TYPES or os.path.splitext(path or "")[1].lower() in _NEVER_SELF_EXTENSIONS:
        return None
    return mime.split("/", 1)[0]
