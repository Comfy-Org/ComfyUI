"""How an asset's preview is reported: a linked preview, the asset itself, or none."""
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


def content_url(asset_id: str) -> str:
    # No query string: clients tell this form apart from /api/view?type=... by that.
    return f"/api/assets/{asset_id}/content"


def preview_fields(
    asset_id: str,
    preview_id: str | None,
    mime_type: str | None,
    file_path: str | None,
    is_missing: bool,
    preview_paths: dict[str, str],
) -> tuple[str | None, str | None]:
    """(preview_id, preview_url); preview_id is only ever sent with a URL."""
    # A self-nomination is ignored: whether a file is its own preview is decided below.
    if preview_id and preview_id != asset_id:
        # A nominated preview is one whatever it holds, so no media check here.
        if preview_id in preview_paths:
            return preview_id, content_url(preview_id)
        return None, None
    if is_missing or not file_path:
        return None, None
    kind = own_preview_kind(mime_type, file_path)
    if kind is None:
        return None, None
    return (asset_id if kind == "image" else None), content_url(asset_id)
