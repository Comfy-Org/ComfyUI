"""How an asset's preview is reported: a linked preview, the asset itself, or none."""
import mimetypes
import os
import urllib.parse

from app.assets.services.path_utils import compute_asset_response_paths


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


# models is deliberately absent: /api/view has no directory type for it.
VIEWABLE_NAMESPACES = frozenset({"input", "output", "temp"})


def view_url(file_path: str | None) -> str | None:
    # /api/view is a FileResponse: byte-range seeking, no user header, no access write.
    if not file_path:
        return None
    paths = compute_asset_response_paths(file_path)
    if not paths:
        return None
    logical_path, relative_path = paths
    namespace = logical_path.split("/", 1)[0]
    if namespace not in VIEWABLE_NAMESPACES or not relative_path:
        return None

    subfolder, _, filename = relative_path.rpartition("/")
    url = f"/api/view?type={namespace}&filename={urllib.parse.quote(filename, safe='')}"
    if subfolder:
        url += f"&subfolder={urllib.parse.quote(subfolder, safe='')}"
    return url


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
            # By id only where /api/view can't serve the file, such as previews/.
            return preview_id, view_url(preview_paths[preview_id]) or content_url(preview_id)
        return None, None
    if is_missing or not file_path:
        return None, None
    kind = own_preview_kind(mime_type, file_path)
    if kind is None:
        return None, None
    url = view_url(file_path)
    if url is None:
        return None, None
    return (asset_id if kind == "image" else None), url
