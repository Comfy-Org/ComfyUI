"""Response headers for serving stored asset bytes, shared by every asset content route."""

import urllib.parse

import folder_paths


def download_headers(content_type: str, filename: str | None, disposition: str, sec_fetch_dest: str | None) -> tuple[str, dict[str, str]]:
    """(content type, headers) for serving bytes stored as ``content_type`` under ``filename``.

    User-controlled asset content must not render inline in the app origin
    (stored XSS via SVG/HTML/XML): dangerous types are forced to an
    octet-stream attachment, except an SVG loaded into an <img>, see
    folder_paths.renders_safely_as_image. That decision reads Sec-Fetch-Dest,
    so those responses carry Vary and no-store: a cache keyed on the URL alone
    could otherwise replay an inline SVG primed by an <img> fetch to a document
    navigation of the same URL.
    """
    headers = {}
    if folder_paths.is_dangerous_content_type(content_type):
        headers["Vary"] = "Sec-Fetch-Dest"
        headers["Cache-Control"] = "no-store"
        if not folder_paths.renders_safely_as_image(content_type, sec_fetch_dest):
            content_type = "application/octet-stream"
            disposition = "attachment"
    # The stored type is uploader-supplied and may carry parameters, which aiohttp rejects in content_type.
    content_type = content_type.split(";", 1)[0].strip() or "application/octet-stream"
    safe_name = (filename or "").replace("\r", "").replace("\n", "")
    headers["Content-Disposition"] = f"{disposition}; filename*=UTF-8''{urllib.parse.quote(safe_name)}"
    headers["X-Content-Type-Options"] = "nosniff"
    return content_type, headers
