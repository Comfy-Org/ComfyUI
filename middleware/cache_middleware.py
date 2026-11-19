"""Cache control middleware for ComfyUI server"""

from aiohttp import web
from typing import Callable, Awaitable

# Time in seconds
ONE_HOUR: int = 3600
ONE_DAY: int = 86400
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


@web.middleware
async def cache_control(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.Response]]
) -> web.Response:
    """Cache control middleware that sets appropriate cache headers based on file type and response status"""
    response: web.Response = await handler(request)

    path_filename = request.path.rsplit("/", 1)[-1]
    is_entry_point = path_filename.startswith("index") and path_filename.endswith(
        ".json"
    )

    if request.path.endswith(".js") or request.path.endswith(".css") or is_entry_point:
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    # Early return for non-image files - no cache headers needed
    if not request.path.lower().endswith(IMG_EXTENSIONS):
        return response

    # Handle image files
    if response.status == 404:
        response.headers.setdefault("Cache-Control", f"public, max-age={ONE_HOUR}")
    elif response.status in (200, 201, 202, 203, 204, 205, 206, 301, 308):
        # Success responses and permanent redirects - cache for 1 day
        response.headers.setdefault("Cache-Control", f"public, max-age={ONE_DAY}")
    elif response.status in (302, 303, 307):
        # Temporary redirects - no cache
        response.headers.setdefault("Cache-Control", "no-cache")
    # Note: 304 Not Modified falls through - no cache headers set

    return response
