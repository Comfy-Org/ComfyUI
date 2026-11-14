from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from comfy_api.latest import ComfyExtension as ComfyExtension_latest


class RouteRequest(TypedDict):
    """An HTTP request to one of a pack's routes, as its handler receives it."""

    method: str
    """The request's method; "HEAD" for a HEAD request to a "GET" route."""
    path: str
    """The decoded path as requested, with the `/api` prefix if the request used it."""
    match_info: dict[str, str]
    """The decoded value each of the route's `{name}` parts matched."""
    query: list[tuple[str, str]]
    """The decoded query parameters in order; a repeated name appears once per value."""
    headers: dict[str, str]
    """Header names in lower case; a repeated header's values joined with ", "."""
    body: bytes
    """The whole body. A body over ComfyUI's upload limit (`--max-upload-size`) is answered 413
    before the handler runs."""
    remote: str | None
    """The client's address as the connection reports it (a proxy's, behind one)."""


class RouteResponse(TypedDict):
    """What a route's handler returns; a missing key gives the client a 500 response."""

    status: int
    headers: dict[str, str]
    """Without "Content-Type", a body is sent as "application/octet-stream"."""
    body: bytes


class Route(TypedDict):
    """One HTTP route of a pack, as `ComfyExtension.get_routes` returns it."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    """A "GET" route also answers HEAD requests."""
    path: str
    """Starts with "/"; `{name}` or `{name:regex}` captures a part into `match_info`."""
    handler: Callable[[RouteRequest], Awaitable[RouteResponse]]
    """Awaited on ComfyUI's event loop for each request; an exception it raises gives a 500
    response."""


class ComfyExtension(ComfyExtension_latest):
    async def get_routes(self) -> list[Route]:
        """Return the HTTP routes ComfyUI serves for this pack at `path` and at `/api` + `path`.

        Read once, at load. ComfyUI disables the pack, with a warning, if a route's method is not
        one `Route` names, its path is malformed or it has no handler; if the pack has routes and
        loads after ComfyUI starts serving; or if a request one of its routes answers could also
        reach another of its routes (a literal path beside a pattern is allowed), a route of a V4
        pack loaded before it, or anything else ComfyUI's server answers: its own routes, static
        folders and sub-applications, and routes of packs written to an older API. Two patterns
        count as reaching one request unless a literal segment, or the number of segments,
        differs before either has a `{name:regex}` part.
        """
        return []
