from aiohttp import web


def is_cross_site_index_navigation(request: web.Request) -> bool:
    """A top-level GET of the index page from another site, such as an OAuth
    sign-in redirecting back to the local frontend. The other site cannot read
    the response and serving the index page has no side effects."""
    return (
        request.method == "GET"
        and request.path == "/"
        and request.headers.get("Sec-Fetch-Mode") == "navigate"
        and request.headers.get("Sec-Fetch-Dest") == "document"
    )
