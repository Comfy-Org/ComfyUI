"""Serves openapi.yaml and renders it as browsable API docs.

Uses Redoc, not Swagger UI: the local server is unauthenticated by default, so
the viewer must not be able to execute requests.
"""

import os

from aiohttp import web

SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "openapi.yaml"
)

# The spec URL is relative so it resolves from both /api-docs and /api/api-docs.
# The viewer is pinned and loaded from a CDN, so the page needs network access;
# if it fails to load, the fallback below points at the locally served spec.
API_DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ComfyUI API Reference</title>
  </head>
  <body style="margin: 0">
    <redoc spec-url="openapi.yaml"></redoc>
    <div id="fallback" style="display: none; margin: 3rem; line-height: 1.6">
      The API docs viewer is loaded from a CDN and could not be reached. The
      specification itself is served locally: <a href="openapi.yaml">openapi.yaml</a>
    </div>
    <script
      src="https://cdn.jsdelivr.net/npm/redoc@2.5.0/bundles/redoc.standalone.js"
      onerror="document.getElementById('fallback').style.display='block'"
    ></script>
  </body>
</html>
"""


def add_api_docs_routes(routes: web.RouteTableDef) -> None:
    """Register /openapi.yaml and /api-docs on the given route table."""

    @routes.get("/openapi.yaml")
    async def get_openapi_spec(request):
        # The spec is edited in place during development, so never let the
        # browser hold a stale copy. no-cache still allows a 304 via the ETag
        # FileResponse sets, which matters for a ~230 KB file.
        return web.FileResponse(SPEC_PATH, headers={
            "Content-Type": "application/yaml",
            "Cache-Control": "no-cache",
        })

    @routes.get("/api-docs")
    async def get_api_docs(request):
        return web.Response(
            text=API_DOCS_HTML,
            content_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )
