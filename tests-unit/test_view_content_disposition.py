"""Regression tests for ComfyUI issue #8914 — /view Content-Disposition
headers must carry an explicit disposition-type per RFC 2183/6266.

Before the fix every /view branch emitted a bare `filename="..."` value with
no disposition-type. That is not a valid Content-Disposition: Go's
mime.ParseMediaType rejects it outright (downloads saved as "view"), and
Python's email parser reads the whole value back as a bogus disposition type.
The fix prefixes the safe paths with `inline;` and leaves the existing
`attachment;` branch (the stored-XSS guard from #15149) untouched.

server.py cannot be imported in a unit test (it pulls in torch and the full
node runtime), so these tests extract the view_image handler from the
server.py AST and execute it directly against the real folder_paths module
with a stub PromptServer — the same server-free approach as
tests-unit/security_test/. No network, no server socket, no real PromptServer.
"""

import ast
import email.message
import mimetypes
import os
import textwrap
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from PIL import Image

import folder_paths

pytestmark = pytest.mark.asyncio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PY = os.path.join(REPO_ROOT, "server.py")

SVG_WITH_SCRIPT = (
    '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
)


def _load_view_handler():
    """Extract the view_image closure from server.py and rebuild it callable.

    view_image is defined (and captured) inside PromptServer.__init__, so the
    only way to unit-test it without a full server is to pull the function
    source out of the AST, wrap it in a tiny factory that injects `self`, and
    exec it with the handler's module-level dependencies resolved to
    lightweight stand-ins: the real folder_paths (its dangerous-type logic is
    part of what these tests pin) plus stubs for the asset-hash resolver and
    the PromptServer instance, neither of which the tested paths reach.
    """
    with open(SERVER_PY, "r", encoding="utf-8") as f:
        source = f.read()

    func_node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "view_image"
    )

    # Drop the @routes.get("/view") decorator line; registering on a real
    # RouteTableDef is neither possible nor needed for direct invocation.
    body = "".join(
        line
        for line in ast.get_source_segment(source, func_node).splitlines(True)
        if not line.strip().startswith("@")
    )
    factory_src = "def _make_view_image(self):\n" + textwrap.indent(
        textwrap.dedent(body), "    "
    )
    # The original closure ends with the (stripped) decorator registration and
    # no return, so the factory has to hand back the rebuilt handler itself.
    # get_source_segment's last line carries no trailing newline.
    if not factory_src.endswith("\n"):
        factory_src += "\n"
    factory_src += "    return view_image\n"

    namespace = {
        "folder_paths": folder_paths,
        "resolve_hash_to_path": MagicMock(return_value=None),
        "web": web,
        "os": os,
        "mimetypes": mimetypes,
        "Image": Image,
        "BytesIO": BytesIO,
    }
    exec(compile(factory_src, SERVER_PY, "exec"), namespace)
    return namespace["_make_view_image"](MagicMock())


def _parse_content_disposition(value):
    """Parse Content-Disposition with Python's RFC 2231/6266 parser.

    Stands in for Go's mime.ParseMediaType (the parser reported broken in
    issue #8914): both expect a disposition-type followed by parameters, so a
    bare `filename=...` value surfaces here as a bogus disposition type
    instead of inline/attachment.
    """
    msg = email.message.Message()
    msg["Content-Disposition"] = value
    return msg.get_content_disposition(), msg.get_filename()


def _write_png(path):
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(path, format="PNG")


def _get(handler, path):
    return handler(make_mocked_request("GET", path))


@pytest.fixture
def view_handler():
    return _load_view_handler()


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    """Point /view's default output directory at a temp dir.

    The handler resolves type=output through folder_paths.get_directory_by_type;
    monkeypatching it keeps the test self-contained without mutating the
    module's global directory state.
    """
    monkeypatch.setattr(
        folder_paths,
        "get_directory_by_type",
        lambda type_name: str(tmp_path) if type_name == "output" else None,
    )
    return tmp_path


async def test_default_path_serves_inline_with_filename(view_handler, output_dir):
    """Case 1: ordinary image, default FileResponse path -> inline; filename=..."""
    _write_png(output_dir / "example.png")

    response = await _get(view_handler, "/view?filename=example.png")

    assert response.headers["Content-Disposition"] == 'inline; filename="example.png"'


@pytest.mark.parametrize(
    "query",
    [
        "filename=example.png&preview=webp",
        "filename=example.png&channel=rgb",
        "filename=example.png&channel=a",
    ],
    ids=["preview_transcode", "channel_rgb_transcode", "channel_alpha_transcode"],
)
async def test_transcode_paths_serve_inline_with_filename(
    view_handler, output_dir, query
):
    """Case 2: the three PIL-transcoded paths -> inline; filename=..."""
    _write_png(output_dir / "example.png")

    response = await _get(view_handler, "/view?" + query)

    assert response.headers["Content-Disposition"] == 'inline; filename="example.png"'


async def test_dangerous_type_keeps_attachment(view_handler, output_dir):
    """Case 3: SVG stays attachment — the #15149 stored-XSS guard must not regress."""
    (output_dir / "evil.svg").write_text(SVG_WITH_SCRIPT)

    response = await _get(view_handler, "/view?filename=evil.svg")

    assert response.headers["Content-Disposition"] == 'attachment; filename="evil.svg"'
    # The #15149 hardening around the dangerous branch stays intact.
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Vary"] == "Sec-Fetch-Dest"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Content-Type"] == "application/octet-stream"


async def test_every_disposition_survives_rfc6266_parse(view_handler, output_dir):
    """Case 4: simulate Go's mime.ParseMediaType on every /view disposition.

    A strict media-type parser must extract both the disposition-type and the
    filename parameter from every header the endpoint emits — exactly what Go
    downloaders failed on in issue #8914.
    """
    _write_png(output_dir / "example.png")
    (output_dir / "evil.svg").write_text(SVG_WITH_SCRIPT)

    cases = {
        "/view?filename=example.png": ("inline", "example.png"),
        "/view?filename=example.png&preview=webp": ("inline", "example.png"),
        "/view?filename=example.png&channel=rgb": ("inline", "example.png"),
        "/view?filename=example.png&channel=a": ("inline", "example.png"),
        "/view?filename=evil.svg": ("attachment", "evil.svg"),
    }
    for path, (expected_type, expected_name) in cases.items():
        response = await _get(view_handler, path)
        disp_type, filename = _parse_content_disposition(
            response.headers["Content-Disposition"]
        )
        assert disp_type == expected_type, path
        assert filename == expected_name, path


async def test_bare_filename_has_no_disposition_type():
    """Negative control: the RFC parser above rejects the old bare filename= form.

    Without a disposition-type the parser hands back garbage (Go's
    mime.ParseMediaType errors out with "no media type" instead) — this is
    exactly the failure reported in issue #8914.
    """
    disp_type, filename = _parse_content_disposition('filename="example.png"')

    assert disp_type not in ("inline", "attachment")
