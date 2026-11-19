"""CI unit tests for FIX #2 of GHSA-779p-m5rp-r4h4.

Path traversal / hardening in app/model_manager.py get_model_preview
(route /experiment/models/preview/{folder}/{path_index}/{filename:.*}).

Reference: https://github.com/Comfy-Org/ComfyUI/security/advisories/GHSA-779p-m5rp-r4h4
"""
import pytest
import yarl
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

@pytest.fixture
def model_manager():
    return ModelFileManager()

@pytest.fixture
def app(model_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_legit_preview_returns_200(aiohttp_client, app, tmp_path):
    """Sanity: a real preview PNG inside the model folder is served as webp 200."""
    img = Image.new('RGB', (16, 16), color=(255, 0, 128))
    img.save(tmp_path / "test_model.png", format='PNG')

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/test_model.png')

        assert response.status == 200
        assert response.content_type == 'image/webp'

        img_bytes = BytesIO(await response.read())
        served = Image.open(img_bytes)
        assert served.format
        assert served.format.lower() == 'webp'
        served.close()


async def test_non_integer_path_index_returns_400(aiohttp_client, app, tmp_path):
    """A non-integer path_index segment must be rejected with 400."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/abc/test_model.png')

        assert response.status == 400


async def test_out_of_range_path_index_returns_404(aiohttp_client, app, tmp_path):
    """A path_index beyond the configured folder list must return 404."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/99/test_model.png')

        assert response.status == 404


async def test_empty_filename_returns_400(aiohttp_client, app, tmp_path):
    """The "{filename:.*}" capture also matches the empty string (trailing
    slash). It would resolve to the folder itself and must be rejected with 400."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/')

        assert response.status == 400


async def test_path_traversal_in_filename_returns_403(aiohttp_client, app, tmp_path):
    """Path traversal in {filename} must be rejected with 403 and must NOT read
    a file outside the configured model directory.

    GOTCHA: aiohttp/yarl collapses literal ``../`` dot-segments out of the URL
    path before it reaches the handler, which would make this test vacuously
    pass (the request would hit a different/non-existent route). We percent-encode
    the dots and slashes (``%2e%2e%2f``) and send the URL with
    ``yarl.URL(..., encoded=True)`` so the bytes survive client-side normalization
    untouched; aiohttp's router then percent-decodes them into ``match_info``,
    delivering the literal ``../`` traversal to the handler's ``{filename:.*}``
    capture.

    Without the fix the handler computes
    ``os.path.normpath(os.path.join(folder, "../../../../etc/hosts"))``, which
    escapes ``tmp_path`` and would be passed straight to get_model_previews ->
    Image.open, serving bytes from outside the model dir (200/served bytes). The
    is_within_directory() containment check is the load-bearing fix that turns
    that escape into a 403.
    """
    # Sanity-anchor: a legit preview exists inside tmp_path, so a 200 path is
    # genuinely reachable — proving the 403 below is the containment check
    # firing, not an unrelated 404.
    img = Image.new('RGB', (16, 16), color=(255, 0, 128))
    img.save(tmp_path / "test_model.png", format='PNG')

    # Percent-encoded "../../../../etc/hosts" so yarl does not collapse the
    # dot-segments before the request leaves the client.
    encoded_traversal = '%2e%2e%2f' * 4 + 'etc%2fhosts'
    raw_path = '/experiment/models/preview/test_folder/0/' + encoded_traversal
    url = yarl.URL(raw_path, encoded=True)

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get(url)

        # Confirm the traversal actually reached the handler intact: a 200 here
        # would mean either normalization stripped the ``../`` (vacuous pass) or
        # the containment check failed open and served outside-dir bytes.
        assert response.status == 403, (
            f"expected 403 from is_within_directory() containment check, "
            f"got {response.status}; traversal may have been normalized away "
            f"or the fix failed open"
        )
        body = await response.read()
        assert body == b"", "403 response must not carry any file bytes"


async def test_symlink_companion_preview_returns_403(aiohttp_client, app, tmp_path):
    """A companion preview file is selected by a glob inside get_model_previews
    and then opened. If that companion is a symlink whose path is in-dir but
    whose target escapes the model folder, it must be rejected with 403 — not
    served. The requested path itself stays in-dir (so the first containment
    check passes); the load-bearing fix is the SECOND is_within_directory check
    on the file actually opened.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    # A real image OUTSIDE the model dir — valid, so without the fix Image.open
    # would succeed and its bytes would be served (200).
    secret = secret_dir / "secret.png"
    Image.new('RGB', (8, 8), color=(0, 0, 0)).save(secret, format='PNG')
    # Companion preview, in-dir by name but a symlink escaping the model dir.
    # (No real model file is needed — get_model_previews globs companions by
    # basename, and omitting a .safetensors avoids the metadata-header read.)
    companion = model_dir / "model.preview.png"
    try:
        companion.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform/filesystem")

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/model.safetensors')

        assert response.status == 403, (
            f"expected 403 — the globbed companion preview is a symlink resolving "
            f"outside the model dir and must not be served; got {response.status}"
        )
        assert await response.read() == b""


async def test_null_byte_in_filename_no_500(aiohttp_client, app, tmp_path):
    """A NUL byte in the filename must yield a clean client rejection, not a 500
    from an uncaught ValueError in is_within_directory's realpath() call."""
    raw_path = '/experiment/models/preview/test_folder/0/' + 'a%00b'
    url = yarl.URL(raw_path, encoded=True)

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get(url)

        assert response.status != 500, (
            f"NUL byte produced a 500 (uncaught ValueError); expected a clean "
            f"4xx rejection, got {response.status}"
        )
        assert 400 <= response.status < 500
