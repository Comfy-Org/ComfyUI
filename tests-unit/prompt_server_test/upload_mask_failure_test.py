import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import folder_paths
import server


def _route(prompt_server, path):
    return next(route.handler for route in prompt_server.routes if route.path == path)


def _post(original_ref=None):
    stream = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 64)).save(stream, format="PNG")
    stream.seek(0)
    post = {"image": SimpleNamespace(filename="mask.png", file=stream)}
    if original_ref is not None:
        post["original_ref"] = json.dumps(original_ref)
    return post


@pytest.fixture
def prompt_server(tmp_path, monkeypatch):
    for name in ("input", "output", "temp", "user"):
        (tmp_path / name).mkdir()
    for name in ("input", "output", "temp", "user"):
        monkeypatch.setattr(folder_paths, f"{name}_directory", str(tmp_path / name))
    monkeypatch.setattr(server.FrontendManager, "init_frontend", lambda _: "")
    assets = MagicMock(enabled=False)
    assets.register_upload.return_value = None
    return server.PromptServer(None, assets)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "status"),
    [
        ({"filename": "/invalid.png"}, 400),
        ({"filename": "original.png", "subfolder": ".."}, 403),
        ({"filename": "missing.png"}, 404),
    ],
)
async def test_rejected_mask_does_not_report_success_or_register_an_asset(
    prompt_server, reference, status
):
    request = SimpleNamespace(post=AsyncMock(return_value=_post(reference)))

    response = await _route(prompt_server, "/upload/mask")(request)

    assert response.status == status
    assert not Path(folder_paths.get_input_directory(), "mask.png").exists()
    prompt_server.asset_manager.register_upload.assert_not_called()


@pytest.mark.asyncio
async def test_successful_mask_saves_alpha_and_registers_once(prompt_server):
    original = Path(folder_paths.get_output_directory(), "original.png")
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(original)
    request = SimpleNamespace(
        post=AsyncMock(return_value=_post({"filename": "original.png"}))
    )

    response = await _route(prompt_server, "/upload/mask")(request)

    assert response.status == 200
    with Image.open(Path(folder_paths.get_input_directory(), "mask.png")) as saved:
        assert saved.getpixel((0, 0)) == (255, 0, 0, 64)
    prompt_server.asset_manager.register_upload.assert_called_once()


@pytest.mark.asyncio
async def test_plain_upload_still_saves_and_registers_once(prompt_server):
    request = SimpleNamespace(post=AsyncMock(return_value=_post()))

    response = await _route(prompt_server, "/upload/image")(request)

    assert response.status == 200
    assert Path(folder_paths.get_input_directory(), "mask.png").is_file()
    prompt_server.asset_manager.register_upload.assert_called_once()
