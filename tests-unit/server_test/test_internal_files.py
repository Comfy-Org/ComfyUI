"""Tests for GET /internal/files/{directory_type}, used by Load Image (from Outputs)."""

import os

import pytest

import folder_paths
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio


@pytest.fixture
def output_dir(tmp_path):
    original = folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(tmp_path))
    yield tmp_path
    folder_paths.set_output_directory(original)


async def test_files_lists_only_images_newest_first(aiohttp_client, output_dir):
    names = ["old.png", "new.webp", "Thumbs.db", ".DS_Store", "clip.mp4", "workflow.json", "notes.txt"]
    for i, name in enumerate(names):
        path = output_dir / name
        path.write_bytes(b"x")
        os.utime(path, (1000 + i, 1000 + i))
    os.utime(output_dir / "new.webp", (5000, 5000))
    (output_dir / "subfolder.png").mkdir()

    client = await aiohttp_client(InternalRoutes(None).get_app())
    resp = await client.get("/files/output")

    assert resp.status == 200
    assert await resp.json() == ["new.webp [output]", "old.png [output]"]
