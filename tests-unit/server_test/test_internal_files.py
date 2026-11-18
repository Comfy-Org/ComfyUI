import os

import pytest
from aiohttp import web

from api_server.routes.internal.internal_routes import InternalRoutes
import folder_paths


@pytest.mark.asyncio
async def test_file_listing_can_filter_images(aiohttp_client, tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    for name, mtime in [
        ("older.jpg", 100),
        ("newer.PNG", 200),
        ("Thumbs.db", 300),
        (".hidden.png", 400),
    ]:
        path = tmp_path / name
        path.touch()
        os.utime(path, (mtime, mtime))
    (tmp_path / "folder.png").mkdir()

    app = web.Application()
    app.add_subapp("/internal", InternalRoutes(None).get_app())
    client = await aiohttp_client(app)

    response = await client.get("/internal/files/output?content_type=image")
    assert response.status == 200
    assert await response.json() == ["newer.PNG [output]", "older.jpg [output]"]

    response = await client.get("/internal/files/output")
    assert response.status == 200
    assert await response.json() == [
        "Thumbs.db [output]",
        "newer.PNG [output]",
        "older.jpg [output]",
    ]
