"""Missing user.css must be an empty stylesheet, not a failed request.

Non-cloud frontends link user.css and api/userdata/user.css on every page.
Most installs have neither file, and a stylesheet link cannot send comfy-user.
"""

import os
from unittest.mock import patch

import pytest
from aiohttp import web

import folder_paths
from app.user_manager import UserManager, user_css_file_response

pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    um.get_request_user_filepath = lambda req, file, **kwargs: (
        os.path.join(tmp_path, file) if file else tmp_path
    )
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


def assert_empty_stylesheet(resp, body):
    assert resp.status == 200
    assert resp.content_type == "text/css"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Content-Disposition"] == "inline"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert body == b""


async def test_missing_userdata_user_css_is_empty_stylesheet(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata/user.css")
    assert_empty_stylesheet(resp, await resp.read())


async def test_present_userdata_user_css_is_served(aiohttp_client, app, tmp_path):
    (tmp_path / "user.css").write_text("body { background: red; }")
    client = await aiohttp_client(app)
    resp = await client.get("/userdata/user.css")
    assert resp.status == 200
    assert resp.content_type == "text/css"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Content-Disposition"] == "inline"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert await resp.text() == "body { background: red; }"


async def test_missing_other_userdata_file_is_still_not_found(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata/workflows/missing.json")
    assert resp.status == 404


async def test_missing_user_css_without_a_user_is_empty_stylesheet(
    aiohttp_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path))
    with patch("app.user_manager.args") as mock_args:
        mock_args.multi_user = True
        um = UserManager()

    app = web.Application()
    routes = web.RouteTableDef()
    um.add_routes(routes)
    app.add_routes(routes)
    client = await aiohttp_client(app)

    with patch("app.user_manager.args") as mock_args:
        mock_args.multi_user = True
        resp = await client.get("/userdata/user.css")
    assert_empty_stylesheet(resp, await resp.read())

    with patch("app.user_manager.args") as mock_args:
        mock_args.multi_user = True
        other = await client.get("/userdata/missing.json", headers={"comfy-user": "nobody"})
    assert other.status == 500


async def test_root_user_css_missing_and_present(aiohttp_client, tmp_path):
    async def get_user_css(request):
        return user_css_file_response(str(tmp_path / "user.css"))

    app = web.Application()
    app.router.add_get("/user.css", get_user_css)
    app.router.add_static("/", tmp_path)
    client = await aiohttp_client(app)

    missing = await client.get("/user.css")
    assert_empty_stylesheet(missing, await missing.read())

    (tmp_path / "user.css").write_text("body { color: blue; }")
    present = await client.get("/user.css")
    assert present.status == 200
    assert present.content_type == "text/css"
    assert present.headers["Cache-Control"] == "no-cache"
    assert await present.text() == "body { color: blue; }"

    (tmp_path / "user.css").unlink()
    outside = tmp_path / "outside.css"
    outside.write_text("root:x:0:0")
    (tmp_path / "user.css").symlink_to(outside)
    linked = await client.get("/user.css")
    assert_empty_stylesheet(linked, await linked.read())

    (tmp_path / "user.css").unlink()
    (tmp_path / "user.css").mkdir()
    directory = await client.get("/user.css")
    assert_empty_stylesheet(directory, await directory.read())
