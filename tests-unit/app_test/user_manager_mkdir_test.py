"""Tests for the userdata mkdir endpoint in user_manager.py."""

import os
from unittest.mock import patch
from urllib.parse import quote

import pytest
from aiohttp import web

import folder_paths
from app.user_manager import UserManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_directory(tmp_path):
    """Point the user directory at a temporary path for the duration of a test."""
    original = folder_paths.get_user_directory()
    folder_paths.set_user_directory(str(tmp_path))
    yield tmp_path
    folder_paths.set_user_directory(original)


@pytest.fixture
def app(user_directory):
    """Build an aiohttp app serving the UserManager routes."""
    with patch("app.user_manager.args") as mock_args:
        mock_args.multi_user = False
        manager = UserManager()
        application = web.Application()
        routes = web.RouteTableDef()
        manager.add_routes(routes)
        application.add_routes(routes)
        yield application


async def test_mkdir_creates_directory(aiohttp_client, app, user_directory):
    """A new directory is created and its relative path is reported back."""
    client = await aiohttp_client(app)
    resp = await client.post("/userdata/workflows%2Fnew_folder/mkdir")
    assert resp.status == 200
    # Separators stay OS-native here, matching the other userdata write routes.
    assert await resp.json() == os.path.join("workflows", "new_folder")
    assert os.path.isdir(user_directory / "default" / "workflows" / "new_folder")


async def test_mkdir_creates_intermediate_directories(aiohttp_client, app, user_directory):
    """Missing parent directories are created along the way."""
    client = await aiohttp_client(app)
    resp = await client.post("/userdata/" + quote("workflows/a/b/c", safe="") + "/mkdir")
    assert resp.status == 200
    assert await resp.json() == os.path.join("workflows", "a", "b", "c")
    assert os.path.isdir(user_directory / "default" / "workflows" / "a" / "b" / "c")


async def test_mkdir_conflicts_with_existing_directory(aiohttp_client, app):
    """Creating a directory that already exists is a conflict, not a silent success."""
    client = await aiohttp_client(app)
    assert (await client.post("/userdata/workflows%2Fdup/mkdir")).status == 200
    resp = await client.post("/userdata/workflows%2Fdup/mkdir")
    assert resp.status == 409


async def test_mkdir_conflicts_with_existing_file(aiohttp_client, app):
    """A file occupying the name is a conflict rather than an overwrite."""
    client = await aiohttp_client(app)
    assert (await client.post("/userdata/workflows%2Ftaken.json", data="{}")).status == 200
    resp = await client.post("/userdata/workflows%2Ftaken.json/mkdir")
    assert resp.status == 409


async def test_mkdir_reports_conflict_when_losing_a_race(aiohttp_client, app):
    """A concurrent creator winning the race still surfaces as a conflict, not a failure."""
    client = await aiohttp_client(app)
    real_makedirs = os.makedirs

    def racing_makedirs(path, *args, **kwargs):
        # Only the leaf create races; parent creation must still work.
        if os.path.basename(path) == "raced":
            raise FileExistsError()
        return real_makedirs(path, *args, **kwargs)

    with patch("os.makedirs", side_effect=racing_makedirs):
        resp = await client.post("/userdata/workflows%2Fraced/mkdir")
    assert resp.status == 409


async def test_mkdir_rejects_path_traversal(aiohttp_client, app, user_directory):
    """A path escaping the user directory is refused and creates nothing."""
    client = await aiohttp_client(app)
    resp = await client.post("/userdata/" + quote("../escaped", safe="") + "/mkdir")
    assert resp.status == 403
    assert not os.path.exists(user_directory / "escaped")


async def test_mkdir_directory_appears_in_listing(aiohttp_client, app):
    """A created directory shows up in the userdata listing."""
    client = await aiohttp_client(app)
    assert (await client.post("/userdata/workflows%2Flisted/mkdir")).status == 200
    resp = await client.get("/v2/userdata?path=workflows")
    assert resp.status == 200
    entries = await resp.json()
    assert any(e["type"] == "directory" and e["name"] == "listed" for e in entries)
