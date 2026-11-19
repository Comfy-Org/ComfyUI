import pytest
import os
from aiohttp import web
from app.user_manager import UserManager
from unittest.mock import patch

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    um.get_request_user_filepath = lambda req, file, **kwargs: os.path.join(
        tmp_path, file
    ) if file else tmp_path
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_listuserdata_empty_directory(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir")
    assert resp.status == 404


async def test_listuserdata_with_files(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir")
    assert resp.status == 200
    assert await resp.json() == ["file1.txt"]


async def test_listuserdata_recursive(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir" / "subdir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")
    with open(tmp_path / "test_dir" / "subdir" / "file2.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&recurse=true")
    assert resp.status == 200
    assert set(await resp.json()) == {"file1.txt", "subdir/file2.txt"}


async def test_listuserdata_full_info(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&full_info=true")
    assert resp.status == 200
    result = await resp.json()
    assert len(result) == 1
    assert result[0]["path"] == "file1.txt"
    assert "size" in result[0]
    assert isinstance(result[0]["modified"], int)
    assert isinstance(result[0]["created"], int)
    # Verify millisecond magnitude (timestamps after year 2000 in ms are > 946684800000)
    assert result[0]["modified"] > 946684800000
    assert result[0]["created"] > 946684800000


async def test_listuserdata_split_path(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir" / "subdir")
    with open(tmp_path / "test_dir" / "subdir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&recurse=true&split=true")
    assert resp.status == 200
    assert await resp.json() == [["subdir/file1.txt", "subdir", "file1.txt"]]


async def test_listuserdata_invalid_directory(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=")
    assert resp.status == 400


async def test_listuserdata_normalized_separator(aiohttp_client, app, tmp_path):
    os_sep = "\\"
    with patch("os.sep", os_sep):
        with patch("os.path.sep", os_sep):
            os.makedirs(tmp_path / "test_dir" / "subdir")
            with open(tmp_path / "test_dir" / "subdir" / "file1.txt", "w") as f:
                f.write("test content")

            client = await aiohttp_client(app)
            resp = await client.get("/userdata?dir=test_dir&recurse=true")
            assert resp.status == 200
            result = await resp.json()
            assert len(result) == 1
            assert "/" in result[0]  # Ensure forward slash is used
            assert "\\" not in result[0]  # Ensure backslash is not present
            assert result[0] == "subdir/file1.txt"

            # Test with full_info
            resp = await client.get(
                "/userdata?dir=test_dir&recurse=true&full_info=true"
            )
            assert resp.status == 200
            result = await resp.json()
            assert len(result) == 1
            assert "/" in result[0]["path"]  # Ensure forward slash is used
            assert "\\" not in result[0]["path"]  # Ensure backslash is not present
            assert result[0]["path"] == "subdir/file1.txt"


async def test_post_userdata_new_file(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    content = b"test content"
    resp = await client.post("/userdata/test.txt", data=content)

    assert resp.status == 200
    assert await resp.text() == '"test.txt"'

    # Verify file was created with correct content
    with open(tmp_path / "test.txt", "rb") as f:
        assert f.read() == content


async def test_post_userdata_overwrite_existing(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "test.txt", "w") as f:
        f.write("initial content")

    client = await aiohttp_client(app)
    new_content = b"updated content"
    resp = await client.post("/userdata/test.txt", data=new_content)

    assert resp.status == 200
    assert await resp.text() == '"test.txt"'

    # Verify file was overwritten
    with open(tmp_path / "test.txt", "rb") as f:
        assert f.read() == new_content


async def test_post_userdata_no_overwrite(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "test.txt", "w") as f:
        f.write("initial content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/test.txt?overwrite=false", data=b"new content")

    assert resp.status == 409

    # Verify original content unchanged
    with open(tmp_path / "test.txt", "r") as f:
        assert f.read() == "initial content"


async def test_post_userdata_full_info(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    content = b"test content"
    resp = await client.post("/userdata/test.txt?full_info=true", data=content)

    assert resp.status == 200
    result = await resp.json()
    assert result["path"] == "test.txt"
    assert result["size"] == len(content)
    assert "modified" in result


async def test_move_userdata(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "source.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt")

    assert resp.status == 200
    assert await resp.text() == '"dest.txt"'

    # Verify file was moved
    assert not os.path.exists(tmp_path / "source.txt")
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "test content"


async def test_move_userdata_no_overwrite(aiohttp_client, app, tmp_path):
    # Create source and destination files
    with open(tmp_path / "source.txt", "w") as f:
        f.write("source content")
    with open(tmp_path / "dest.txt", "w") as f:
        f.write("destination content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt?overwrite=false")

    assert resp.status == 409

    # Verify files remain unchanged
    with open(tmp_path / "source.txt", "r") as f:
        assert f.read() == "source content"
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "destination content"


async def test_move_userdata_full_info(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "source.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt?full_info=true")

    assert resp.status == 200
    result = await resp.json()
    assert result["path"] == "dest.txt"
    assert result["size"] == len("test content")
    assert "modified" in result

    # Verify file was moved
    assert not os.path.exists(tmp_path / "source.txt")
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "test content"


async def test_listuserdata_v2_empty_root(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v2/userdata")
    assert resp.status == 200
    assert await resp.json() == []


async def test_listuserdata_v2_nonexistent_subdirectory(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/v2/userdata?path=does_not_exist")
    assert resp.status == 404


async def test_listuserdata_v2_default(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir" / "subdir")
    (tmp_path / "test_dir" / "file1.txt").write_text("content")
    (tmp_path / "test_dir" / "subdir" / "file2.txt").write_text("content")

    client = await aiohttp_client(app)
    resp = await client.get("/v2/userdata?path=test_dir")
    assert resp.status == 200
    data = await resp.json()
    file_paths = {item["path"] for item in data if item["type"] == "file"}
    assert file_paths == {"test_dir/file1.txt", "test_dir/subdir/file2.txt"}


async def test_listuserdata_v2_normalized_separators(aiohttp_client, app, tmp_path, monkeypatch):
    # Force backslash as os separator
    monkeypatch.setattr(os, 'sep', '\\')
    monkeypatch.setattr(os.path, 'sep', '\\')
    os.makedirs(tmp_path / "test_dir" / "subdir")
    (tmp_path / "test_dir" / "subdir" / "file1.txt").write_text("x")

    client = await aiohttp_client(app)
    resp = await client.get("/v2/userdata?path=test_dir")
    assert resp.status == 200
    data = await resp.json()
    for item in data:
        assert "/" in item["path"]
        assert "\\" not in item["path"]\

async def test_listuserdata_v2_url_encoded_path(aiohttp_client, app, tmp_path):
    # Create a directory with a space in its name and a file inside
    os.makedirs(tmp_path / "my dir")
    (tmp_path / "my dir" / "file.txt").write_text("content")

    client = await aiohttp_client(app)
    # Use URL-encoded space in path parameter
    resp = await client.get("/v2/userdata?path=my%20dir&recurse=false")
    assert resp.status == 200
    data = await resp.json()
    assert len(data) == 1
    entry = data[0]
    assert entry["name"] == "file.txt"
    # Ensure the path is correctly decoded and uses forward slash
    assert entry["path"] == "my dir/file.txt"
