import asyncio
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import torch
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import create_engine, exc as sa_exc
from sqlalchemy.orm import Session as SASession

import folder_paths
from comfy.cli_args import args

# comfy.model_management picks its device at import time; a build with no CUDA driver raises there.
_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import server
finally:
    args.cpu = _original_cpu
from app.assets import manager as manager_module
from app.assets.database.models import Base
from app.assets.manager import AssetRegistrationError, AssetsEnabled
from app.assets.services.ingest import register_file_in_place


class _ArgsStub:
    enable_assets = True
    enable_asset_hashing = False


def _locked_error() -> sa_exc.OperationalError:
    return sa_exc.OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))


@pytest.fixture
def input_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "input"
    path.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(path))
    return path


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A real file database, as in production, with a short busy timeout so a held lock fails fast.
    path = tmp_path / "assets.sqlite3"
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"timeout": 0.1, "check_same_thread": False}
    )
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(engine)

    @contextmanager
    def _create_session():
        with SASession(engine) as session:
            yield session

    monkeypatch.setattr("app.assets.services.ingest.create_session", _create_session)
    monkeypatch.setattr(manager_module, "_LOCKED_RETRY_PAUSE_SECONDS", 0.1)
    yield path
    engine.dispose()


@contextmanager
def _write_lock(db_path: Path):
    holder = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    released = threading.Event()

    def release():
        if not released.is_set():
            released.set()
            holder.execute("ROLLBACK")

    try:
        yield release
    finally:
        release()
        holder.close()


def _asset_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]


@pytest_asyncio.fixture
async def client(input_dir: Path, db_path: Path):
    prompt_server = server.PromptServer(None, AssetsEnabled(_ArgsStub()))

    async def ping(request):
        return web.Response(text="pong")

    prompt_server.app.router.add_get("/test-ping", ping)
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as test_client:
        yield test_client


async def _upload(client: TestClient, data: bytes, name: str = "photo.png"):
    body = FormData()
    body.add_field("image", data, filename=name, content_type="image/png")
    body.add_field("type", "input")
    return await client.post("/upload/image", data=body)


@pytest.mark.asyncio
async def test_upload_registers_once_a_brief_lock_clears(client, db_path, input_dir):
    with _write_lock(db_path) as release:
        threading.Timer(0.15, release).start()
        resp = await _upload(client, b"brief lock")

    body = await resp.json()
    assert resp.status == 200, body
    assert body["asset"]["id"]
    assert _asset_count(db_path) == 1


@pytest.mark.asyncio
async def test_upload_returns_503_while_locked_and_a_retry_reuses_the_saved_file(
    client, db_path, input_dir
):
    with _write_lock(db_path):
        resp = await _upload(client, b"held lock")
        body = await resp.json()
        assert resp.status == 503, body
        assert "busy" in body["error"]
        assert "asset" not in body
    assert (input_dir / "photo.png").read_bytes() == b"held lock"
    assert _asset_count(db_path) == 0

    retry = await _upload(client, b"held lock")

    retry_body = await retry.json()
    assert retry.status == 200, retry_body
    assert retry_body["name"] == "photo.png"
    assert retry_body["asset"]["id"]
    assert sorted(p.name for p in input_dir.iterdir()) == ["photo.png"]
    assert _asset_count(db_path) == 1


@pytest.mark.asyncio
async def test_upload_returns_500_when_registration_fails_for_another_reason(
    client, input_dir, monkeypatch
):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise OSError("disk I/O error")

    monkeypatch.setattr(manager_module, "register_file_in_place", fail)

    resp = await _upload(client, b"broken db")

    body = await resp.json()
    assert resp.status == 500, body
    assert body["error"] == "Asset registration failed"
    assert len(calls) == 1, "a non-lock failure is not retried"
    assert (input_dir / "photo.png").exists()


@pytest.mark.asyncio
async def test_locked_registration_does_not_block_other_requests(client, db_path):
    with _write_lock(db_path):
        upload = asyncio.ensure_future(_upload(client, b"slow upload"))
        await asyncio.sleep(0.05)
        ping = await client.get("/test-ping")
        assert ping.status == 200
        assert not upload.done(), "the server answered while the upload was still retrying"
        resp = await upload
        assert resp.status == 503


def _register(manager: AssetsEnabled, path: Path):
    return manager.register_upload(
        str(path), name=path.name, upload_type="input", subfolder="", content_written=True
    )


def test_register_upload_retries_a_locked_database_then_succeeds(
    input_dir, db_path, monkeypatch
):
    path = input_dir / "retry.png"
    path.write_bytes(b"retry")
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise _locked_error()
        return register_file_in_place(**kwargs)

    monkeypatch.setattr(manager_module, "register_file_in_place", flaky)

    view = _register(AssetsEnabled(_ArgsStub()), path)

    assert view is not None and view.asset.id
    assert len(attempts) == 3


def test_register_upload_gives_up_after_the_bound(input_dir, db_path, monkeypatch):
    path = input_dir / "stuck.png"
    path.write_bytes(b"stuck")
    attempts = []

    def locked(**kwargs):
        attempts.append(kwargs)
        raise _locked_error()

    monkeypatch.setattr(manager_module, "register_file_in_place", locked)

    with pytest.raises(AssetRegistrationError) as raised:
        _register(AssetsEnabled(_ArgsStub()), path)

    assert raised.value.locked is True
    assert len(attempts) == 3


def test_register_upload_does_not_retry_other_errors(input_dir, db_path, monkeypatch):
    path = input_dir / "gone.png"
    attempts = []

    def missing(**kwargs):
        attempts.append(kwargs)
        raise FileNotFoundError(kwargs["abs_path"])

    monkeypatch.setattr(manager_module, "register_file_in_place", missing)

    with pytest.raises(AssetRegistrationError) as raised:
        _register(AssetsEnabled(_ArgsStub()), path)

    assert raised.value.locked is False
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_concurrent_identical_uploads_each_register(client, db_path, input_dir):
    # Registration runs in worker threads, so a burst of the same image registers in parallel.
    responses = await asyncio.gather(*(_upload(client, b"same bytes") for _ in range(8)))

    bodies = [await r.json() for r in responses]
    assert [r.status for r in responses] == [200] * 8, bodies
    assert all(b["asset"]["id"] for b in bodies)
    assert sorted(p.name for p in input_dir.iterdir()) == ["photo.png"]
    assert _asset_count(db_path) == 8
