import asyncio
import io
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import torch
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
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


async def _serve(manager):
    prompt_server = server.PromptServer(None, manager)

    async def ping(request):
        return web.Response(text="pong")

    prompt_server.app.router.add_get("/test-ping", ping)
    prompt_server.app.add_routes(prompt_server.routes)
    return TestClient(TestServer(prompt_server.app))


@pytest_asyncio.fixture
async def client(input_dir: Path, db_path: Path):
    async with await _serve(AssetsEnabled(_ArgsStub())) as test_client:
        yield test_client


async def _upload(client: TestClient, data: bytes, name: str = "photo.png", **fields: str):
    body = FormData()
    body.add_field("image", data, filename=name, content_type="image/png")
    body.add_field("type", "input")
    for key, value in fields.items():
        body.add_field(key, value)
    return await client.post("/upload/image", data=body)


def _wrap_registration(monkeypatch: pytest.MonkeyPatch, around):
    """Route the manager's registration calls through ``around(attempt_number, call)``."""
    attempts = []

    def wrapped(**kwargs):
        attempts.append(kwargs)
        return around(len(attempts), lambda: register_file_in_place(**kwargs))

    monkeypatch.setattr(manager_module, "register_file_in_place", wrapped)
    return attempts


def _png(color: tuple[int, int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGBA", (4, 4), color).save(out, format="PNG")
    return out.getvalue()


@pytest.mark.asyncio
async def test_upload_registers_once_a_brief_lock_clears(client, db_path, monkeypatch):
    with _write_lock(db_path) as release:

        def release_after_first_failure(attempt, register):
            try:
                return register()
            except Exception:
                release()
                raise

        attempts = _wrap_registration(monkeypatch, release_after_first_failure)
        resp = await _upload(client, b"brief lock")

    body = await resp.json()
    assert resp.status == 200, body
    assert body["asset"]["id"]
    assert len(attempts) == 2, "the first attempt hit the lock and the retry registered"
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
        raise OSError("registration failed")

    monkeypatch.setattr(manager_module, "register_file_in_place", fail)

    resp = await _upload(client, b"broken db")

    body = await resp.json()
    assert resp.status == 500, body
    assert body["error"] == "Asset registration failed"
    assert len(calls) == 1, "a non-lock failure is not retried"
    assert (input_dir / "photo.png").exists()


@pytest.mark.asyncio
async def test_locked_registration_does_not_block_other_requests(client, db_path, monkeypatch):
    entered, exited = threading.Event(), threading.Event()

    def observe(attempt, register):
        entered.set()
        try:
            return register()
        finally:
            exited.set()

    _wrap_registration(monkeypatch, observe)
    with _write_lock(db_path):
        upload = asyncio.ensure_future(_upload(client, b"slow upload"))
        assert await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5)
        ping = await client.get("/test-ping")
        assert ping.status == 200
        assert not exited.is_set(), "the server answered while registration was still waiting"
        resp = await upload
        assert resp.status == 503


@pytest.mark.asyncio
async def test_an_overwrite_waits_for_the_upload_still_registering(
    client, db_path, input_dir, monkeypatch
):
    first_registering = threading.Event()

    def slow_first(attempt, register):
        if attempt == 1:
            first_registering.set()
            time.sleep(0.3)
        return register()

    _wrap_registration(monkeypatch, slow_first)
    first = asyncio.ensure_future(_upload(client, b"first bytes"))
    assert await asyncio.get_running_loop().run_in_executor(None, first_registering.wait, 5)
    second = await _upload(client, b"second bytes", overwrite="true")
    first = await first

    first_body, second_body = await first.json(), await second.json()
    assert first.status == second.status == 200, (first_body, second_body)
    assert first_body["asset"]["asset_hash"] != second_body["asset"]["asset_hash"]
    assert (input_dir / "photo.png").read_bytes() == b"second bytes"
    with sqlite3.connect(db_path) as conn:
        live = conn.execute(
            "SELECT hash FROM asset_contents WHERE path = ? AND is_missing = 0",
            (str(input_dir / "photo.png"),),
        ).fetchall()
    assert live == [(second_body["asset"]["asset_hash"],)]


@pytest.mark.asyncio
async def test_mask_upload_registers_the_composited_file(client, db_path, input_dir):
    (input_dir / "original.png").write_bytes(_png((255, 0, 0, 255)))
    body = FormData()
    body.add_field("image", _png((0, 0, 0, 128)), filename="mask.png", content_type="image/png")
    body.add_field("type", "input")
    body.add_field("original_ref", json.dumps({"filename": "original.png", "type": "input"}))

    resp = await client.post("/upload/mask", data=body)

    data = await resp.json()
    assert resp.status == 200, data
    assert data["asset"]["id"]
    with Image.open(input_dir / "mask.png") as saved:
        assert saved.getpixel((0, 0)) == (255, 0, 0, 128)


class _AssetsDisabled:
    enabled = False

    def register_routes(self, app, user_manager):
        pass

    def set_event_sink(self, sink):
        pass

    def register_upload(self, abs_path, name, upload_type, subfolder, *, content_written):
        return None


@pytest.mark.asyncio
async def test_upload_with_assets_disabled_returns_200_without_an_asset(input_dir):
    async with await _serve(_AssetsDisabled()) as test_client:
        resp = await _upload(test_client, b"no assets")
        body = await resp.json()

    assert resp.status == 200, body
    assert body == {"name": "photo.png", "subfolder": "", "type": "input"}
    assert (input_dir / "photo.png").read_bytes() == b"no assets"


def _register(manager: AssetsEnabled, path: Path):
    return manager.register_upload(
        str(path), name=path.name, upload_type="input", subfolder="", content_written=True
    )


def _fail_with(monkeypatch: pytest.MonkeyPatch, *errors: Exception):
    attempts = []

    def failing(**kwargs):
        attempts.append(kwargs)
        if len(attempts) <= len(errors):
            raise errors[len(attempts) - 1]
        return register_file_in_place(**kwargs)

    monkeypatch.setattr(manager_module, "register_file_in_place", failing)
    return attempts


def test_register_upload_retries_a_locked_database_then_succeeds(
    input_dir, db_path, monkeypatch
):
    path = input_dir / "retry.png"
    path.write_bytes(b"retry")
    attempts = _fail_with(monkeypatch, _locked_error(), _locked_error())

    view = _register(AssetsEnabled(_ArgsStub()), path)

    assert view is not None and view.asset.id
    assert len(attempts) == 3


def test_register_upload_gives_up_after_the_bound(input_dir, db_path, monkeypatch):
    path = input_dir / "stuck.png"
    path.write_bytes(b"stuck")
    attempts = _fail_with(monkeypatch, *(_locked_error() for _ in range(4)))

    with pytest.raises(AssetRegistrationError) as raised:
        _register(AssetsEnabled(_ArgsStub()), path)

    assert raised.value.locked is True
    assert len(attempts) == 3


def test_register_upload_does_not_retry_other_errors(input_dir, db_path, monkeypatch):
    path = input_dir / "gone.png"
    attempts = _fail_with(monkeypatch, FileNotFoundError(str(path)))

    with pytest.raises(AssetRegistrationError) as raised:
        _register(AssetsEnabled(_ArgsStub()), path)

    assert raised.value.locked is False
    assert len(attempts) == 1


def test_register_upload_reports_the_last_failure(input_dir, db_path, monkeypatch):
    path = input_dir / "mixed.png"
    path.write_bytes(b"mixed")
    attempts = _fail_with(monkeypatch, _locked_error(), OSError("registration failed"))

    with pytest.raises(AssetRegistrationError) as raised:
        _register(AssetsEnabled(_ArgsStub()), path)

    assert raised.value.locked is False
    assert len(attempts) == 2
