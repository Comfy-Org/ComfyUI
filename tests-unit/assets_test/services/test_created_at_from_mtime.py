import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from sqlalchemy.orm import Session

import folder_paths
from app.assets.api import routes
from app.assets.database.models import Asset
from app.assets.helpers import get_utc_now
from app.assets.scanner import SeedAssetSpec, insert_asset_specs
from app.assets.services.ingest import register_executed_output, register_file_in_place

_EPOCH = datetime(1970, 1, 1)


@pytest.fixture
def output_dir(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(temp_dir))
    return temp_dir


@pytest.fixture
def one_database(db_engine, mock_create_session, monkeypatch: pytest.MonkeyPatch):
    @contextmanager
    def _create_session():
        with Session(db_engine) as session:
            yield session

    monkeypatch.setattr("app.assets.scanner.create_session", _create_session)
    monkeypatch.setattr("app.assets.scanner.create_write_session", _create_session)
    monkeypatch.setattr(routes, "create_session", lambda: Session(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    return db_engine


def _write(directory: Path, name: str, when: datetime | None = None, extra_ns: int = 0) -> Path:
    path = directory / name
    path.write_bytes(name.encode())
    if when is not None:
        mtime_ns = (when - _EPOCH) // timedelta(microseconds=1) * 1000 + extra_ns
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _spec(path: Path) -> SeedAssetSpec:
    stat_result = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "info_name": path.name,
        "tags": ["output"],
        "fname": path.name,
        "metadata": None,
        "mime_type": "image/png",
        "job_id": None,
    }


def _scan(*paths: Path) -> None:
    created, error = insert_asset_specs([_spec(path) for path in paths], set())
    assert (created, error) == (len(paths), None)


async def _list(params: dict[str, str]) -> dict:
    response = await routes.list_assets_route(
        make_mocked_request("GET", f"/api/assets?{urlencode(params)}")
    )
    assert isinstance(response, web.Response)
    return json.loads(response.body)


async def _listed_names(**params: str) -> list[str]:
    body = await _list(params)
    return [asset["name"] for asset in body["assets"]]


def _created_at(engine, name: str) -> datetime:
    with Session(engine) as session:
        return session.query(Asset).filter(Asset.name == name).one().created_at


@pytest.mark.asyncio
async def test_scanned_files_list_newest_mtime_first(one_database, output_dir: Path):
    old = _write(output_dir, "old.png", datetime(2024, 1, 1))
    middle = _write(output_dir, "middle.png", datetime(2025, 1, 1, 0, 0, 0, 123456), extra_ns=789)
    recent = _write(output_dir, "recent.png", datetime(2026, 1, 1))

    # Newest first, so stamping the scan time would list them the other way round.
    _scan(recent, middle, old)

    assert await _listed_names() == ["recent.png", "middle.png", "old.png"]
    assert _created_at(one_database, "middle.png") == datetime(2025, 1, 1, 0, 0, 0, 123456)  # floored


@pytest.mark.asyncio
async def test_new_generation_stays_on_top_while_the_first_scan_is_still_inserting(
    one_database, output_dir: Path
):
    old = [_write(output_dir, f"old-{i}.png", datetime(2025, 1, 1 + i)) for i in range(3)]
    generated = _write(output_dir, "generated.png")
    assert register_executed_output(str(generated), job_id="job-1") is not None

    # The scan reaches the older files only after the prompt registered its output.
    _scan(*old)

    assert (await _listed_names())[0] == "generated.png"


@pytest.mark.asyncio
async def test_upload_after_a_scan_lists_first(one_database, output_dir: Path):
    _scan(_write(output_dir, "old.png", datetime(2025, 1, 1)))
    uploaded = _write(output_dir, "uploaded.png", datetime(2020, 1, 1))

    register_file_in_place(str(uploaded), "uploaded.png", ["output"])

    assert (await _listed_names())[0] == "uploaded.png"


@pytest.mark.asyncio
async def test_future_mtime_is_capped_at_now(
    one_database, output_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # Pinned a second back, so a coarse clock can't tie the scan with the generation below.
    scan_time = get_utc_now() - timedelta(seconds=1)
    with monkeypatch.context() as patched:
        patched.setattr("app.assets.helpers.get_utc_now", lambda: scan_time)
        _scan(_write(output_dir, "future.png", datetime(2200, 1, 1)))
    generated = _write(output_dir, "generated.png")
    assert register_executed_output(str(generated), job_id="job-1") is not None

    assert _created_at(one_database, "future.png") == scan_time
    assert (await _listed_names())[0] == "generated.png"


@pytest.mark.asyncio
async def test_future_mtime_file_dates_from_its_arrival_on_a_later_scan(
    one_database, output_dir: Path
):
    future = _write(output_dir, "future.png", datetime(2200, 1, 1))
    time.sleep(0.05)  # past a coarse clock tick (Windows)
    generated = _write(output_dir, "generated.png")
    assert register_executed_output(str(generated), job_id="job-1") is not None

    # The rescan after the prompt reaches the file only now.
    _scan(future)

    assert (await _listed_names())[0] == "generated.png"
    arrived = _EPOCH + timedelta(microseconds=future.stat().st_ctime_ns // 1000)
    assert _created_at(one_database, "future.png") == arrived


@pytest.mark.asyncio
async def test_cursor_walks_files_sharing_an_mtime_once_each_in_order(
    one_database, output_dir: Path
):
    shared = datetime(2025, 6, 1, 12, 0, 0, 654321)
    paths = [_write(output_dir, f"copy-{i}.png", shared) for i in range(5)]
    paths.append(_write(output_dir, "older.png", datetime(2025, 1, 1)))
    _scan(*paths)
    expected = await _listed_names(limit="50")

    walked: list[str] = []
    params = {"limit": "1"}
    for _ in range(len(paths)):
        body = await _list(params)
        walked += [asset["name"] for asset in body["assets"]]
        if not body["has_more"]:
            break
        params = {"limit": "1", "after": body["next_cursor"]}
    else:
        pytest.fail(f"cursor walk did not finish in {len(paths)} pages: {walked}")

    assert walked == expected
    assert sorted(walked) == sorted(path.name for path in paths)
    assert walked[-1] == "older.png"
