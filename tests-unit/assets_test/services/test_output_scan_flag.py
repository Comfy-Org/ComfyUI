import argparse
import json
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import folder_paths
import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.assets import lifecycle, mode, scanner, seeder as seeder_module
from app.assets import manager as manager_module
from app.assets.api import routes
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.manager import AssetsEnabled
from app.assets.seeder import _AssetSeeder as AssetSeeder
from app.database.models import Base
from comfy.cli_args import ASSETS_OUTPUT_SCAN_DEFAULT, parse_bool, parser


class _Args:
    enable_assets = True
    enable_asset_hashing = False

    def __init__(self, output_scan: bool) -> None:
        self.enable_assets_output_scan = output_scan


@pytest.fixture
def output_scan_off() -> _Args:
    args = _Args(output_scan=False)
    mode.init(args)
    return args


@pytest.fixture
def fresh_seeder(monkeypatch: pytest.MonkeyPatch) -> Iterator[AssetSeeder]:
    seeder = AssetSeeder()
    monkeypatch.setattr(seeder_module, "asset_seeder", seeder)
    monkeypatch.setattr(manager_module, "asset_seeder", seeder)
    monkeypatch.setattr(routes, "asset_seeder", seeder)
    yield seeder
    _ = seeder.shutdown()


@pytest.fixture
def asset_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    output_dir.mkdir()
    input_dir.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_dir))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return output_dir, input_dir


@pytest.fixture
def threaded_create_session(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[], AbstractContextManager[Session]]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    @contextmanager
    def _create_session() -> Generator[SASession, None, None]:
        with SASession(engine) as session:
            yield session

    monkeypatch.setattr(seeder_module, "create_session", _create_session)
    monkeypatch.setattr(scanner, "create_session", _create_session)
    monkeypatch.setattr("app.assets.services.ingest.create_session", _create_session)
    monkeypatch.setattr("app.database.db.WriteSession", sessionmaker(bind=engine))
    yield _create_session
    engine.dispose()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ASSETS_OUTPUT_SCAN_DEFAULT),
        (["--enable-assets-output-scan"], True),
        (["--enable-assets-output-scan", "true"], True),
        (["--enable-assets-output-scan", "False"], False),
        (["--enable-assets-output-scan=false"], False),
    ],
)
def test_flag_parsing(argv: list[str], expected: bool) -> None:
    assert parser.parse_args(argv).enable_assets_output_scan is expected


def test_flag_rejects_non_boolean_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("sometimes")


def test_scannable_roots_keeps_output_when_enabled() -> None:
    mode.init(_Args(output_scan=True))
    assert mode.scannable_roots(("models", "input", "output")) == (
        "models",
        "input",
        "output",
    )


def test_scannable_roots_drops_output_when_disabled(output_scan_off: _Args) -> None:
    assert mode.scannable_roots(("models", "input", "output")) == ("models", "input")
    assert mode.scannable_roots(("output",)) == ()


def test_startup_scan_excludes_output_but_still_prunes(
    output_scan_off: _Args, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeder_start = MagicMock(return_value=True)
    monkeypatch.setattr(seeder_module.asset_seeder, "start", seeder_start)

    assert lifecycle.start_asset_seeder()

    seeder_start.assert_called_once_with(
        roots=("models", "input"), prune_first=True, compute_hashes=False
    )


def test_lazy_scan_excludes_output(
    output_scan_off: _Args, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeder_start = MagicMock()
    monkeypatch.setattr(manager_module.asset_seeder, "start", seeder_start)

    AssetsEnabled(output_scan_off).ensure_scan_started()

    seeder_start.assert_called_once_with(roots=("models", "input"))


def test_end_of_prompt_output_scan_not_queued(
    output_scan_off: _Args, fresh_seeder: AssetSeeder, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue = MagicMock()
    monkeypatch.setattr(fresh_seeder, "enqueue_scan", enqueue)

    AssetsEnabled(output_scan_off).queue_output_scan()

    enqueue.assert_not_called()


def test_end_of_prompt_output_scan_queued_when_enabled(
    fresh_seeder: AssetSeeder, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _Args(output_scan=True)
    mode.init(args)
    enqueue = MagicMock()
    monkeypatch.setattr(fresh_seeder, "enqueue_scan", enqueue)

    AssetsEnabled(args).queue_output_scan()

    enqueue.assert_called_once()
    assert enqueue.call_args.kwargs["roots"] == ("output",)


@pytest.mark.asyncio
async def test_seed_route_rejects_output_only_request(
    output_scan_off: _Args, fresh_seeder: AssetSeeder, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeder_start = MagicMock()
    monkeypatch.setattr(fresh_seeder, "start", seeder_start)
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    request = make_mocked_request("POST", "/api/assets/seed")
    monkeypatch.setattr(request, "json", _json_body({"roots": ["output"]}))

    response = await routes.seed_assets(request)

    assert isinstance(response, web.Response)
    assert response.status == 400
    assert isinstance(response.body, bytes)
    assert json.loads(response.body)["error"]["code"] == "OUTPUT_SCAN_DISABLED"
    seeder_start.assert_not_called()


@pytest.mark.asyncio
async def test_seed_route_drops_output_from_mixed_request(
    output_scan_off: _Args, fresh_seeder: AssetSeeder, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeder_start = MagicMock(return_value=True)
    monkeypatch.setattr(fresh_seeder, "start", seeder_start)
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    request = make_mocked_request("POST", "/api/assets/seed")
    monkeypatch.setattr(request, "json", _json_body({"roots": ["input", "output"]}))

    response = await routes.seed_assets(request)

    assert isinstance(response, web.Response)
    assert response.status == 202
    assert seeder_start.call_args.kwargs["roots"] == ("input",)


def test_startup_scan_keeps_catalogued_outputs(
    output_scan_off: _Args,
    fresh_seeder: AssetSeeder,
    asset_roots: tuple[Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
) -> None:
    output_dir, input_dir = asset_roots
    kept = output_dir / "kept.png"
    kept.write_bytes(b"catalogued output")
    deleted = output_dir / "deleted.png"
    uncatalogued = output_dir / "uncatalogued.png"
    uncatalogued.write_bytes(b"never registered")
    (input_dir / "source.png").write_bytes(b"input file")
    with threaded_create_session() as session:
        for path in (kept, deleted):
            content = create_content(session, path=str(path))
            create_record(session, content_id=content.id, name=path.name, tags=["output"])
        session.commit()

    assert lifecycle.start_asset_seeder()
    assert fresh_seeder.wait(timeout=5)
    assert fresh_seeder.get_status().errors == []

    with threaded_create_session() as session:
        contents = {c.path: c for c in session.scalars(select(AssetContent))}
    assert contents[str(kept)].is_missing is False
    # Output is not swept, so a vanished output file is not noticed either.
    assert contents[str(deleted)].is_missing is False
    assert str(uncatalogued) not in contents
    assert str(input_dir / "source.png") in contents


def _json_body(body: dict[str, object]) -> Callable[[], object]:
    async def _json() -> object:
        return json.loads(json.dumps(body))

    return _json
