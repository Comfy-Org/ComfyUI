"""A file whose name is not valid UTF-8 is left out of the catalog without taking the
rest of its insert batch with it. Run through the seeder's real fast phase on an
in-memory catalog."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent, Base
from app.assets.scanner import insert_asset_specs
from app.assets.scanner_admission import _WATCH_LIST

N_FILES = 1000
BAD_NAME = b"bad_\xff\xfe.png"
EVENT = "[assets-event] scanner.name_not_utf8"


@pytest.fixture
def roots(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dirs = {name: temp_dir / name for name in ("input", "output")}
    for path in dirs.values():
        path.mkdir()
    try:
        _write_bad(dirs["input"]).unlink()
    except (OSError, UnicodeError):
        pytest.skip("filesystem does not accept names that are not valid UTF-8")
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(dirs["output"]))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(dirs["input"]))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return dirs


@pytest.fixture
def db_engine():
    """One in-memory database shared with the seeder's scan thread."""
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()


def _scan(roots: tuple[str, ...], phase=seeder_module.ScanPhase.FAST, seeder=None) -> None:
    seeder = seeder or seeder_module._AssetSeeder()
    assert seeder.start(roots=roots, phase=phase)
    assert seeder.wait(timeout=60)


def _write_files(directory: Path, prefix: str, count: int) -> list[Path]:
    paths = [directory / f"{prefix}_{i:04d}.png" for i in range(count)]
    for path in paths:
        path.write_bytes(b"png-bytes")
    return paths


def _write_bad(directory: Path) -> Path:
    path = Path(os.fsdecode(os.path.join(os.fsencode(directory), BAD_NAME)))
    path.write_bytes(b"png-bytes")
    return path


def _live_paths(session) -> set[str]:
    session.expire_all()
    return set(session.scalars(sa.select(AssetContent.path).where(AssetContent.is_missing.is_(False))))


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def test_full_scan_catalogs_every_other_file_and_warns_once(roots, session, caplog):
    good = _write_files(roots["input"], "in", N_FILES // 2) + _write_files(
        roots["output"], "out", N_FILES // 2
    )
    _write_bad(roots["input"])
    # os.walk lists a folder's files before its subfolders, so this bad file comes last,
    # in the third insert batch, never in the input one's.
    (roots["output"] / "z").mkdir()
    _write_bad(roots["output"] / "z")

    with caplog.at_level(logging.INFO):
        _scan(("models", "input", "output"))

    assert _live_paths(session) == {str(p) for p in good}
    warnings = [m for m in _messages(caplog, logging.WARNING) if "not valid UTF-8" in m]
    assert len(warnings) == 1
    assert "Skipped 2 file(s)" in warnings[0] and repr(BAD_NAME)[2:-1] in warnings[0]
    assert [m for m in _messages(caplog, logging.INFO) if m.startswith(EVENT)] == [f"{EVENT} count=2"]
    assert not [m for m in _messages(caplog, logging.ERROR) if "Batch insert" in m]


def test_full_scan_reports_names_rejected_during_enrichment(roots, monkeypatch, caplog):
    """A file still being written at fast-phase admission settles off the watch list in
    the enrich phase, and is rejected there."""
    seeder = seeder_module._AssetSeeder()

    def enrich_rejecting_a_name(_roots):
        seeder._scan_state.names_not_utf8.append(b"/late/" + BAD_NAME)
        return False, 0

    monkeypatch.setattr(seeder, "_run_enrich_phase", enrich_rejecting_a_name)
    with caplog.at_level(logging.INFO):
        _scan(("input",), phase=seeder_module.ScanPhase.FULL, seeder=seeder)

    assert [m for m in _messages(caplog, logging.INFO) if m.startswith(EVENT)] == [f"{EVENT} count=1"]


def test_a_path_found_twice_is_counted_once(roots, monkeypatch, caplog):
    """One models folder registered under two categories lists its files twice."""
    _write_bad(roots["input"])
    real_collect = scanner.collect_paths_for_roots
    monkeypatch.setattr(
        seeder_module,
        "collect_paths_for_roots",
        lambda r, progress=None: (paths := real_collect(r, progress)) + paths,
    )
    with caplog.at_level(logging.INFO):
        _scan(("input",))

    assert [m for m in _messages(caplog, logging.INFO) if m.startswith(EVENT)] == [f"{EVENT} count=1"]


def test_output_rescan_catalogs_new_files_and_only_logs_at_debug(roots, session, caplog):
    first = _write_files(roots["output"], "first", 10)
    _scan(("output",))
    added = _write_files(roots["output"], "added", N_FILES)
    _write_bad(roots["output"])

    with caplog.at_level(logging.DEBUG):
        _scan(("output",))

    assert _live_paths(session) == {str(p) for p in first + added}
    assert not [m for m in _messages(caplog, logging.WARNING) if "not valid UTF-8" in m]
    assert not [m for m in caplog.messages if m.startswith(EVENT)]
    assert [m for m in _messages(caplog, logging.DEBUG) if "not valid UTF-8" in m]


@pytest.mark.parametrize("hashing", [False, True], ids=["hashing_off", "hashing_on"])
def test_insert_skips_the_bad_name_without_a_batch_error(roots, session, hashing):
    good = _write_files(roots["input"], "in", 5)
    bad = _write_bad(roots["input"])
    paths = good[:2] + [bad] + good[2:]
    specs = scanner.build_asset_specs([str(p) for p in paths], set())[0]
    progress = seeder_module._ScanState()

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=hashing):
        created, error = insert_asset_specs(specs, set(), progress)

    assert error is None
    assert created == 5
    assert _live_paths(session) == {str(p) for p in good}
    assert progress.names_not_utf8 == [os.fsencode(bad)]
