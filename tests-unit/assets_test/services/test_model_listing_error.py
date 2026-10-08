"""A model category whose folder_paths listing raises (e.g. one of its folders was listed
earlier and has since gone away) is skipped: the asset scan completes and catalogues
everything else. Run through the seeder's real scan loop and folder_paths listing on an
in-memory catalog."""

import errno
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

import folder_paths
from app.assets import seeder as seeder_module
from app.assets.database.models import Asset, AssetContent
from app.assets.event_log import TAG
from app.assets.scanner_admission import _WATCH_LIST

ALL_ROOTS = ("models", "input", "output")


def events_named(caplog: pytest.LogCaptureFixture, event: str) -> list[dict]:
    prefix = f"{TAG} {event}"
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message == prefix or message.startswith(prefix + " "):
            pairs = (pair.split("=", 1) for pair in message[len(prefix):].split())
            out.append({k: int(v) if v.isdigit() else v for k, v in pairs})
    return out


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


@pytest.fixture
def layout(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """checkpoints in the install, an extra_model_paths-style shared folder, and one
    configured after it."""
    dirs = {
        "checkpoints": temp_dir / "models" / "checkpoints",
        "shared": temp_dir / "shared" / "models" / "checkpoints",
        "late": temp_dir / "late" / "checkpoints",
        "loras": temp_dir / "models" / "loras",
        "input": temp_dir / "input",
        "output": temp_dir / "output",
        "temp": temp_dir / "temp",
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    exts = {".safetensors"}
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(dirs["checkpoints"]), str(dirs["shared"]), str(dirs["late"])], exts),
        "loras": ([str(dirs["loras"])], exts),
    })
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(dirs["input"]))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(dirs["output"]))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(dirs["temp"]))
    return dirs


def _write(path: Path, payload: bytes = b"model-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _startup_scan(caplog: pytest.LogCaptureFixture) -> dict:
    """One startup scan once the node list is served (prune first, then a fast scan);
    returns its scan_completed fields."""
    seeder = seeder_module._AssetSeeder()
    seeder._state = seeder_module.State.RUNNING
    seeder._scan_state = seeder_module._ScanState()
    seeder._roots = ALL_ROOTS
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._prune_first = True
    seeder._prune_pending = seeder._node_list_served = True
    seeder._run_gate.set()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        seeder._run_scan()
    assert events_named(caplog, "seeder.scan_failed") == []
    (completed,) = events_named(caplog, "seeder.scan_completed")
    return completed


def _missing(session) -> dict[str, bool]:
    """path -> missing, for every record."""
    session.expire_all()
    return dict(session.execute(
        sa.select(AssetContent.path, AssetContent.is_missing).join(
            Asset, Asset.content_id == AssetContent.id
        )
    ).all())


def _seed(layout: dict[str, Path], session, caplog) -> list[Path]:
    shared_files = [_write(layout["shared"] / f"shared_{i}.safetensors") for i in range(3)]
    _write(layout["checkpoints"] / "local.safetensors")
    _write(layout["input"] / "photo.png", b"png")
    _startup_scan(caplog)
    assert _missing(session) == {str(p): False for p in [
        *shared_files, layout["checkpoints"] / "local.safetensors", layout["input"] / "photo.png",
    ]}
    return shared_files


def _add_new_files(layout: dict[str, Path]) -> list[Path]:
    return [
        _write(layout["loras"] / "new_lora.safetensors"),
        _write(layout["output"] / "ComfyUI_00001_.png", b"png"),
    ]


def test_a_model_folder_that_vanishes_after_being_listed_does_not_abort_the_scan(
    layout, session, caplog
):
    """folder_paths' cached listing raises once a folder it recorded is gone (and checked
    before any folder that changed). The category's other folders are still catalogued."""
    shared_files = _seed(layout, session, caplog)
    layout["shared"].rename(layout["shared"].with_name("checkpoints-away"))
    added = [*_add_new_files(layout), _write(layout["late"] / "new_checkpoint.safetensors")]

    completed = _startup_scan(caplog)

    assert completed["created"] == 3
    missing = _missing(session)
    assert all(missing[str(p)] is False for p in added)
    assert missing[str(layout["checkpoints"] / "local.safetensors")] is False
    # Unchanged: the gone folder's rows are marked missing, as they are on master.
    assert all(missing[str(p)] is True for p in shared_files)

    # Every later scan still lists the category rather than skipping it.
    later = _write(layout["late"] / "later_checkpoint.safetensors")
    assert _startup_scan(caplog)["created"] == 1
    assert _missing(session)[str(later)] is False


def test_a_model_category_whose_listing_raises_an_oserror_is_skipped(
    layout, session, caplog, monkeypatch
):
    """Any OSError, e.g. Windows' WinError 433 (device does not exist), that a fresh
    listing hits too: that category is skipped, the rest catalogued."""
    shared_files = _seed(layout, session, caplog)
    for name in ("get_filename_list", "get_filename_list_"):
        real = getattr(folder_paths, name)

        def raising(folder_name, real=real):
            if folder_name == "checkpoints":
                raise OSError(errno.EINVAL, "A device which does not exist was specified")
            return real(folder_name)

        monkeypatch.setattr(folder_paths, name, raising)
    added = _add_new_files(layout)

    completed = _startup_scan(caplog)

    assert completed["created"] == 2
    missing = _missing(session)
    assert all(missing[str(p)] is False for p in [*added, *shared_files])
    assert any("skipping model category checkpoints" in r.getMessage() for r in caplog.records)


def test_a_model_folder_absent_at_startup_still_has_its_rows_marked_missing(
    layout, session, caplog
):
    """Unchanged from master: nothing raises here, and the sync retires the rows."""
    shared_files = _seed(layout, session, caplog)
    layout["shared"].rename(layout["shared"].with_name("checkpoints-away"))
    folder_paths.filename_list_cache.clear()

    completed = _startup_scan(caplog)

    assert completed["created"] == 0
    missing = _missing(session)
    assert all(missing[str(p)] is True for p in shared_files)
    assert missing[str(layout["checkpoints"] / "local.safetensors")] is False
