"""A registered model folder that can't be listed (moved install, unplugged drive,
permission denied) is skipped by the startup scan: the scan completes, every other
folder is catalogued, and the skipped folder's rows are neither retired nor duplicated.
Run through the seeder's real scan loop and folder_paths listing on an in-memory catalog."""

import errno
import logging
import os
import shutil
import sys
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
    """checkpoints in the install and in an extra_model_paths-style shared folder, plus a
    registered folder that was never created, as stock installs have."""
    dirs = {
        "checkpoints": temp_dir / "models" / "checkpoints",
        "shared": temp_dir / "shared" / "models" / "checkpoints",
        "loras": temp_dir / "models" / "loras",
        "input": temp_dir / "input",
        "output": temp_dir / "output",
        "temp": temp_dir / "temp",
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    exts = {".safetensors"}
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(dirs["checkpoints"]), str(dirs["shared"])], exts),
        "loras": ([str(dirs["loras"])], exts),
        "classifiers": ([str(temp_dir / "models" / "classifiers")], exts),
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


def _startup_scan(caplog: pytest.LogCaptureFixture, roots=ALL_ROOTS) -> dict:
    """One startup scan (prune first, then a fast scan); returns its scan_completed fields.
    ``roots=("output",)`` is instead the rescan queued after each prompt."""
    seeder = seeder_module._AssetSeeder()
    seeder._state = seeder_module.State.RUNNING
    seeder._scan_state = seeder_module._ScanState()
    seeder._roots = roots
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._prune_first = roots == ALL_ROOTS
    seeder._run_gate.set()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        seeder._run_scan()
    assert events_named(caplog, "seeder.scan_failed") == []
    assert events_named(caplog, "scanner.fast_scan_failed") == []
    (completed,) = events_named(caplog, "seeder.scan_completed")
    return completed


def _rows(session) -> dict[str, list[tuple[str, bool]]]:
    """path -> [(record id, missing)] for every record."""
    session.expire_all()
    out: dict[str, list[tuple[str, bool]]] = {}
    for record_id, path, missing in session.execute(
        sa.select(Asset.id, AssetContent.path, AssetContent.is_missing).join(
            AssetContent, Asset.content_id == AssetContent.id
        )
    ):
        out.setdefault(path, []).append((record_id, missing))
    return out


def _seed(layout: dict[str, Path], session, caplog) -> tuple[list[Path], dict]:
    shared_files = [_write(layout["shared"] / f"shared_{i}.safetensors") for i in range(3)]
    _write(layout["checkpoints"] / "local.safetensors")
    _write(layout["input"] / "photo.png", b"png")
    assert _startup_scan(caplog)["skipped_folders_count"] == 0
    before = _rows(session)
    assert len(before) == 5
    assert not any(missing for rows in before.values() for _id, missing in rows)
    return shared_files, before


def _assert_unchanged(before: dict, after: dict, paths: list[Path]) -> None:
    for path in paths:
        assert after[str(path)] == before[str(path)], path


@pytest.mark.parametrize("fresh_process", [False, True], ids=["listed-then-gone", "gone-at-startup"])
def test_a_missing_model_folder_is_skipped_without_retiring_its_rows(
    layout, session, caplog, fresh_process
):
    shared_files, before = _seed(layout, session, caplog)
    away = layout["shared"].parent / "checkpoints-away"
    layout["shared"].rename(away)
    if fresh_process:
        folder_paths.filename_list_cache.clear()
    added_model = _write(layout["loras"] / "new_lora.safetensors")
    added_input = _write(layout["input"] / "new.png", b"png2")

    completed = _startup_scan(caplog)

    assert completed["skipped_folders_count"] == 1
    assert completed["created"] == 2
    after = _rows(session)
    _assert_unchanged(before, after, shared_files)
    assert after[str(added_model)][0][1] is False
    assert after[str(added_input)][0][1] is False
    assert events_named(caplog, "seeder.marked_missing") == [{"count": 0, "stage": "pruning"}]

    # The folder comes back: the same records, still live, and no new ones.
    away.rename(layout["shared"])
    completed = _startup_scan(caplog)

    assert completed["skipped_folders_count"] == 0
    assert completed["created"] == 0
    assert _rows(session) == after


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions that apply to the test user",
)
def test_an_unreadable_model_folder_is_skipped_without_retiring_its_rows(
    layout, session, caplog
):
    shared_files, before = _seed(layout, session, caplog)
    mode = layout["shared"].stat().st_mode
    layout["shared"].chmod(0)
    try:
        completed = _startup_scan(caplog)
    finally:
        layout["shared"].chmod(mode)

    assert completed["skipped_folders_count"] == 1
    assert completed["created"] == 0
    _assert_unchanged(before, _rows(session), shared_files)


def test_an_io_error_on_a_file_in_a_listable_folder_leaves_its_row_live(
    layout, session, caplog, monkeypatch
):
    """The folder probe passes, but stat'ing the stored files fails (a flaky share, a
    device error): that says nothing about whether they exist, so nothing is retired."""
    shared_files, before = _seed(layout, session, caplog)
    real_stat = os.stat
    flaky = {str(path) for path in shared_files}

    def flaky_stat(path, *args, **kwargs):
        if str(path) in flaky:
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky_stat)
    completed = _startup_scan(caplog)
    monkeypatch.setattr(os, "stat", real_stat)

    assert completed["skipped_folders_count"] == 0
    _assert_unchanged(before, _rows(session), shared_files)


def test_a_deleted_file_in_a_listable_folder_is_still_marked_missing(layout, session, caplog):
    """The skip is per folder: a folder that lists still retires the files it lost."""
    shared_files, before = _seed(layout, session, caplog)
    shared_files[0].unlink()

    completed = _startup_scan(caplog)

    assert completed["skipped_folders_count"] == 0
    after = _rows(session)
    assert after[str(shared_files[0])] == [(before[str(shared_files[0])][0][0], True)]
    _assert_unchanged(before, after, shared_files[1:])


def test_a_whole_model_root_that_cant_be_listed_still_scans_input(layout, session, caplog):
    shared_files, before = _seed(layout, session, caplog)
    local = layout["checkpoints"] / "local.safetensors"
    for name in ("checkpoints", "shared", "loras"):
        shutil.rmtree(layout[name])
    photo = _write(layout["input"] / "new.png", b"png2")

    completed = _startup_scan(caplog)

    # loras and classifiers hold no rows, so only the two checkpoints folders count.
    assert completed["skipped_folders_count"] == 2
    after = _rows(session)
    _assert_unchanged(before, after, [*shared_files, local])
    assert after[str(photo)][0][1] is False


def test_a_registered_folder_that_was_never_created_is_neither_counted_nor_warned(
    layout, session, caplog
):
    _seed(layout, session, caplog)

    completed = _startup_scan(caplog)

    assert completed["skipped_folders_count"] == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "classifiers" in r.getMessage()]



def _register_output_checkpoints(layout: dict[str, Path]) -> Path:
    """main.py registers output/checkpoints as a checkpoints folder."""
    nested = layout["output"] / "checkpoints"
    folder_paths.folder_names_and_paths["checkpoints"][0].append(str(nested))
    return nested


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs a privilege on Windows")
def test_a_linked_folder_inside_output_keeps_its_rows_when_the_link_target_is_gone(
    layout, session, caplog, temp_dir
):
    """output lists fine, but output/checkpoints links to a drive that's gone: neither the
    output sync nor the per-prompt output rescan may retire the rows behind the link."""
    drive = temp_dir / "drive" / "checkpoints"
    saved = _write(drive / "saved.safetensors")
    nested = _register_output_checkpoints(layout)
    nested.symlink_to(drive, target_is_directory=True)
    _startup_scan(caplog)
    before = _rows(session)[str(nested / saved.name)]
    assert before[0][1] is False
    unplugged = drive.parent.with_name("drive-away")
    drive.parent.rename(unplugged)

    assert _startup_scan(caplog)["skipped_folders_count"] == 1
    assert _rows(session)[str(nested / saved.name)] == before
    _startup_scan(caplog, roots=("output",))
    assert _rows(session)[str(nested / saved.name)] == before

    unplugged.rename(drive.parent)
    assert _startup_scan(caplog)["created"] == 0
    assert _rows(session)[str(nested / saved.name)] == before


def test_a_registered_folder_deleted_from_inside_output_keeps_its_rows(layout, session, caplog):
    """The same trade-off as any other registered folder whose path is absent."""
    nested = _register_output_checkpoints(layout)
    saved = _write(nested / "saved.safetensors")
    _startup_scan(caplog)
    before = _rows(session)[str(saved)]
    shutil.rmtree(nested)

    assert _startup_scan(caplog)["skipped_folders_count"] == 1
    _startup_scan(caplog, roots=("output",))
    assert _rows(session)[str(saved)] == before
