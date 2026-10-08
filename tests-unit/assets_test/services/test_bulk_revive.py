"""A folder that goes away and comes back within the revive window brings its rows
back in bulk, with their history, when each file is listed again at the same size.
Run through the seeder's real fast phase on an in-memory catalog, hashing off unless
a test says otherwise."""

import os
import shutil
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

from app.assets import mode, scanner, scanner_admission, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    ensure_tag,
    ensure_tag_link,
    mark_content_missing,
    revive_contents,
    unset_content_missing,
)
from app.assets.helpers import get_utc_now
from app.assets.scanner_admission import _WATCH_LIST


@pytest.fixture
def root(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root; the tests move ``root / "batch"`` away and back."""
    root = temp_dir / "output"
    (root / "batch").mkdir(parents=True)
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(root))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir / "input"))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return root


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
def bulk_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the per-file revive off, so a test sees what the bulk pass alone revives."""
    monkeypatch.setattr(scanner, "recover_missing_content_by_stat", lambda *_args: "no_match")


def _scan(roots=("output",)) -> seeder_module._ScanState:
    """A fast phase; ``("output",)`` takes the after-prompt listing rescan, so the
    default exercises the path that never stats a listed file."""
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    seeder._run_fast_phase(roots)
    return seeder._scan_state


def _populate(folder: Path, count: int = 5) -> list[Path]:
    files = []
    for i in range(count):
        path = folder / f"image_{i}.png"
        path.write_bytes(b"x" * (10 + i))
        files.append(path)
    return files


def _customise(session) -> dict[str, tuple[str, dict | None, str | None, set[str]]]:
    session.expire_all()
    ensure_tag(session, "favourite")
    for i, record in enumerate(session.scalars(sa.select(Asset).order_by(Asset.name))):
        record.name = f"renamed {i}"
        record.user_metadata = {"note": i}
        record.job_id = f"job-{i}"
        ensure_tag_link(session, asset_id=record.id, tag_name="favourite", origin="manual")
    session.commit()
    return _records(session)


def _records(session) -> dict[str, tuple[str, dict | None, str | None, set[str]]]:
    session.expire_all()
    return {
        record.id: (
            record.name,
            record.user_metadata,
            record.job_id,
            set(session.scalars(sa.select(AssetTag.tag_name).where(AssetTag.asset_id == record.id))),
        )
        for record in session.scalars(sa.select(Asset))
    }


def _contents(session) -> list[AssetContent]:
    session.expire_all()
    return list(session.scalars(sa.select(AssetContent).order_by(AssetContent.path)))


def _move_away(folder: Path) -> Path:
    """Park the folder outside every root, as an unplugged drive or a move would."""
    parked = folder.parent.parent / (folder.name + ".away")
    folder.rename(parked)
    return parked


def _copy_back(parked: Path, folder: Path) -> None:
    """Restore the way a copy does: same bytes, new mtimes."""
    shutil.copytree(parked, folder, copy_function=shutil.copy)
    shutil.rmtree(parked)
    for path in folder.iterdir():
        stat_result = path.stat()
        os.utime(path, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000))


def _gone_and_copied_back(root: Path, session) -> tuple[list[Path], dict]:
    files = _populate(root / "batch")
    _scan()
    edits = _customise(session)
    parked = _move_away(root / "batch")
    assert _scan().missing_marked == len(files)
    _copy_back(parked, root / "batch")
    return files, edits


@pytest.mark.parametrize("roots", [("output",), ("models", "input", "output")])
def test_a_copied_back_folder_keeps_its_rows_and_history(root, session, roots):
    files, edits = _gone_and_copied_back(root, session)

    state = _scan(roots)

    # The per-file revive needs an exact mtime match, so only the bulk revive gets these.
    assert state.recovered == len(files)
    assert _records(session) == edits
    contents = _contents(session)
    assert len(contents) == len(files)
    assert [(c.is_missing, c.missing_since, c.mtime_ns) for c in contents] == [
        (False, None, path.stat().st_mtime_ns) for path in sorted(files)
    ]


def test_a_revived_row_loses_its_missing_tag_and_its_hash_when_the_mtime_moved(root, session):
    _populate(root / "batch", 1)
    _scan()
    (content,) = _contents(session)
    content.hash = "blake3:" + "0" * 64
    session.commit()
    parked = _move_away(root / "batch")
    _scan()
    assert "missing" in next(iter(_records(session).values()))[-1]
    _copy_back(parked, root / "batch")

    _scan()

    (content,) = _contents(session)
    assert content.hash is None
    assert "missing" not in next(iter(_records(session).values()))[-1]


def test_scan_marks_are_stamped_and_single_row_marks_are_not(root, session):
    files = _populate(root / "batch")
    _scan()
    files[0].unlink()
    before = get_utc_now()

    _scan()

    stamped = {c.path: c.missing_since for c in _contents(session) if c.is_missing}
    assert list(stamped) == [str(files[0])]
    assert stamped[str(files[0])] >= before
    live = next(c for c in _contents(session) if not c.is_missing)
    mark_content_missing(session, live.id)
    session.commit()
    assert session.get(AssetContent, live.id).missing_since is None


def test_the_per_file_unmark_clears_the_stamp(root, session):
    _populate(root / "batch", 1)
    _scan()
    _move_away(root / "batch")
    _scan()
    (content,) = _contents(session)
    assert content.missing_since is not None

    unset_content_missing(session, content.id)
    session.commit()

    assert session.get(AssetContent, content.id).missing_since is None


def test_a_folder_recreated_empty_revives_nothing_and_keeps_the_stamps(root, session):
    files = _populate(root / "batch")
    _scan()
    parked = _move_away(root / "batch")
    _scan()
    (root / "batch").mkdir()  # what a save into the missing folder does

    assert _scan().recovered == 0
    assert all(c.is_missing and c.missing_since is not None for c in _contents(session))

    (root / "batch").rmdir()
    _copy_back(parked, root / "batch")
    assert _scan().recovered == len(files)


def test_a_file_back_at_a_different_size_is_not_revived(root, session, bulk_only):
    files, edits = _gone_and_copied_back(root, session)
    files[0].write_bytes(b"y" * 999)

    state = _scan()

    assert state.recovered == len(files) - 1
    rows = [c for c in _contents(session) if c.path == str(files[0])]
    assert sorted((c.is_missing, c.size_bytes) for c in rows) == [(False, 999), (True, 10)]


def test_rows_marked_before_the_window_are_left_to_the_per_file_revive(root, session):
    files = _populate(root / "batch")
    _scan()
    parked = _move_away(root / "batch")
    _scan()
    session.execute(
        sa.update(AssetContent).values(missing_since=get_utc_now() - scanner.REVIVE_WINDOW - timedelta(minutes=1))
    )
    session.commit()
    _copy_back(parked, root / "batch")

    state = _scan()

    assert state.recovered == 0
    contents = _contents(session)
    assert sum(1 for c in contents if c.is_missing) == len(files)
    assert sum(1 for c in contents if not c.is_missing) == len(files)  # re-created


def test_an_unstamped_missing_row_is_not_revived_in_bulk(root, session):
    files = _populate(root / "batch", 1)
    _scan()
    (content,) = _contents(session)
    mark_content_missing(session, content.id)  # a single-row mark: no stamp
    session.commit()
    stat_result = files[0].stat()
    os.utime(files[0], ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000))

    assert _scan().recovered == 0
    assert session.get(AssetContent, content.id).is_missing


def test_a_file_deleted_from_a_folder_that_stays_is_marked_once(root, session):
    files = _populate(root / "batch")
    _scan()
    files[0].unlink()
    assert _scan().missing_marked == 1

    state = _scan()

    assert (state.recovered, state.missing_marked) == (0, 0)


def test_a_live_row_at_the_path_blocks_the_revive(root, session, bulk_only):
    files, _ = _gone_and_copied_back(root, session)
    squatter = create_content(session, path=str(files[0]), size_bytes=10, mtime_ns=1)
    create_record(session, content_id=squatter.id, name="new")
    session.commit()

    state = _scan()

    assert state.recovered == len(files) - 1
    rows = [c for c in _contents(session) if c.path == str(files[0])]
    assert sorted((c.id == squatter.id, c.is_missing) for c in rows) == [(False, True), (True, False)]


def test_the_newest_row_with_a_record_wins_over_a_newer_one_without(root, session, bulk_only):
    files = _populate(root / "batch", 1)
    _scan()
    (users,) = _contents(session)
    users_record = session.scalar(sa.select(Asset.id))
    parked = _move_away(root / "batch")
    _scan()
    orphan = create_content(session, path=str(files[0]), size_bytes=10, mtime_ns=1)
    session.flush()
    orphan.is_missing = True
    orphan.missing_since = get_utc_now() + timedelta(seconds=1)
    session.commit()
    _copy_back(parked, root / "batch")

    assert _scan().recovered == 1

    assert not session.get(AssetContent, users.id).is_missing
    assert session.get(AssetContent, orphan.id).is_missing
    assert session.get(Asset, users_record).content_id == users.id


def test_only_the_returned_part_of_a_nested_folder_revives(root, session):
    top = _populate(root / "batch", 2)
    (root / "batch" / "sub").mkdir()
    nested = _populate(root / "batch" / "sub", 2)
    _scan()
    parked = _move_away(root / "batch")
    _scan()
    (root / "batch").mkdir()
    for path in top:
        shutil.copy(parked / path.name, path)  # new mtimes, so only the bulk revive takes them

    assert _scan().recovered == len(top)
    assert {c.path for c in _contents(session) if c.is_missing} == {str(p) for p in nested}


def test_hashing_on_keeps_the_per_file_revive(root, session):
    files, _ = _gone_and_copied_back(root, session)

    class _HashingOn:
        enable_asset_hashing = True

    mode.init(_HashingOn())
    scanner.revive_returned_references_safely("output")

    assert all(c.is_missing for c in _contents(session))
    assert len(files) == len(_contents(session))


def test_the_newest_of_two_rows_with_records_wins(root, session, bulk_only):
    files = _populate(root / "batch", 1)
    _scan()
    (older,) = _contents(session)
    older.created_at = older.created_at - timedelta(days=1)
    older.is_missing, older.missing_since = True, get_utc_now()
    session.commit()
    newer = create_content(session, path=str(files[0]), size_bytes=10, mtime_ns=1)
    create_record(session, content_id=newer.id, name="newer")
    newer.is_missing, newer.missing_since = True, get_utc_now()
    session.commit()
    stat_result = files[0].stat()
    os.utime(files[0], ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000))

    assert _scan().recovered == 1

    assert not session.get(AssetContent, newer.id).is_missing
    assert session.get(AssetContent, older.id).is_missing


def test_a_row_whose_record_went_before_the_write_is_not_revived(root, session):
    _populate(root / "batch", 1)
    _scan()
    parked = _move_away(root / "batch")
    _scan()
    _copy_back(parked, root / "batch")
    (content,) = _contents(session)
    candidates = scanner._revival_candidates(
        session, [str(root)], get_utc_now() - scanner.REVIVE_WINDOW
    )
    assert list(candidates.values()) == [[(content.id, 10)]]
    session.execute(sa.delete(AssetTag))
    session.execute(sa.delete(Asset))  # the user deletes it while the scan is paused
    session.commit()

    assert revive_contents(session, {content.id: 1}) == []
    assert session.get(AssetContent, content.id).is_missing


def test_a_pre_epoch_file_does_not_stop_the_rest_reviving(root, session, bulk_only):
    files, edits = _gone_and_copied_back(root, session)
    os.utime(files[0], ns=(0, -1_000_000_000))

    state = _scan()

    assert state.recovered == len(files) - 1
    assert all(not c.is_missing for c in _contents(session) if c.path != str(files[0]))


def test_a_cancel_during_the_revive_leaves_the_rows_missing(root, session):
    files, _ = _gone_and_copied_back(root, session)
    calls = 0

    def cancelled_at_the_second_file() -> bool:
        """The revive asks once for the directory, then once per file; cancel on the
        second file, before anything is written."""
        nonlocal calls
        calls += 1
        return calls >= 3

    scanner.revive_returned_references_safely("output", should_stop=cancelled_at_the_second_file)

    assert sum(1 for c in _contents(session) if c.is_missing) == len(files)


def test_a_failing_revive_leaves_the_rest_of_the_scan_running(root, session, monkeypatch, caplog):
    files, _ = _gone_and_copied_back(root, session)
    (root / "new.png").write_bytes(b"new")

    def broken(*_args):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(scanner, "_revival_candidates", broken)
    state = _scan()

    assert state.recovered == len(files)  # the per-file revive, with the same window rule
    assert any(c.path == str(root / "new.png") and not c.is_missing for c in _contents(session))
    assert "bulk revive failed" in caplog.text


def test_the_newest_row_of_the_returned_size_wins(root, session, bulk_only):
    files = _populate(root / "batch", 1)
    _scan()
    (original,) = _contents(session)
    original.created_at = original.created_at - timedelta(days=1)
    original.is_missing, original.missing_since = True, get_utc_now()
    session.commit()
    replacement = create_content(session, path=str(files[0]), size_bytes=999, mtime_ns=1)
    create_record(session, content_id=replacement.id, name="replacement")
    replacement.is_missing, replacement.missing_since = True, get_utc_now()
    session.commit()
    stat_result = files[0].stat()
    os.utime(files[0], ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000))

    assert _scan().recovered == 1

    assert not session.get(AssetContent, original.id).is_missing
    assert session.get(AssetContent, replacement.id).is_missing


def test_a_revive_keeps_the_hash_when_the_mtime_did_not_move(root, session):
    _populate(root / "batch", 1)
    _scan()
    (content,) = _contents(session)
    content.hash = "blake3:" + "0" * 64
    session.commit()
    parked = _move_away(root / "batch")
    _scan()
    shutil.move(parked, root / "batch")  # a move keeps the mtime

    assert _scan().recovered == 1

    (content,) = _contents(session)
    assert (content.is_missing, content.hash) == (False, "blake3:" + "0" * 64)


def test_rows_marked_before_the_upgrade_still_revive_per_file(root, session):
    files = _populate(root / "batch")
    _scan()
    edits = _customise(session)
    for content in _contents(session):
        mark_content_missing(session, content.id)  # no stamp, like a row marked before 0009
    session.commit()

    state = _scan()

    assert state.recovered == len(files)
    assert _records(session) == edits
    assert all(not c.is_missing and c.missing_since is None for c in _contents(session))


def test_the_revive_writes_in_batches_and_counts_what_committed(root, session, monkeypatch):
    files, _ = _gone_and_copied_back(root, session)
    monkeypatch.setattr(scanner, "WRITE_BATCH_ROWS", 2)
    batches: list[int] = []

    def revive(session, mtimes):
        batches.append(len(mtimes))
        if len(batches) == 2:
            raise RuntimeError("database is locked")
        return revive_contents(session, mtimes)

    monkeypatch.setattr(scanner, "revive_contents", revive)
    progress = seeder_module._ScanState()
    scanner.revive_returned_references_safely("output", progress)

    assert batches == [2, 2]
    assert progress.recovered == 2  # the first batch committed; the failure stopped the rest
    assert sum(1 for c in _contents(session) if not c.is_missing) == 2
    assert _scan().recovered == len(files) - 2


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_an_unreadable_entry_does_not_hide_the_rest_of_its_directory(root, session, bulk_only):
    files, edits = _gone_and_copied_back(root, session)
    locked = root.parent / "locked"
    (locked / "inner").mkdir(parents=True)
    (root / "batch" / "link").symlink_to(locked / "inner")
    locked.chmod(0)  # stat'ing the link's target now raises PermissionError
    try:
        # A full scan: the output rescan reads names from its walk and never lists the folder itself.
        assert _scan(("models", "input", "output")).recovered == len(files)
    finally:
        locked.chmod(0o755)
    assert _records(session) == edits


def test_a_file_the_bulk_pass_missed_still_keeps_its_row(root, session, monkeypatch):
    """A file still being copied when the bulk pass looks is admitted later through the
    per-file revive, which applies the same window and size rule."""
    files, edits = _gone_and_copied_back(root, session)
    monkeypatch.setattr(scanner, "_returned_files", lambda *_args: {})

    state = _scan(("models", "input", "output"))

    assert state.recovered == len(files)
    assert _records(session) == edits
    assert [(c.is_missing, c.mtime_ns) for c in _contents(session)] == [
        (False, path.stat().st_mtime_ns) for path in sorted(files)
    ]


def test_a_file_still_being_written_waits_on_the_watch_list(root, session, monkeypatch):
    files, _ = _gone_and_copied_back(root, session)

    def still_copying(_seconds):
        stat_result = files[0].stat()  # the copy is still writing into a preallocated file
        os.utime(files[0], ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(scanner_admission.time, "sleep", still_copying)
    progress = seeder_module._ScanState()
    scanner.revive_returned_references_safely("output", progress)

    assert progress.recovered == len(files) - 1
    missing = [c.path for c in _contents(session) if c.is_missing]
    assert missing == [str(files[0])]
    assert [entry.path for entry in _WATCH_LIST] == [str(files[0])]
