"""A file briefly absent when a full scan stats the catalogued paths, and back by the time
the scan walks the folders (a sync client re-downloading it, say), keeps its record. Only
a path the walk misses too is retired. Run through the seeder's real fast phase on an
in-memory catalog, with hashing off unless a test says otherwise."""

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

from app.assets import mode, scanner, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import ensure_tag, ensure_tag_link
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.services import path_utils
from app.assets.scanner_changes import clear_pending_verifications, drain_pending_verifications

ROOTS = ("input", "output")
EVENT_LINE = re.compile(r"^\[assets-event\] (?P<event>\S+)(?P<fields>(?: \S+)*)$")
Hook = Callable[[], None]


@pytest.fixture
def drive(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    drive = temp_dir / "drive"
    for name in ROOTS:
        (drive / name).mkdir(parents=True)
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(drive / "output"))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(drive / "input"))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return drive


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    clear_pending_verifications()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()
    clear_pending_verifications()


def _seeder() -> seeder_module._AssetSeeder:
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    return seeder


@contextmanager
def _around_the_walk(after_check: Hook | None, after_walk: Hook | None):
    """``after_check`` runs once every root's stored paths are stat'ed and before the
    walk; ``after_walk`` once the walk has listed the folders."""
    real_collect = seeder_module.collect_paths_for_roots

    def collect(walk_roots, *args, **kwargs):
        if after_check is not None:
            after_check()
        paths = real_collect(walk_roots, *args, **kwargs)
        if after_walk is not None:
            after_walk()
        return paths

    with patch.object(seeder_module, "collect_paths_for_roots", collect):
        yield


def _scan(
    roots=ROOTS,
    after_check: Hook | None = None,
    after_walk: Hook | None = None,
    seeder: seeder_module._AssetSeeder | None = None,
) -> seeder_module._ScanState:
    seeder = seeder or _seeder()
    with _around_the_walk(after_check, after_walk):
        seeder._run_fast_phase(roots)
    return seeder._scan_state


@contextmanager
def _absent_for_the_check(*paths: Path):
    """Each path is gone when the scan stats the catalogue. The caller puts it back."""
    real_sync = seeder_module.sync_root_safely

    def sync(root, progress=None, **kwargs):
        for path in paths:
            if path.exists():
                path.rename(_parked(path))
        return real_sync(root, progress, **kwargs)

    with patch.object(seeder_module, "sync_root_safely", sync):
        yield


def _parked(path: Path) -> Path:
    """Hidden, so the walk never lists the parked copy."""
    return path.with_name("." + path.name + ".parked")


def _put_back(path: Path, mtime_bump_ns: int = 0, data: bytes | None = None) -> None:
    parked = _parked(path)
    if parked.exists():
        parked.rename(path)
    if data is not None:
        path.write_bytes(data)
    if mtime_bump_ns:
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + mtime_bump_ns))


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, str]]:
    found = []
    for record in caplog.records:
        match = EVENT_LINE.match(record.getMessage())
        if match is not None and match.group("event") == name:
            found.append(dict(pair.split("=", 1) for pair in match.group("fields").split()))
    return found


def _populate(drive: Path, roots=ROOTS) -> list[Path]:
    files = []
    for name in roots:
        for i in range(3):
            path = drive / name / f"{name}_{i}.png"
            path.write_bytes(f"{name}-{i}".encode() * (i + 1))
            files.append(path)
    return files


def _customise(session) -> dict[str, tuple]:
    """Give every record a user's edits and a job link; returns them keyed by record id."""
    session.expire_all()
    for i, record in enumerate(session.scalars(sa.select(Asset).order_by(Asset.name))):
        record.name = f"renamed {i}"
        record.user_metadata = {"note": i}
        record.job_id = f"job-{i}"
        ensure_tag(session, "favourite")
        ensure_tag_link(session, asset_id=record.id, tag_name="favourite", origin="manual")
    session.commit()
    return _records(session)


def _records(session) -> dict[str, tuple]:
    session.expire_all()
    return {
        record.id: (
            record.content_id,
            record.name,
            record.user_metadata,
            record.job_id,
            set(session.scalars(sa.select(AssetTag.tag_name).where(AssetTag.asset_id == record.id))),
        )
        for record in session.scalars(sa.select(Asset))
    }


def _rows(session, path: Path) -> list[AssetContent]:
    session.expire_all()
    return list(session.scalars(sa.select(AssetContent).where(AssetContent.path == str(path))))


def _live(session, path: Path) -> AssetContent:
    (row,) = [row for row in _rows(session, path) if not row.is_missing]
    return row


def _content_count(session) -> int:
    return session.scalar(sa.select(sa.func.count()).select_from(AssetContent))


def test_a_file_rewritten_in_place_during_the_scan_keeps_its_record(drive, session, caplog):
    files = _populate(drive)
    _scan()
    edits = _customise(session)
    target = files[0]
    before = _live(session, target)
    before.hash = "blake3:" + "0" * 64
    session.commit()
    old_mtime = before.mtime_ns

    with _absent_for_the_check(target), caplog.at_level(logging.INFO):
        state = _scan(after_check=lambda: _put_back(target, mtime_bump_ns=1_000_000_000))

    assert state.missing_marked == 0
    assert state.recovered == 0
    assert state.skipped == len(files)  # the walk treats it as the live row it is
    assert _events(caplog, "seeder.marked_missing") == []
    assert (state.found_again, state.found_again_mtime_changed, state.found_again_size_changed) == (1, 1, 0)
    assert len(_rows(session, target)) == 1
    row = _live(session, target)
    assert row.id == before.id
    assert row.mtime_ns == old_mtime + 1_000_000_000
    assert row.hash is None  # the refreshed stat can't vouch for the old digest
    assert _records(session) == edits

    # And it stays that way: the next ordinary scan creates and retires nothing.
    again = _scan()
    assert again.missing_marked == 0
    assert _records(session) == edits
    assert _content_count(session) == len(files)


def test_the_found_again_counts_reach_the_scan_completed_line(drive, session, caplog):
    files = _populate(drive)
    _scan()
    seeder = _seeder()
    seeder._roots = ROOTS
    seeder._prune_first = False

    def put_back():
        _put_back(files[0], mtime_bump_ns=1_000_000_000)
        _put_back(files[1])

    with _absent_for_the_check(files[0], files[1]), _around_the_walk(put_back, None), \
         caplog.at_level(logging.INFO):
        seeder._run_scan()

    (completed,) = _events(caplog, "seeder.scan_completed")
    assert completed["found_again_count"] == "2"
    assert completed["found_again_mtime_changed_count"] == "1"
    assert completed["found_again_size_changed_count"] == "0"
    assert completed["found_again_cloud_count"] == "0"
    assert completed["missing_marked_count"] == "0"


def test_a_file_back_unchanged_is_left_alone(drive, session):
    files = _populate(drive)
    _scan()
    edits = _customise(session)

    with _absent_for_the_check(files[0]):
        state = _scan(after_check=lambda: _put_back(files[0]))

    assert (state.missing_marked, state.recovered, state.found_again) == (0, 0, 1)
    assert state.found_again_mtime_changed == 0
    assert _records(session) == edits
    assert _content_count(session) == len(files)


def test_a_file_back_with_new_bytes_splits_as_a_live_row_does(drive, session):
    files = _populate(drive)
    _scan()
    target = files[0]
    old = _live(session, target)
    edits = _customise(session)
    (old_record,) = [rid for rid, rec in edits.items() if rec[0] == old.id]

    with _absent_for_the_check(target):
        state = _scan(after_check=lambda: _put_back(target, data=b"different content, longer"))

    assert state.missing_marked == 0
    assert state.found_again_size_changed == 1
    rows = _rows(session, target)
    assert sorted(row.is_missing for row in rows) == [False, True]
    assert [row.id for row in rows if row.is_missing] == [old.id]
    assert _live(session, target).size_bytes == len(b"different content, longer")
    # The old record keeps the user's edits, gaining only the "missing" tag.
    records = _records(session)
    assert records[old_record][:4] == edits[old_record][:4]
    assert records[old_record][4] == edits[old_record][4] | {"missing"}


def test_hashing_on_verifies_a_found_again_file_instead_of_re_creating_it(drive, session):
    class _HashingOn:
        enable_asset_hashing = True

    mode.init(_HashingOn())
    files = _populate(drive)
    _scan()
    edits = _customise(session)
    target = files[0]
    before = _live(session, target)

    with _absent_for_the_check(target):
        state = _scan(after_check=lambda: _put_back(target, mtime_bump_ns=1_000_000_000))

    assert (state.missing_marked, state.found_again) == (0, 1)
    assert drain_pending_verifications(session) == 1
    session.commit()
    row = _live(session, target)
    assert row.id == before.id
    assert row.hash is not None
    assert _records(session) == edits


def test_a_genuinely_deleted_file_is_retired_after_the_walk(drive, session, caplog):
    files = _populate(drive)
    _scan()
    events_at_walk: list[int] = []

    def note_events():
        events_at_walk.append(len(_events(caplog, "seeder.marked_missing")))

    files[0].unlink()
    with caplog.at_level(logging.INFO):
        state = _scan(after_check=note_events)

    assert events_at_walk == [0]  # nothing was retired before the walk
    assert state.missing_marked == 1
    assert state.found_again == 0
    assert [row.is_missing for row in _rows(session, files[0])] == [True]
    assert _events(caplog, "seeder.marked_missing") == [
        {"count": "1", "root": "input", "stage": "fast_scan"}
    ]


def test_a_partial_folder_keeps_what_came_back_and_retires_the_rest(drive, session):
    files = _populate(drive)
    _scan()
    edits = _customise(session)
    back, gone = files[:2], files[2]

    with _absent_for_the_check(*files[:3]):
        state = _scan(after_check=lambda: [_put_back(p, mtime_bump_ns=2_000_000_000) for p in back])

    assert (state.found_again, state.missing_marked) == (2, 1)
    assert [[row.is_missing for row in _rows(session, p)] for p in back] == [[False], [False]]
    assert [row.is_missing for row in _rows(session, gone)] == [True]
    # Every record survives; only the retired one gains the "missing" tag.
    records = _records(session)
    gone_record = next(rid for rid, rec in edits.items() if rec[0] == _rows(session, gone)[0].id)
    assert records.pop(gone_record)[4] == edits.pop(gone_record)[4] | {"missing"}
    assert records == edits


def test_a_file_back_under_another_name_is_still_retired_and_re_created(drive, session):
    files = _populate(drive)
    _scan()
    target = files[0]
    renamed = target.with_name("renamed.png")

    with _absent_for_the_check(target):
        state = _scan(
            after_check=lambda: _parked(target).rename(renamed)
        )

    assert (state.found_again, state.missing_marked) == (0, 1)
    assert [row.is_missing for row in _rows(session, target)] == [True]
    assert [row.is_missing for row in _rows(session, renamed)] == [False]


def test_a_file_listed_by_the_walk_but_gone_again_is_retired(drive, session):
    files = _populate(drive)
    _scan()
    target = files[0]

    with _absent_for_the_check(target):
        state = _scan(after_check=lambda: _put_back(target), after_walk=target.unlink)

    assert (state.found_again, state.missing_marked) == (0, 1)
    assert [row.is_missing for row in _rows(session, target)] == [True]


def test_an_undecided_restat_leaves_the_row_live_and_inserts_nothing(drive, session, monkeypatch):
    files = _populate(drive)
    _scan()
    target = files[0]
    real_stat = os.stat

    def locked(path, *args, **kwargs):
        if os.path.abspath(path) == str(target):
            raise PermissionError(13, "locked")
        return real_stat(path, *args, **kwargs)

    with _absent_for_the_check(target):
        state = _scan(
            after_check=lambda: _put_back(target, mtime_bump_ns=1_000_000_000),
            after_walk=lambda: monkeypatch.setattr(os, "stat", locked),
        )
    monkeypatch.setattr(os, "stat", real_stat)

    assert (state.found_again, state.missing_marked, state.permission_denied) == (0, 0, 1)
    assert [row.is_missing for row in _rows(session, target)] == [False]
    assert _content_count(session) == len(files)


def test_a_row_another_writer_changed_meanwhile_is_left_to_it(drive, session):
    files = _populate(drive)
    _scan()
    target = files[0]
    row_id = _live(session, target).id

    def another_writer():
        _put_back(target, mtime_bump_ns=1_000_000_000)
        session.get(AssetContent, row_id).mtime_ns = 12345
        session.commit()

    with _absent_for_the_check(target):
        _scan(after_check=another_writer)

    rows = _rows(session, target)
    assert [(row.id, row.is_missing, row.mtime_ns) for row in rows] == [(row_id, False, 12345)]


def test_a_cancel_before_the_walk_retires_nothing(drive, session):
    files = _populate(drive)
    _scan()
    seeder = _seeder()
    files[0].unlink()

    real_sync = seeder_module.sync_root_safely

    def sync_then_cancel(root, progress=None, **kwargs):
        survivors = real_sync(root, progress, **kwargs)
        seeder._cancel_event.set()
        return survivors

    with patch.object(seeder_module, "sync_root_safely", sync_then_cancel):
        state = _scan(seeder=seeder)

    assert state.missing_marked == 0
    assert [row.is_missing for row in _rows(session, files[0])] == [False]


def test_a_cancel_during_the_walk_retires_nothing(drive, session):
    files = _populate(drive)
    _scan()
    seeder = _seeder()
    files[0].unlink()

    state = _scan(seeder=seeder, after_walk=seeder._cancel_event.set)

    assert state.missing_marked == 0
    assert [row.is_missing for row in _rows(session, files[0])] == [False]


def test_a_held_input_row_resolves_while_models_scans_normally(drive, session, caplog, monkeypatch):
    models = drive / "models"
    models.mkdir()
    folders = lambda: [("checkpoints", [str(models)], set())]  # noqa: E731
    monkeypatch.setattr(scanner, "get_comfy_models_folders", folders)
    monkeypatch.setattr(path_utils, "get_comfy_models_folders", folders)
    monkeypatch.setattr(
        scanner, "collect_models_files", lambda: [str(p) for p in sorted(models.iterdir())]
    )
    roots = ("models", "input", "output")
    files = _populate(drive)
    checkpoint = models / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    new_checkpoint = models / "added.safetensors"
    _scan(roots)
    edits = _customise(session)
    target = files[0]
    checkpoint.unlink()

    def back_and_add():
        _put_back(target, mtime_bump_ns=1_000_000_000)
        new_checkpoint.write_bytes(b"more weights")

    with _absent_for_the_check(target), caplog.at_level(logging.INFO):
        state = _scan(roots, after_check=back_and_add)

    assert (state.found_again, state.missing_marked) == (1, 1)
    assert _events(caplog, "seeder.marked_missing") == [
        {"count": "1", "root": "models", "stage": "fast_scan"}
    ]
    assert [row.is_missing for row in _rows(session, target)] == [False]
    assert [row.is_missing for row in _rows(session, checkpoint)] == [True]
    assert [row.is_missing for row in _rows(session, new_checkpoint)] == [False]
    records = _records(session)
    retired = next(rid for rid, rec in edits.items() if rec[0] == _rows(session, checkpoint)[0].id)
    assert records.pop(retired)[4] == edits.pop(retired)[4] | {"missing"}
    assert {rid: records.pop(rid) for rid in edits} == edits
    assert len(records) == 1  # only the added checkpoint is new


def test_temp_references_are_still_retired_at_once(temp_dir, session, monkeypatch):
    temp_root = temp_dir / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(scanner, "get_temp_prefixes", lambda: [str(temp_root)])
    path = temp_root / "preview.png"
    path.write_bytes(b"preview")
    from app.assets.database.queries.records import create_content, create_record

    st = path.stat()
    content = create_content(session, path=str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    create_record(session, content_id=content.id, name=path.name)
    session.commit()
    path.unlink()

    scanner.sync_temp_references_safely()

    assert [row.is_missing for row in _rows(session, path)] == [True]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        (0x400000, True),  # RECALL_ON_DATA_ACCESS: a dehydrated placeholder
        (0x80000 | 0x20, True),  # PINNED: "always keep on this device"
        (0x100000, True),  # UNPINNED
        (0x40000, True),  # RECALL_ON_OPEN
        (0x20, False),  # ARCHIVE alone: an ordinary file
        (0x400 | 0x20, False),  # a reparse point with no placeholder state (compressed, dedup)
        (None, False),  # no st_file_attributes: not Windows
    ],
)
def test_cloud_file_attributes(attributes, expected):
    stat_result = SimpleNamespace() if attributes is None else SimpleNamespace(st_file_attributes=attributes)
    assert scanner._is_cloud_file(stat_result) is expected


def test_a_found_again_cloud_file_is_counted(drive, session, monkeypatch):
    files = _populate(drive)
    _scan()
    target = files[0]
    real_stat = os.stat

    def windows_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.path.abspath(path) != str(target):
            return result
        return SimpleNamespace(
            st_size=result.st_size, st_mtime_ns=result.st_mtime_ns, st_file_attributes=0x80000
        )

    with _absent_for_the_check(target):
        state = _scan(
            after_check=lambda: _put_back(target, mtime_bump_ns=1_000_000_000),
            after_walk=lambda: monkeypatch.setattr(scanner.os, "stat", windows_stat),
        )
    monkeypatch.setattr(scanner.os, "stat", real_stat)

    assert (state.found_again, state.found_again_cloud) == (1, 1)



def test_a_gone_row_the_walk_does_not_list_is_not_stat_ed_again(drive, session, monkeypatch):
    # A share that is offline lists nothing, and each stat of it can take seconds.
    files = _populate(drive)
    _scan()
    target = files[0]
    target.unlink()
    real_stat = os.stat
    stats_after_walk: list[str] = []

    def counting(path, *args, **kwargs):
        stats_after_walk.append(os.path.abspath(path))
        return real_stat(path, *args, **kwargs)

    state = _scan(after_walk=lambda: monkeypatch.setattr(scanner.os, "stat", counting))
    monkeypatch.setattr(scanner.os, "stat", real_stat)

    assert state.missing_marked == 1
    assert str(target) not in stats_after_walk


def test_the_re_stat_counts_as_a_file_stat(drive, session):
    files = _populate(drive)
    _scan()
    baseline = _scan().files_statted

    with _absent_for_the_check(files[0]):
        state = _scan(after_check=lambda: _put_back(files[0], mtime_bump_ns=1_000_000_000))

    # The stored-path stat that missed it, plus the re-stat after the walk.
    assert state.files_statted == baseline + 1
