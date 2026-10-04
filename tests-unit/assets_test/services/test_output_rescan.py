"""Output-only rescans diff directory listings against the catalog instead of stat'ing
every live row. Run through the seeder's real fast phase on an in-memory catalog."""

import logging
import os
import re
import shutil
import stat as stat_module
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.services import file_utils

OUTPUT_ONLY = ("output",)
ALL_ROOTS = ("models", "input", "output")
N_FILES = 30


@pytest.fixture
def roots(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dirs = {name: temp_dir / name for name in ("input", "output")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(dirs["output"]))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(dirs["input"]))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return dirs


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


def _scan(roots=OUTPUT_ONLY) -> tuple[int, int, int]:
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    return seeder._run_fast_phase(roots)


def _write(path: Path, payload: bytes = b"png-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _warm_catalog(output: Path) -> list[Path]:
    """N files over the root and two levels of nesting."""
    subdirs = [output, output / "a", output / "b" / "c"]
    return [_write(subdirs[i % 3] / f"img_{i:03d}.png") for i in range(N_FILES)]


def _rows(session, path: Path) -> list[AssetContent]:
    session.expire_all()
    return list(session.scalars(sa.select(AssetContent).where(AssetContent.path == str(path))))


def _live_paths(session) -> set[str]:
    session.expire_all()
    return set(session.scalars(sa.select(AssetContent.path).where(AssetContent.is_missing.is_(False))))


_LISTING_LINE = re.compile(
    r"output listing: (\d+) dirs listed, (\d+) rows retired, (\d+) rows skipped"
)


def _listing_counts(caplog: pytest.LogCaptureFixture) -> tuple[int, int, int]:
    """(dirs listed, rows retired, rows skipped) from the one rescan in ``caplog``."""
    matches = [_LISTING_LINE.search(r.getMessage()) for r in caplog.records]
    found = [m for m in matches if m]
    assert len(found) == 1, [r.getMessage() for r in caplog.records]
    listed, retired, skipped = (int(g) for g in found[0].groups())
    return listed, retired, skipped


def _scan_logged(caplog: pytest.LogCaptureFixture, roots=OUTPUT_ONLY) -> tuple[int, int, int]:
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        return _scan(roots)


def test_output_rescan_catalogs_new_files_and_retires_deleted_ones(roots, session, caplog):
    output = roots["output"]
    files = _warm_catalog(output)
    _scan()
    assert _live_paths(session) == {str(p) for p in files}

    deleted = files[:10]
    for path in deleted:
        path.unlink()
    added = [_write(output / ("a" if i % 2 else "b/c") / f"new_{i}.png") for i in range(10)]

    created, _skipped, total = _scan_logged(caplog)

    assert created == 10
    assert total == N_FILES
    assert _live_paths(session) == {str(p) for p in files[10:] + added}
    for path in deleted:
        (row,) = _rows(session, path)
        assert row.is_missing is True
    _listed, retired, skipped = _listing_counts(caplog)
    assert (retired, skipped) == (10, 0)


def test_in_place_overwrite_is_not_detected_by_an_output_only_rescan(roots, session):
    """The documented, accepted gap: without a per-row stat, a same-path overwrite is
    invisible to an output-only rescan until the next scan that stats."""
    target = _warm_catalog(roots["output"])[0]
    _scan()
    (before,) = _rows(session, target)

    _write(target, b"a different, longer payload")
    _scan()

    (after,) = _rows(session, target)
    assert after.id == before.id
    assert after.is_missing is False
    assert after.size_bytes == before.size_bytes != target.stat().st_size


def test_a_scan_that_is_not_output_only_still_stats_every_row(roots, session, caplog):
    files = _warm_catalog(roots["output"])
    _write(roots["input"] / "in.png")
    _scan(ALL_ROOTS)
    (before,) = _rows(session, files[0])

    _write(files[0], b"a different, longer payload")
    files[1].unlink()
    _scan_logged(caplog, ALL_ROOTS)

    assert not [r for r in caplog.records if _LISTING_LINE.search(r.getMessage())]
    # The stat path ran: the overwrite split the row, the deletion retired its row.
    assert _rows(session, files[1])[0].is_missing is True
    rows = _rows(session, files[0])
    assert {row.id: row.is_missing for row in rows}[before.id] is True
    assert len(rows) == 2


def _catalog_directly(session, path: Path) -> str:
    """A live row the walk would never produce itself."""
    stat = path.stat()
    content = create_content(session, str(path), None, stat.st_size, stat.st_mtime_ns)
    create_record(session, content.id, path.name)
    session.commit()
    return content.id


def test_row_whose_stored_spelling_differs_from_the_entry_stays_live(roots, session, monkeypatch):
    # A node that saves to "Portraits/" reuses an existing "portraits/" on NTFS or APFS and
    # reports its own spelling, so the row's path never matches the listing verbatim.
    real = _write(roots["output"] / "portraits" / "img.png")
    respelled = roots["output"] / "Portraits" / "img.png"
    stat = real.stat()
    content = create_content(session, str(respelled), None, stat.st_size, stat.st_mtime_ns)
    create_record(session, content.id, respelled.name)
    session.commit()
    # Resolve either spelling, as a case-insensitive filesystem does, on any host.
    real_stat = os.stat
    monkeypatch.setattr(
        os, "stat",
        lambda p, *a, **k: real_stat(os.fspath(p).replace(str(respelled.parent), str(real.parent)), *a, **k),
    )

    _scan()

    assert session.get(AssetContent, content.id).is_missing is False


def test_row_the_listing_misses_but_cannot_be_statted_stays_live(roots, session, monkeypatch):
    real = _write(roots["output"] / "portraits" / "img.png")
    respelled = roots["output"] / "Portraits" / "img.png"
    stat = real.stat()
    content = create_content(session, str(respelled), None, stat.st_size, stat.st_mtime_ns)
    create_record(session, content.id, respelled.name)
    session.commit()
    real_stat = os.stat

    def stat_denying_the_respelled_path(path, *args, **kwargs):
        if os.fspath(path) == str(respelled):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_denying_the_respelled_path)

    _scan()

    assert session.get(AssetContent, content.id).is_missing is False


def test_rows_under_hidden_paths_stay_live(roots, session, caplog):
    output = roots["output"]
    _warm_catalog(output)
    hidden = [_write(output / ".cache" / "x.png"), _write(output / "a" / ".dotfile.png")]
    for path in hidden:
        _catalog_directly(session, path)

    _scan_logged(caplog)

    for path in hidden:
        (row,) = _rows(session, path)
        assert row.is_missing is False
    _listed, retired, skipped = _listing_counts(caplog)
    assert (retired, skipped) == (0, 2)


def test_dir_failing_to_list_stats_its_rows_while_siblings_are_diffed(
    roots, session, monkeypatch, caplog
):
    output = roots["output"]
    files = _warm_catalog(output)
    _scan()
    in_failing = [p for p in files if p.parent == output / "a"]
    in_sibling = [p for p in files if p.parent == output / "b" / "c"]
    in_failing[0].unlink()
    in_sibling[0].unlink()

    real_list = file_utils._list_visible_entries

    def failing_list(dirpath: str, should_stop=None):
        if dirpath == str(output / "a"):
            raise OSError("transient listing failure")
        return real_list(dirpath, should_stop)

    monkeypatch.setattr(file_utils, "_list_visible_entries", failing_list)
    _scan_logged(caplog)

    for path in in_failing:
        (row,) = _rows(session, path)
        assert row.is_missing is (path == in_failing[0])
    (gone,) = _rows(session, in_sibling[0])
    assert gone.is_missing is True
    _listed, retired, skipped = _listing_counts(caplog)
    assert (retired, skipped) == (2, len(in_failing) - 1)


def test_deleted_output_under_a_hidden_path_is_retired(roots, session):
    # SaveImage accepts prefixes like "foo/.bar/img" (hidden dir) and "foo/.bar"
    # (hidden file), and the node registers the result like any other output.
    output = roots["output"]
    _warm_catalog(output)
    hidden = [_write(output / "foo" / ".bar" / "img.png"), _write(output / "foo" / ".bar_00001_.png")]
    for path in hidden:
        _catalog_directly(session, path)
    _scan()
    for path in hidden:
        path.unlink()

    _scan()

    for path in hidden:
        (row,) = _rows(session, path)
        assert row.is_missing is True


def test_unlistable_output_root_stats_every_row(roots, session, monkeypatch):
    output = roots["output"]
    files = _warm_catalog(output)
    _scan()
    files[0].unlink()
    real_list = file_utils._list_visible_entries

    def failing_root(dirpath: str, should_stop=None):
        if dirpath == str(output):
            raise OSError("root unreadable")
        return real_list(dirpath, should_stop)

    monkeypatch.setattr(file_utils, "_list_visible_entries", failing_root)
    _scan()

    assert _live_paths(session) == {str(p) for p in files[1:]}


def test_row_under_an_unvisited_symlink_alias_is_stat_checked(roots, session):
    output = roots["output"]
    real = _write(output / "real" / "x.png")
    (output / "alias").symlink_to(output / "real")
    _scan()
    (cataloged,) = _live_paths(session)
    # The cycle guard walks the directory once, under whichever name came first.
    visited, unvisited = (
        (real, output / "alias" / "x.png")
        if cataloged == str(real)
        else (output / "alias" / "x.png", real)
    )
    alias_id = _catalog_directly(session, unvisited)

    _scan()

    assert session.get(AssetContent, alias_id).is_missing is False
    assert _live_paths(session) == {str(visited), str(unvisited)}

    real.unlink()
    _scan()

    assert session.get(AssetContent, alias_id).is_missing is True
    assert _live_paths(session) == set()


DEPTH = 20


def _deep_tree(output: Path) -> tuple[list[Path], list[Path]]:
    """A chain of DEPTH nested dirs under output with one file per level, plus
    three unchanged sibling dirs at the root. Returns (dirs, files), dirs[0] the root."""
    dirs = [output]
    for level in range(1, DEPTH + 1):
        dirs.append(dirs[-1] / f"d{level:02d}")
    dirs += [output / f"sib{i}" for i in range(3)]
    files = [_write(d / f"f_{i:02d}.png") for i, d in enumerate(dirs)]
    return dirs, files


@pytest.fixture
def dir_stats(monkeypatch: pytest.MonkeyPatch):
    """Counts os.stat calls per directory path; files are not counted."""
    counts: dict[str, int] = {}
    real_stat = os.stat

    def counting_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if stat_module.S_ISDIR(result.st_mode):
            counts[os.fspath(path)] = counts.get(os.fspath(path), 0) + 1
        return result

    monkeypatch.setattr(os, "stat", counting_stat)
    return counts


def test_deep_unreported_add_is_admitted(roots, session):
    dirs, files = _deep_tree(roots["output"])
    _scan()
    added = _write(dirs[DEPTH] / "deep_new.png")

    created, _skipped, _total = _scan()

    assert created == 1
    assert _live_paths(session) == {str(p) for p in files + [added]}


def test_deep_deletions_mark_exactly_those_rows_missing(roots, session, caplog):
    _dirs, files = _deep_tree(roots["output"])
    _scan()
    deleted = [files[10], files[15], files[20]]  # depths 10, 15 and 20, one per dir
    for path in deleted:
        path.unlink()

    _scan_logged(caplog)

    session.expire_all()
    missing = set(session.scalars(sa.select(AssetContent.path).where(AssetContent.is_missing.is_(True))))
    assert missing == {str(p) for p in deleted}
    assert _live_paths(session) == {str(p) for p in files if p not in deleted}
    _listed, retired, skipped = _listing_counts(caplog)
    assert (retired, skipped) == (3, 0)


def test_removing_a_whole_dir_retires_every_row_beneath_it(roots, session, caplog):
    dirs, files = _deep_tree(roots["output"])
    _scan()
    shutil.rmtree(dirs[5])  # d05 and the 15 levels under it, one file each

    _scan_logged(caplog)

    gone = {str(p) for p in files[5:DEPTH + 1]}
    assert _live_paths(session) == {str(p) for p in files} - gone
    _listed, retired, skipped = _listing_counts(caplog)
    assert (retired, skipped) == (len(gone), 0)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as e:  # Windows without the symlink privilege
        pytest.skip(f"cannot create symlinks here: {e}")


def test_symlink_whose_target_is_removed_is_retired(roots, temp_dir, session):
    target = _write(temp_dir / "elsewhere" / "t.png")
    link = roots["output"] / "linked.png"
    _symlink_or_skip(link, target)
    kept = _write(roots["output"] / "kept.png")
    _scan()
    assert _live_paths(session) == {str(link), str(kept)}

    target.unlink()
    _scan()

    assert _live_paths(session) == {str(kept)}


def test_every_dir_is_stated_and_listed_once_per_rescan(roots, caplog, dir_stats):
    dirs, _files = _deep_tree(roots["output"])
    _scan()

    for _ in range(2):
        dir_stats.clear()
        _scan_logged(caplog)
        assert dir_stats == {str(d): 1 for d in dirs}
        assert _listing_counts(caplog)[0] == len(dirs)


def test_the_rescan_yields_the_gil_per_dir_entry_and_row(roots, monkeypatch):
    dirs, files = _deep_tree(roots["output"])
    _scan()
    walk_yields: list[None] = []
    row_yields: list[None] = []
    monkeypatch.setattr(file_utils, "yield_gil", lambda run=None: run == file_utils.RESCAN_YIELD_RUN and walk_yields.append(None))
    monkeypatch.setattr(scanner, "yield_gil", lambda run=None: run == file_utils.RESCAN_YIELD_RUN and row_yields.append(None))

    _scan()

    entries = len(files) + len(dirs) - 1  # every file, and every dir but the root
    # once per dir walked, per entry listed, and per file path built
    assert len(walk_yields) == len(dirs) + entries + len(files)
    assert len(row_yields) == 2 * len(files)  # reading the live rows, then diffing them


def test_output_rescan_counts_its_listings_and_row_stats(roots, session):
    files = _warm_catalog(roots["output"])
    _scan()
    files[0].unlink()
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST

    seeder._run_fast_phase(OUTPUT_ONLY)

    assert state.dirs_listed == 4  # output, a, b, b/c
    assert state.files_statted == 1  # only the row its listing lacks is stat'ed
