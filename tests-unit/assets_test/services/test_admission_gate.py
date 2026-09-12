import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.assets import scanner_admission
from app.assets.database.models import AssetContent
from app.assets.scanner import (
    _WATCH_LIST,
    _WatchEntry,
    _should_skip_extension,
    _two_stat_admit,
    tick_watch_list,
)


@pytest.fixture(autouse=True)
def clear_watch_list() -> Iterator[None]:
    _WATCH_LIST.clear()
    yield
    _WATCH_LIST.clear()


def test_part_suffix_never_admitted():
    assert _should_skip_extension("model.safetensors.part") is True


def test_size_drift_watch_listed(temp_dir: Path, monkeypatch):
    path = temp_dir / "download.bin"
    path.write_bytes(b"first")
    first_stat = path.stat()

    def changed_stat(_: float) -> None:
        path.write_bytes(b"changed")

    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", changed_stat)
    admitted, watched = _two_stat_admit([(str(path), first_stat)])

    assert admitted == []
    assert watched == [str(path)]


def test_watch_list_is_bounded_when_distinct_paths_keep_changing(temp_dir: Path, monkeypatch):
    candidates: list[tuple[str, os.stat_result]] = []
    for index in range(1000):
        path = temp_dir / f"changing-{index}.bin"
        path.write_bytes(b"before")
        candidates.append((str(path), path.stat()))
        path.write_bytes(b"after-change")
    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", lambda _: None)

    _two_stat_admit(candidates)

    assert scanner_admission._WATCH_LIST_MAX_SIZE == 256
    assert len(_WATCH_LIST) <= scanner_admission._WATCH_LIST_MAX_SIZE


def test_refreshing_watched_path_preserves_ticks_and_replaces_stat(temp_dir: Path, monkeypatch):
    path = temp_dir / "changing.bin"
    path.write_bytes(b"first")
    first_stat = path.stat()
    path.write_bytes(b"second-version")
    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", lambda _: None)
    _two_stat_admit([(str(path), first_stat)])
    _WATCH_LIST[0].ticks = 7
    refresh_first_stat = path.stat()
    path.write_bytes(b"third-version-is-longer")
    refreshed_stat = path.stat()

    _two_stat_admit([(str(path), refresh_first_stat)])

    assert len(_WATCH_LIST) == 1
    assert _WATCH_LIST[0].last_stat == refreshed_stat
    assert _WATCH_LIST[0].ticks == 7


def test_never_stabilizes_dropped_after_cap(session, temp_dir: Path):
    path = temp_dir / "moving.bin"
    path.write_bytes(b"0")
    _WATCH_LIST[:] = [_WatchEntry(str(path), path.stat())]
    for index in range(scanner_admission._WATCH_SCAN_RETRIES):
        path.write_bytes(str(index + 1).encode())
        tick_watch_list(session)

    assert _WATCH_LIST == []
    assert session.scalars(select(AssetContent)).all() == []


def test_stable_scan_admission_removes_watch_entry_before_next_tick(session, temp_dir: Path, monkeypatch):
    path = temp_dir / "stable.bin"
    path.write_bytes(b"complete")
    current_stat = path.stat()
    _WATCH_LIST[:] = [_WatchEntry(str(path), current_stat, ticks=4)]
    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", lambda _: None)

    admitted, watched = _two_stat_admit([(str(path), current_stat)])
    entries_after_admission = len(_WATCH_LIST)
    with (
        patch("app.assets.scanner_admission.compute_loader_path", return_value="stable.bin"),
        patch(
            "app.assets.scanner_admission.get_name_and_tags_from_asset_path",
            return_value=("stable.bin", []),
        ),
        patch("app.assets.scanner.seed_asset_specs") as seed_asset_specs,
    ):
        tick_watch_list(session)

    assert admitted == [str(path)]
    assert watched == []
    assert entries_after_admission == 0
    seed_asset_specs.assert_not_called()


def test_evicted_path_is_admitted_by_later_stable_scan(temp_dir: Path, monkeypatch):
    monkeypatch.setattr(scanner_admission, "_WATCH_LIST_MAX_SIZE", 2, raising=False)
    monkeypatch.setattr("app.assets.scanner_admission.time.sleep", lambda _: None)
    paths: list[Path] = []
    candidates: list[tuple[str, os.stat_result]] = []
    for index in range(3):
        path = temp_dir / f"overflow-{index}.bin"
        path.write_bytes(b"before")
        candidates.append((str(path), path.stat()))
        path.write_bytes(b"after-change")
        paths.append(path)
    _two_stat_admit(candidates)
    entries_after_overflow = [entry.path for entry in _WATCH_LIST]
    stable_stat = paths[0].stat()

    admitted, watched = _two_stat_admit([(str(paths[0]), stable_stat)])

    assert entries_after_overflow == [str(paths[1]), str(paths[2])]
    assert admitted == [str(paths[0])]
    assert watched == []


def test_empty_candidate_batch_returns_without_paying_stability_gap(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "app.assets.scanner_admission.time.sleep", lambda seconds: sleeps.append(seconds)
    )

    admitted, watched = _two_stat_admit([])

    assert admitted == []
    assert watched == []
    assert sleeps == []


def test_nonempty_candidate_batch_still_pays_stability_gap(temp_dir: Path, monkeypatch):
    path = temp_dir / "stable.bin"
    path.write_bytes(b"complete")
    first_stat = path.stat()
    sleeps: list[float] = []
    monkeypatch.setattr(
        "app.assets.scanner_admission.time.sleep", lambda seconds: sleeps.append(seconds)
    )

    admitted, watched = _two_stat_admit([(str(path), first_stat)])

    assert sleeps == [0.1]
    assert admitted == [str(path)]
    assert watched == []
