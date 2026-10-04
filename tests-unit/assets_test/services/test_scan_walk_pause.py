"""The fast scan's folder walk and its per-file stat and spec loops check the pause gate
for every directory and file, so a prompt that starts during them stops the scan at once
instead of after the whole library has been walked and stat'ed."""

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent, Base
from app.assets.scanner_admission import _WATCH_LIST, _two_stat_admit
from app.assets.services.file_utils import list_files_recursively

FILES = 6


class _Gate:
    """A fake ShouldStop: call ``block_at`` parks until released, calls from ``stop_at`` on return True."""

    def __init__(self, block_at: int | None = None, stop_at: int | None = None) -> None:
        self.calls = 0
        self.block_at = block_at
        self.stop_at = stop_at
        self.blocked = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> bool:
        self.calls += 1
        if self.calls == self.block_at:
            self.blocked.set()
            assert self.release.wait(5)
        return self.stop_at is not None and self.calls >= self.stop_at


class _Counts:
    dirs_listed = 0
    files_statted = 0

    def mark_emitted(self, key: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def clear_watch_list():
    _WATCH_LIST.clear()
    yield
    _WATCH_LIST.clear()


@pytest.fixture
def tree(temp_dir, monkeypatch) -> Path:
    """The output directory: one file in each of FILES subdirectories."""
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(temp_dir))
    for d in range(FILES):
        sub = temp_dir / f"d{d}"
        sub.mkdir()
        (sub / "a.png").write_bytes(b"x" * (d + 1))
    return temp_dir


def _in_thread(fn):
    result: list = []
    worker = threading.Thread(target=lambda: result.append(fn()), daemon=True)
    worker.start()
    return worker, result


def test_walk_blocks_on_pause_and_resumes_with_the_same_files(tree):
    expected = list_files_recursively(str(tree))
    gate = _Gate(block_at=3)
    counts = _Counts()

    worker, result = _in_thread(lambda: list_files_recursively(str(tree), counts, gate))
    assert gate.blocked.wait(5)
    time.sleep(0.1)
    assert counts.dirs_listed == 2  # parked before the third directory
    gate.release.set()
    worker.join(5)

    assert result == [expected]
    assert gate.calls == FILES + 1


def test_walk_stops_on_cancel(tree):
    counts = _Counts()
    gate = _Gate(stop_at=3)
    files = list_files_recursively(str(tree), counts, gate)
    assert counts.dirs_listed == 2
    assert gate.calls == 3  # stopped walking, not just skipping the rest
    assert len(files) < FILES


def _paths(tree: Path) -> list[str]:
    return sorted(list_files_recursively(str(tree)))


def _specs(tree: Path, **kwargs):
    return scanner.build_asset_specs(_paths(tree), set(), enable_metadata_extraction=False, **kwargs)


# Gate call numbers: the first stat loop makes calls 1..FILES, the second stat FILES+1..2*FILES,
# the spec loop 2*FILES+1..3*FILES.
@pytest.mark.parametrize("loop", ["first_stat", "second_stat", "spec"])
def test_spec_loops_block_on_pause_and_resume_with_the_same_specs(tree, loop):
    expected = _specs(tree)
    start = {"first_stat": 0, "second_stat": FILES, "spec": 2 * FILES}[loop]
    gate = _Gate(block_at=start + 3)
    counts = _Counts()
    named: list[str] = []
    real_name = scanner.get_name_and_tags_from_asset_path

    def name(path):
        named.append(path)
        return real_name(path)

    with patch("app.assets.scanner.get_name_and_tags_from_asset_path", name):
        worker, result = _in_thread(lambda: _specs(tree, progress=counts, should_stop=gate))
        assert gate.blocked.wait(5)
        time.sleep(0.1)
        # Parked before the third item of that loop.
        assert counts.files_statted == min(start, FILES) + (2 if loop != "spec" else FILES)
        assert len(named) == (2 if loop == "spec" else 0)
        gate.release.set()
        worker.join(5)

    assert result == [expected]
    assert gate.calls == 3 * FILES


@pytest.mark.parametrize("loop", ["first_stat", "second_stat", "spec"])
def test_spec_loops_return_nothing_on_cancel(tree, loop):
    start = {"first_stat": 0, "second_stat": FILES, "spec": 2 * FILES}[loop]
    counts = _Counts()
    specs, tags, _ = _specs(tree, progress=counts, should_stop=_Gate(stop_at=start + 3))
    assert (specs, tags) == ([], set())
    assert counts.files_statted == min(start, FILES) + (2 if loop != "spec" else FILES)


def test_second_stat_returns_nothing_on_cancel(tree):
    candidates = [(p, os.stat(p)) for p in _paths(tree)]
    assert _two_stat_admit(candidates, None, _Gate(stop_at=3)) == ([], [])


@pytest.fixture
def catalog():
    engine = sa.create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    @contextmanager
    def _create_session():
        with SASession(engine) as sess:
            yield sess

    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.assets.scanner.create_write_session", sessionmaker(bind=engine)):
        yield engine


class _HookedState(seeder_module._ScanState):
    """Calls ``on_count(name, value)`` as each directory or stat is counted, so a test can
    land a prompt part way through the walk or the stats."""

    on_count = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("dirs_listed", "files_statted") and self.on_count is not None:
            self.on_count(name, value)


@pytest.fixture
def scan(tree, catalog, monkeypatch, tmp_path):
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    instance = seeder_module._AssetSeeder()
    instance._state = seeder_module.State.RUNNING
    instance._scan_state = _HookedState()
    instance._phase = seeder_module.ScanPhase.FAST
    instance._run_gate.set()
    events: list[str] = []
    instance.set_event_sink(lambda kind, _data: events.append(kind))
    yield instance, events
    instance.cancel()  # frees a scan thread a failed assertion left parked


def _rows(engine) -> int:
    with SASession(engine) as sess:
        return sess.scalar(sa.select(sa.func.count()).select_from(AssetContent))


# Pause on the 3rd directory listed, or the 3rd and the 9th file stat'ed (the first and
# second stat loops; the reference sync stats nothing on an empty catalog).
@pytest.mark.parametrize("counter,at,parked_at", [
    ("dirs_listed", 3, (3, 0)),
    ("files_statted", 3, (FILES + 2, 3)),
    ("files_statted", FILES + 3, (FILES + 2, FILES + 3)),
])
def test_a_prompt_starting_mid_walk_or_stat_parks_the_scan(scan, catalog, counter, at, parked_at):
    instance, events = scan
    state = instance._scan_state
    paused = threading.Event()

    def on_count(name, n):
        if name == counter and n == at:
            assert instance.pause()
            paused.set()

    state.on_count = on_count
    worker, result = _in_thread(lambda: instance._run_fast_phase(("input", "output")))
    assert paused.wait(5)
    time.sleep(0.2)
    assert (state.dirs_listed, state.files_statted) == parked_at
    assert "assets.seed.paused" in events
    assert instance.resume()
    worker.join(5)

    assert result[0][0] == FILES
    assert _rows(catalog) == FILES
    assert state.paused_s > 0.1  # the pause was timed; margin for the worker's lead-in


def test_a_cancel_mid_walk_ends_the_scan_before_it_starts_seeding(scan, catalog):
    instance, events = scan
    state = instance._scan_state
    state.on_count = lambda name, n: name == "dirs_listed" and n == 3 and instance.cancel()

    assert instance._run_fast_phase(("input", "output")) == (0, 0, 0)
    assert state.dirs_listed == 3  # the walk itself stopped
    assert state.files_statted == 0
    assert "assets.seed.started" not in events
    assert state.cancel_stage == seeder_module._ScanStage.FAST_SCAN.value
    assert _rows(catalog) == 0
