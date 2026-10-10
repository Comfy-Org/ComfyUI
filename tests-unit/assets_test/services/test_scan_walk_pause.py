"""The fast scan's folder walk and its per-file stat and spec loops check the pause gate
for every directory entry and file, so a prompt that starts during them stops the scan at
once instead of after the whole library (or one huge folder) has been listed and stat'ed."""

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
from app.assets.services import file_utils
from app.assets.services.file_utils import walk_listings

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


class _ScandirCounter:
    """Stands in for file_utils' os module, counting the directory entries scandir hands
    out (calling ``on_entry(count)`` as each one is), and how many directories are open."""

    def __init__(self) -> None:
        self.entries = 0
        self.open = 0
        self.on_entry = None

    def __getattr__(self, name):
        return getattr(os, name)

    def scandir(self, path):
        counter, real = self, os.scandir(path)
        counter.open += 1

        class _Entries:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                real.close()
                counter.open -= 1

            def __iter__(self):
                return self

            def __next__(self):
                entry = next(real)
                counter.entries += 1
                if counter.on_entry is not None:
                    counter.on_entry(counter.entries)
                return entry

        return _Entries()


@pytest.fixture
def entries(monkeypatch) -> _ScandirCounter:
    counter = _ScandirCounter()
    monkeypatch.setattr(file_utils, "os", counter)
    return counter


@pytest.fixture
def flat(temp_dir) -> Path:
    """One folder of 10 files."""
    for i in range(10):
        (temp_dir / f"f{i}.png").write_bytes(b"x")
    return temp_dir


class _Pause:
    """The seeder's gate in miniature: ``interrupted`` is the non-blocking "pause or cancel
    requested?", ``should_stop`` blocks while paused and returns True once cancelled."""

    def __init__(self) -> None:
        self.requested = False
        self.cancelled = False
        self.interrupted_calls = 0
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.parked_after: list[int] = []  # probe() at each pause honoured
        self.probe = lambda: 0

    def interrupted(self) -> bool:
        self.interrupted_calls += 1
        return self.requested or self.cancelled

    def is_cancelled(self) -> bool:
        return self.cancelled

    def walk(self, base: Path):
        return walk_listings(str(base), self.should_stop, self.interrupted, self.is_cancelled)

    def should_stop(self) -> bool:
        if self.requested and not self.cancelled:
            self.parked_after.append(self.probe())
            self.blocked.set()
            assert self.release.wait(5)
            self.requested = False
        return self.cancelled


def test_walk_closes_the_folder_while_paused_part_way_and_lists_it_again(flat, entries):
    expected = walk_listings(str(flat))
    entries.entries = 0
    pause = _Pause()
    entries.on_entry = lambda n: n == 4 and setattr(pause, "requested", True)

    worker, result = _in_thread(lambda: pause.walk(flat))
    assert pause.blocked.wait(5)
    time.sleep(0.1)
    assert entries.entries == 4  # stopped on the 4th entry
    assert entries.open == 0  # nothing held open while paused
    calls_when_parked = pause.interrupted_calls
    pause.release.set()
    worker.join(5)

    assert result == [expected]
    assert entries.entries == 4 + 10  # the folder was read again from the start
    assert pause.interrupted_calls == calls_when_parked  # in one go, so it can't restart forever


def test_walk_abandons_a_folder_cancelled_part_way(flat, entries):
    pause = _Pause()
    entries.on_entry = lambda n: n == 4 and setattr(pause, "cancelled", True)
    walk = pause.walk(flat)
    assert entries.entries == 4  # stopped reading, not just skipping the rest
    assert (walk.files, walk.listings) == ([], {})  # the half-read folder vouches for nothing


def test_a_pause_during_the_second_listing_waits_for_it_but_a_cancel_does_not(flat, entries):
    """The trade-off that keeps a slow folder from restarting forever: once a pause has
    interrupted a folder, its second listing goes on through another pause, but a cancel
    still stops it part way."""
    pause = _Pause()

    def on_entry(n):
        if n == 4:
            pause.requested = True  # the first pause: the folder is closed, then listed again
        elif n == 4 + 2:
            pause.requested = True  # lands during the second listing: not honoured until it ends
        elif n == 4 + 6:
            pause.cancelled = True  # a cancel is

    entries.on_entry = on_entry
    pause.release.set()  # let the first pause through at once
    walk = pause.walk(flat)
    assert entries.entries == 4 + 6  # the second listing stopped on the cancel, not at its end (4 + 10)
    assert (walk.files, walk.listings) == ([], {})


def test_a_pause_during_the_second_listing_takes_effect_when_it_ends(flat, entries):
    (flat / "sub").mkdir()
    (flat / "sub" / "s.png").write_bytes(b"x")
    pause = _Pause()
    pause.release.set()
    pause.probe = lambda: entries.entries
    entries.on_entry = lambda n: n in (4, 4 + 2) and setattr(pause, "requested", True)
    walk = pause.walk(flat)
    # The second pause: as soon as the folder's 11 entries were listed, before "sub".
    assert pause.parked_after == [4, 4 + 11]
    assert len(walk.files) == 11


def test_a_subfolder_interrupted_twice_is_listed_in_walk_order(tree, entries):
    """Only the folder a pause interrupted is listed again without pause checks: a later
    folder still stops part way, and the walk keeps os.walk's order."""
    for d in range(FILES):  # six entries in every subfolder, whichever order scandir gives
        for i in range(5):
            (tree / f"d{d}" / f"more{i}.png").write_bytes(b"x")
    expected = walk_listings(str(tree))
    entries.entries = 0
    pause = _Pause()
    pause.release.set()
    # The root's FILES entries; 3 of the 1st subfolder before the pause, then all 6 again;
    # then 3 into the 2nd subfolder.
    first, second = FILES + 3, FILES + 3 + 6 + 3
    pause.probe = lambda: entries.entries
    entries.on_entry = lambda n: n in (first, second) and setattr(pause, "requested", True)
    walk = pause.walk(tree)
    assert walk == expected
    # Each pause took effect on the entry it landed on, not at the end of its folder.
    assert pause.parked_after == [first, second]


def test_a_folder_renamed_while_the_walk_is_paused_in_it_is_skipped(flat, entries):
    """Nothing is held open while paused, so the folder can be renamed; the walk then
    finds it gone and skips it rather than failing."""
    pause = _Pause()
    entries.on_entry = lambda n: n == 4 and setattr(pause, "requested", True)
    sub = flat / "inner"
    sub.mkdir()
    for i in range(10):
        (sub / f"g{i}.png").write_bytes(b"x")
    renamed = flat.parent / (flat.name + "-renamed-inner")

    worker, result = _in_thread(lambda: pause.walk(sub))
    try:
        assert pause.blocked.wait(5)
        os.rename(sub, renamed)  # would fail on Windows with the directory still open
        pause.release.set()
        worker.join(5)
        assert result[0].files == [] and result[0].listings == {}
    finally:
        pause.release.set()
        if renamed.exists():
            os.rename(renamed, sub)


# should_stop is called once before each directory.
def test_walk_parks_between_folders_and_resumes_with_the_same_listing(tree, entries):
    expected = walk_listings(str(tree))
    entries.entries = 0
    gate = _Gate(block_at=3)  # the root, a subfolder, then the next

    worker, result = _in_thread(lambda: walk_listings(str(tree), gate))
    assert gate.blocked.wait(5)
    time.sleep(0.1)
    assert entries.entries == FILES + 1
    gate.release.set()
    worker.join(5)

    assert result == [expected]


def test_walk_cancelled_between_folders_keeps_only_whole_listings(tree, entries):
    walk = walk_listings(str(tree), _Gate(stop_at=3))
    assert entries.entries == FILES + 1  # no entry of the next folder read
    [first] = set(walk.listings) - {str(tree)}  # whichever subfolder scandir gave first
    assert str(tree) in walk.listings
    assert walk.files == [os.path.join(first, "a.png")]
    assert walk.dirs_listed == 2


def _paths(tree: Path) -> list[str]:
    return sorted(walk_listings(str(tree)).files)


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
    """Calls ``on_count(name, value)`` as each file stat is counted, so a test can land a
    prompt part way through the stats."""

    on_count = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "files_statted" and self.on_count is not None:
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


@pytest.fixture
def hooked(scan, entries):
    """Lets a test land a prompt on the n-th directory entry read, or the n-th file stat'ed."""
    instance, _events = scan
    state = instance._scan_state

    def on(counter, at, action):
        if counter == "entries":
            entries.on_entry = lambda n: n == at and action()
        else:
            state.on_count = lambda name, n: name == counter and n == at and action()

    return on


# Pause on the 3rd entry the output walk reads (after the empty input root's one listing),
# or the 3rd and the 9th file stat'ed (the first and second stat loops; the reference sync
# stats nothing on an empty catalog).
@pytest.mark.parametrize("counter,at,parked_at", [
    ("entries", 3, (1, 0, 3)),
    ("files_statted", 3, (FILES + 2, 3, 2 * FILES)),
    ("files_statted", FILES + 3, (FILES + 2, FILES + 3, 2 * FILES)),
])
def test_a_prompt_starting_mid_walk_or_stat_parks_the_scan(
    scan, catalog, entries, hooked, counter, at, parked_at
):
    instance, events = scan
    state = instance._scan_state
    parked = threading.Event()

    def sink(kind, _data):
        events.append(kind)
        if kind == "assets.seed.paused":
            parked.set()

    instance.set_event_sink(sink)
    hooked(counter, at, instance.pause)
    worker, result = _in_thread(lambda: instance._run_fast_phase(("input", "output")))
    assert parked.wait(5)
    time.sleep(0.2)
    assert (state.dirs_listed, state.files_statted, entries.entries) == parked_at
    assert entries.open == 0  # no directory held open while paused
    assert instance.resume()
    worker.join(5)

    assert result[0][0] == FILES
    assert _rows(catalog) == FILES
    assert state.paused_s > 0.1  # the pause was timed; margin for the worker's lead-in


def test_a_cancel_mid_walk_ends_the_scan_before_it_starts_seeding(scan, catalog, entries, hooked):
    instance, events = scan
    state = instance._scan_state
    hooked("entries", 3, instance.cancel)

    assert instance._run_fast_phase(("input", "output")) == (0, 0, 0)
    assert entries.entries == 3  # no entry read after the cancel
    assert state.dirs_listed == 1  # only the input root's listing completed
    assert state.files_statted == 0
    assert "assets.seed.started" not in events
    assert state.cancel_stage == seeder_module._ScanStage.FAST_SCAN.value
    assert _rows(catalog) == 0


def test_a_cancel_during_the_second_listing_after_a_pause_stops_it(scan, catalog, entries):
    """After a pause closes a folder part way, the walk lists it again without pause
    checks, but a cancel (shutdown) must still stop it part way."""
    instance, events = scan
    parked = threading.Event()

    def sink(kind, _data):
        events.append(kind)
        if kind == "assets.seed.paused":
            parked.set()

    instance.set_event_sink(sink)

    def on_entry(n):
        if n == 3:  # the output root, part way: pause
            assert instance.pause()
        elif n == 3 + 3:  # part way through listing it again: cancel
            assert instance.cancel()

    entries.on_entry = on_entry
    worker, result = _in_thread(lambda: instance._run_fast_phase(("input", "output")))
    assert parked.wait(5)
    assert instance.resume()
    worker.join(5)

    assert result == [(0, 0, 0)]
    assert entries.entries == 3 + 3  # stopped on the cancel, not after the whole folder (3 + FILES)
    assert _rows(catalog) == 0


def test_a_second_prompt_during_the_second_listing_lets_it_finish(scan, catalog, entries):
    """The fast scan passes only a cancel check to a folder's second listing, so a folder
    slower to list than the gap between prompts still gets listed."""
    instance, events = scan
    parked = threading.Event()

    def sink(kind, _data):
        events.append(kind)
        if kind == "assets.seed.paused":
            parked.set()

    instance.set_event_sink(sink)
    entries.on_entry = lambda n: n in (3, 3 + 2) and instance.pause()
    worker, result = _in_thread(lambda: instance._run_fast_phase(("input", "output")))
    assert parked.wait(5)
    assert entries.entries == 3
    parked.clear()
    assert instance.resume()
    assert parked.wait(5)
    time.sleep(0.1)
    assert entries.entries == 3 + FILES  # the second pause waited for the whole folder
    assert instance.resume()
    worker.join(5)
    assert result[0][0] == FILES


def _live_rows(engine) -> int:
    with SASession(engine) as sess:
        return sess.scalar(
            sa.select(sa.func.count()).select_from(AssetContent).where(AssetContent.is_missing.is_(False))
        )


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_a_prompt_or_cancel_mid_rescan_listing_marks_nothing_missing(
    scan, catalog, entries, hooked, monkeypatch, action
):
    """The output-only rescan retires rows its listings lack, so a listing cut short must
    never read as files having vanished. The stat that normally double-checks a row before
    retiring it is made to say "gone", so only the listing decides."""
    instance, events = scan
    state = instance._scan_state
    assert instance._run_fast_phase(("input", "output"))[0] == FILES
    monkeypatch.setattr(scanner, "_is_gone", lambda _path: True)
    parked = threading.Event()

    def sink(kind, _data):
        events.append(kind)
        if kind == "assets.seed.paused":
            parked.set()

    instance.set_event_sink(sink)
    entries.entries = 0
    hooked("entries", 3, getattr(instance, action))

    worker, result = _in_thread(lambda: instance._run_fast_phase(("output",)))
    if action == "pause":
        assert parked.wait(5)
        time.sleep(0.1)
        assert entries.entries == 3  # parked mid-listing
        assert entries.open == 0
        assert instance.resume()
    worker.join(5)

    assert result[0] == (0, FILES if action == "pause" else 0, FILES if action == "pause" else 0)
    assert state.missing_marked == 0
    assert _live_rows(catalog) == FILES
