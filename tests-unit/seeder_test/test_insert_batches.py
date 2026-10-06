"""The fast scan inserts in small batches with a pause after each, so a foreground write
(an upload, an output being registered) gets the write lock while it runs."""

import os
import threading
import time
from types import SimpleNamespace

import folder_paths
import pytest

from app.assets import mode
from app.assets import scanner as scanner_module
from app.assets import seeder as seeder_module
from app.assets.seeder import ScanPhase, State, _AssetSeeder, _ScanState
from app.assets.services import ingest


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch) -> _AssetSeeder:
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._scan_state = _ScanState()
    instance._roots = ("input",)
    instance._phase = ScanPhase.FAST
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    return instance


def _run_fast_phase(scan_seeder, monkeypatch, count, fail_batch=None):
    """Run the fast phase over ``count`` specs with insert_asset_specs stubbed out;
    returns the result, each batch's length, and the pauses taken."""
    batches: list[int] = []
    sleeps: list[float] = []

    def insert(batch, _tags, _progress):
        batches.append(len(batch))
        if len(batches) == fail_batch:
            raise RuntimeError("database is locked")
        return len(batch), None

    specs = [{"tags": []} for _ in range(count)]
    monkeypatch.setattr(seeder_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(seeder_module, "insert_asset_specs", insert)
    monkeypatch.setattr(seeder_module, "sync_root_safely", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(seeder_module, "collect_paths_for_roots", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(seeder_module, "build_asset_specs", lambda *_args, **_kwargs: (specs, set(), 0))
    monkeypatch.setattr(seeder_module, "tick_watch_list", lambda _progress=None: None)
    result = scan_seeder._run_fast_phase(("input",))
    return result, batches, sleeps


def test_inserts_go_in_batches_with_a_pause_after_all_but_the_last(scan_seeder, monkeypatch):
    (created, _, _), batches, sleeps = _run_fast_phase(scan_seeder, monkeypatch, 450)

    assert created == 450
    assert batches == [seeder_module.INSERT_BATCH_SIZE] * 2 + [50]
    assert sleeps == [seeder_module.INSERT_PAUSE_SECONDS] * 2


def test_a_failed_batch_is_still_followed_by_a_pause(scan_seeder, monkeypatch):
    # A batch that lost the lock to a foreground write must still let that write go first.
    (created, _, _), batches, sleeps = _run_fast_phase(scan_seeder, monkeypatch, 450, fail_batch=2)

    assert len(batches) == 3
    assert sleeps == [seeder_module.INSERT_PAUSE_SECONDS] * 2
    assert created == seeder_module.INSERT_BATCH_SIZE + 50
    assert scan_seeder._errors == [
        f"Batch insert encountered an error at offset {seeder_module.INSERT_BATCH_SIZE} after creating 0: database is locked"
    ]


def test_the_pause_outlasts_the_busy_handlers_longest_poll():
    # SQLite's default busy handler sleeps at most 100 ms between retries, and Windows'
    # default timer tick (15.6 ms) can stretch that sleep; a shorter pause can fall
    # between two retries, and a waiting write never sees the lock free.
    assert seeder_module.INSERT_PAUSE_SECONDS > 0.100 + 0.0156


# --- real threads, real WAL database ---------------------------------------------

# Per scanned file inside the write transaction: 500 files in one transaction, as before
# this change, hold the lock for over 5 s. The test's small batches keep each hold under
# ~1.5 s even on a slow CI runner.
_SLOW_RECORD_SECONDS = 0.01
_TEST_BATCH_SIZE = 20
# A foreground write that gets in at a pause waits about one batch plus a pause.
_FOREGROUND_WAIT_LIMIT_SECONDS = 4.0


@pytest.fixture
def hashing_off():
    mode.init(SimpleNamespace(enable_asset_hashing=False))
    yield
    mode.init(None)


def test_output_registration_gets_in_while_the_scan_inserts(hashing_off, file_db, tmp_path, monkeypatch):
    library = tmp_path / "input"
    library.mkdir()
    for i in range(600):
        (library / f"f{i}.png").write_bytes(b"x" * (i + 1))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(library))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    old = time.time() - 3600
    for path in library.iterdir():
        os.utime(path, (old, old))

    scanning = threading.Event()
    create_record = scanner_module.create_record
    records = []

    def slow_create_record(*args, **kwargs):
        records.append(1)
        # Partway into the second batch, so the writes contend with batches in progress.
        if len(records) == _TEST_BATCH_SIZE + 10:
            scanning.set()
        time.sleep(_SLOW_RECORD_SECONDS)
        return create_record(*args, **kwargs)

    monkeypatch.setattr(scanner_module, "create_record", slow_create_record)
    monkeypatch.setattr(seeder_module, "INSERT_BATCH_SIZE", _TEST_BATCH_SIZE)
    seeder = _AssetSeeder()
    assert seeder.start(roots=("input",), phase=ScanPhase.FAST)
    try:
        assert scanning.wait(30)
        waits = []
        for i in range(3):
            path = output / f"out{i}.png"
            path.write_bytes(b"output")
            started = time.perf_counter()
            registered = ingest.register_executed_output(str(path), job_id="job")
            waits.append(time.perf_counter() - started)
            assert registered is not None
        status = seeder.get_status()
    finally:
        seeder.cancel()
        stopped = seeder.wait(timeout=60)

    assert stopped
    assert status.state == State.RUNNING, "the scan ended before the foreground writes; nothing was contended"
    assert max(waits) < _FOREGROUND_WAIT_LIMIT_SECONDS, waits
    assert status.errors == []  # the scan's own batches got the lock too
