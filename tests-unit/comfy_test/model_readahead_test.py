import time

import pytest

import comfy.model_readahead as readahead


@pytest.fixture(autouse=True)
def small_limits(monkeypatch):
    monkeypatch.setattr(readahead, "_MIN_FILE_BYTES", 1)
    monkeypatch.setattr(readahead, "_CHUNK_BYTES", 64 * 1024)
    monkeypatch.setattr(readahead, "_BASE_RESERVE_BYTES", 0)
    monkeypatch.setattr(readahead, "_MODEL_RESERVE_RATIO", 0.0)
    monkeypatch.setattr(readahead, "_MODEL_RESERVE_MIN_BYTES", 0)
    monkeypatch.setattr(readahead, "_MODEL_RESERVE_MAX_BYTES", 0)
    monkeypatch.setattr(readahead, "_MIN_BUDGET_BYTES", 1)
    monkeypatch.setattr(readahead, "_STOP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(readahead, "_PROCESS_WAIT_S", 0.2)
    monkeypatch.setattr(readahead, "_MONITOR_INTERVAL_S", 0.01)
    monkeypatch.setattr(readahead, "_available_memory", lambda: 1024 * 1024 * 1024)


def make_file(tmp_path, name="model.safetensors", size=4 * 1024 * 1024):
    path = tmp_path / name
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def wait_for(reader, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    with reader._cv:
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            reader._cv.wait(timeout=min(0.02, remaining))
    return True


def test_candidate_filters_extension_size_and_missing_file(tmp_path):
    good = make_file(tmp_path, "good.safetensors", 1024)
    bad_ext = make_file(tmp_path, "bad.bin", 1024)

    assert readahead._SequentialReadAhead._candidate(str(good)) is not None
    assert readahead._SequentialReadAhead._candidate(str(bad_ext)) is None
    assert readahead._SequentialReadAhead._candidate(str(tmp_path / "missing.safetensors")) is None

    readahead._MIN_FILE_BYTES = 2048
    assert readahead._SequentialReadAhead._candidate(str(good)) is None


def test_plan_respects_ram_reserve_and_minimum_budget(monkeypatch):
    monkeypatch.setattr(readahead, "_available_memory", lambda: None)
    assert readahead._SequentialReadAhead._plan(1024) is None

    monkeypatch.setattr(readahead, "_available_memory", lambda: 100)
    readahead._BASE_RESERVE_BYTES = 100
    assert readahead._SequentialReadAhead._plan(1024) is None

    readahead._BASE_RESERVE_BYTES = 32
    readahead._CHUNK_BYTES = 16
    readahead._MIN_BUDGET_BYTES = 1
    budget, reserve, available = readahead._SequentialReadAhead._plan(1024)
    assert (budget, reserve, available) == (64, 32, 100)


def test_sequential_read_completes_and_worker_shuts_down(tmp_path):
    path = make_file(tmp_path, size=2 * 1024 * 1024)
    reader = readahead._SequentialReadAhead()
    try:
        assert reader.request(str(path))
        assert wait_for(reader, lambda: reader._active is None and reader._pending is None)
    finally:
        assert reader.shutdown()
    assert not reader._worker_thread.is_alive()


def test_stop_for_sampling_terminates_child(tmp_path):
    path = make_file(tmp_path)
    reader = readahead._SequentialReadAhead()
    reader._CHILD_READER_CODE = "import time; time.sleep(10)"
    try:
        assert reader.request(str(path))
        assert wait_for(reader, lambda: reader._active_process is not None)
        with reader._cv:
            process = reader._active_process
        assert process is not None
        assert reader.stop_for_sampling()
        assert process.poll() is not None
        assert wait_for(reader, lambda: reader._active is None)
    finally:
        reader.shutdown()


def test_newer_request_cancels_previous_child(tmp_path):
    first = make_file(tmp_path, "first.safetensors")
    second = make_file(tmp_path, "second.safetensors")
    reader = readahead._SequentialReadAhead()
    reader._CHILD_READER_CODE = (
        "import sys,time; "
        "time.sleep(10) if sys.argv[1].endswith('first.safetensors') else None; "
        "print(1, flush=True)"
    )
    try:
        assert reader.request(str(first))
        assert wait_for(reader, lambda: reader._active_process is not None)
        with reader._cv:
            first_process = reader._active_process
        assert first_process is not None

        assert reader.request(str(second))
        assert wait_for(reader, lambda: reader._active is None and reader._pending is None)
        assert first_process.poll() is not None
    finally:
        reader.shutdown()


def test_ram_guard_stops_active_child(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    readahead._BASE_RESERVE_BYTES = 1
    reader = readahead._SequentialReadAhead()
    reader._CHILD_READER_CODE = "import time; time.sleep(10)"

    low_memory = False

    def available_memory():
        return 0 if low_memory else 1024 * 1024 * 1024

    monkeypatch.setattr(readahead, "_available_memory", available_memory)
    try:
        assert reader.request(str(path))
        assert wait_for(reader, lambda: reader._active_process is not None)
        low_memory = True
        assert wait_for(reader, lambda: reader._active is None and reader._pending is None)
    finally:
        reader.shutdown()


def test_shutdown_terminates_child_and_worker(tmp_path):
    path = make_file(tmp_path)
    reader = readahead._SequentialReadAhead()
    reader._CHILD_READER_CODE = "import time; time.sleep(10)"

    try:
        assert reader.request(str(path))
        assert wait_for(reader, lambda: reader._active_process is not None)
        with reader._cv:
            process = reader._active_process
        assert process is not None

        assert reader.shutdown()
        assert process.poll() is not None
        assert not reader._worker_thread.is_alive()
    finally:
        reader.shutdown()


def test_public_api_is_noop_when_disabled(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    monkeypatch.setattr(readahead, "_reader", None)

    assert readahead.request(str(path)) is False
    assert readahead.stop_for_sampling() is True
    assert readahead.shutdown() is True
