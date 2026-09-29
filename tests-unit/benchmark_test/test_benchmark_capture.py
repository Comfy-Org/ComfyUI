"""Tests for the in-core benchmark capture spike (comfy.benchmark).

Two halves, per the spike brief:
  (a) "benchmark of the benchmark" -- prove that when a run does NOT opt in,
      absolutely nothing runs: no context, no wrappers installed, no sampler
      thread spawned, and finish() is a no-op.
  (b) schema/shape validation -- when a run IS on, drive the recording API with
      a mocked sampler + fake executors (no GPU/model/torch) and assert the
      emitted event and the on-disk JSON match the documented contract.

These tests deliberately avoid importing ``execution`` so they run without
torch/psutil/GPU installed.
"""
import json
import threading

import pytest

import comfy.benchmark as benchmark
import comfy.patcher_extension as pe
from comfy.cli_args import args


class FakeSampler:
    """Stand-in for HardwareSampler: no thread, canned series/peak/env."""

    def __init__(self, interval_s=0.25):
        self.backend = "cpu"
        self.interval_s = interval_s
        self.series = [
            {"t_ms": 0.0, "cpu_percent": 10.0, "ram_used_mb": 1000.0,
             "vram_used_mb": 500.0, "vram_util_percent": 20.0, "power_w": 50.0},
            {"t_ms": 250.0, "cpu_percent": 40.0, "ram_used_mb": 1200.0,
             "vram_used_mb": 900.0, "vram_util_percent": 80.0, "power_w": 120.0},
        ]
        self.started = False
        self.stopped = False
        self.joined = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True

    def peak(self):
        return {"vram_used_mb": 900.0, "ram_used_mb": 1200.0,
                "cpu_percent": 40.0, "vram_util_percent": 80.0, "power_w": 120.0}

    def total_vram_mb(self):
        return 24564.0


@pytest.fixture(autouse=True)
def _clean_state():
    """Restore global capture state + get_all_wrappers around every test."""
    orig_get_all_wrappers = pe.get_all_wrappers
    orig_active = benchmark._active
    orig_flag = getattr(args, "benchmark", False)
    benchmark._active = None
    yield
    benchmark._active = orig_active
    pe.get_all_wrappers = orig_get_all_wrappers
    args.benchmark = orig_flag


@pytest.fixture
def fake_sampler(monkeypatch):
    holder = {}

    def _factory(interval_s=0.25):
        holder["instance"] = FakeSampler(interval_s)
        return holder["instance"]

    monkeypatch.setattr("comfy.benchmark.sampler.HardwareSampler", _factory)
    return holder


# --------------------------------------------------------------------------
# (a) The gate + "benchmark of the benchmark": zero overhead when off.
# --------------------------------------------------------------------------

def test_gate_off_by_default():
    assert benchmark.should_capture({}) is False


def test_gate_per_run_optin():
    assert benchmark.should_capture({"benchmark": True}) is True
    assert benchmark.should_capture({"benchmark": 1}) is True
    assert benchmark.should_capture({"benchmark": False}) is False
    assert benchmark.should_capture({"benchmark": 0}) is False
    assert benchmark.should_capture({"benchmark": None}) is False


def test_gate_global_flag(monkeypatch):
    monkeypatch.setattr(args, "benchmark", True)
    assert benchmark.should_capture({}) is True


def test_zero_overhead_when_off():
    """When neither activation is on: no context, no wrappers, no thread."""
    orig_get_all_wrappers = pe.get_all_wrappers
    threads_before = threading.active_count()

    ctx = benchmark.start("prompt-1", {})

    assert ctx is None, "start() must return None when not opted in"
    assert benchmark.get_active() is None, "no active context should exist"
    assert pe.get_all_wrappers is orig_get_all_wrappers, (
        "get_all_wrappers must be untouched when capture is off"
    )
    assert threading.active_count() == threads_before, (
        "no sampler thread should be spawned when capture is off"
    )


def test_finish_is_noop_when_off():
    class Spy:
        def __init__(self):
            self.calls = []

        def add_message(self, event, data, broadcast):
            self.calls.append((event, data, broadcast))

    spy = Spy()
    assert benchmark.finish(None, spy) is None
    assert spy.calls == [], "finish(None, ...) must not emit any message"


# --------------------------------------------------------------------------
# (b) When on: wrappers install, recording works, schema validates.
# --------------------------------------------------------------------------

def test_wrappers_installed_only_when_on(fake_sampler):
    orig = pe.get_all_wrappers
    ctx = benchmark.start("prompt-2", {"benchmark": True})
    assert ctx is not None
    assert benchmark.get_active() is ctx
    assert pe.get_all_wrappers is not orig, "get_all_wrappers should be wrapped"

    predict = pe.get_all_wrappers(pe.WrappersMP.PREDICT_NOISE, {}, is_model_options=True)
    sample = pe.get_all_wrappers(pe.WrappersMP.SAMPLER_SAMPLE, {}, is_model_options=True)
    other = pe.get_all_wrappers(pe.WrappersMP.CALC_COND_BATCH, {}, is_model_options=True)

    assert ctx._predict_noise_wrapper in predict
    assert ctx._sampler_sample_wrapper in sample
    assert other == [], "unrelated wrapper types must be untouched"

    # Finalizing restores the original function.
    ctx.finalize()
    assert pe.get_all_wrappers is orig


def test_wrappers_measure_steps_and_sampler(fake_sampler):
    ctx = benchmark.start("prompt-3", {"benchmark": True})

    # Simulate the WrapperExecutor invoking our wrappers: predict_noise once per
    # denoise step, sampler_sample once around the whole loop.
    def fake_executor(*a, **k):
        return "result"

    for _ in range(4):
        assert ctx._predict_noise_wrapper(fake_executor) == "result"
    assert ctx._sampler_sample_wrapper(fake_executor) == "result"

    event = ctx.finalize()
    assert event["sampling"]["step_count"] == 4
    assert len(event["sampling"]["step_durations_ms"]) == 4
    assert len(event["sampling"]["per_step_it_per_s"]) == 4
    assert event["durations"]["sampler_ms"] >= 0.0


def test_event_schema_shape(fake_sampler):
    ctx = benchmark.start("prompt-4", {"benchmark": True})
    ctx.record_node("10", "KSampler", 123.456)
    ctx.record_node("11", "VAEDecode", 7.89)
    for _ in range(2):
        ctx._predict_noise_wrapper(lambda: None)
    ctx._sampler_sample_wrapper(lambda: None)

    event = ctx.finalize()

    # Top-level envelope.
    assert event["type"] == "benchmark"
    assert event["capture_schema_version"] == benchmark.CAPTURE_SCHEMA_VERSION == 1
    assert event["collector_id"] == "comfyui-core"
    assert event["prompt_id"] == "prompt-4"
    assert isinstance(event["timestamp"], str) and event["timestamp"].endswith("Z")

    # device / env snapshot: keys must always be present (values may be None).
    for key in ("backend", "gpu_model", "driver_version", "vram_is_unified",
                "pytorch_version", "comfyui_version", "os", "platform", "arch",
                "cpu_model", "cpu_cores_physical", "cpu_cores_logical",
                "total_vram_mb", "total_ram_mb"):
        assert key in event["device"], f"missing device.{key}"

    # durations.
    for key in ("total_run_ms", "sampler_ms", "node_total_ms"):
        assert isinstance(event["durations"][key], (int, float))

    # nodes timeline.
    assert event["nodes"] == [
        {"node_id": "10", "class_type": "KSampler", "elapsed_ms": 123.456},
        {"node_id": "11", "class_type": "VAEDecode", "elapsed_ms": 7.89},
    ]
    assert event["durations"]["node_total_ms"] == pytest.approx(131.346)

    # sampling.
    s = event["sampling"]
    assert s["step_count"] == 2
    assert isinstance(s["step_durations_ms"], list) and len(s["step_durations_ms"]) == 2
    assert isinstance(s["per_step_it_per_s"], list)
    assert s["avg_it_per_s"] is None or isinstance(s["avg_it_per_s"], (int, float))

    # resources: series + peak + interval.
    r = event["resources"]
    assert r["sample_interval_ms"] == 250
    assert isinstance(r["series"], list) and len(r["series"]) == 2
    for point in r["series"]:
        for key in ("t_ms", "cpu_percent", "ram_used_mb", "vram_used_mb",
                    "vram_util_percent", "power_w"):
            assert key in point
    assert r["peak"]["vram_used_mb"] == 900.0

    # Whole event must be JSON-serializable (it is emitted + optionally written).
    json.dumps(event)


def test_json_report_written(fake_sampler, monkeypatch, tmp_path):
    monkeypatch.setattr(args, "benchmark", True)  # soak mode -> write JSON
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    ctx = benchmark.start("prompt-5", {})  # global flag drives capture
    assert ctx is not None and ctx.write_json is True
    ctx.record_node("1", "CheckpointLoaderSimple", 42.0)

    class Spy:
        def __init__(self):
            self.events = []

        def add_message(self, event, data, broadcast):
            self.events.append((event, data))

    spy = Spy()
    event = benchmark.finish(ctx, spy)

    # Emitted on the event stream.
    assert spy.events and spy.events[0][0] == "benchmark"
    assert spy.events[0][1] == event

    # Written to disk under output/benchmarks/.
    files = list((tmp_path / "benchmarks").glob("*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text(encoding="utf-8"))
    assert on_disk["prompt_id"] == "prompt-5"
    assert on_disk["collector_id"] == "comfyui-core"
