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

    def __init__(self, interval_s=0.25, backend="cpu"):
        self.backend = backend
        self.interval_s = interval_s
        # Two samples 1s apart carrying the full v2 metric set so energy
        # integration and the new peak fields have something to chew on.
        self.series = [
            {"t_ms": 0.0, "cpu_percent": 10.0, "ram_used_mb": 1000.0,
             "vram_used_mb": 500.0, "vram_util_percent": 20.0, "power_w": 50.0,
             "temperature_c": 45.0, "sm_clock_mhz": 900.0, "mem_clock_mhz": 5000.0,
             "power_limit_w": 600.0},
            {"t_ms": 1000.0, "cpu_percent": 40.0, "ram_used_mb": 1200.0,
             "vram_used_mb": 900.0, "vram_util_percent": 80.0, "power_w": 590.0,
             "temperature_c": 84.0, "sm_clock_mhz": 2500.0, "mem_clock_mhz": 10000.0,
             "power_limit_w": 600.0},
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
                "cpu_percent": 40.0, "vram_util_percent": 80.0, "power_w": 590.0,
                "temperature_c": 84.0, "sm_clock_mhz": 2500.0,
                "mem_clock_mhz": 10000.0, "power_limit_w": 600.0, "throttled": True}

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

    def _factory(interval_s=0.25, backend=None):
        holder["instance"] = FakeSampler(interval_s, backend=backend or "cpu")
        return holder["instance"]

    monkeypatch.setattr("comfy.benchmark.sampler.HardwareSampler", _factory)
    return holder


@pytest.fixture
def fake_sampler_cuda(monkeypatch):
    """A CUDA-flavored sampler + a canned CUDA env snapshot (no GPU needed)."""
    holder = {}

    def _factory(interval_s=0.25, backend=None):
        holder["instance"] = FakeSampler(interval_s, backend="cuda")
        return holder["instance"]

    def _env(backend=None):
        return {
            "backend": "cuda", "gpu_model": "NVIDIA GeForce RTX 5090",
            "driver_version": "555.85", "vram_is_unified": False,
            "pytorch_version": "2.5.1+cu124", "comfyui_version": "0.38.0",
            "os": "Windows-11", "platform": "windows", "arch": "AMD64",
            "cpu_model": "AMD Ryzen 9", "cpu_cores_physical": 16,
            "cpu_cores_logical": 32, "total_vram_mb": 32607.0,
            "total_ram_mb": 65413.0, "vram_state": "NORMAL_VRAM",
            "offloaded": False, "weight_dtype": "float16",
            "compute_dtype": "float16", "attention_impl": "pytorch",
            "cuda_version": "12.4", "cudnn_version": 90100,
            "compute_capability": "12.0", "is_laptop": False,
            "pcie_gen": 5, "pcie_width": 16,
        }

    monkeypatch.setattr("comfy.benchmark.sampler.HardwareSampler", _factory)
    monkeypatch.setattr("comfy.benchmark.sampler.env_snapshot", _env)
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
    assert event["capture_schema_version"] == benchmark.CAPTURE_SCHEMA_VERSION == 2
    assert event["collector_id"] == "comfyui-core"
    assert event["prompt_id"] == "prompt-4"
    assert isinstance(event["timestamp"], str) and event["timestamp"].endswith("Z")

    # device / env snapshot: keys must always be present (values may be None).
    for key in ("backend", "gpu_model", "driver_version", "vram_is_unified",
                "pytorch_version", "comfyui_version", "os", "platform", "arch",
                "cpu_model", "cpu_cores_physical", "cpu_cores_logical",
                "total_vram_mb", "total_ram_mb",
                # v2 additions:
                "vram_state", "offloaded", "weight_dtype", "compute_dtype",
                "attention_impl", "cuda_version", "cudnn_version",
                "compute_capability", "is_laptop", "pcie_gen", "pcie_width",
                "baseline"):
        assert key in event["device"], f"missing device.{key}"

    # v2 run group: always present, keys stable (values may be None).
    for key in ("status", "image_count", "batch_size", "benchmark_id",
                "benchmark_version", "warmup_runs", "measured_runs", "seed"):
        assert key in event["run"], f"missing run.{key}"
    assert event["run"]["status"] == "success"

    # v2 workflow group: structural params only.
    for key in ("resolution", "steps", "sampler", "scheduler", "cfg",
                "denoise", "seed", "samplers"):
        assert key in event["workflow"], f"missing workflow.{key}"
    assert set(event["workflow"]["resolution"].keys()) == {"width", "height"}
    assert isinstance(event["workflow"]["samplers"], list)

    # v2 summary group.
    for key in ("energy_wh_per_image", "sec_per_image", "throttled"):
        assert key in event["summary"], f"missing summary.{key}"

    # durations (incl. v2 model_load_ms which may be None).
    for key in ("total_run_ms", "sampler_ms", "node_total_ms"):
        assert isinstance(event["durations"][key], (int, float))
    assert "model_load_ms" in event["durations"]

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
                    "vram_util_percent", "power_w",
                    # v2 additions:
                    "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
                    "power_limit_w"):
            assert key in point
    assert r["peak"]["vram_used_mb"] == 900.0
    for key in ("temperature_c", "sm_clock_mhz", "mem_clock_mhz",
                "power_limit_w", "throttled"):
        assert key in r["peak"], f"missing peak.{key}"

    # Whole event must be JSON-serializable (it is emitted + optionally written).
    json.dumps(event)


class _Spy:
    def __init__(self):
        self.events = []

    def add_message(self, event, data, broadcast):
        self.events.append((event, data))


def test_json_report_written_under_global_flag(fake_sampler, monkeypatch, tmp_path):
    monkeypatch.setattr(args, "benchmark", True)  # soak mode
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    ctx = benchmark.start("prompt-5", {})  # global flag drives capture
    assert ctx is not None and ctx.write_json is True
    ctx.record_node("1", "CheckpointLoaderSimple", 42.0)

    spy = _Spy()
    event = benchmark.finish(ctx, spy)

    # ws event still emitted (unchanged channel).
    assert spy.events and spy.events[0][0] == "benchmark"
    assert spy.events[0][1] == event

    # Per-run file keyed by prompt_id.
    path = tmp_path / "benchmarks" / "prompt-5.json"
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["prompt_id"] == "prompt-5"
    assert on_disk["collector_id"] == "comfyui-core"


def test_json_report_written_per_run_optin(fake_sampler, monkeypatch, tmp_path):
    """extra_data opt-in (no global flag) must also produce the per-run file."""
    assert getattr(args, "benchmark", False) is False  # global flag OFF
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    ctx = benchmark.start("run-abc-123", {"benchmark": True})
    assert ctx is not None and ctx.write_json is True
    ctx.record_node("1", "KSampler", 10.0)

    spy = _Spy()
    benchmark.finish(ctx, spy)

    # File named exactly <prompt_id>.json so a poller can fetch it deterministically.
    path = tmp_path / "benchmarks" / "run-abc-123.json"
    assert path.exists()
    assert spy.events and spy.events[0][0] == "benchmark"  # ws still emitted
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["prompt_id"] == "run-abc-123"


def test_no_json_file_when_off(monkeypatch, tmp_path):
    """Off-path: no capture, no file work at all."""
    assert getattr(args, "benchmark", False) is False
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    ctx = benchmark.start("prompt-off", {})
    assert ctx is None
    assert benchmark.finish(ctx, _Spy()) is None

    # Nothing should have been created under output/benchmarks/.
    assert not (tmp_path / "benchmarks").exists()


# --------------------------------------------------------------------------
# (c) v2 additions: run metadata, workflow params, device/series enrichment.
# --------------------------------------------------------------------------

_SAMPLE_PROMPT = {
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 768, "batch_size": 3},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42, "steps": 25, "cfg": 7.5, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
            # links must be ignored, not captured:
            "model": ["4", 0], "positive": ["6", 0], "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
}


def test_workflow_params_parsed(fake_sampler):
    ctx = benchmark.start("wf-1", {"benchmark": True}, _SAMPLE_PROMPT)
    event = ctx.finalize()

    wf = event["workflow"]
    assert wf["resolution"] == {"width": 1024, "height": 768}
    assert wf["steps"] == 25
    assert wf["sampler"] == "euler"
    assert wf["scheduler"] == "normal"
    assert wf["cfg"] == 7.5
    assert wf["denoise"] == 1.0
    assert wf["seed"] == 42
    assert len(wf["samplers"]) == 1
    assert wf["samplers"][0]["node_id"] == "3"
    assert wf["samplers"][0]["class_type"] == "KSampler"

    # batch_size flows into run; image_count falls back to it (no SaveImage run).
    assert event["run"]["batch_size"] == 3
    assert event["run"]["image_count"] == 3


def test_workflow_ignores_linked_inputs(fake_sampler):
    """A widget wired from another node (a link) must be captured as None."""
    prompt = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"steps": ["9", 0], "seed": 1, "cfg": 8.0,
                       "sampler_name": "dpmpp_2m", "scheduler": "karras",
                       "denoise": 1.0},
        },
    }
    ctx = benchmark.start("wf-2", {"benchmark": True}, prompt)
    event = ctx.finalize()
    assert event["workflow"]["steps"] is None  # link, not a literal
    assert event["workflow"]["sampler"] == "dpmpp_2m"


def test_run_metadata_object_optin(fake_sampler):
    """The v2 object opt-in form stamps id/version/warmup/measured/seed."""
    opt = {"benchmark": {"id": "sdxl-baseline", "version": "3",
                         "warmup_runs": 1, "measured_runs": 5, "seed": 12345}}
    assert benchmark.should_capture(opt) is True  # object form still gates on

    ctx = benchmark.start("run-obj", opt, _SAMPLE_PROMPT)
    event = ctx.finalize()
    run = event["run"]
    assert run["benchmark_id"] == "sdxl-baseline"
    assert run["benchmark_version"] == "3"
    assert run["warmup_runs"] == 1
    assert run["measured_runs"] == 5
    assert run["seed"] == 12345


def test_run_metadata_legacy_truthy_optin(fake_sampler):
    """Legacy ``benchmark: true`` still captures; metadata fields are None."""
    ctx = benchmark.start("run-legacy", {"benchmark": True}, _SAMPLE_PROMPT)
    event = ctx.finalize()
    run = event["run"]
    assert run["benchmark_id"] is None
    assert run["benchmark_version"] is None
    assert run["warmup_runs"] is None


def test_status_defaults_success_and_is_settable(fake_sampler):
    ctx = benchmark.start("run-status", {"benchmark": True})
    assert ctx.status == "success"
    ctx.set_status("interrupted")
    event = ctx.finalize()
    assert event["run"]["status"] == "interrupted"


def test_model_load_ms_from_first_loader_node(fake_sampler):
    ctx = benchmark.start("run-load", {"benchmark": True})
    ctx.record_node("4", "CheckpointLoaderSimple", 812.5)
    ctx.record_node("3", "KSampler", 4000.0)
    event = ctx.finalize()
    assert event["durations"]["model_load_ms"] == 812.5


def test_produced_image_count_from_executor(fake_sampler):
    ctx = benchmark.start("run-img", {"benchmark": True}, _SAMPLE_PROMPT)

    class ExecutorWithHistory(_Spy):
        history_result = {"outputs": {"9": {"images": [{"a": 1}, {"a": 2}]}}}

    event = benchmark.finish(ctx, ExecutorWithHistory())
    # Actual SaveImage count (2) overrides the batch_size fallback (3).
    assert event["run"]["image_count"] == 2


def test_cuda_device_and_energy_fields(fake_sampler_cuda):
    ctx = benchmark.start("run-cuda", {"benchmark": True}, _SAMPLE_PROMPT)
    event = ctx.finalize()

    dev = event["device"]
    assert dev["backend"] == "cuda"
    assert dev["cuda_version"] == "12.4"
    assert dev["cudnn_version"] == 90100
    assert dev["compute_capability"] == "12.0"
    assert dev["pcie_gen"] == 5 and dev["pcie_width"] == 16
    assert dev["vram_state"] == "NORMAL_VRAM"
    assert dev["weight_dtype"] == "float16"
    assert dev["attention_impl"] == "pytorch"
    assert dev["baseline"] is None or isinstance(dev["baseline"], dict)

    # peak carries the v2 GPU metrics + a throttle verdict.
    peak = event["resources"]["peak"]
    assert peak["temperature_c"] == 84.0
    assert peak["throttled"] is True

    # Energy integrates power over the 1s series (50W->590W) / 3 images.
    # trapezoid: 0.5*(50+590)*1s = 320 Ws = 0.08889 Wh; /3 images.
    summ = event["summary"]
    assert summ["energy_wh_per_image"] == pytest.approx(0.08889 / 3, rel=1e-2)
    assert summ["throttled"] is True
    assert summ["sec_per_image"] is not None


def test_null_case_cpu_energy_none(fake_sampler):
    """CPU/no-power backend: energy is None, device GPU fields are None-safe."""
    ctx = benchmark.start("run-cpu", {"benchmark": True})  # no prompt
    inst = fake_sampler["instance"]
    # Wipe power so integration has nothing to work with (CPU-only shape).
    for s in inst.series:
        s["power_w"] = None
    event = ctx.finalize()
    assert event["summary"]["energy_wh_per_image"] is None
    # No prompt -> workflow fully null, image_count None.
    assert event["run"]["image_count"] is None
    assert event["workflow"]["resolution"] == {"width": None, "height": None}
