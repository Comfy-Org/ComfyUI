# In-Core Benchmark Capture — SPIKE

Status: **spike / RFC reference.** This proves the vertical cleanly for an
in-core benchmarking capability. It is local-only: **no network calls, no
upload, no consent logic** (those live in the orchestrator/consumer, "Track B").

This document is the **contract** the downstream consumer builds against. The
emitted `benchmark` event is the canonical corpus record shape. It is versioned
via `capture_schema_version` (currently `1`); any breaking change to field names,
types, or semantics must bump that integer.

---

## 1. What it does

When a run **opts in**, ComfyUI core captures rich benchmark metrics for that run
and emits a single versioned `benchmark` event at the end of execution. When a
run does **not** opt in, literally nothing runs — no wrappers are installed, no
sampler thread is spawned, no context object is created. The only off-path cost
is one dict lookup in the gate plus one `None` check per node.

Captured per opted-in run:

- **Device / environment snapshot** (A5): backend, GPU model, driver, PyTorch &
  ComfyUI versions, OS/platform/arch, CPU model/cores, total VRAM/RAM.
- **Per-node timeline** (A2): `[{node_id, class_type, elapsed_ms}]`.
- **Per-step sampler timing + it/s** (A3): step count, per-step durations, and
  derived iterations-per-second.
- **Resource series** (A4): CPU% + RAM always; VRAM used/util/power on CUDA;
  unified-memory VRAM on Apple/MPS. Sampled on a fixed interval on its own
  thread, joined at run end. Peak values summarized.

---

## 2. How to activate (both opt-in, off by default)

### 2a. Per-run (primary primitive)

Set a truthy `benchmark` key in `extra_data` on the `/prompt` submission:

```jsonc
{
  "prompt": { /* ...graph... */ },
  "extra_data": { "benchmark": true },
  "client_id": "…"
}
```

Orchestrators (e.g. ComfyUI Desktop) set this per run. This is the primary way
to capture a single run without global side effects.

### 2b. Global (soak mode)

Launch with the CLI flag:

```bash
python main.py --benchmark
```

This forces capture on for **every** run. Per-run `extra_data["benchmark"]`
still works independently.

### 2c. Per-run JSON sink (poll-friendly channel)

Whenever capture is active — **either** the per-run `extra_data["benchmark"]`
opt-in **or** the global `--benchmark` flag — the same event payload is written
to a file keyed by prompt id:

```
output/benchmarks/<prompt_id>.json
```

One file per run, named deterministically by `prompt_id` (the id is sanitized
for filesystem safety). This exists so orchestrators that poll (e.g. the Desktop
runner over `/api/jobs`, with no websocket tap) can fetch a specific run's record
without listening to the event stream. The websocket `benchmark` event is still
emitted, unchanged — this is an additional, reliable file channel, not a
replacement.

The zero-overhead-when-off guarantee is unaffected: when a run does not opt in,
no directory is created and no file is written.

### 2d. The gate

A single function decides everything:

```python
comfy.benchmark.should_capture(extra_data) -> bool
#   True  iff  args.benchmark is set  OR  extra_data["benchmark"] is truthy
```

`comfy.benchmark.start(prompt_id, extra_data)` returns `None` (and does nothing)
when the gate is `False`. This is the zero-overhead guarantee, verified by
`tests-unit/benchmark_test/test_benchmark_capture.py::test_zero_overhead_when_off`.

---

## 3. Event schema (contract)

The payload is delivered through **two channels**, both carrying the identical
shape below:

1. **Websocket event** named **`benchmark`** (via `PromptExecutor.add_message`,
   `broadcast=False`).
2. **Per-run file** at `output/benchmarks/<prompt_id>.json`, written whenever
   capture is active for that run (see §2c). Poll-friendly for consumers without
   a websocket tap.

| Field | Type | Filled by | Notes |
|-------|------|-----------|-------|
| `type` | `str` | always | Constant `"benchmark"`. |
| `capture_schema_version` | `int` | always | Currently `1`. Bump on breaking change. |
| `collector_id` | `str` | always | Constant `"comfyui-core"`. |
| `prompt_id` | `str` | always | The prompt/run id. |
| `timestamp` | `str` | always | ISO-8601 UTC, `YYYY-MM-DDTHH:MM:SSZ`. |
| `device` | `object` | always | Device/env snapshot — see below. |
| `durations` | `object` | always | Run-level durations — see below. |
| `nodes` | `array` | always | Per-node timeline — see below. |
| `sampling` | `object` | always | Per-step sampler metrics — see below. |
| `resources` | `object` | always | Hardware sample series + peak — see below. |

### 3a. `device` (env snapshot)

| Field | Type | Backend | Notes |
|-------|------|---------|-------|
| `backend` | `str` | all | One of `cuda`, `mps`, `xpu`, `cpu`, `unknown`. |
| `gpu_model` | `str \| null` | CUDA/MPS | CUDA: `nvidia-smi name` (falls back to `torch.cuda.get_device_name`). MPS: chip name. `null` on CPU. |
| `driver_version` | `str \| null` | CUDA/MPS | CUDA: driver version. MPS: macOS version. `null` otherwise. |
| `vram_is_unified` | `bool` | all | `true` on Apple/MPS, else `false`. |
| `pytorch_version` | `str \| null` | all | `torch.__version__`. |
| `comfyui_version` | `str \| null` | all | From `comfyui_version.__version__`. |
| `os` | `str` | all | `platform.platform()`. |
| `platform` | `str` | all | `platform.system().lower()` (`windows`/`linux`/`darwin`). |
| `arch` | `str` | all | `platform.machine()`. |
| `cpu_model` | `str \| null` | all | `platform.processor()`. |
| `cpu_cores_physical` | `int \| null` | all | `psutil.cpu_count(logical=False)`. |
| `cpu_cores_logical` | `int \| null` | all | `psutil.cpu_count(logical=True)`. |
| `total_vram_mb` | `number \| null` | CUDA/MPS | CUDA: `nvidia-smi memory.total`. MPS: unified RAM total. `null` on CPU. |
| `total_ram_mb` | `number \| null` | all | `psutil.virtual_memory().total`. |

All keys are **always present**; unavailable values are `null` (never omitted),
so the consumer sees a stable shape on every backend.

### 3b. `durations`

| Field | Type | Notes |
|-------|------|-------|
| `total_run_ms` | `number` | Wall time of the captured `execute_async`, measured from capture start to finalize. |
| `sampler_ms` | `number` | Sum of time spent inside the `SAMPLER_SAMPLE` wrapper (whole sampling loop(s)). |
| `node_total_ms` | `number` | Sum of `nodes[].elapsed_ms`. |

### 3c. `nodes` (array of objects)

| Field | Type | Notes |
|-------|------|-------|
| `node_id` | `str` | Unique node id in the (dynamic) prompt. |
| `class_type` | `str` | Node class, e.g. `KSampler`. |
| `elapsed_ms` | `number` | Wall time of that node's `get_output_data`, in run order. |

Only executed (non-cached) nodes appear. Order is execution order.

### 3d. `sampling`

| Field | Type | Notes |
|-------|------|-------|
| `step_count` | `int` | Number of `PREDICT_NOISE` calls = denoise steps across the run. |
| `step_durations_ms` | `number[]` | One entry per step, in order. |
| `per_step_it_per_s` | `(number \| null)[]` | Index-aligned with `step_durations_ms`; `1000/ms` per step, `null` if a step was 0ms. |
| `avg_it_per_s` | `number \| null` | `step_count / (sampler_ms/1000)`; `null` if no steps/time. |

For a multi-sampler graph, `step_count` aggregates all sampler passes and
`sampler_ms` sums their wrapped time.

### 3e. `resources`

| Field | Type | Notes |
|-------|------|-------|
| `sample_interval_ms` | `int` | Sampler cadence (default 250). |
| `series` | `object[]` | Time series of hardware samples (see below). |
| `peak` | `object` | Max over the series for each metric. |

Each `series[]` point:

| Field | Type | Backend | Notes |
|-------|------|---------|-------|
| `t_ms` | `number` | all | ms since capture start. |
| `cpu_percent` | `number \| null` | all | `psutil.cpu_percent`. |
| `ram_used_mb` | `number \| null` | all | System RAM used. |
| `vram_used_mb` | `number \| null` | CUDA/MPS | CUDA: `nvidia-smi memory.used`. MPS: `torch.mps.current_allocated_memory()`. `null` on CPU. |
| `vram_util_percent` | `number \| null` | CUDA | `nvidia-smi utilization.gpu`. `null` on MPS/CPU. |
| `power_w` | `number \| null` | CUDA | `nvidia-smi power.draw` (if the GPU reports it). `null` otherwise. |

`peak` mirrors `vram_used_mb`, `ram_used_mb`, `cpu_percent`, `vram_util_percent`,
`power_w` — each the max of non-null samples, or `null` if none.
**Peak VRAM = `resources.peak.vram_used_mb`.**

---

## 4. Copy-paste example event (CUDA)

```json
{
  "type": "benchmark",
  "capture_schema_version": 1,
  "collector_id": "comfyui-core",
  "prompt_id": "f7a1c2e0-1234-4abc-9def-0123456789ab",
  "timestamp": "2026-09-29T18:42:07Z",
  "device": {
    "backend": "cuda",
    "gpu_model": "NVIDIA GeForce RTX 5090",
    "driver_version": "555.85",
    "vram_is_unified": false,
    "pytorch_version": "2.5.1+cu124",
    "comfyui_version": "0.38.0",
    "os": "Windows-11-10.0.26200-SP0",
    "platform": "windows",
    "arch": "AMD64",
    "cpu_model": "AMD Ryzen 9 7950X 16-Core Processor",
    "cpu_cores_physical": 16,
    "cpu_cores_logical": 32,
    "total_vram_mb": 32607.0,
    "total_ram_mb": 65413.0
  },
  "durations": {
    "total_run_ms": 5401.22,
    "sampler_ms": 4012.9,
    "node_total_ms": 5303.57
  },
  "nodes": [
    { "node_id": "4", "class_type": "CheckpointLoaderSimple", "elapsed_ms": 812.44 },
    { "node_id": "6", "class_type": "CLIPTextEncode", "elapsed_ms": 41.02 },
    { "node_id": "7", "class_type": "CLIPTextEncode", "elapsed_ms": 38.71 },
    { "node_id": "3", "class_type": "KSampler", "elapsed_ms": 4120.88 },
    { "node_id": "8", "class_type": "VAEDecode", "elapsed_ms": 233.19 },
    { "node_id": "9", "class_type": "SaveImage", "elapsed_ms": 57.33 }
  ],
  "sampling": {
    "step_count": 20,
    "step_durations_ms": [201.3, 199.8, 200.1, 200.5, 199.9, 200.2, 200.0, 201.1, 199.7, 200.4, 200.0, 199.6, 200.8, 200.2, 199.9, 200.1, 200.3, 199.8, 200.0, 200.2],
    "per_step_it_per_s": [4.9677, 5.005, 4.9975, 4.9875, 5.0025, 4.995, 5.0, 4.9727, 5.0075, 4.99, 5.0, 5.01, 4.9801, 4.995, 5.0025, 4.9975, 4.9925, 5.005, 5.0, 4.995],
    "avg_it_per_s": 4.9839
  },
  "resources": {
    "sample_interval_ms": 250,
    "series": [
      { "t_ms": 0.0,   "cpu_percent": 8.3,  "ram_used_mb": 14320.5, "vram_used_mb": 1830.0,  "vram_util_percent": 3.0,  "power_w": 41.2 },
      { "t_ms": 250.1, "cpu_percent": 22.7, "ram_used_mb": 15980.2, "vram_used_mb": 21874.0, "vram_util_percent": 99.0, "power_w": 528.6 },
      { "t_ms": 500.2, "cpu_percent": 19.4, "ram_used_mb": 16010.9, "vram_used_mb": 22140.0, "vram_util_percent": 98.0, "power_w": 531.0 }
    ],
    "peak": {
      "vram_used_mb": 22140.0,
      "ram_used_mb": 16010.9,
      "cpu_percent": 22.7,
      "vram_util_percent": 99.0,
      "power_w": 531.0
    }
  }
}
```

On **Apple/MPS** the same shape applies, with: `device.backend = "mps"`,
`device.vram_is_unified = true`, `vram_util_percent` / `power_w` = `null` in every
series point, and `vram_used_mb` sourced from `torch.mps.current_allocated_memory()`.
On **CPU-only**, all VRAM/util/power fields are `null`.

---

## 5. Implementation map (seams touched)

Small surface: one gate + one node-timing wrap + one sampler module + one event.

| File | Change |
|------|--------|
| `comfy/cli_args.py` | Add `--benchmark` flag. |
| `comfy/benchmark/__init__.py` | Gate (`should_capture`), `BenchmarkContext`, event assembly, JSON writer, wrapper install/restore. |
| `comfy/benchmark/sampler.py` | `HardwareSampler` thread + backend-aware `env_snapshot()`. |
| `execution.py` | `start()` after `execution_start`; `finish()` in the `finally`; per-node `perf_counter` bracket around `get_output_data`. |
| `tests-unit/benchmark_test/` | Zero-overhead + schema tests. |

### Reality note — wrapper registration

The brief assumed a global registry for `SAMPLER_SAMPLE` / `PREDICT_NOISE`
wrappers. In this repo there is **none**: `comfy.patcher_extension.get_all_wrappers`
reads wrappers out of each model's `model_options` at sampling time
(`samplers.py` L1214 / L1235 / L1211). To register capture wrappers for an
arbitrary run without mutating every model patcher, `BenchmarkContext` temporarily
wraps `patcher_extension.get_all_wrappers` for the duration of the run and restores
it in `finalize()`. This install happens **only while capture is active**, so the
off-path is untouched and zero-overhead holds.

### No new dependencies

Uses `psutil` (already in `requirements.txt`) and stdlib `subprocess` (for
`nvidia-smi`). No new deps added.

---

## 6. Running the tests

```bash
python -m pytest tests-unit/benchmark_test/ -v
```

`test_zero_overhead_when_off` is the "benchmark of the benchmark": it asserts
`start({})` returns `None`, no active context exists, `get_all_wrappers` is the
exact original function object, and no extra thread was spawned. The remaining
tests validate wrapper install/measurement and the emitted event / JSON schema
using a mocked sampler and fake executors (no GPU/model/torch required).
```
