# In-Core Benchmark Capture — SPIKE

Status: **spike / RFC reference.** This proves the vertical cleanly for an
in-core benchmarking capability. It is local-only: **no network calls, no
upload, no consent logic** (those live in the orchestrator/consumer, "Track B").

This document is the **contract** the downstream consumer builds against. The
emitted `benchmark` event is the canonical corpus record shape. It is versioned
via `capture_schema_version` (currently `2`); any breaking change to field names,
types, or semantics must bump that integer.

**v2 is additive**: every v1 field is intact with identical semantics. v2 adds
the `run` and `workflow` and `summary` groups, enriches `device`, `durations`,
and each `resources` sample, and accepts an object form of the opt-in. Capture is
generous by design (local + opt-in, corpus can't be backfilled): **every field is
best-effort — the key is always present, the value is `null` when unavailable, and
capture never raises into the run.**

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
- **(v2) Run metadata**: outcome (`success`/`error`/`interrupted`), image/batch
  counts, and the opt-in harness metadata (`benchmark_id`, `version`, warmup/
  measured runs, seed).
- **(v2) Structural workflow params**: resolution, steps, sampler, scheduler, cfg,
  denoise, seed — parsed from the graph, **never prompt text**.
- **(v2) Richer device snapshot**: vram_state/offload, weight & compute dtypes,
  attention impl, CUDA/cuDNN versions, compute capability, PCIe link, laptop hint,
  and a pre-work idle `baseline`.
- **(v2) Deeper resource series**: GPU temperature, SM/mem clocks, power limit, a
  throttle verdict, plus a derived `energy_wh_per_image` / `sec_per_image` summary.

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

**v2 object form (optional).** `extra_data["benchmark"]` may instead be an object
carrying corpus metadata. Any subset of keys is allowed; missing keys are stamped
`null`. The truthy back-compat is preserved — `true` still opts in — because a
non-empty object is itself truthy, so the same gate captures both forms:

```jsonc
{
  "prompt": { /* ...graph... */ },
  "extra_data": {
    "benchmark": {
      "id": "sdxl-baseline",     // -> run.benchmark_id
      "version": "3",            // -> run.benchmark_version
      "warmup_runs": 1,          // -> run.warmup_runs
      "measured_runs": 5,        // -> run.measured_runs
      "seed": 12345              // -> run.seed
    }
  },
  "client_id": "…"
}
```

### 2b. Global (soak mode)

Launch with the CLI flag:

```bash
python main.py --benchmark
```

This forces capture on for **every** run. Per-run `extra_data["benchmark"]`
still works independently.

### 2c. Two channels: event (canonical) vs. file (local artifact)

The record is delivered through **two channels**:

- **Websocket `benchmark` event — the canonical channel.** Always emitted when
  capture is active. This is what orchestrators and **cloud** consume: it needs no
  shared filesystem and survives ephemeral containers.
- **Per-run JSON file — a local artifact.** Written to
  `output/benchmarks/<prompt_id>.json` (one file per run, named deterministically
  by the sanitized `prompt_id`). This is for **local** poll-based consumers — the
  Desktop `/api/jobs` runner and soak mode — that have no websocket tap and want
  to fetch a specific run's record off disk.

Whenever capture is active — **either** the per-run `extra_data["benchmark"]`
opt-in **or** the global `--benchmark` flag — both channels fire by default.

**Disabling the file sink (cloud).** In cloud (ephemeral containers, no shared
FS) the file is dead weight, so it can be turned off while the event keeps firing:

```bash
python main.py --benchmark --benchmark-no-file   # CLI flag
COMFYUI_BENCHMARK_NO_FILE=1 python main.py         # or env var
```

Either disables **only** the file; the websocket `benchmark` event is unaffected.
Default keeps the file **on** (Desktop's poller relies on it).

**Retention cap (bounded disk).** To keep `output/benchmarks/` from growing
without bound over many runs (soak mode / long-lived Desktop installs), after each
write only the newest **N** `*.json` (by mtime) are kept; older ones are pruned.
`N` defaults to **50** and is overridable via `COMFYUI_BENCHMARK_RETENTION`
(a negative value disables pruning entirely). Pruning is best-effort — a prune
failure is swallowed and never raised into the run, and the file just written
always survives its own prune (it has the newest mtime).

The zero-overhead-when-off guarantee is unaffected: when a run does not opt in,
no directory is created and no file is written.

### 2d. The gate

A single function decides everything:

```python
comfy.benchmark.should_capture(extra_data) -> bool
#   True  iff  args.benchmark is set  OR  extra_data["benchmark"] is truthy
#   (truthy covers both the legacy `true` and the v2 non-empty object form)
```

`comfy.benchmark.start(prompt_id, extra_data)` returns `None` (and does nothing)
when the gate is `False`. This is the zero-overhead guarantee, verified by
`tests-unit/benchmark_test/test_benchmark_capture.py::test_zero_overhead_when_off`.

---

## 3. Event schema (contract)

The payload is delivered through the two channels of §2c, both carrying the
identical shape below:

1. **Websocket event** named **`benchmark`** (via `PromptExecutor.add_message`,
   `broadcast=False`) — the **canonical** channel; always emitted when capture is
   active. This is the channel cloud/orchestrators consume.
2. **Per-run file** at `output/benchmarks/<prompt_id>.json` — a **local artifact**,
   written whenever capture is active *unless* the file sink is disabled
   (`--benchmark-no-file` / `COMFYUI_BENCHMARK_NO_FILE=1`; see §2c). Poll-friendly
   for local consumers without a websocket tap; retained newest-N (see §2c).

| Field | Type | Filled by | Notes |
|-------|------|-----------|-------|
| `type` | `str` | always | Constant `"benchmark"`. |
| `capture_schema_version` | `int` | always | Currently `3`. Bump on breaking change. **v3:** the constant power cap moved from every `resources.series[]` sample (and from `peak`) to a single `device.power_limit_w`; the throttle rollup now lives only in `summary.throttled` (no longer mirrored in `peak`). |
| `collector_id` | `str` | always | Constant `"comfyui-core"`. |
| `prompt_id` | `str` | always | The prompt/run id. |
| `timestamp` | `str` | always | ISO-8601 UTC, `YYYY-MM-DDTHH:MM:SSZ`. |
| `run` | `object` | always | **(v2)** Run outcome, image counts, opt-in metadata — see below. |
| `workflow` | `object` | always | **(v2)** Structural workflow params (never prompt text) — see below. |
| `device` | `object` | always | Device/env snapshot — see below. |
| `durations` | `object` | always | Run-level durations — see below. |
| `nodes` | `array` | always | Per-node timeline — see below. |
| `sampling` | `object` | always | Per-step sampler metrics — see below. |
| `resources` | `object` | always | Hardware sample series + peak — see below. |
| `summary` | `object` | always | **(v2)** Derived per-image summaries — see below. |

### 3.0 `run` (v2 — run metadata)

| Field | Type | Filled by | Source / Notes |
|-------|------|-----------|----------------|
| `status` | `str` | always | `"success"` \| `"error"` \| `"interrupted"`. From `execution.py` outcome (`handle_execution_error` classifies error vs `InterruptProcessingException`). |
| `image_count` | `int \| null` | best-effort | Images actually produced: sum of `images`/`gifs` across output nodes in `executor.history_result`; falls back to `run.batch_size` when no output node recorded any; `null` if neither. |
| `batch_size` | `int \| null` | best-effort | Latent `batch_size` widget from the first EmptyLatentImage-style node. `null` if wired from a link or absent. |
| `benchmark_id` | `str \| null` | best-effort | From `extra_data.benchmark.id` (object form). `null` for legacy `true`. |
| `benchmark_version` | `str \| null` | best-effort | From `extra_data.benchmark.version`. |
| `warmup_runs` | `int \| null` | best-effort | From `extra_data.benchmark.warmup_runs`. |
| `measured_runs` | `int \| null` | best-effort | From `extra_data.benchmark.measured_runs`. |
| `seed` | `int \| null` | best-effort | From `extra_data.benchmark.seed` (the harness's requested seed; distinct from `workflow.seed`). |

### 3.0b `workflow` (v2 — structural params, NEVER prompt text)

Parsed statically from the prompt graph. **Only literal widget values are read**;
an input wired from another node (a `[node_id, index]` link) is captured as `null`.
CLIP/prompt text is never inspected.

| Field | Type | Source / Notes |
|-------|------|----------------|
| `resolution` | `object` | `{ "width": int\|null, "height": int\|null }` from the first EmptyLatentImage/EmptySD3LatentImage (or any node exposing `width`+`height`). |
| `steps` | `int \| null` | Primary sampler's `steps`. |
| `sampler` | `str \| null` | Primary sampler's `sampler_name`. |
| `scheduler` | `str \| null` | Primary sampler's `scheduler`. |
| `cfg` | `number \| null` | Primary sampler's `cfg`. |
| `denoise` | `number \| null` | Primary sampler's `denoise`. |
| `seed` | `int \| null` | Primary sampler's `seed` (or `noise_seed` for KSamplerAdvanced). |
| `samplers` | `object[]` | One entry per sampler-type node (class_type containing `KSampler` or `SamplerCustom`): `{node_id, class_type, steps, sampler, scheduler, cfg, denoise, seed}`. The primary/first is promoted to the fields above. |

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
| `vram_state` | `str \| null` | all | **(v2)** `comfy.model_management.vram_state.name` — `NORMAL_VRAM`/`LOW_VRAM`/`NO_VRAM`/`HIGH_VRAM`/`SHARED`/`DISABLED`. |
| `offloaded` | `bool \| null` | all | **(v2)** `true` if `vram_state ∈ {NO_VRAM, LOW_VRAM, SHARED}` (weights moved on/off device). `null` if state unreadable. |
| `weight_dtype` | `str \| null` | all | **(v2)** Loaded diffusion model's stored dtype (`base.get_dtype()`), e.g. `float16`/`bfloat16`/`float8_e4m3fn`. `null` if no model loaded. |
| `compute_dtype` | `str \| null` | all | **(v2)** Loaded model's inference dtype (`base.get_dtype_inference()`, honors manual cast); falls back to `model_management.unet_dtype()`. |
| `attention_impl` | `str \| null` | all | **(v2)** `sage`/`flash`/`xformers`/`pytorch` from `model_management.*_attention_enabled()` (checked in that order). `null` if none report. |
| `cuda_version` | `str \| null` | CUDA | **(v2)** `torch.version.cuda`. |
| `cudnn_version` | `int \| null` | CUDA | **(v2)** `torch.backends.cudnn.version()`. |
| `compute_capability` | `str \| null` | CUDA | **(v2)** `"{major}.{minor}"` from `torch.cuda.get_device_capability(0)`. |
| `is_laptop` | `bool \| null` | all | **(v2)** Hint: `psutil.sensors_battery() is not None`. A battery implies portable; absence does not prove desktop. `null` if undeterminable. |
| `pcie_gen` | `int \| null` | CUDA | **(v2)** `nvidia-smi pcie.link.gen.current`. |
| `pcie_width` | `int \| null` | CUDA | **(v2)** `nvidia-smi pcie.link.width.current` (lane count). |
| `baseline` | `object \| null` | all | **(v2)** One idle sample taken at capture start, BEFORE any work: `{ vram_used_mb, vram_util_percent, temperature_c, ram_used_mb, cpu_percent }` (each `null` per backend rules). Detects a contaminated run. `null` if the pre-sample failed. |

All keys are **always present**; unavailable values are `null` (never omitted),
so the consumer sees a stable shape on every backend.

### 3b. `durations`

| Field | Type | Notes |
|-------|------|-------|
| `total_run_ms` | `number` | Wall time of the captured `execute_async`, measured from capture start to finalize. |
| `sampler_ms` | `number` | Sum of time spent inside the `SAMPLER_SAMPLE` wrapper (whole sampling loop(s)). |
| `node_total_ms` | `number` | Sum of `nodes[].elapsed_ms`. |
| `model_load_ms` | `number \| null` | **(v2)** Best-effort cold model-load proxy: wall time of the first executed node whose `class_type` contains `Loader` or `Checkpoint` (CheckpointLoaderSimple, UNETLoader, VAELoader, CLIPLoader, …). ComfyUI loads weights lazily inside `model_management` at sample time, so this is an *approximation* (loader-node wall time), not an isolated GPU-transfer measurement. `null` if no loader node executed (e.g. cached). |

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
| `sample_interval_ms` | `int` | Sampler cadence. Default **500** (widened from 250 to reduce the observer effect); overridable via `COMFYUI_BENCHMARK_INTERVAL_MS`. Reported from the live sampler instance. |
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
| `temperature_c` | `number \| null` | CUDA | **(v2)** `nvidia-smi temperature.gpu`. |
| `sm_clock_mhz` | `number \| null` | CUDA | **(v2)** `nvidia-smi clocks.sm`. |
| `mem_clock_mhz` | `number \| null` | CUDA | **(v2)** `nvidia-smi clocks.mem`. |
| ~~`power_limit_w`~~ | — | — | **(v3: removed from series)** the enforced cap is a device constant — reported once as `device.power_limit_w`, not per sample. |

**GPU sampling path (fidelity vs. observer effect).** All GPU metrics come from a
single probe per sample. On CUDA, the sampler prefers a persistent **NVML** handle
opened once via a *soft* `pynvml` import: each sample is then an in-process,
microsecond-cost call for `vram_used_mb` / `vram_util_percent` / `power_w` /
`temperature_c` / `sm_clock_mhz` / `mem_clock_mhz` / `power_limit_w` (and the
one-shot `env` fields `gpu_model` / `driver_version` / `total_vram_mb` /
`pcie_gen` / `pcie_width`). When `pynvml` is **not** importable it transparently
falls back to the original **`nvidia-smi` subprocess** (one combined `--query-gpu`
per sample), so `pynvml` stays a soft/optional dep — no new hard dependency.

Why this matters: `nvidia-smi` forks a process on every sample. At 4×/sec (the old
250ms cadence) that runs *concurrently with the generation being measured* and
perturbs the very numbers it records (CPU scheduling, driver contention). The NVML
path removes the subprocess entirely; the cadence was also widened to 500ms as
defense-in-depth. Fidelity tradeoff: NVML is both cheaper **and** more accurate;
the `nvidia-smi` fallback preserves the metric set but carries the subprocess
overhead — acceptable because it only engages on machines without `pynvml`.

`peak` mirrors `vram_used_mb`, `ram_used_mb`, `cpu_percent`, `vram_util_percent`,
`power_w`, plus **(v2)** `temperature_c`, `sm_clock_mhz`, `mem_clock_mhz` — each the
max of non-null samples, or `null` if none. **(v3)** `power_limit_w` is no longer in
`peak` (it is a device constant, see `device.power_limit_w`), and the throttle
rollup is no longer mirrored here (see `summary.throttled`).
**Peak VRAM = `resources.peak.vram_used_mb`.**

`summary.throttled` **(v2; v3 sole home)** — `bool \| null`. Best-effort heuristic:
`true` if peak `temperature_c ≥ 83°C` **or** any sample's `power_w ≥ 98%` of the
enforced `device.power_limit_w` (power-capped). `null` when neither temperature nor
power-limit data is available. We cannot read the card's base clock, so "clocks
below base" is intentionally *not* part of this heuristic.

### 3f. `summary` (v2 — derived per-image)

| Field | Type | Notes |
|-------|------|-------|
| `energy_wh_per_image` | `number \| null` | Trapezoidal integral of `power_w` over `resources.series` (using `t_ms`), in watt-hours, divided by `run.image_count`. `null` if there are no usable power samples or no image count. |
| `sec_per_image` | `number \| null` | `durations.total_run_ms / 1000 / run.image_count`. `null` if no image count. |
| `throttled` | `bool \| null` | **(v3: sole home)** throttle rollup (was mirrored in `resources.peak` in v2). See heuristic above. |

---

## 4. Copy-paste example event (CUDA)

```json
{
  "type": "benchmark",
  "capture_schema_version": 2,
  "collector_id": "comfyui-core",
  "prompt_id": "f7a1c2e0-1234-4abc-9def-0123456789ab",
  "timestamp": "2026-09-29T18:42:07Z",
  "run": {
    "status": "success",
    "image_count": 1,
    "batch_size": 1,
    "benchmark_id": "sdxl-baseline",
    "benchmark_version": "3",
    "warmup_runs": 1,
    "measured_runs": 5,
    "seed": 12345
  },
  "workflow": {
    "resolution": { "width": 1024, "height": 1024 },
    "steps": 20,
    "sampler": "euler",
    "scheduler": "normal",
    "cfg": 7.5,
    "denoise": 1.0,
    "seed": 42,
    "samplers": [
      { "node_id": "3", "class_type": "KSampler", "steps": 20, "sampler": "euler", "scheduler": "normal", "cfg": 7.5, "denoise": 1.0, "seed": 42 }
    ]
  },
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
    "total_ram_mb": 65413.0,
    "vram_state": "NORMAL_VRAM",
    "offloaded": false,
    "weight_dtype": "float16",
    "compute_dtype": "float16",
    "attention_impl": "pytorch",
    "cuda_version": "12.4",
    "cudnn_version": 90100,
    "compute_capability": "12.0",
    "is_laptop": false,
    "pcie_gen": 5,
    "pcie_width": 16,
    "power_limit_w": 600.0,
    "baseline": {
      "vram_used_mb": 1830.0,
      "vram_util_percent": 3.0,
      "temperature_c": 38.0,
      "ram_used_mb": 14320.5,
      "cpu_percent": 6.1
    }
  },
  "durations": {
    "total_run_ms": 5401.22,
    "sampler_ms": 4012.9,
    "node_total_ms": 5303.57,
    "model_load_ms": 812.44
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
    "sample_interval_ms": 500,
    "series": [
      { "t_ms": 0.0,   "cpu_percent": 8.3,  "ram_used_mb": 14320.5, "vram_used_mb": 1830.0,  "vram_util_percent": 3.0,  "power_w": 41.2,  "temperature_c": 39.0, "sm_clock_mhz": 420.0,  "mem_clock_mhz": 405.0 },
      { "t_ms": 250.1, "cpu_percent": 22.7, "ram_used_mb": 15980.2, "vram_used_mb": 21874.0, "vram_util_percent": 99.0, "power_w": 528.6, "temperature_c": 71.0, "sm_clock_mhz": 2520.0, "mem_clock_mhz": 10501.0 },
      { "t_ms": 500.2, "cpu_percent": 19.4, "ram_used_mb": 16010.9, "vram_used_mb": 22140.0, "vram_util_percent": 98.0, "power_w": 591.0, "temperature_c": 84.0, "sm_clock_mhz": 2490.0, "mem_clock_mhz": 10501.0 }
    ],
    "peak": {
      "vram_used_mb": 22140.0,
      "ram_used_mb": 16010.9,
      "cpu_percent": 22.7,
      "vram_util_percent": 99.0,
      "power_w": 591.0,
      "temperature_c": 84.0,
      "sm_clock_mhz": 2520.0,
      "mem_clock_mhz": 10501.0
    }
  },
  "summary": {
    "energy_wh_per_image": 0.076,
    "sec_per_image": 5.40122,
    "throttled": true
  }
}
```

On **Apple/MPS** the same shape applies, with: `device.backend = "mps"`,
`device.vram_is_unified = true`, `vram_util_percent` / `power_w` and all v2 GPU
metrics (`temperature_c`, `sm_clock_mhz`, `mem_clock_mhz`, `power_limit_w`,
`cuda_version`, `cudnn_version`, `compute_capability`, `pcie_gen`, `pcie_width`) =
`null`, and `vram_used_mb` sourced from `torch.mps.current_allocated_memory()`.
On **CPU-only**, all VRAM/util/power/clock/temp fields are `null`, so
`summary.energy_wh_per_image` and `peak.throttled` are `null` too. The `run`,
`workflow`, `durations.model_load_ms`, and `device.baseline` groups still populate
from the graph and CPU/RAM probes on every backend.

---

## 5. Implementation map (seams touched)

Small surface: one gate + one node-timing wrap + one sampler module + one event.

| File | Change |
|------|--------|
| `comfy/cli_args.py` | Add `--benchmark` and `--benchmark-no-file` flags. |
| `comfy/benchmark/__init__.py` | Gate (`should_capture`), file-sink gate (`file_sink_disabled`), opt-in metadata parse, `BenchmarkContext`, workflow parser, energy/summary derivation, event assembly, JSON writer + retention prune, wrapper install/restore (defensive), cadence resolution (`_sample_interval_s`), singleton guard + series snapshot. |
| `comfy/benchmark/sampler.py` | `HardwareSampler` thread (v2 GPU metrics via persistent NVML handle w/ nvidia-smi fallback), `baseline_sample()`, `_device_runtime_snapshot()` + backend-aware `env_snapshot()`. |
| `execution.py` | `start(prompt_id, extra_data, prompt)` after `execution_start`; `finish()` in the `finally` (counts produced images); per-node `perf_counter` bracket around `get_output_data`; `set_status()` in `handle_execution_error` (error vs interrupted). |
| `tests-unit/benchmark_test/` | Zero-overhead + schema tests (+ v2 run/workflow/device/summary, CUDA + null-case). |

### v2 field sourcing at a glance

- **run**: `status` from `execution.py` outcome; `image_count` from
  `executor.history_result` (SaveImage `images`/`gifs`) or `batch_size` fallback;
  `benchmark_*`/`warmup_runs`/`measured_runs`/`seed` from the `extra_data.benchmark`
  object.
- **workflow**: static parse of the prompt graph — literals only, links -> `null`,
  never prompt text.
- **device (v2)**: `torch` (`version.cuda`, `backends.cudnn.version()`,
  `cuda.get_device_capability()`), `nvidia-smi` (`pcie.link.*`),
  `model_management` (`vram_state`, dtypes via loaded model, attention helpers),
  `psutil.sensors_battery()` (laptop hint), and one pre-work `baseline` sample.
- **resources (v2)**: per-sample GPU probe (`vram/util/power/temp/clocks/
  power.limit`) via a persistent NVML handle (soft `pynvml`), falling back to a
  single extended `nvidia-smi` query when `pynvml` is absent; `peak.throttled`
  heuristic. Default cadence 500ms (`COMFYUI_BENCHMARK_INTERVAL_MS`).
- **summary**: trapezoidal power integral / image count; `sec_per_image`.

### Reality note — wrapper registration

The brief assumed a global registry for `SAMPLER_SAMPLE` / `PREDICT_NOISE`
wrappers. In this repo there is **none**: `comfy.patcher_extension.get_all_wrappers`
reads wrappers out of each model's `model_options` at sampling time
(`samplers.py` L1214 / L1235 / L1211). To register capture wrappers for an
arbitrary run without mutating every model patcher, `BenchmarkContext` temporarily
wraps `patcher_extension.get_all_wrappers` for the duration of the run and restores
it in `finalize()`. This install happens **only while capture is active**, so the
off-path is untouched and zero-overhead holds.

### No new (hard) dependencies

Uses `psutil` (already in `requirements.txt`) and stdlib `subprocess` (for the
`nvidia-smi` fallback). `pynvml` is used **only if already importable** (soft
import) — it is not added to `requirements.txt` and its absence changes nothing
except that GPU sampling uses the `nvidia-smi` subprocess fallback. No new hard
dep added.

### Environment variables (all optional)

| Var | Default | Effect |
|-----|---------|--------|
| `COMFYUI_BENCHMARK_NO_FILE` | unset | `1`/`true`/`yes`/`on` disables the per-run JSON file sink (event still emitted). Same as `--benchmark-no-file`. |
| `COMFYUI_BENCHMARK_RETENTION` | `50` | Max per-run JSON files kept under `output/benchmarks/` (newest by mtime). Negative disables pruning. |
| `COMFYUI_BENCHMARK_INTERVAL_MS` | `500` | Hardware sampler cadence in ms. |

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
