"""In-core benchmark capture for ComfyUI (SPIKE).

Design goal: **zero overhead when not activated.** When a run does not opt in,
``start()`` returns ``None`` and installs nothing -- no wrappers are registered,
no sampler thread is spawned, no context object is created. The only cost on the
off-path is a single dict lookup in ``should_capture`` plus a ``None`` check per
node in ``execution.execute``.

Activation (both opt-in, off by default):
  1. Per-run:  a truthy ``extra_data["benchmark"]`` on the ``/prompt`` submission.
  2. Global:   the ``--benchmark`` CLI flag -> capture every run *and* also write
               a JSON report to ``output/benchmarks/<timestamp>.json`` (soak mode).

The emitted ``benchmark`` event is the canonical corpus record shape. Its schema
is versioned via ``capture_schema_version`` and documented in ``SPIKE.md`` -- that
document is the contract downstream consumers build against.

NOTE ON WRAPPER REGISTRATION (reality differs from the original brief):
ComfyUI has no *global* wrapper registry -- ``comfy.patcher_extension`` reads
SAMPLER_SAMPLE / PREDICT_NOISE wrappers out of each model's ``model_options`` at
sampling time. To register capture wrappers for an arbitrary run without touching
every model patcher, we temporarily wrap ``patcher_extension.get_all_wrappers``
for the duration of the run and restore it at the end. This install happens only
when capture is active, so the off-path is untouched.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Any, Optional

CAPTURE_SCHEMA_VERSION = 1
COLLECTOR_ID = "comfyui-core"

# Sampling cadence for the hardware sampler thread.
_SAMPLE_INTERVAL_S = 0.25

# The single active capture context, or None when nothing is being captured.
_active: Optional["BenchmarkContext"] = None


def should_capture(extra_data: dict) -> bool:
    """The single gate. Returns True iff this run should be benchmarked.

    Global ``--benchmark`` wins for every run; otherwise honor the per-run
    ``extra_data["benchmark"]`` opt-in. Kept dependency-light so the off-path
    stays cheap.
    """
    try:
        from comfy.cli_args import args
        if getattr(args, "benchmark", False):
            return True
    except Exception:
        pass
    return bool(extra_data.get("benchmark"))


def get_active() -> Optional["BenchmarkContext"]:
    """Return the live capture context, or None when capture is off."""
    return _active


def start(prompt_id: str, extra_data: dict) -> Optional["BenchmarkContext"]:
    """Begin capture for a run if opted in; otherwise do nothing and return None."""
    global _active
    if not should_capture(extra_data):
        return None  # zero-overhead path: nothing is created, nothing runs.

    write_json = False
    try:
        from comfy.cli_args import args
        write_json = bool(getattr(args, "benchmark", False))
    except Exception:
        pass

    ctx = BenchmarkContext(prompt_id=prompt_id, write_json=write_json)
    try:
        ctx.start()
    except Exception:
        # Capture must never break a run. Roll back any partial setup and
        # continue as if benchmarking was off.
        logging.exception("benchmark: failed to start capture; disabling for this run")
        try:
            ctx._restore_wrappers()
        except Exception:
            pass
        return None
    _active = ctx
    return ctx


def finish(ctx: Optional["BenchmarkContext"], executor) -> Optional[dict]:
    """Finalize capture: emit the versioned event and, in soak mode, write JSON.

    ``executor`` is the ``PromptExecutor`` whose ``add_message`` reuses the
    existing message/event stream. Safe to call with ``ctx is None``.
    """
    global _active
    if ctx is None:
        return None
    try:
        try:
            event = ctx.finalize()
        except Exception:
            # Never let capture teardown break a run; ensure wrappers are restored.
            logging.exception("benchmark: failed to finalize capture")
            try:
                ctx._restore_wrappers()
            except Exception:
                pass
            return None
        try:
            executor.add_message("benchmark", event, broadcast=False)
        except Exception:
            logging.exception("benchmark: failed to emit event")
        if ctx.write_json:
            try:
                _write_json_report(event)
            except Exception:
                logging.exception("benchmark: failed to write JSON report")
        return event
    finally:
        _active = None


def _write_json_report(event: dict) -> str:
    import folder_paths
    out_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    ts = event.get("timestamp", "").replace(":", "-") or datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = os.path.join(out_dir, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    logging.info("benchmark: wrote report %s", path)
    return path


class BenchmarkContext:
    """Holds all metrics for one captured run and manages the capture machinery."""

    def __init__(self, prompt_id: str, write_json: bool = False):
        self.prompt_id = prompt_id
        self.write_json = write_json

        # A2: per-node timeline.
        self.node_timeline: list[dict[str, Any]] = []
        # A3: per-step timing (one entry per PREDICT_NOISE call = one denoise step).
        self.step_durations_ms: list[float] = []
        self._sampler_ms: float = 0.0  # sum of SAMPLER_SAMPLE wrapped time.

        self._run_t0: Optional[float] = None
        self._orig_get_all_wrappers = None
        self.sampler = None  # HardwareSampler, created in start()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._run_t0 = time.perf_counter()
        self._install_wrappers()
        from .sampler import HardwareSampler
        self.sampler = HardwareSampler(interval_s=_SAMPLE_INTERVAL_S)
        self.sampler.start()

    def finalize(self) -> dict:
        self._restore_wrappers()
        total_run_ms = (
            (time.perf_counter() - self._run_t0) * 1000.0 if self._run_t0 else 0.0
        )

        series: list = []
        peak: dict = {}
        env: dict = {}
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler.join(timeout=5.0)  # joined at run end
            series = self.sampler.series
            peak = self.sampler.peak()
            from .sampler import env_snapshot
            env = env_snapshot(self.sampler.backend)
            if env.get("total_vram_mb") is None and self.sampler.total_vram_mb() is not None:
                env["total_vram_mb"] = self.sampler.total_vram_mb()

        step_count = len(self.step_durations_ms)
        sampler_s = self._sampler_ms / 1000.0
        avg_it_per_s = (
            round(step_count / sampler_s, 4) if sampler_s > 0 and step_count > 0 else None
        )
        # One entry per step, index-aligned with step_durations_ms (None when a
        # step duration is zero/unmeasurable).
        per_step_it_per_s = [
            round(1000.0 / d, 4) if d > 0 else None for d in self.step_durations_ms
        ]
        node_total_ms = round(
            sum(n["elapsed_ms"] for n in self.node_timeline), 3
        )

        return {
            "type": "benchmark",
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "collector_id": COLLECTOR_ID,
            "prompt_id": self.prompt_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "device": env,
            "durations": {
                "total_run_ms": round(total_run_ms, 3),
                "sampler_ms": round(self._sampler_ms, 3),
                "node_total_ms": node_total_ms,
            },
            "nodes": self.node_timeline,
            "sampling": {
                "step_count": step_count,
                "step_durations_ms": self.step_durations_ms,
                "per_step_it_per_s": per_step_it_per_s,
                "avg_it_per_s": avg_it_per_s,
            },
            "resources": {
                "sample_interval_ms": round(_SAMPLE_INTERVAL_S * 1000.0),
                "series": series,
                "peak": peak,
            },
        }

    # -- recording (called from the hot path only when active) -------------
    def record_node(self, node_id: str, class_type: str, elapsed_ms: float) -> None:
        self.node_timeline.append(
            {
                "node_id": str(node_id),
                "class_type": class_type,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    def record_step(self, elapsed_ms: float) -> None:
        self.step_durations_ms.append(round(elapsed_ms, 3))

    def add_sampler_time(self, elapsed_ms: float) -> None:
        self._sampler_ms += elapsed_ms

    # -- wrapper install / restore (see module docstring) ------------------
    def _install_wrappers(self) -> None:
        import comfy.patcher_extension as pe

        self._orig_get_all_wrappers = pe.get_all_wrappers
        orig = pe.get_all_wrappers
        ctx = self

        def patched_get_all_wrappers(wrapper_type, transformer_options, is_model_options=False):
            wl = orig(wrapper_type, transformer_options, is_model_options=is_model_options)
            if wrapper_type == pe.WrappersMP.PREDICT_NOISE:
                return wl + [ctx._predict_noise_wrapper]
            if wrapper_type == pe.WrappersMP.SAMPLER_SAMPLE:
                return wl + [ctx._sampler_sample_wrapper]
            return wl

        pe.get_all_wrappers = patched_get_all_wrappers

    def _restore_wrappers(self) -> None:
        if self._orig_get_all_wrappers is not None:
            import comfy.patcher_extension as pe
            pe.get_all_wrappers = self._orig_get_all_wrappers
            self._orig_get_all_wrappers = None

    def _predict_noise_wrapper(self, executor, *args, **kwargs):
        # One PREDICT_NOISE call == one denoise step.
        t = time.perf_counter()
        try:
            return executor(*args, **kwargs)
        finally:
            self.record_step((time.perf_counter() - t) * 1000.0)

    def _sampler_sample_wrapper(self, executor, *args, **kwargs):
        # Brackets the whole sampling loop -> total sampler wall time.
        t = time.perf_counter()
        try:
            return executor(*args, **kwargs)
        finally:
            self.add_sampler_time((time.perf_counter() - t) * 1000.0)
