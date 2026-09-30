"""In-core benchmark capture for ComfyUI (SPIKE).

Design goal: **zero overhead when not activated.** When a run does not opt in,
``start()`` returns ``None`` and installs nothing -- no wrappers are registered,
no sampler thread is spawned, no context object is created. The only cost on the
off-path is a single dict lookup in ``should_capture`` plus a ``None`` check per
node in ``execution.execute``.

Activation (both opt-in, off by default):
  1. Per-run:  a truthy ``extra_data["benchmark"]`` on the ``/prompt`` submission.
  2. Global:   the ``--benchmark`` CLI flag -> capture every run (soak mode).

Whenever capture is active (either activation), the event payload is also written
to a per-run file ``output/benchmarks/<prompt_id>.json`` so local orchestrators
that poll (no websocket tap) can fetch a specific run's record deterministically.
The file sink can be disabled (``--benchmark-no-file`` / ``COMFYUI_BENCHMARK_NO_FILE``)
for cloud/ephemeral containers while the websocket event still fires; the file
directory is retention-capped (``COMFYUI_BENCHMARK_RETENTION``, default 50) so it
never grows without bound. No file work happens when capture is off.

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

CAPTURE_SCHEMA_VERSION = 2
COLLECTOR_ID = "comfyui-core"

# Default number of per-run JSON reports to retain under output/benchmarks/.
_DEFAULT_RETENTION = 50


def _sample_interval_s() -> float:
    """Sampling cadence for the hardware sampler thread, in seconds.

    Defaults to 500ms (widened from the 250ms spike default to further reduce the
    observer effect of hardware sampling running concurrently with generation).
    Overridable via ``COMFYUI_BENCHMARK_INTERVAL_MS``.
    """
    try:
        v = os.environ.get("COMFYUI_BENCHMARK_INTERVAL_MS")
        if v:
            ms = float(v)
            if ms > 0:
                return ms / 1000.0
    except Exception:
        pass
    return 0.5


# Resolved once at import; the schema reports the *actual* interval used per run
# (BenchmarkContext.finalize reads it off the live sampler instance).
_SAMPLE_INTERVAL_S = _sample_interval_s()

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
    # Truthy back-compat: ``benchmark: true`` still opts in. The v2 object form
    # ``benchmark: { "id": ..., "version": ..., ... }`` is truthy too, so the
    # same gate captures both without special-casing.
    return bool(extra_data.get("benchmark"))


def _env_truthy(name: str) -> bool:
    """True if env var ``name`` is set to a truthy value (1/true/yes/on)."""
    v = os.environ.get(name)
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def file_sink_disabled() -> bool:
    """Whether the per-run JSON file sink is disabled for this process.

    The websocket ``benchmark`` event is *always* emitted; this only gates the
    ``output/benchmarks/<prompt_id>.json`` local artifact. Disabled via the
    ``--benchmark-no-file`` CLI flag or the ``COMFYUI_BENCHMARK_NO_FILE`` env var.
    Intended for cloud (ephemeral containers, no shared FS) where the event is the
    canonical channel and the file would be dead weight. Default: file stays on
    (Desktop's poller relies on it).
    """
    try:
        from comfy.cli_args import args
        if getattr(args, "benchmark_no_file", False):
            return True
    except Exception:
        pass
    return _env_truthy("COMFYUI_BENCHMARK_NO_FILE")


def _parse_optin_metadata(extra_data: dict) -> dict:
    """Extract the optional v2 opt-in metadata from ``extra_data["benchmark"]``.

    Accepts either the legacy truthy form (``true``/``1``) or the v2 object form
    ``{ "id", "version", "warmup_runs", "measured_runs", "seed" }``. Every field
    is best-effort: missing keys and the legacy form yield ``None``. Never raises.
    """
    meta = {
        "benchmark_id": None,
        "benchmark_version": None,
        "warmup_runs": None,
        "measured_runs": None,
        "seed": None,
    }
    try:
        opt = extra_data.get("benchmark")
        if isinstance(opt, dict):
            meta["benchmark_id"] = opt.get("id")
            meta["benchmark_version"] = opt.get("version")
            meta["warmup_runs"] = opt.get("warmup_runs")
            meta["measured_runs"] = opt.get("measured_runs")
            meta["seed"] = opt.get("seed")
    except Exception:
        pass
    return meta


def get_active() -> Optional["BenchmarkContext"]:
    """Return the live capture context, or None when capture is off."""
    return _active


def start(prompt_id: str, extra_data: dict, prompt: Optional[dict] = None) -> Optional["BenchmarkContext"]:
    """Begin capture for a run if opted in; otherwise do nothing and return None.

    ``prompt`` is the raw graph (node_id -> {class_type, inputs}); when provided
    it is parsed for structural workflow params (resolution/steps/sampler/etc.).
    It is never inspected for prompt text.
    """
    global _active
    if not should_capture(extra_data):
        return None  # zero-overhead path: nothing is created, nothing runs.

    # Singleton guard: only one capture context may be live at a time. If a prior
    # context was never finalized (e.g. a crash between start/finish left a stray
    # _active), restore its wrappers and drop it so it can't leak a wrapper layer
    # onto this run's install/restore.
    if _active is not None:
        logging.warning(
            "benchmark: an active capture context already exists (prompt_id=%s); "
            "resetting it before starting %s",
            getattr(_active, "prompt_id", "?"), prompt_id,
        )
        try:
            _active._restore_wrappers()
        except Exception:
            pass
        _active = None

    # A per-run JSON sink is written whenever capture is active (either the
    # per-run extra_data["benchmark"] opt-in or the global --benchmark flag),
    # UNLESS the file sink is disabled (--benchmark-no-file / env), so
    # orchestrators that poll (e.g. the Desktop /api/jobs runner) can read a
    # run's record without tapping the websocket. Keyed by prompt_id. In cloud the
    # event is the canonical channel and the file is disabled.
    ctx = BenchmarkContext(
        prompt_id=prompt_id,
        write_json=not file_sink_disabled(),
        prompt=prompt,
        optin_metadata=_parse_optin_metadata(extra_data),
    )
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
    """Finalize capture: emit the versioned event and write the per-run JSON file.

    ``executor`` is the ``PromptExecutor`` whose ``add_message`` reuses the
    existing message/event stream. Safe to call with ``ctx is None``.
    """
    global _active
    if ctx is None:
        return None
    try:
        try:
            ctx.set_produced_images(_count_produced_images(executor))
        except Exception:
            pass
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


def _count_produced_images(executor) -> Optional[int]:
    """Best-effort count of images actually produced this run.

    Reads the executor's ``history_result`` outputs and sums the length of any
    ``images`` (and ``gifs``) lists that output nodes (e.g. SaveImage) emitted.
    Returns ``None`` when nothing image-like was recorded so the consumer can
    fall back to ``run.batch_size``.
    """
    try:
        history = getattr(executor, "history_result", None) or {}
        outputs = history.get("outputs", {}) or {}
        total = 0
        found = False
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs"):
                seq = node_out.get(key)
                if isinstance(seq, list):
                    found = True
                    total += len(seq)
        return total if found else None
    except Exception:
        return None


def _literal(v):
    """Return a literal widget value, or None if it's a graph link.

    In the prompt graph an input is either a literal (int/float/str/bool) or a
    ``[node_id, output_index]`` link. Only literals are captured; links resolve
    at runtime and are not structural. Never raises.
    """
    if isinstance(v, list):
        return None
    return v


# Sampler-node class_types we know carry structural widget params. Matched by
# substring so *Advanced / vendor variants are still recognized.
_SAMPLER_HINTS = ("KSampler", "SamplerCustom")


def _extract_sampler(class_type: str, inputs: dict) -> dict:
    """Pull structural sampling params (never prompt text) from a sampler node."""
    seed = inputs.get("seed")
    if seed is None:
        seed = inputs.get("noise_seed")
    return {
        "node_id": None,  # filled by caller
        "class_type": class_type,
        "steps": _literal(inputs.get("steps")),
        "sampler": _literal(inputs.get("sampler_name")),
        "scheduler": _literal(inputs.get("scheduler")),
        "cfg": _literal(inputs.get("cfg")),
        "denoise": _literal(inputs.get("denoise")),
        "seed": _literal(seed),
    }


def _parse_workflow(prompt: Optional[dict]):
    """Extract structural-only workflow params from the prompt graph.

    NEVER reads prompt/CLIP text -- only numeric/enum widget values. Returns
    ``(workflow_dict, batch_size)``. The primary (first) sampler is promoted to
    top-level fields for convenience; ``workflow["samplers"]`` lists them all.
    Fully defensive: any parse failure yields a fully-null shape.
    """
    workflow = {
        "resolution": {"width": None, "height": None},
        "steps": None,
        "sampler": None,
        "scheduler": None,
        "cfg": None,
        "denoise": None,
        "seed": None,
        "samplers": [],
    }
    batch_size = None
    if not isinstance(prompt, dict):
        return workflow, batch_size

    try:
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type") or ""
            inputs = node.get("inputs") or {}
            if not isinstance(inputs, dict):
                continue

            # Resolution + batch size from an EmptyLatentImage-style node: any
            # node exposing width/height/batch_size widgets.
            is_latent = (
                class_type in ("EmptyLatentImage", "EmptySD3LatentImage")
                or ("width" in inputs and "height" in inputs)
            )
            if is_latent and workflow["resolution"]["width"] is None:
                w = _literal(inputs.get("width"))
                h = _literal(inputs.get("height"))
                if w is not None:
                    workflow["resolution"]["width"] = w
                if h is not None:
                    workflow["resolution"]["height"] = h
            if batch_size is None:
                bs = _literal(inputs.get("batch_size"))
                if bs is not None:
                    batch_size = bs

            # Sampler nodes.
            if any(h in class_type for h in _SAMPLER_HINTS):
                s = _extract_sampler(class_type, inputs)
                s["node_id"] = str(node_id)
                workflow["samplers"].append(s)

        if workflow["samplers"]:
            primary = workflow["samplers"][0]
            for key in ("steps", "sampler", "scheduler", "cfg", "denoise", "seed"):
                workflow[key] = primary.get(key)
    except Exception:
        logging.exception("benchmark: failed to parse workflow params")

    return workflow, batch_size


def _energy_wh_per_image(series: list, image_count: Optional[int]) -> Optional[float]:
    """Trapezoidal integration of power_w over the series, in Wh, per image.

    Uses consecutive samples where both endpoints report ``power_w``. Returns
    ``None`` when there are no usable power samples or no image count.
    """
    if not series or not image_count or image_count <= 0:
        return None
    try:
        total_wh = 0.0
        used = False
        for a, b in zip(series, series[1:]):
            pa, pb = a.get("power_w"), b.get("power_w")
            ta, tb = a.get("t_ms"), b.get("t_ms")
            if pa is None or pb is None or ta is None or tb is None:
                continue
            dt_h = (tb - ta) / 1000.0 / 3600.0  # ms -> s -> h
            if dt_h <= 0:
                continue
            total_wh += 0.5 * (pa + pb) * dt_h
            used = True
        if not used:
            return None
        return round(total_wh / image_count, 6)
    except Exception:
        return None


def _safe_filename(name: str) -> str:
    """Make a prompt_id safe to use as a filename (prompt_ids are normally UUIDs)."""
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in str(name))
    return cleaned or "run"


def _retention_limit() -> int:
    """Max number of per-run JSON reports to keep. Env-overridable.

    Defaults to ``_DEFAULT_RETENTION`` (50). Overridable via
    ``COMFYUI_BENCHMARK_RETENTION``. A negative value disables pruning
    (unbounded); ``0`` keeps none but the just-written file.
    """
    try:
        v = os.environ.get("COMFYUI_BENCHMARK_RETENTION")
        if v is not None and v.strip() != "":
            return int(v)
    except Exception:
        pass
    return _DEFAULT_RETENTION


def _prune_reports(out_dir: str, keep: int) -> None:
    """Keep only the newest ``keep`` ``*.json`` files in ``out_dir`` by mtime.

    Best-effort: never raises. A negative ``keep`` disables pruning. Prevents the
    benchmarks directory from growing without bound over many runs (soak mode /
    long-lived Desktop installs).
    """
    try:
        if keep is None or keep < 0:
            return  # pruning disabled -> unbounded (opt-out).
        import glob as _glob
        paths = _glob.glob(os.path.join(out_dir, "*.json"))
        if len(paths) <= keep:
            return
        # Newest first; delete everything past the retention window.
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for stale in paths[keep:]:
            try:
                os.remove(stale)
            except Exception:
                pass  # a concurrent reader/lock -> skip, retry next run.
    except Exception:
        # Retention is best-effort; a prune failure must never break the run.
        logging.debug("benchmark: report pruning failed", exc_info=True)


def _write_json_report(event: dict) -> str:
    """Write one file per run, keyed by prompt_id: output/benchmarks/<prompt_id>.json.

    After writing, prunes the directory to the newest ``_retention_limit()`` files
    so disk usage stays bounded across runs (B1).
    """
    import folder_paths
    out_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    name = _safe_filename(event.get("prompt_id", "") or "run")
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    logging.info("benchmark: wrote report %s", path)
    # Retention cap: the file just written has the newest mtime, so it survives.
    _prune_reports(out_dir, _retention_limit())
    return path


class BenchmarkContext:
    """Holds all metrics for one captured run and manages the capture machinery."""

    def __init__(
        self,
        prompt_id: str,
        write_json: bool = False,
        prompt: Optional[dict] = None,
        optin_metadata: Optional[dict] = None,
    ):
        self.prompt_id = prompt_id
        self.write_json = write_json
        self.prompt = prompt
        self.optin_metadata = optin_metadata or {}

        # A2: per-node timeline.
        self.node_timeline: list[dict[str, Any]] = []
        # A3: per-step timing (one entry per PREDICT_NOISE call = one denoise step).
        self.step_durations_ms: list[float] = []
        self._sampler_ms: float = 0.0  # sum of SAMPLER_SAMPLE wrapped time.

        # v2: run-level outcome + counts. Defaults to success; execution.py flips
        # this to "error"/"interrupted" via set_status() when a node fails/aborts.
        self.status: str = "success"
        self.produced_images: Optional[int] = None

        self._run_t0: Optional[float] = None
        self._orig_get_all_wrappers = None
        self.sampler = None  # HardwareSampler, created in start()
        self.baseline: Optional[dict[str, Any]] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._run_t0 = time.perf_counter()
        self._install_wrappers()
        from .sampler import HardwareSampler, baseline_sample, detect_backend
        backend = detect_backend()
        # One idle sample BEFORE any work, so the consumer can detect a
        # contaminated run (VRAM already full / GPU already busy at start).
        try:
            self.baseline = baseline_sample(backend)
        except Exception:
            self.baseline = None
        self.sampler = HardwareSampler(interval_s=_SAMPLE_INTERVAL_S, backend=backend)
        self.sampler.start()

    # -- v2 run-level setters (called from execution.py) -------------------
    def set_status(self, status: str) -> None:
        self.status = status

    def set_produced_images(self, count: Optional[int]) -> None:
        self.produced_images = count

    def finalize(self) -> dict:
        self._restore_wrappers()
        total_run_ms = (
            (time.perf_counter() - self._run_t0) * 1000.0 if self._run_t0 else 0.0
        )

        series: list = []
        peak: dict = {}
        env: dict = {}
        interval_s = _SAMPLE_INTERVAL_S
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler.join(timeout=5.0)  # joined at run end
            # Snapshot the series AFTER join so a late daemon-thread append can't
            # mutate the list mid-iteration while we summarize/serialize it.
            series = list(self.sampler.series)
            interval_s = getattr(self.sampler, "interval_s", _SAMPLE_INTERVAL_S)
            peak = self.sampler.peak()
            from .sampler import env_snapshot
            env = env_snapshot(self.sampler.backend)
            if env.get("total_vram_mb") is None and self.sampler.total_vram_mb() is not None:
                env["total_vram_mb"] = self.sampler.total_vram_mb()
        # v2: idle baseline sample taken at capture start (see start()).
        env["baseline"] = self.baseline

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

        # v2 groups: structural workflow params, run metadata, model-load proxy,
        # and derived per-image summaries.
        workflow, batch_size = _parse_workflow(self.prompt)
        image_count = self.produced_images if self.produced_images is not None else batch_size
        model_load_ms = self._model_load_ms()
        energy_wh_per_image = _energy_wh_per_image(series, image_count)
        sec_per_image = (
            round((total_run_ms / 1000.0) / image_count, 4)
            if image_count and image_count > 0 else None
        )

        return {
            "type": "benchmark",
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "collector_id": COLLECTOR_ID,
            "prompt_id": self.prompt_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "run": {
                "status": self.status,
                "image_count": image_count,
                "batch_size": batch_size,
                "benchmark_id": self.optin_metadata.get("benchmark_id"),
                "benchmark_version": self.optin_metadata.get("benchmark_version"),
                "warmup_runs": self.optin_metadata.get("warmup_runs"),
                "measured_runs": self.optin_metadata.get("measured_runs"),
                "seed": self.optin_metadata.get("seed"),
            },
            "workflow": workflow,
            "device": env,
            "durations": {
                "total_run_ms": round(total_run_ms, 3),
                "sampler_ms": round(self._sampler_ms, 3),
                "node_total_ms": node_total_ms,
                "model_load_ms": model_load_ms,
            },
            "nodes": self.node_timeline,
            "sampling": {
                "step_count": step_count,
                "step_durations_ms": self.step_durations_ms,
                "per_step_it_per_s": per_step_it_per_s,
                "avg_it_per_s": avg_it_per_s,
            },
            "resources": {
                "sample_interval_ms": round(interval_s * 1000.0),
                "series": series,
                "peak": peak,
            },
            "summary": {
                "energy_wh_per_image": energy_wh_per_image,
                "sec_per_image": sec_per_image,
                "throttled": peak.get("throttled") if isinstance(peak, dict) else None,
            },
        }

    def _model_load_ms(self) -> Optional[float]:
        """Best-effort cold model-load proxy.

        We cannot cleanly isolate the GPU transfer (ComfyUI loads weights lazily
        inside model_management at sample time), so this approximates it as the
        wall time of the first executed model-loading node -- any node whose
        class_type contains "Loader" or "Checkpoint" (CheckpointLoaderSimple,
        UNETLoader, VAELoader, CLIPLoader, ...). ``None`` if none executed.
        """
        for n in self.node_timeline:
            ct = n.get("class_type") or ""
            if "Loader" in ct or "Checkpoint" in ct:
                return n.get("elapsed_ms")
        return None

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
        # One PREDICT_NOISE call == one denoise step. Recording is best-effort:
        # a capture bug must never break the wrapped generation (m4).
        t = time.perf_counter()
        try:
            return executor(*args, **kwargs)
        finally:
            try:
                self.record_step((time.perf_counter() - t) * 1000.0)
            except Exception:
                logging.debug("benchmark: record_step failed", exc_info=True)

    def _sampler_sample_wrapper(self, executor, *args, **kwargs):
        # Brackets the whole sampling loop -> total sampler wall time. Recording is
        # best-effort so it can never break the wrapped sampling loop (m4).
        t = time.perf_counter()
        try:
            return executor(*args, **kwargs)
        finally:
            try:
                self.add_sampler_time((time.perf_counter() - t) * 1000.0)
            except Exception:
                logging.debug("benchmark: add_sampler_time failed", exc_info=True)
