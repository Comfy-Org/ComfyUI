"""Backend-aware hardware sampler for in-core benchmark capture (SPIKE).

This module is imported lazily by ``comfy.benchmark`` and only ever instantiated
when a run has opted in. It must never be touched on the zero-overhead off-path.

The sampler runs on its own daemon thread, polling CPU / RAM / VRAM on a fixed
interval, and is joined at the end of the run. All device probing is defensive:
a missing tool (e.g. ``nvidia-smi`` not on PATH) degrades to ``None`` fields
rather than raising into the hot path.

Schema v2 extends each series point with GPU temperature / clocks / power-limit
(one nvidia-smi call per sample, same overhead profile), adds a one-shot idle
``baseline`` sample, and enriches the device snapshot (CUDA/cuDNN versions,
compute capability, PCIe link, vram_state, dtypes, attention impl, laptop hint).
Every field is best-effort: the key is always present, ``None`` when unavailable.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from typing import Any, Optional

# Backend identifiers (kept as plain strings so the schema is stable/portable).
BACKEND_CUDA = "cuda"
BACKEND_MPS = "mps"
BACKEND_XPU = "xpu"
BACKEND_CPU = "cpu"
BACKEND_UNKNOWN = "unknown"

# Per-sample nvidia-smi query fields, in order. Extended in v2 with temperature,
# SM/mem clocks and the power limit so the resource series can show throttling.
_SMI_SAMPLE_FIELDS = [
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
]

# Heuristic throttle thresholds (best-effort; documented in SPIKE.md).
_THROTTLE_TEMP_C = 83.0          # NVIDIA consumer thermal-throttle neighborhood.
_THROTTLE_POWER_FRACTION = 0.98  # sustained draw this close to the cap == capped.


def _bytes_to_mb(n: Optional[float]) -> Optional[float]:
    if n is None:
        return None
    return round(n / (1024.0 * 1024.0), 2)


def _dtype_str(dt: Any) -> Optional[str]:
    """Normalize a torch dtype (or anything) to a short string like ``fp16``.

    ``torch.float16`` -> ``"float16"``; unknown/None -> ``None``. Never raises.
    """
    if dt is None:
        return None
    try:
        s = str(dt)
    except Exception:
        return None
    if s.startswith("torch."):
        s = s[len("torch."):]
    return s or None


def detect_backend() -> str:
    """Best-effort backend detection reusing ComfyUI's model_management state."""
    try:
        import comfy.model_management as mm
        state = mm.cpu_state
        if state == mm.CPUState.MPS:
            return BACKEND_MPS
        if state == mm.CPUState.CPU:
            return BACKEND_CPU
        # GPU state: distinguish CUDA vs XPU.
        try:
            if mm.is_intel_xpu():
                return BACKEND_XPU
        except Exception:
            pass
        import torch
        if torch.cuda.is_available():
            return BACKEND_CUDA
        return BACKEND_UNKNOWN
    except Exception:
        return BACKEND_UNKNOWN


def _nvidia_smi_query(fields: list[str]) -> Optional[list[str]]:
    """Run a single-GPU nvidia-smi query. Returns the split values or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        return [v.strip() for v in line.split(",")]
    except Exception:
        return None


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if f != f:  # NaN guard
        return None
    return f


def _to_int(v: Optional[str]) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None


def _query_gpu(backend: str) -> dict[str, Any]:
    """One-shot GPU metric probe shared by the sampler thread and the baseline.

    Returns the full v2 metric set with ``None`` for anything unavailable. On
    CUDA this is a single nvidia-smi call; on MPS only ``vram_used_mb`` is known;
    otherwise everything is ``None``.
    """
    out: dict[str, Any] = {
        "vram_used_mb": None,
        "vram_util_percent": None,
        "power_w": None,
        "temperature_c": None,
        "sm_clock_mhz": None,
        "mem_clock_mhz": None,
        "power_limit_w": None,
    }
    if backend == BACKEND_CUDA:
        vals = _nvidia_smi_query(_SMI_SAMPLE_FIELDS)
        if vals is None:
            return out

        def _at(i):
            return vals[i] if len(vals) > i else None

        out["vram_used_mb"] = _to_float(_at(0))
        out["vram_util_percent"] = _to_float(_at(1))
        out["power_w"] = _to_float(_at(2))
        out["temperature_c"] = _to_float(_at(3))
        out["sm_clock_mhz"] = _to_float(_at(4))
        out["mem_clock_mhz"] = _to_float(_at(5))
        out["power_limit_w"] = _to_float(_at(6))
        return out
    if backend == BACKEND_MPS:
        try:
            import torch
            out["vram_used_mb"] = _bytes_to_mb(torch.mps.current_allocated_memory())
        except Exception:
            pass
        return out
    # CPU / XPU / unknown: no discrete VRAM reading in the spike.
    return out


def _sample_cpu_ram():
    try:
        import psutil
        cpu = psutil.cpu_percent(None)
        vm = psutil.virtual_memory()
        return cpu, _bytes_to_mb(vm.used)
    except Exception:
        return None, None


def baseline_sample(backend: Optional[str] = None) -> dict[str, Any]:
    """A single ``/system_stats``-style idle sample taken BEFORE any work.

    Lets the consumer detect a contaminated run (e.g. VRAM already full, GPU
    already hot/busy when capture started). Fully defensive; keys always present.
    """
    backend = backend or detect_backend()
    cpu_percent, ram_used_mb = _sample_cpu_ram()
    gpu = _query_gpu(backend)
    return {
        "vram_used_mb": gpu["vram_used_mb"],
        "vram_util_percent": gpu["vram_util_percent"],
        "temperature_c": gpu["temperature_c"],
        "ram_used_mb": ram_used_mb,
        "cpu_percent": cpu_percent,
    }


class HardwareSampler(threading.Thread):
    """Samples CPU/RAM/VRAM on an interval on its own thread.

    Series entries are plain dicts with a stable shape (see SPIKE.md). Missing
    metrics for a backend are recorded as ``None`` rather than omitted, so the
    consumer sees a consistent schema regardless of hardware.
    """

    def __init__(self, interval_s: float = 0.25, backend: Optional[str] = None):
        super().__init__(name="comfy-benchmark-sampler", daemon=True)
        self.interval_s = interval_s
        self.backend = backend or detect_backend()
        self.series: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self.vram_is_unified = self.backend == BACKEND_MPS
        # Cache total VRAM once (does not change across the run).
        self._total_vram_mb = self._probe_total_vram_mb()

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(None)  # prime the interval-based reading
        except Exception:
            pass
        # Sample immediately, then on each interval, then once more on stop so
        # short runs still yield a couple of points.
        self.series.append(self._sample())
        while not self._stop.wait(self.interval_s):
            self.series.append(self._sample())
        self.series.append(self._sample())

    def stop(self) -> None:
        self._stop.set()

    # -- sampling ----------------------------------------------------------
    def _sample(self) -> dict[str, Any]:
        t_ms = round((time.perf_counter() - self._t0) * 1000.0, 2)
        cpu_percent, ram_used_mb = _sample_cpu_ram()
        gpu = _query_gpu(self.backend)
        return {
            "t_ms": t_ms,
            "cpu_percent": cpu_percent,
            "ram_used_mb": ram_used_mb,
            "vram_used_mb": gpu["vram_used_mb"],
            "vram_util_percent": gpu["vram_util_percent"],
            "power_w": gpu["power_w"],
            "temperature_c": gpu["temperature_c"],
            "sm_clock_mhz": gpu["sm_clock_mhz"],
            "mem_clock_mhz": gpu["mem_clock_mhz"],
            "power_limit_w": gpu["power_limit_w"],
        }

    # -- summaries ---------------------------------------------------------
    def _probe_total_vram_mb(self) -> Optional[float]:
        if self.backend == BACKEND_CUDA:
            vals = _nvidia_smi_query(["memory.total"])
            if vals:
                return _to_float(vals[0])
            return None
        if self.backend == BACKEND_MPS:
            try:
                import torch
                # Recommended max working set; falls back to unified RAM total.
                return _bytes_to_mb(torch.mps.driver_allocated_memory())
            except Exception:
                return None
        return None

    def _throttled(self) -> Optional[bool]:
        """Best-effort throttle heuristic over the series.

        True if peak temp reaches the throttle neighborhood, or sustained power
        draw sits at/above ~98% of the reported limit (power-capped). ``None``
        when neither signal is available. We cannot see the card's base clock,
        so "clocks below base" is not part of this heuristic.
        """
        temps = [s["temperature_c"] for s in self.series if s.get("temperature_c") is not None]
        hot = None
        if temps:
            hot = max(temps) >= _THROTTLE_TEMP_C
        power_capped = None
        for s in self.series:
            pw = s.get("power_w")
            pl = s.get("power_limit_w")
            if pw is not None and pl is not None and pl > 0:
                if power_capped is None:
                    power_capped = False
                if pw >= _THROTTLE_POWER_FRACTION * pl:
                    power_capped = True
                    break
        if hot is None and power_capped is None:
            return None
        return bool(hot) or bool(power_capped)

    def peak(self) -> dict[str, Any]:
        def _max(key):
            vals = [s[key] for s in self.series if s.get(key) is not None]
            return max(vals) if vals else None
        return {
            "vram_used_mb": _max("vram_used_mb"),
            "ram_used_mb": _max("ram_used_mb"),
            "cpu_percent": _max("cpu_percent"),
            "vram_util_percent": _max("vram_util_percent"),
            "power_w": _max("power_w"),
            "temperature_c": _max("temperature_c"),
            "sm_clock_mhz": _max("sm_clock_mhz"),
            "mem_clock_mhz": _max("mem_clock_mhz"),
            "power_limit_w": _max("power_limit_w"),
            "throttled": self._throttled(),
        }

    def total_vram_mb(self) -> Optional[float]:
        return self._total_vram_mb


def _device_runtime_snapshot() -> dict[str, Any]:
    """model_management-derived fields (vram_state, dtypes, attention impl).

    Read at finalize so models are loaded. Every probe is independent and
    defensive so one missing helper never blanks the rest.
    """
    snap: dict[str, Any] = {
        "vram_state": None,
        "offloaded": None,
        "weight_dtype": None,
        "compute_dtype": None,
        "attention_impl": None,
    }
    try:
        import comfy.model_management as mm
    except Exception:
        return snap

    try:
        state = mm.vram_state
        snap["vram_state"] = getattr(state, "name", str(state))
        # Offload/low-vram modes move weights on/off the device during the run.
        snap["offloaded"] = state in (
            mm.VRAMState.NO_VRAM, mm.VRAMState.LOW_VRAM, mm.VRAMState.SHARED,
        )
    except Exception:
        pass

    # Prefer the actually-loaded diffusion model's dtypes; fall back to helpers.
    try:
        loaded = mm.loaded_models()
        base = None
        for patcher in loaded:
            candidate = getattr(patcher, "model", None)
            if candidate is not None and hasattr(candidate, "get_dtype"):
                base = candidate
                break
        if base is not None:
            snap["weight_dtype"] = _dtype_str(base.get_dtype())
            if hasattr(base, "get_dtype_inference"):
                snap["compute_dtype"] = _dtype_str(base.get_dtype_inference())
    except Exception:
        pass
    if snap["compute_dtype"] is None:
        try:
            snap["compute_dtype"] = _dtype_str(mm.unet_dtype())
        except Exception:
            pass

    try:
        if mm.sage_attention_enabled():
            snap["attention_impl"] = "sage"
        elif mm.flash_attention_enabled():
            snap["attention_impl"] = "flash"
        elif mm.xformers_enabled():
            snap["attention_impl"] = "xformers"
        elif mm.pytorch_attention_enabled():
            snap["attention_impl"] = "pytorch"
    except Exception:
        pass

    return snap


def env_snapshot(backend: Optional[str] = None) -> dict[str, Any]:
    """One-shot device/env snapshot. Backend-aware, fully defensive."""
    backend = backend or detect_backend()
    snap: dict[str, Any] = {
        "backend": backend,
        "gpu_model": None,
        "driver_version": None,
        "vram_is_unified": backend == BACKEND_MPS,
        "pytorch_version": None,
        "comfyui_version": None,
        "os": platform.platform(),
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "cpu_model": platform.processor() or None,
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "total_vram_mb": None,
        "total_ram_mb": None,
        # -- v2 additions (Tier 2) ----------------------------------------
        "vram_state": None,
        "offloaded": None,
        "weight_dtype": None,
        "compute_dtype": None,
        "attention_impl": None,
        "cuda_version": None,
        "cudnn_version": None,
        "compute_capability": None,
        "is_laptop": None,
        "pcie_gen": None,
        "pcie_width": None,
    }

    try:
        import torch
        snap["pytorch_version"] = torch.__version__
    except Exception:
        pass

    try:
        from comfyui_version import __version__ as cv
        snap["comfyui_version"] = cv
    except Exception:
        pass

    try:
        import psutil
        snap["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        snap["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        snap["total_ram_mb"] = _bytes_to_mb(psutil.virtual_memory().total)
    except Exception:
        pass

    # Laptop hint: a battery usually means a portable machine. Absence of a
    # battery does not prove desktop, but its presence is a strong signal.
    try:
        import psutil
        snap["is_laptop"] = psutil.sensors_battery() is not None
    except Exception:
        snap["is_laptop"] = None

    # model_management-derived runtime fields (all backends).
    snap.update(_device_runtime_snapshot())

    if backend == BACKEND_CUDA:
        vals = _nvidia_smi_query([
            "name", "driver_version", "memory.total",
            "pcie.link.gen.current", "pcie.link.width.current",
        ])
        if vals:
            snap["gpu_model"] = vals[0] if len(vals) > 0 and vals[0] else None
            snap["driver_version"] = vals[1] if len(vals) > 1 and vals[1] else None
            snap["total_vram_mb"] = _to_float(vals[2] if len(vals) > 2 else None)
            snap["pcie_gen"] = _to_int(vals[3] if len(vals) > 3 else None)
            snap["pcie_width"] = _to_int(vals[4] if len(vals) > 4 else None)
        if snap["gpu_model"] is None:
            try:
                import torch
                snap["gpu_model"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        try:
            import torch
            snap["cuda_version"] = torch.version.cuda
        except Exception:
            pass
        try:
            import torch
            snap["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            pass
        try:
            import torch
            cap = torch.cuda.get_device_capability(0)
            snap["compute_capability"] = "{}.{}".format(cap[0], cap[1])
        except Exception:
            pass
    elif backend == BACKEND_MPS:
        # Apple Silicon: unified memory, chip name via platform.
        snap["gpu_model"] = platform.processor() or "Apple Silicon"
        snap["driver_version"] = platform.mac_ver()[0] or None
        snap["total_vram_mb"] = snap["total_ram_mb"]  # unified

    return snap
