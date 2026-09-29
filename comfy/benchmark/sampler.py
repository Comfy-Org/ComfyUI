"""Backend-aware hardware sampler for in-core benchmark capture (SPIKE).

This module is imported lazily by ``comfy.benchmark`` and only ever instantiated
when a run has opted in. It must never be touched on the zero-overhead off-path.

The sampler runs on its own daemon thread, polling CPU / RAM / VRAM on a fixed
interval, and is joined at the end of the run. All device probing is defensive:
a missing tool (e.g. ``nvidia-smi`` not on PATH) degrades to ``None`` fields
rather than raising into the hot path.
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


def _bytes_to_mb(n: Optional[float]) -> Optional[float]:
    if n is None:
        return None
    return round(n / (1024.0 * 1024.0), 2)


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
        cpu_percent, ram_used_mb = self._sample_cpu_ram()
        vram_used_mb, vram_util_percent, power_w = self._sample_vram()
        return {
            "t_ms": t_ms,
            "cpu_percent": cpu_percent,
            "ram_used_mb": ram_used_mb,
            "vram_used_mb": vram_used_mb,
            "vram_util_percent": vram_util_percent,
            "power_w": power_w,
        }

    def _sample_cpu_ram(self):
        try:
            import psutil
            cpu = psutil.cpu_percent(None)
            vm = psutil.virtual_memory()
            return cpu, _bytes_to_mb(vm.used)
        except Exception:
            return None, None

    def _sample_vram(self):
        if self.backend == BACKEND_CUDA:
            vals = _nvidia_smi_query(["memory.used", "utilization.gpu", "power.draw"])
            if vals is None:
                return None, None, None
            used = _to_float(vals[0] if len(vals) > 0 else None)
            util = _to_float(vals[1] if len(vals) > 1 else None)
            power = _to_float(vals[2] if len(vals) > 2 else None)
            return used, util, power
        if self.backend == BACKEND_MPS:
            try:
                import torch
                used = _bytes_to_mb(torch.mps.current_allocated_memory())
                return used, None, None
            except Exception:
                return None, None, None
        # CPU / XPU / unknown: no discrete VRAM reading in the spike.
        return None, None, None

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
        }

    def total_vram_mb(self) -> Optional[float]:
        return self._total_vram_mb


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

    if backend == BACKEND_CUDA:
        vals = _nvidia_smi_query(["name", "driver_version", "memory.total"])
        if vals:
            snap["gpu_model"] = vals[0] if len(vals) > 0 and vals[0] else None
            snap["driver_version"] = vals[1] if len(vals) > 1 and vals[1] else None
            snap["total_vram_mb"] = _to_float(vals[2] if len(vals) > 2 else None)
        if snap["gpu_model"] is None:
            try:
                import torch
                snap["gpu_model"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    elif backend == BACKEND_MPS:
        # Apple Silicon: unified memory, chip name via platform.
        snap["gpu_model"] = platform.processor() or "Apple Silicon"
        snap["driver_version"] = platform.mac_ver()[0] or None
        snap["total_vram_mb"] = snap["total_ram_mb"]  # unified

    return snap
