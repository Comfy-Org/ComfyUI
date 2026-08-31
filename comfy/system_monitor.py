from __future__ import annotations

import copy
import math
import threading
import time

import psutil

import comfy.model_management as model_management

try:
    import pynvml
except ImportError:
    pynvml = None


SAMPLE_INTERVAL_SECONDS = 0.25
MAX_VOLUMES = 64
MAX_ACCELERATORS = 64
MAX_LABEL_LENGTH = 128
MAX_NAME_LENGTH = 256
MAX_BYTES = 2**63 - 1

_cache_lock = threading.Lock()
_cached_at = -math.inf
_cached_snapshot: dict | None = None
_nvml_ready: bool | None = None


def _text(value, limit: int, fallback: str) -> str:
    text = " ".join(str(value).replace("\x00", "").split())[:limit]
    return text or fallback


def _bytes(value) -> int:
    return max(0, min(int(value), MAX_BYTES))


def _percent(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(number, 100.0))


def _temperature(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not -273.15 <= number <= 1000.0:
        return None
    return number


def _volume_label(mountpoint: str, index: int) -> str:
    logical = str(mountpoint).replace("\\", "/").rstrip("/")
    name = logical.rsplit("/", 1)[-1] if logical else "Root"
    return _text(name, MAX_LABEL_LENGTH, f"Volume {index + 1}")


def _volumes(psutil_module=psutil) -> list[dict]:
    partitions = sorted(
        psutil_module.disk_partitions(all=False),
        key=lambda item: str(item.mountpoint),
    )[:MAX_VOLUMES]
    result = []
    for index, partition in enumerate(partitions):
        try:
            usage = psutil_module.disk_usage(partition.mountpoint)
        except (OSError, PermissionError):
            continue
        total = _bytes(usage.total)
        available = min(_bytes(usage.free), total)
        result.append({
            "id": f"volume-{index}",
            "label": _volume_label(partition.mountpoint, index),
            "total": total,
            "available": available,
        })
    return result


def _nvml_available(nvml_module) -> bool:
    global _nvml_ready
    if nvml_module is None:
        return False
    if nvml_module is not pynvml:
        return True
    if _nvml_ready is None:
        try:
            nvml_module.nvmlInit()
            _nvml_ready = True
        except Exception:
            _nvml_ready = False
    return _nvml_ready


def _nvml_values(nvml_module, index: int) -> tuple:
    if not _nvml_available(nvml_module):
        return None, None, None, None, None
    try:
        handle = nvml_module.nvmlDeviceGetHandleByIndex(index)
        memory = nvml_module.nvmlDeviceGetMemoryInfo(handle)
        utilization = nvml_module.nvmlDeviceGetUtilizationRates(handle).gpu
        temperature = nvml_module.nvmlDeviceGetTemperature(
            handle, nvml_module.NVML_TEMPERATURE_GPU)
        name = nvml_module.nvmlDeviceGetName(handle)
        return memory.total, memory.free, utilization, temperature, name
    except Exception:
        return None, None, None, None, None


def _accelerators(model_management_module=model_management, nvml_module=pynvml) -> list[dict]:
    primary = model_management_module.get_torch_device()
    devices = list(model_management_module.get_all_torch_devices())
    if primary in devices:
        devices = [primary, *(device for device in devices if device != primary)]
    else:
        devices.insert(0, primary)

    devices = [
        device for device in devices
        if getattr(device, "type", None) != "cpu"
    ][:MAX_ACCELERATORS]
    result = []
    for position, device in enumerate(devices):
        total, _torch_total = model_management_module.get_total_memory(
            device, torch_total_too=True)
        available, _torch_available = model_management_module.get_free_memory(
            device, torch_free_too=True)
        utilization = None
        temperature = None
        nvml_name = None
        if getattr(device, "type", None) == "cuda":
            index = getattr(device, "index", None)
            nvml_index = 0 if index is None else int(index)
            nvml_total, nvml_free, utilization, temperature, nvml_name = (
                _nvml_values(nvml_module, nvml_index))
            if nvml_total is not None:
                total, available = nvml_total, nvml_free
        name = nvml_name or model_management_module.get_torch_device_name(device)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        total = _bytes(total)
        available = min(_bytes(available), total)
        result.append({
            "id": f"accelerator-{position}",
            "name": _text(name, MAX_NAME_LENGTH, f"Accelerator {position + 1}"),
            "memory_total": total,
            "memory_available": available,
            "utilization_percent": _percent(utilization),
            "temperature_c": _temperature(temperature),
        })
    return result


def _collect_snapshot(
    psutil_module=psutil,
    model_management_module=model_management,
    nvml_module=pynvml,
) -> dict:
    memory = psutil_module.virtual_memory()
    total = _bytes(memory.total)
    available = min(_bytes(memory.available), total)
    return {
        "cpu": {"utilization_percent": _percent(psutil_module.cpu_percent())},
        "memory": {"total": total, "available": available},
        "volumes": _volumes(psutil_module),
        "accelerators": _accelerators(model_management_module, nvml_module),
    }


def get_system_monitor_snapshot() -> dict:
    global _cached_at, _cached_snapshot
    now = time.monotonic()
    with _cache_lock:
        if _cached_snapshot is None or now - _cached_at >= SAMPLE_INTERVAL_SECONDS:
            _cached_snapshot = _collect_snapshot()
            _cached_at = now
        return copy.deepcopy(_cached_snapshot)
