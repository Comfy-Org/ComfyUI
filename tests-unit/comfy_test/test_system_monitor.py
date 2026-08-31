from types import SimpleNamespace

from comfy import system_monitor


class _Psutil:
    cpu_calls = 0

    @classmethod
    def cpu_percent(cls):
        cls.cpu_calls += 1
        return 37.5

    @staticmethod
    def virtual_memory():
        return SimpleNamespace(total=1000, available=400)

    @staticmethod
    def disk_partitions(all=False):
        assert all is False
        return [
            SimpleNamespace(mountpoint="/"),
            SimpleNamespace(mountpoint="/Volumes/Private\x00 Data"),
        ]

    @staticmethod
    def disk_usage(mountpoint):
        return SimpleNamespace(total=2000, free=500)


class _Device:
    type = "cuda"
    index = 0


class _Models:
    device = _Device()

    @classmethod
    def get_torch_device(cls):
        return cls.device

    @classmethod
    def get_all_torch_devices(cls):
        return [cls.device]

    @staticmethod
    def get_total_memory(device, torch_total_too=False):
        assert torch_total_too
        return 3000, 2500

    @staticmethod
    def get_free_memory(device, torch_free_too=False):
        assert torch_free_too
        return 1000, 900

    @staticmethod
    def get_torch_device_name(device):
        return "Fallback GPU"


class _Nvml:
    NVML_TEMPERATURE_GPU = 0

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index):
        return index

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        return SimpleNamespace(total=4000, free=1500)

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        return SimpleNamespace(gpu=61)

    @staticmethod
    def nvmlDeviceGetTemperature(handle, sensor):
        return 72

    @staticmethod
    def nvmlDeviceGetName(handle):
        return b"Example GPU"


def test_projection_contains_bounded_metrics_without_mount_paths():
    snapshot = system_monitor._collect_snapshot(_Psutil, _Models, _Nvml)

    assert snapshot == {
        "cpu": {"utilization_percent": 37.5},
        "memory": {"total": 1000, "available": 400},
        "volumes": [
            {"id": "volume-0", "label": "Root", "total": 2000, "available": 500},
            {"id": "volume-1", "label": "Private Data", "total": 2000, "available": 500},
        ],
        "accelerators": [{
            "id": "accelerator-0",
            "name": "Example GPU",
            "memory_total": 4000,
            "memory_available": 1500,
            "utilization_percent": 61.0,
            "temperature_c": 72.0,
        }],
    }
    assert "/Volumes" not in repr(snapshot)


def test_snapshot_is_cached_and_returned_as_an_independent_value(monkeypatch):
    times = iter([1.0, 1.1, 1.3])
    monkeypatch.setattr(system_monitor.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        system_monitor,
        "_collect_snapshot",
        lambda: {"sample": _Psutil.cpu_percent()},
    )
    monkeypatch.setattr(system_monitor, "_cached_at", -1.0)
    monkeypatch.setattr(system_monitor, "_cached_snapshot", None)
    _Psutil.cpu_calls = 0

    first = system_monitor.get_system_monitor_snapshot()
    first["sample"] = -1
    second = system_monitor.get_system_monitor_snapshot()
    third = system_monitor.get_system_monitor_snapshot()

    assert second == {"sample": 37.5}
    assert third == {"sample": 37.5}
    assert _Psutil.cpu_calls == 2


def test_missing_optional_sensors_are_null():
    snapshot = system_monitor._collect_snapshot(_Psutil, _Models, None)
    accelerator = snapshot["accelerators"][0]

    assert accelerator["name"] == "Fallback GPU"
    assert accelerator["utilization_percent"] is None
    assert accelerator["temperature_c"] is None


def test_byte_totals_fit_the_cross_language_safe_integer_contract():
    assert system_monitor._bytes(2**63) == 2**53 - 1
    assert system_monitor._bytes(-1) == 0
