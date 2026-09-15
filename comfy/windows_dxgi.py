from __future__ import annotations

import ctypes
import logging
import platform
import time
import uuid
from dataclasses import dataclass


DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1
DXGI_ERROR_NOT_FOUND = 0x887A0002


class DXGIUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMemoryInfo:
    budget: int
    current_usage: int
    available_for_reservation: int = 0
    current_reservation: int = 0


def safe_headroom(info, reserve):
    return max(0, info.budget - info.current_usage - reserve)


class LivePinBudget:
    def __init__(self, provider, reserve, hysteresis):
        self.provider = provider
        self.reserve = reserve
        self.hysteresis = hysteresis
        self.last_info = None
        self.last_headroom = None
        self.evicted = 0
        self.query_count = 0
        self.query_ns = 0
        self.max_query_ns = 0

    def query(self):
        start = time.perf_counter_ns()
        try:
            return self.provider.query()
        finally:
            elapsed = time.perf_counter_ns() - start
            self.query_count += 1
            self.query_ns += elapsed
            self.max_query_ns = max(self.max_query_ns, elapsed)

    def ensure(self, size, evict):
        info = self.query()
        self.last_info = info
        margin = info.budget - info.current_usage - self.reserve
        self.last_headroom = max(0, margin)
        if margin >= size:
            return True

        self.evicted += evict(size - margin + self.hysteresis)
        info = self.query()
        self.last_info = info
        margin = info.budget - info.current_usage - self.reserve
        self.last_headroom = max(0, margin)
        return margin >= size


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, value):
        parsed = uuid.UUID(value)
        super().__init__(parsed.time_low, parsed.time_mid, parsed.time_hi_version, (ctypes.c_ubyte * 8)(*parsed.bytes[8:]))


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


_IID_IDXGIFACTORY1 = _GUID("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_IDXGIADAPTER3 = _GUID("645967a4-1392-4310-a798-8053ce3e93fd")
_COMFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


def _com_method(pointer, index, result_type, *argument_types):
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return _COMFUNCTYPE(result_type, ctypes.c_void_p, *argument_types)(vtable[index])


def _release(pointer):
    if pointer:
        _com_method(pointer, 2, ctypes.c_ulong)(pointer)


class _DXGIAdapter:
    def __init__(self, pointer, luid, description):
        self.pointer = pointer
        self.luid = luid
        self.description = description

    def close(self):
        pointer = self.pointer
        self.pointer = None
        _release(pointer)

    def query(self, node_index=0):
        info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        result = _com_method(
            self.pointer,
            14,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
        )(self.pointer, node_index, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, ctypes.byref(info))
        if result != 0:
            raise DXGIUnavailable(f"QueryVideoMemoryInfo failed with HRESULT 0x{result & 0xffffffff:08x}")
        return VideoMemoryInfo(info.Budget, info.CurrentUsage, info.AvailableForReservation, info.CurrentReservation)

    def __del__(self):
        self.close()


def _enumerate_adapters():
    dxgi = ctypes.WinDLL("dxgi.dll")
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    result = create_factory(ctypes.byref(_IID_IDXGIFACTORY1), ctypes.byref(factory))
    if result != 0:
        raise DXGIUnavailable(f"CreateDXGIFactory1 failed with HRESULT 0x{result & 0xffffffff:08x}")

    adapters = []
    try:
        enum_adapters = _com_method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        index = 0
        while True:
            adapter1 = ctypes.c_void_p()
            result = enum_adapters(factory, index, ctypes.byref(adapter1))
            if result & 0xffffffff == DXGI_ERROR_NOT_FOUND:
                break
            if result != 0:
                raise DXGIUnavailable(f"EnumAdapters1 failed with HRESULT 0x{result & 0xffffffff:08x}")

            try:
                desc = _DXGI_ADAPTER_DESC1()
                result = _com_method(adapter1, 10, ctypes.c_long, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(adapter1, ctypes.byref(desc))
                if result != 0:
                    raise DXGIUnavailable(f"GetDesc1 failed with HRESULT 0x{result & 0xffffffff:08x}")

                adapter3 = ctypes.c_void_p()
                result = _com_method(adapter1, 0, ctypes.c_long, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))(
                    adapter1, ctypes.byref(_IID_IDXGIADAPTER3), ctypes.byref(adapter3)
                )
                if result == 0:
                    luid = ctypes.string_at(ctypes.byref(desc.AdapterLuid), ctypes.sizeof(desc.AdapterLuid))
                    adapters.append(_DXGIAdapter(adapter3, luid, desc.Description))
            finally:
                _release(adapter1)
            index += 1
    except Exception:
        for adapter in adapters:
            adapter.close()
        raise
    finally:
        _release(factory)
    return adapters


def _cuda_device_luid(device_index):
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceGetLuid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
    cuda.cuDeviceGetLuid.restype = ctypes.c_int

    if cuda.cuInit(0) != 0:
        raise DXGIUnavailable("cuInit failed")
    device = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(device), device_index) != 0:
        raise DXGIUnavailable(f"CUDA device {device_index} is unavailable")
    luid = (ctypes.c_ubyte * 8)()
    node_mask = ctypes.c_uint()
    if cuda.cuDeviceGetLuid(ctypes.byref(luid), ctypes.byref(node_mask), device) != 0:
        raise DXGIUnavailable(f"CUDA device {device_index} has no Windows LUID")
    if node_mask.value == 0 or node_mask.value & (node_mask.value - 1):
        raise DXGIUnavailable(f"CUDA device {device_index} has ambiguous DXGI node mask 0x{node_mask.value:x}")
    return bytes(luid), node_mask.value.bit_length() - 1


class DXGINonLocalBudgetProvider:
    def __init__(self, device_index, cuda_luid_getter=_cuda_device_luid, adapter_enumerator=_enumerate_adapters):
        self.adapter = None
        cuda_mapping = cuda_luid_getter(device_index)
        if isinstance(cuda_mapping, tuple):
            cuda_luid, self.node_index = cuda_mapping
        else:
            cuda_luid, self.node_index = cuda_mapping, 0
        adapters = adapter_enumerator()
        matches = [adapter for adapter in adapters if adapter.luid == cuda_luid]
        if len(matches) != 1:
            for adapter in adapters:
                adapter.close()
            raise DXGIUnavailable(f"CUDA device {device_index} matched {len(matches)} DXGI adapters")
        self.adapter = matches[0]
        for adapter in adapters:
            if adapter is not self.adapter:
                adapter.close()

    def query(self):
        return self.adapter.query(self.node_index)

    def close(self):
        adapter = self.adapter
        self.adapter = None
        if adapter is not None:
            adapter.close()

    def __del__(self):
        self.close()


def create_dxgi_budget_provider(device_index):
    if platform.system() != "Windows":
        return None
    try:
        return DXGINonLocalBudgetProvider(device_index)
    except (AttributeError, OSError, DXGIUnavailable) as err:
        logging.debug("DXGI NON_LOCAL pin budget unavailable for CUDA device %s: %s", device_index, err)
        return None
