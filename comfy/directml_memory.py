"""Dedicated VRAM reporting for the DirectML backend.

torch-directml exposes no usable memory API: ``torch_directml.gpu_memory()``
returns zeros on the adapters we have tested, so ComfyUI historically fell back
to a hardcoded 1 GiB. That badly under-reports real hardware (an RX 580 has
8 GB), which makes every model look too large to keep resident and shows a
wrong number in the UI.

This module asks DXGI directly, which reports true dedicated video memory. It is
only imported when DirectML is actually enabled, so the ctypes/DXGI path never
costs anything for CUDA/ROCm/CPU users.
"""

import functools
import sys

from comfy.cli_args import args


def _dxgi_vram():
    """Return {adapter_index: dedicated_vram_bytes} for every DXGI adapter."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8),
        ]

    class DXGI_ADAPTER_DESC1(ctypes.Structure):
        _fields_ = [
            ("Description", wintypes.WCHAR * 128),
            ("VendorId", wintypes.DWORD), ("DeviceId", wintypes.DWORD),
            ("SubSysId", wintypes.DWORD), ("Revision", wintypes.DWORD),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", ctypes.c_ulonglong),
            ("Flags", wintypes.DWORD),
        ]

    iid_factory1 = GUID(
        0x770AAE78, 0xF26F, 0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )

    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory1), ctypes.byref(factory)) != 0:
        return {}

    # IDXGIFactory1 vtable slot 12 is EnumAdapters1 and IDXGIAdapter1 slot 10 is
    # GetDesc1. Indexing the vtable is fragile, but it is the only way to reach
    # these without pulling in a full COM binding for one value.
    vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    enum_adapters1 = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
    )(vtable[12])

    result = {}
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        if enum_adapters1(factory, index, ctypes.byref(adapter)) != 0:
            break
        try:
            avtable = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            get_desc1 = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)
            )(avtable[10])
            desc = DXGI_ADAPTER_DESC1()
            if get_desc1(adapter, ctypes.byref(desc)) == 0:
                result[index] = desc.DedicatedVideoMemory
        except (OSError, ValueError, ctypes.ArgumentError):
            pass
        index += 1
    return result


@functools.lru_cache(maxsize=1)
def _dxgi_vram_cached():
    return _dxgi_vram()


def get_total_vram(device=None):
    """Dedicated VRAM for a DirectML device, in bytes, or None if undeterminable.

    Deliberately reports dedicated memory only, not the shared-system-memory
    heap DirectML can also spill into. Overstating the budget is worse than
    understating it here: DirectML's allocator aborts the process outright when
    a large allocation cannot be satisfied, so telling ComfyUI it has more room
    than the card really has turns a graceful fallback into a hard crash. Users
    who know their setup can raise it with --directml-vram-gb.
    """
    if sys.platform != "win32":
        return None

    override = getattr(args, "directml_vram_gb", None)
    if override is not None and override > 0:
        return int(override * (1024 ** 3))

    try:
        per_adapter = _dxgi_vram_cached()
    except Exception:
        return None

    if not per_adapter:
        return None

    # DirectML and DXGI enumerate adapters in the same order, so the indices
    # line up. If they ever do not (hybrid-GPU laptops can order them
    # differently), fall back to the largest discrete adapter rather than
    # reporting nothing.
    index = getattr(device, "index", None)
    dedicated = per_adapter.get(0 if index is None else int(index))
    if not dedicated:
        dedicated = max(per_adapter.values())
    if not dedicated or dedicated <= 0:
        return None

    return int(dedicated)
