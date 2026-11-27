"""Read-only HIP VMM capability gate for automatic ComfyUI DynamicVRAM.

AMD HIP documentation: hipDeviceGetAttribute(
    &value, hipDeviceAttributeVirtualMemoryManagementSupported, device_index).
Attribute ID 89 is verified against ROCm HIP's Python enum.
Only an ALREADY LOADED HIP runtime is used. No allocation or GPU launch occurs.
"""
import ctypes
import logging
import os
import sys

HIP_ATTR_VMM = 89


def _loaded_hip_runtime():
    """Find the HIP runtime already resident in this process, not a different
    AMD driver or unrelated system-wide HIP installation."""
    if os.name == "nt":
        # Windows ROCm may load amdhip64_7.dll under PyTorch; never load an
        # unrelated System32 copy by name.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        lookup = kernel.GetModuleHandleW
        lookup.argtypes = [ctypes.c_wchar_p]
        lookup.restype = ctypes.c_void_p
        for name in ("amdhip64_7.dll", "amdhip64.dll"):
            existing = lookup(name)
            if existing:
                return ctypes.CDLL(None, handle=existing)
        return None

    if sys.platform.startswith("linux"):
        # RTLD_NOLOAD is critical: querying support must not inject a second
        # HIP runtime into the Torch process.
        flag = getattr(os, "RTLD_NOLOAD", None)
        if flag is None:
            return None
        candidates = []
        try:
            with open("/proc/self/maps", encoding="utf-8") as mapped:
                for line in mapped:
                    fields = line.split()
                    if len(fields) >= 6:
                        path = fields[-1]
                        basename = os.path.basename(path)
                        if basename == "libamdhip64.so" or basename.startswith("libamdhip64.so."):
                            candidates.append(path)
        except OSError:
            pass
        candidates.extend(("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"))
        for name in dict.fromkeys(candidates):
            try:
                return ctypes.CDLL(name, mode=os.RTLD_NOW | flag)
            except OSError:
                continue
    return None


def _query_hip_vmm(device_index):
    runtime = _loaded_hip_runtime()
    if runtime is None:
        raise RuntimeError("loaded HIP runtime not found")
    try:
        query = runtime.hipDeviceGetAttribute
    except AttributeError as exc:
        raise RuntimeError("HIP runtime lacks hipDeviceGetAttribute") from exc
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    value = ctypes.c_int(-1)
    status = query(ctypes.byref(value), HIP_ATTR_VMM, device_index)
    return int(status), int(value.value)


def hip_vmm_supported(devices, *, query=None):
    """True only if every target HIP device successfully reports VMM=1.

    The optional query hook is only for isolated tests; production calls the
    native hipDeviceGetAttribute without allocating GPU memory.
    """
    try:
        selected = list(devices)
    except Exception as exc:
        logging.warning("DynamicVRAM: cannot enumerate AMD devices (%s); using legacy ModelPatcher", exc)
        return False
    if not selected:
        logging.warning("DynamicVRAM: no AMD devices enumerated; using legacy ModelPatcher")
        return False
    query = _query_hip_vmm if query is None else query
    seen = set()
    for device in selected:
        idx = getattr(device, "index", None)
        if (getattr(device, "type", None) != "cuda"
                or type(idx) is not int or idx < 0):
            logging.warning("DynamicVRAM: invalid HIP device index; using legacy ModelPatcher")
            return False
        if idx in seen:
            continue
        seen.add(idx)
        try:
            status, support = query(idx)
        except Exception as exc:
            logging.warning("DynamicVRAM: device %s HIP VMM query failed (%s); using legacy ModelPatcher",
                            idx, type(exc).__name__)
            return False
        if status != 0 or support != 1:
            logging.warning("DynamicVRAM: device %s VMM not confirmed (HIP status %s, support %s); "
                            "using legacy ModelPatcher", idx, status, support)
            return False
    return True
