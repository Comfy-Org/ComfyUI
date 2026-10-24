import operator
import os
import sys
import tempfile
import uuid

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel32.DeviceIoControl.restype = wintypes.BOOL
    _kernel32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetFileInformationByHandle.restype = wintypes.BOOL


def create_anonymous_file(size=0):
    """Return an unbuffered binary file owned by the caller, initially sized in bytes.

    Uses Linux memfd, a Windows temporary-cache/delete-on-close file, or an
    unlinked temporary file elsewhere.
    Windows may still write to disk; Linux memfd may be swapped.
    Initial sizing is sparse with zero-filled holes; Windows requires a filesystem
    supporting sparse files. Keep the file open through all I/O and close
    mappings before closing it. No persistent pathname is provided.
    """
    size = operator.index(size)
    if size < 0 or size > 0x7fffffffffffffff:
        raise ValueError("size must fit a nonnegative signed 64-bit file offset")

    if os.name == "nt":
        flags = os.O_TEMPORARY | os.O_SHORT_LIVED
        path = os.path.join(tempfile.gettempdir(), f"aimdo-{uuid.uuid4().hex}")
        file = open(path, "x+b", buffering=0,
                    opener=lambda path, mode: os.open(path, mode | flags, 0o600))
    elif sys.platform == "linux":
        file = open("aimdo", "w+b", buffering=0,
                    opener=lambda name, mode: os.memfd_create(name, os.MFD_CLOEXEC))
    else:
        file = tempfile.TemporaryFile(buffering=0)

    try:
        if os.name == "nt":
            handle = msvcrt.get_osfhandle(file.fileno())
            transferred = wintypes.DWORD()
            # FSCTL_SET_SPARSE: unwritten ranges stay unallocated and read as zeros.
            if not _kernel32.DeviceIoControl(handle, 0x900c4, None, 0, None, 0,
                                             ctypes.byref(transferred), None):
                raise ctypes.WinError(ctypes.get_last_error())
            # FileEndOfFileInfo avoids the CRT truncate implementation's zero-fill loop.
            end = ctypes.c_longlong(size)
            if not _kernel32.SetFileInformationByHandle(handle, 6,
                                                       ctypes.byref(end), ctypes.sizeof(end)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            file.truncate(size)
        return file
    except BaseException:
        file.close()
        raise
