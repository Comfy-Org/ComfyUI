import ctypes
import operator
import os
import sys
import tempfile
import uuid

if os.name == "nt":
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel32.DeviceIoControl.restype = wintypes.BOOL
    _kernel32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    _kernel32.GetVolumeInformationByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    _kernel32.GetVolumeInformationByHandleW.restype = wintypes.BOOL
elif sys.platform == "linux":
    # os.memfd_create is missing from Pythons built against glibc < 2.27.
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    _libc.memfd_create.restype = ctypes.c_int


def create_anonymous_file(size=0):
    """Return an unbuffered binary file owned by the caller, initially sized in bytes.

    Uses Linux memfd or a Windows temporary-cache/delete-on-close file.
    Windows may still write to disk; Linux memfd may be swapped.
    Initial sizing is sparse with zero-filled holes where the filesystem supports
    sparse files; otherwise it is fully allocated. Keep the file open through all
    I/O and close mappings before closing it. No persistent pathname is provided.
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
        fd = _libc.memfd_create(b"aimdo", 1)  # MFD_CLOEXEC
        if fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        file = open(fd, "w+b", buffering=0)
    else:
        raise NotImplementedError("anonymous files require Linux or Windows")

    try:
        if os.name == "nt":
            handle = msvcrt.get_osfhandle(file.fileno())
            transferred = wintypes.DWORD()
            fs_flags = wintypes.DWORD()
            # SMB shares can't report volume flags; treat them like volumes without sparse support.
            has_flags = _kernel32.GetVolumeInformationByHandleW(handle, None, 0, None, None, ctypes.byref(fs_flags), None, 0)
            # FSCTL_SET_SPARSE: unwritten ranges stay unallocated and read as zeros.
            # FAT32 and exFAT lack FILE_SUPPORTS_SPARSE_FILES (0x40); the file is then fully allocated.
            if has_flags and fs_flags.value & 0x40 and not _kernel32.DeviceIoControl(handle, 0x900c4, None, 0, None, 0,
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
