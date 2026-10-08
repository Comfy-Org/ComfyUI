import os
import platform
import re
import unicodedata


FONT_NAME_LIMIT = 4096
FONT_SCAN_LIMIT = 100000
FONT_NAME_BYTES_LIMIT = 1024 * 1024


def _system_font_directory():
    system = platform.system()
    if system == "Windows":
        root = os.environ.get("SystemRoot")
        return os.path.join(root, "Fonts") if root else None
    if system == "Linux":
        return "/usr/share/fonts/truetype"
    if system == "Darwin":
        return "/System/Library/Fonts"
    return None


def font_names(folder="system", prefix="", folder_paths_module=None):
    """Portable immediate TTF basenames in host directory order; no font bytes."""
    if not isinstance(folder, str) or folder not in {"system", "input"}:
        raise ValueError("font folder must be system or input")
    if not isinstance(prefix, str) or len(prefix.encode("utf-8")) > 1024:
        raise ValueError("font prefix must be a bounded logical directory")
    logical = prefix.replace("\\", "/")
    if ("\x00" in logical or logical.startswith("/")
            or re.match(r"^[A-Za-z]:", logical)
            or any(p == ".." for p in logical.split("/"))):
        raise ValueError("font prefix escapes the managed directory")
    if folder == "system":
        if logical:
            raise ValueError("system font catalogue does not accept a prefix")
        base = _system_font_directory()
        if base is None:
            return []
    else:
        if folder_paths_module is None:
            import folder_paths as folder_paths_module
        base = folder_paths_module.get_input_directory()
    root = os.path.realpath(os.path.abspath(base))
    directory = os.path.realpath(os.path.join(root, logical))
    if os.path.commonpath((root, directory)) != root:
        raise ValueError("font prefix escapes the managed directory")
    if not os.path.exists(directory):
        return []
    if not os.path.isdir(directory):
        raise NotADirectoryError("font prefix is not a directory")
    names = []
    name_bytes = 0
    with os.scandir(directory) as entries:
        for scanned, entry in enumerate(entries, start=1):
            if scanned > FONT_SCAN_LIMIT:
                raise ValueError("font catalogue scan limit exceeded")
            if not entry.name.lower().endswith(".ttf"):
                continue
            if ("/" in entry.name or "\\" in entry.name
                    or any(unicodedata.category(char) == "Cc" for char in entry.name)):
                continue
            target = os.path.realpath(entry.path)
            if os.path.commonpath((root, target)) != root or not entry.is_file():
                continue
            size = len(entry.name.encode("utf-8"))
            if size > 255:
                raise ValueError("font name exceeds 255 UTF-8 bytes")
            names.append(entry.name)
            name_bytes += size
            if len(names) > FONT_NAME_LIMIT or name_bytes > FONT_NAME_BYTES_LIMIT:
                raise ValueError("font catalogue name limit exceeded")
    return names
