"""ComfyUI's own engine extensions for ``eng.files``, ``eng.output``, ``eng.store`` and ``eng.cache``: ComfyUI's folders, and what a pack keeps between runs."""
import collections
import json
import os
import re
import threading
import time
from comfy_api.latest import _runtime


FOLDERS = ("input", "output", "temp")


def _root(folder):
    import folder_paths

    if folder not in FOLDERS:
        raise ValueError("folder must be 'input', 'output' or 'temp'")
    return os.path.abspath(folder_paths.get_directory_by_type(folder))


def _resolve(ref):
    import folder_paths

    if not isinstance(ref, str) or not ref or "\x00" in ref:
        raise ValueError("a file ref is a non-empty string")
    path = os.path.abspath(folder_paths.get_annotated_filepath(ref))
    if not any(path.startswith(_root(f) + os.sep) for f in FOLDERS):
        raise PermissionError(f"{ref!r} is outside the input, output and temp folders")
    return path


class FilesOps:
    """Reading ComfyUI's input, output and temp folders.

A ref is a file name as ComfyUI's loaders take it: ``"photo.png"`` (input folder),
``"sub/photo.png [input]"``, ``"result_00001_.png [output]"`` or ``"x.png [temp]"``.
Refs stay inside those folders."""

    def read(self, ref):
        with open(_resolve(ref), "rb") as file:
            return file.read()

    def exists(self, ref):
        try:
            return os.path.isfile(_resolve(ref))
        except (ValueError, PermissionError):
            return False

    def list(self, folder="input", subfolder="", recursive=False, extensions=None):
        root = _root(folder)
        start = os.path.abspath(os.path.join(root, subfolder))
        if start != root and not start.startswith(root + os.sep):
            raise PermissionError("subfolder is outside the folder")
        wanted = {("." + e.lower().lstrip(".")) for e in extensions} if extensions else None
        found = []
        for directory, dirs, names in os.walk(start):
            dirs.sort()
            for name in sorted(names):
                if wanted is not None and os.path.splitext(name)[1].lower() not in wanted:
                    continue
                path = os.path.join(directory, name)
                stat = os.stat(path)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                found.append({"ref": f"{relative} [{folder}]", "name": relative, "size": stat.st_size,
                              "mtime": stat.st_mtime})
            if not recursive:
                break
        return found


OUTPUT_FOLDERS = ("output", "temp")


def _check(data, suffix):
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes the node encoded")
    suffix = str(suffix).lstrip(".")
    if not suffix or os.sep in suffix or "/" in suffix or len(suffix) > 16:
        raise ValueError("suffix must be a short file extension such as 'png'")
    return suffix


class OutputOps:
    """Writing files to ComfyUI's output and temp folders."""

    def write(self, data, filename_prefix, suffix, folder="output", width=0, height=0, trailing_underscore=True):
        """One file (``data`` is bytes) or a batch (``data`` is a list of bytes), named as ComfyUI's
        save nodes name them: ``<prefix>_<counter>_.<suffix>``. A batch shares one counter run and
        replaces ``%batch_num%`` in the prefix per file, as ``SaveImage`` does."""
        import folder_paths

        if folder not in OUTPUT_FOLDERS:
            raise ValueError("folder must be 'output' or 'temp'")
        batch = isinstance(data, (list, tuple))
        files = list(data) if batch else [data]
        if not files:
            return []
        suffix = _check(files[0], suffix)
        for item in files[1:]:
            _check(item, suffix)
        root = folder_paths.get_output_directory() if folder == "output" else folder_paths.get_temp_directory()
        # Raises if the prefix leaves the folder.
        full_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, root, width, height)
        end = "_" if trailing_underscore else ""
        # ComfyUI's counter can't see past %batch_num% names, so those overwrite as SaveImage does;
        # every other name is new, and an existing file is never replaced.
        mode = "wb" if "%batch_num%" in name else "xb"
        results = []
        for batch_number, item in enumerate(files):
            filename = f"{name.replace('%batch_num%', str(batch_number))}_{counter:05}{end}.{suffix}"
            with open(os.path.join(full_folder, filename), mode) as file:
                file.write(bytes(item))
            results.append({"filename": filename, "subfolder": subfolder, "type": folder})
            counter += 1
        return results if batch else results[0]


SCOPES = ("pack", "workflow", "node")
TENANT = "default"  # the storage tenant when none is configured


def _pack_name():
    rt = _runtime.current_runtime()
    parts = [p for p in (rt.pack or "").replace("\\", "/").replace("/", ".").split(".") if p not in ("", "custom_nodes")]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", parts[0] if parts else "pack")


def _scope_id(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope is one of {SCOPES}, got {scope!r}")
    pack = _pack_name()
    if scope == "pack":
        return pack
    plan = _runtime.current_runtime().plan
    workflow = ((getattr(plan, "extra_pnginfo", None) or {}).get("workflow") or {}).get("id")
    if not workflow:
        raise ValueError(f"scope {scope!r} needs the workflow's id, and this prompt has none")
    return f"{pack}/workflow/{workflow}" if scope == "workflow" else f"{pack}/node/{workflow}/{getattr(plan, 'node_id', '')}"


def _store():
    from comfy_api.latest.pack_storage import pack_store
    return pack_store()


class StoreOps:
    """The ``eng.store`` rows, small values a pack keeps between runs.

Kept by pack storage (``comfy_api/latest/pack_storage.py``: atomic, bounded files
under ``user/secure-node-storage``), so a pack's values are the same in ComfyUI's process, a pack
process and the sandbox. Scopes: ``"pack"`` (every workflow), ``"workflow"`` (this workflow, by its id)
and ``"node"`` (this node in this workflow). Values are plain data, stored as JSON: keys up to 128
bytes, values up to 1 MB, and 256 entries and 8 MB per scope."""

    def get(self, key, scope="pack", default=None):
        text = _store().get(TENANT, _scope_id(scope), str(key))
        return default if text is None else json.loads(text)

    def set(self, key, value, scope="pack"):
        _store().set(TENANT, _scope_id(scope), str(key), json.dumps(value))  # plain data only

    def delete(self, key, scope="pack"):
        return _store().delete(TENANT, _scope_id(scope), str(key))

    def keys(self, scope="pack"):
        return _store().keys(TENANT, _scope_id(scope))


_entries = collections.OrderedDict()  # (pack, key) -> (value, size, expires)
_lock = threading.Lock()


def _budget():
    return int(float(os.environ.get("COMFY_PACK_CACHE_MB", "2048")) * (1 << 20))


def _size(value):
    if _runtime._looks_like_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values()) + 64
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value) + 64
    if isinstance(value, (str, bytes)):
        return len(value)
    model_size = getattr(value, "model_size", None)  # a ModelPatcher
    if callable(model_size):
        try:
            return int(model_size())
        except Exception:
            pass
    return 64


def _key(key):
    return (_runtime.current_runtime().pack or "", str(key))


class CacheOps:
    """The ``eng.cache`` rows, large values kept in ComfyUI's memory between runs.

Per pack, least recently used first out, within ``COMFY_PACK_CACHE_MB`` (default 2048) across packs;
an entry may expire after ``ttl`` seconds. Values are tensors, plain data, or engine objects (a model
the pack loaded); they stay in ComfyUI's process, so a pack running elsewhere gets them back as it
gets any value. Nothing survives a restart: it's a cache."""

    def get(self, key, default=None):
        with _lock:
            entry = _entries.get(_key(key))
            if entry is None:
                return default
            if entry[2] is not None and entry[2] < time.monotonic():
                del _entries[_key(key)]
                return default
            _entries.move_to_end(_key(key))
            return entry[0]

    def set(self, key, value, ttl=None):
        if ttl is not None and not (isinstance(ttl, (int, float)) and ttl > 0):
            raise ValueError("ttl is a number of seconds")
        size = _size(value)
        budget = _budget()
        if size > budget:
            raise ValueError("the value is larger than the whole cache (COMFY_PACK_CACHE_MB)")
        with _lock:
            _entries.pop(_key(key), None)
            _entries[_key(key)] = (value, size, time.monotonic() + ttl if ttl else None)
            total = sum(entry[1] for entry in _entries.values())
            while total > budget:
                _, (_, dropped, _) = _entries.popitem(last=False)
                total -= dropped

    def delete(self, key):
        with _lock:
            return _entries.pop(_key(key), None) is not None


EXTENSIONS = {"files": FilesOps, "output": OutputOps, "store": StoreOps, "cache": CacheOps}
