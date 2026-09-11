import os
import sys

import psutil

CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"
PROC_SELF_CGROUP = "/proc/self/cgroup"

_cgroup_dirs = None


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _read_int(path):
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_stat(path, key):
    raw = _read_text(path)
    if raw is None:
        return None
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _lineage(root, path):
    dirs = [root]
    for part in path.split("/"):
        if part:
            dirs.append(os.path.join(dirs[-1], part))
    return dirs[::-1]


def _own_cgroup_dirs():
    raw = _read_text(PROC_SELF_CGROUP)
    if raw is None:
        return []
    dirs = []
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1], parts[2]
        if controllers == "":
            root = CGROUP_V2_ROOT
        elif "memory" in controllers.split(","):
            root = CGROUP_V1_MEMORY_ROOT
        else:
            continue
        for directory in _lineage(root, path):
            if directory not in dirs:
                dirs.append(directory)
    return dirs


def _cgroup_directories():
    global _cgroup_dirs
    if _cgroup_dirs is None:
        _cgroup_dirs = _own_cgroup_dirs() or [CGROUP_V2_ROOT, CGROUP_V1_MEMORY_ROOT]
    return _cgroup_dirs


def _limit_in(directory):
    for name in ("memory.max", "memory.limit_in_bytes"):
        value = _read_int(os.path.join(directory, name))
        if value is not None and value > 0:
            return value
    return None


def _working_set_in(directory):
    usage = _read_int(os.path.join(directory, "memory.current"))
    key = "inactive_file"
    if usage is None:
        usage = _read_int(os.path.join(directory, "memory.usage_in_bytes"))
        key = "total_inactive_file"
    if usage is None:
        return None
    inactive_file = _read_stat(os.path.join(directory, "memory.stat"), key) or 0
    return max(0, usage - inactive_file)


def _limited(host_total):
    if not sys.platform.startswith("linux"):
        return []
    limited = []
    for directory in _cgroup_directories():
        limit = _limit_in(directory)
        if limit is not None and limit < host_total:
            limited.append((limit, directory))
    return limited


def cgroup_memory_limit():
    return min((limit for limit, _ in _limited(psutil.virtual_memory().total)), default=None)


def virtual_memory_total():
    host = psutil.virtual_memory()
    return min((limit for limit, _ in _limited(host.total)), default=host.total)


def virtual_memory_available():
    host = psutil.virtual_memory()
    available = host.available
    for limit, directory in _limited(host.total):
        used = _working_set_in(directory)
        available = min(available, limit if used is None else limit - used)
    return max(0, available)
