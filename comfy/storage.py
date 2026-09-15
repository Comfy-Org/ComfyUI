import functools
import logging
import os
import platform
import re

import comfy_aimdo.storage


_NVME_NAMESPACE = re.compile(r"^(nvme\d+)n\d+$")


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _physical_block_devices(name):
    partition = f"/sys/class/block/{name}/partition"
    if os.path.exists(partition):
        name = os.path.basename(os.path.dirname(os.path.realpath(f"/sys/class/block/{name}")))

    slaves = f"/sys/class/block/{name}/slaves"
    try:
        children = os.listdir(slaves)
    except OSError:
        children = []
    if children:
        devices = []
        for child in children:
            devices.extend(_physical_block_devices(child))
        return devices
    return [name]


def _fast_nvme(name):
    match = _NVME_NAMESPACE.match(name)
    if match is None:
        return False
    controller = match.group(1)
    speed = _read(f"/sys/class/nvme/{controller}/device/current_link_speed")
    width = _read(f"/sys/class/nvme/{controller}/device/current_link_width")
    if speed is None or width is None:
        return None
    try:
        speed_gts = float(speed.split()[0])
        width = int(width)
    except ValueError:
        return None
    return (speed_gts >= 8.0 and width >= 4) or (speed_gts >= 32.0 and width >= 2)


@functools.lru_cache(maxsize=None)
def _linux_fast_storage(device):
    sys_device = f"/sys/dev/block/{os.major(device)}:{os.minor(device)}"
    if not os.path.exists(sys_device):
        return None
    name = os.path.basename(os.path.realpath(sys_device))
    devices = _physical_block_devices(name)
    results = [_fast_nvme(x) for x in devices]
    if any(x is None for x in results):
        return None
    return all(results)


def fast_storage(path):
    system = platform.system()
    if system == "Linux":
        try:
            device = os.stat(os.path.realpath(path)).st_dev
        except OSError:
            return None
        return _linux_fast_storage(device)
    if system == "Windows":
        return comfy_aimdo.storage.fast_disk(path)
    return None


def annotate_state_dict(state_dict, path):
    path = os.path.realpath(path)
    for value in state_dict.values():
        untyped_storage = getattr(value, "untyped_storage", None)
        if untyped_storage is not None:
            untyped_storage()._comfy_source_path = path


def state_dict_fast_disk(state_dict):
    state_dicts = state_dict if isinstance(state_dict, (list, tuple)) else (state_dict,)
    paths = set()
    for sd in state_dicts:
        for value in sd.values():
            untyped_storage = getattr(value, "untyped_storage", None)
            if untyped_storage is not None:
                path = getattr(untyped_storage(), "_comfy_source_path", None)
                if path is not None:
                    paths.add(path)
    return model_fast_disk(sorted(paths)) if paths else False


def model_fast_disk(paths):
    results = [fast_storage(path) for path in paths]
    fast = bool(results) and all(result is True for result in results)
    logging.info("Model storage policy: fast_disk=%s paths=%s", fast, [os.path.realpath(path) for path in paths])
    return fast
