"""The model weights a pack's manifest declares: present and verified before the pack runs.

Each weight is ``{"folder", "filename", "repo_id"?, "revision"?, "sha256"?}``. One with a ``repo_id``
is installed by the secure overlay's tested downloader (``hf_weights``), at the path converted packs
use; one without must be in its model folder already, and is checked against its sha256. A verified
file isn't hashed again until its size or modification time changes
(``user/pack_weights_verified.json``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class WeightsError(RuntimeError):
    """A declared weight isn't available, or isn't the file the manifest names."""


def _folder_root(folder: str) -> str:
    import folder_paths
    from ..manifest import check_folder_name
    check_folder_name(folder)  # a plain name under ComfyUI's models directory, never a path the manifest picks
    try:
        paths = folder_paths.get_folder_paths(folder)
    except KeyError:
        paths = []
    if paths:
        return paths[0]
    path = os.path.join(folder_paths.models_dir, folder)  # a folder ComfyUI doesn't know yet
    os.makedirs(path, exist_ok=True)
    folder_paths.add_model_folder_path(folder, path)
    return path


def _find(folder: str, filename: str) -> Optional[str]:
    import folder_paths
    try:
        return folder_paths.get_full_path(folder, filename)
    except Exception:
        return None


def _verified_file() -> str:
    import folder_paths
    return os.path.join(folder_paths.get_user_directory(), "pack_weights_verified.json")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(weight: dict) -> str:
    """The secure overlay's downloader (``hf_weights``): its path in the model folder."""
    import folder_paths
    from . import hf_weights

    logical = hf_weights.download(weight)
    path = folder_paths.get_full_path(weight["folder"], logical)
    if path is None:
        raise WeightsError(f"{logical} was installed but isn't in models/{weight['folder']}")
    return path


def ensure(pack: str, weights: list[dict], download: Optional[Callable[[dict], str]] = None) -> list[str]:
    """Every declared weight, present and verified: their paths. Raises ``WeightsError``.

    A weight with a ``repo_id`` is installed by the secure overlay's downloader at
    ``huggingface/<repo>/<revision>/<file>`` in its folder (verified structurally and by sha256);
    one without must already be in its folder. A verified file isn't hashed again until it changes."""
    download = download or _download
    paths = []
    with _lock:
        try:
            with open(_verified_file()) as file:
                verified = json.load(file)
        except (FileNotFoundError, ValueError):
            verified = {}
        changed = False
        for weight in weights:
            if weight.get("repo_id"):
                logical = f"huggingface/{weight['repo_id']}/{weight.get('revision') or 'main'}/{weight['filename']}"
                path = _find(weight["folder"], logical)
                stamp = _stamp(path, weight) if path else None
                if path is None or verified.get(path) != stamp:
                    try:
                        path = download(weight)
                    except WeightsError:
                        raise
                    except Exception as error:
                        raise WeightsError(f"{pack} needs {weight['filename']} from {weight['repo_id']}: {error}") from error
                    verified[path] = _stamp(path, weight)
                    changed = True
                paths.append(path)
                continue
            path = _find(weight["folder"], weight["filename"])
            if path is None:
                raise WeightsError(f"{pack} needs {weight['filename']} in models/{weight['folder']}; "
                                   "its manifest doesn't say where to download it from")
            if weight.get("sha256"):
                stamp = _stamp(path, weight)
                if verified.get(path) != stamp:
                    actual = _sha256(path)
                    if actual != weight["sha256"]:
                        raise WeightsError(f"{path} isn't the file {pack} declares: sha256 {actual}, "
                                           f"expected {weight['sha256']}")
                    verified[path] = stamp
                    changed = True
            paths.append(path)
        if changed:
            with open(_verified_file() + ".tmp", "w") as file:
                json.dump(verified, file)
            os.replace(_verified_file() + ".tmp", _verified_file())
    return paths


def _stamp(path: str, weight: dict) -> list:
    stat = os.stat(path)
    return [stat.st_size, stat.st_mtime_ns, weight.get("sha256")]
