"""Where a downloaded model has to land, and whether it actually got there.

This is the part of the feature that fixes BE-10028. ``comfy model download``
resolves paths against its own workspace, which is not necessarily the ComfyUI
being driven, so a transfer can report 100% into a directory no loader
enumerates. ComfyUI is the only process that knows its own ``folder_paths``, so
it picks the destination here and hands the backend an absolute path.

"Completed" then means :func:`inspect` found the running server listing the
file, not merely that the bytes arrived.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import folder_paths
from app.assets.services.path_utils import get_comfy_models_folders


class DestinationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _model_folders() -> dict[str, set[str]]:
    """Writable model categories and the extensions their loaders enumerate.

    Shared with the upload endpoint on purpose. ``folder_names_and_paths`` also
    holds ``custom_nodes``, whose contents ComfyUI *imports* at startup, so
    reading it directly here would turn a download into arbitrary code
    execution. ``get_comfy_models_folders`` is the allowlist that already
    excludes it.
    """
    return {name: extensions for name, _paths, extensions in get_comfy_models_folders()}


def known_folder(folder_name: str) -> str:
    resolved = folder_paths.map_legacy(folder_name.strip())
    if resolved not in _model_folders():
        raise DestinationError(
            "UNKNOWN_MODEL_FOLDER",
            f"'{folder_name}' is not a model folder this server can download into.",
        )
    return resolved


def directory(folder_name: str) -> str:
    """The configured directory new files for ``folder_name`` belong in.

    The first configured path is the default one: ``add_model_folder_path``
    moves a default root to the front of the list.
    """
    roots = folder_paths.get_folder_paths(known_folder(folder_name))
    if not roots:
        raise DestinationError("UNKNOWN_MODEL_FOLDER", f"No directory is configured for '{folder_name}'.")
    return os.path.abspath(roots[0])


def resolve(folder_name: str, filename: str) -> str:
    """Absolute path a model named ``filename`` must occupy to be loadable.

    ``filename`` may contain forward-slash subfolders, which loaders do list.
    The extension must be one the folder enumerates, and a folder that
    enumerates nothing refuses every name: ``get_filename_list`` filters on
    that set, so anything outside it can never reach a loader no matter how
    well the transfer goes.
    """
    folder_name = known_folder(folder_name)
    relative = _safe_relative_name(filename)
    extensions = _model_folders()[folder_name]
    if os.path.splitext(relative)[1].lower() not in extensions:
        raise DestinationError(
            "UNSUPPORTED_EXTENSION",
            f"'{filename}' does not end in an extension that '{folder_name}' loaders list "
            f"({', '.join(sorted(extensions)) or 'none'}).",
        )

    root = directory(folder_name)
    resolved = os.path.abspath(os.path.join(root, relative))
    if not folder_paths.is_within_directory(root, resolved):
        raise DestinationError("INVALID_FILENAME", f"'{filename}' escapes the '{folder_name}' directory.")
    return resolved


def _safe_relative_name(filename: str) -> str:
    name = filename.strip().replace("\\", "/")
    if not name or name.startswith("/") or os.path.isabs(name):
        raise DestinationError("INVALID_FILENAME", "A filename is required and must be relative.")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise DestinationError("INVALID_FILENAME", f"'{filename}' is not a usable filename.")
    return os.path.join(*parts)


def folder_for_path(path: str) -> str | None:
    """Which model folder owns ``path``, or None if no model root contains it.

    Downloads started outside ComfyUI -- the local agent shelling out to
    ``comfy model download`` -- arrive with a destination and no folder name.
    Roots are compared resolved, because a model directory is very often a
    symlink onto another volume.
    """
    target = os.path.realpath(path)
    best: tuple[int, str] | None = None
    for folder_name in _model_folders():
        for root in folder_paths.get_folder_paths(folder_name):
            if folder_paths.is_within_directory(root, target):
                depth = len(os.path.realpath(root))
                if best is None or depth > best[0]:
                    best = (depth, folder_name)
    return best[1] if best else None


@dataclass(frozen=True)
class Visibility:
    """Whether a finished download is a model this server can actually load."""

    folder: str | None
    name: str | None
    problem: str | None

    @property
    def ok(self) -> bool:
        return self.problem is None


def inspect(path: str) -> Visibility:
    """Check that this server now lists ``path``, and say why if it does not.

    A miss forces the listing to be rebuilt before it is believed: the cache
    validates itself on directory mtimes, and a file written into a directory
    that was already listed in the same second does not move one.
    """
    if not os.path.isfile(path):
        return Visibility(None, None, f"the file is no longer at {path}")
    folder = folder_for_path(path)
    if folder is None:
        return Visibility(None, None, f"{path} is outside every configured model directory")

    name = relative_name(folder, path)
    if name in folder_paths.get_filename_list(folder):
        return Visibility(folder, name, None)
    folder_paths.invalidate_filename_list_cache(folder)
    if name in folder_paths.get_filename_list(folder):
        return Visibility(folder, name, None)

    extensions = ", ".join(sorted(_model_folders().get(folder, set()))) or "none"
    return Visibility(folder, name, f"'{name}' is not listed by '{folder}' loaders, which load {extensions}")


def relative_name(folder_name: str, path: str) -> str | None:
    """The name a loader would show for ``path``, or None if it is outside.

    Spelled the way ``get_filename_list`` spells it -- native separators, no
    normalisation -- because its only use is membership in that list.
    """
    target = os.path.realpath(path)
    for root in folder_paths.get_folder_paths(folder_name):
        if folder_paths.is_within_directory(root, target):
            return os.path.relpath(target, os.path.realpath(root))
    return None
