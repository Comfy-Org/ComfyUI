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


class DestinationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _model_folders() -> dict[str, set[str]]:
    """Writable model categories and the extensions their loaders enumerate.

    ``folder_names_and_paths`` also holds ``custom_nodes``, whose contents
    ComfyUI *imports* at startup, so writing into a category straight off that
    mapping would turn a download into arbitrary code execution.
    ``non_model_folder_names`` is the exclusion the upload endpoint applies for
    the same reason; it is read from ``folder_paths`` so the two cannot drift.
    """
    return {
        name: set(extensions)
        for name, (paths, extensions) in folder_paths.folder_names_and_paths.items()
        if name not in folder_paths.non_model_folder_names and paths
    }


def _forbidden_roots() -> list[str]:
    """Directories no download may land in, whatever folder name reaches them.

    The exclusion cannot be keyed on the folder name alone.
    ``extra_model_paths.yaml`` and ``add_model_folder_path`` both let a second
    name be registered for a directory that already has one, so an alias such
    as ``node_packs: custom_nodes/`` would pass a name check and write
    importable code into ``custom_nodes``.
    """
    roots = []
    for name in folder_paths.non_model_folder_names:
        paths = folder_paths.folder_names_and_paths.get(name)
        if paths:
            roots.extend(paths[0])
    return roots


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
    The extension must be one the folder enumerates, matching
    ``filter_files_extensions``: an empty set there means the folder lists
    every file it holds, so it accepts any name rather than none.
    """
    folder_name = known_folder(folder_name)
    relative = _safe_relative_name(filename)
    if not _lists_extension_of(relative, _model_folders()[folder_name]):
        listed = ", ".join(sorted(_model_folders()[folder_name]))
        raise DestinationError(
            "UNSUPPORTED_EXTENSION",
            f"'{filename}' does not end in an extension that '{folder_name}' loaders list ({listed}).",
        )

    root = directory(folder_name)
    resolved = os.path.abspath(os.path.join(root, relative))
    if not folder_paths.is_within_directory(root, resolved):
        raise DestinationError("INVALID_FILENAME", f"'{filename}' escapes the '{folder_name}' directory.")
    for forbidden in _forbidden_roots():
        if folder_paths.is_within_directory(forbidden, resolved):
            raise DestinationError(
                "UNKNOWN_MODEL_FOLDER",
                f"'{folder_name}' leads into {forbidden}, which downloads may not write to.",
            )
    return resolved


def _lists_extension_of(name: str, extensions: set[str]) -> bool:
    """Whether ``get_filename_list`` would include a file called ``name``.

    Mirrors ``filter_files_extensions``, including its rule that an empty set
    matches everything.
    """
    return not extensions or os.path.splitext(name)[1].lower() in extensions


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
    """Check that a loader asking for this model would actually get this file.

    Membership in ``get_filename_list`` is not enough. That listing is a union
    over every root configured for the folder, while ``get_full_path`` returns
    the *first* root holding the name -- so a download into a secondary root
    whose name already exists in the preferred one reads as present while every
    loader goes on opening the other file. Several stock folders have two roots
    (``diffusion_models`` is unet + diffusion_models, ``text_encoders`` is
    text_encoders + clip) and ``extra_model_paths.yaml`` adds more, so this is
    ordinary, not exotic. Resolve the name the way a loader resolves it and
    compare the file it lands on.
    """
    if not os.path.isfile(path):
        return Visibility(None, None, f"the file is no longer at {path}")
    folder = folder_for_path(path)
    if folder is None:
        return Visibility(None, None, f"it landed at {path}, outside every configured model directory")

    name = relative_name(folder, path)
    extensions = _model_folders().get(folder, set())
    if not _lists_extension_of(name, extensions):
        return Visibility(
            folder,
            name,
            f"it landed at {path}, but '{folder}' loaders only list {', '.join(sorted(extensions))}",
        )

    resolved = folder_paths.get_full_path(folder, name)
    if resolved is None or os.path.realpath(resolved) != os.path.realpath(path):
        return Visibility(
            folder,
            name,
            f"it landed at {path}, but '{folder}' loaders resolve '{name}' to "
            f"{resolved or 'nothing'}, so they would never read it",
        )
    return Visibility(folder, name, None)


def refresh_listing(folder_name: str) -> None:
    """Drop the cached file listing so the next node refresh shows the model.

    Not needed for correctness -- :func:`inspect` resolves against the
    filesystem -- but without it the loader combo keeps serving a listing built
    before the download, which is the stale-catalog half of BE-10028.
    """
    folder_paths.invalidate_filename_list_cache(folder_name)


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
