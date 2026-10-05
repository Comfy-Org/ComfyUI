"""Where a downloaded model has to land, and whether it actually got there.

This is the part of the feature that fixes BE-10028. ``comfy model download``
resolves paths against its own workspace, which is not necessarily the ComfyUI
being driven, so a transfer can report 100% into a directory no loader
enumerates. ComfyUI is the only process that knows its own ``folder_paths``, so
it picks the destination here and hands the backend an absolute path.

"Completed" then means :func:`is_visible` -- the running server lists the file
for that folder -- not merely that the bytes arrived.
"""

from __future__ import annotations

import os

import folder_paths

# Extension sets that carry no filename information: a folder holding
# directories, or one that accepts anything. Checking an extension against these
# would reject valid names.
_UNCHECKABLE_EXTENSIONS = frozenset({"", "folder"})


class DestinationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def known_folder(folder_name: str) -> str:
    """Normalize a model folder name, rejecting one this server has no path for."""
    resolved = folder_paths.map_legacy(folder_name.strip())
    if resolved not in folder_paths.folder_names_and_paths:
        raise DestinationError(
            "UNKNOWN_MODEL_FOLDER",
            f"'{folder_name}' is not a model folder this server knows about.",
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
    The extension check is not cosmetic: ``get_filename_list`` filters on the
    folder's extension set, so a name outside it can never appear in a loader
    no matter how well the transfer goes.
    """
    folder_name = known_folder(folder_name)
    relative = _safe_relative_name(filename)

    extensions = folder_paths.folder_names_and_paths[folder_name][1]
    if extensions and not set(extensions) & _UNCHECKABLE_EXTENSIONS:
        if os.path.splitext(relative)[1].lower() not in extensions:
            raise DestinationError(
                "UNSUPPORTED_EXTENSION",
                f"'{filename}' does not end in an extension that '{folder_name}' loaders list "
                f"({', '.join(sorted(extensions))}).",
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
    """Which model folder owns ``path``, or None if no configured root contains it.

    Downloads started outside ComfyUI -- the local agent shelling out to
    ``comfy model download`` -- arrive with a destination and no folder name, and
    their caches still have to be invalidated on completion.
    """
    target = os.path.abspath(path)
    best: tuple[int, str] | None = None
    for folder_name, (roots, _extensions) in folder_paths.folder_names_and_paths.items():
        for root in roots:
            if folder_paths.is_within_directory(root, target):
                depth = len(os.path.abspath(root))
                if best is None or depth > best[0]:
                    best = (depth, folder_name)
    return best[1] if best else None


def is_visible(folder_name: str, destination: str) -> bool:
    """Whether this server now lists ``destination`` among ``folder_name``'s files.

    A miss forces the listing to be rebuilt before it is believed: the cache
    validates itself on directory mtimes, and a file written into a directory
    that was already listed in the same second does not move one.
    """
    if not os.path.isfile(destination):
        return False
    name = relative_name(folder_name, destination)
    if name is None:
        return False
    if name in folder_paths.get_filename_list(folder_name):
        return True
    folder_paths.invalidate_filename_list_cache(folder_name)
    return name in folder_paths.get_filename_list(folder_name)


def relative_name(folder_name: str, destination: str) -> str | None:
    """The name a loader would show for ``destination``, or None if it is outside."""
    target = os.path.abspath(destination)
    for root in folder_paths.get_folder_paths(folder_name):
        if folder_paths.is_within_directory(root, target):
            return os.path.relpath(target, root).replace(os.sep, "/")
    return None
