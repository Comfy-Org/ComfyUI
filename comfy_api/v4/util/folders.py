"""ComfyUI's folders: where inputs, outputs, temporary files and model files are.

These calls are ComfyUI's `folder_paths`, for packs written against it; ComfyUI answers each.
The calls that take a path take any path, as `folder_paths` does.

A pack that runs in its own process gets the output, input, temp and user folders that process
was started with.
"""

import os
from collections.abc import Callable, Collection, Mapping
from functools import partial
from types import MappingProxyType
from typing import Literal

import folder_paths as _folder_paths

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting;
# only `map_legacy`, which takes one, is called by position.

base_path: str
"""ComfyUI's base folder: `--base-directory`, or the folder ComfyUI is installed in."""
models_dir: str
"""The folder holding the model folders: `--models-directory`, or `models` in `base_path`."""
supported_pt_extensions: set[str]
"""The weight file extensions of folders such as `checkpoints` and `loras`; a new set on every read."""
folder_names_and_paths: Mapping[str, tuple[tuple[str, ...], frozenset[str]]]
"""ComfyUI's model folders, `{name: (directories, extensions)}`: a read-only copy, made on every read.

A folder with no extensions lists every file. `add_model_folder_path` adds a directory.
"""


def __getattr__(name: str):
    match name:
        case "base_path":
            return _folder_paths.base_path
        case "models_dir":
            return _folder_paths.models_dir
        case "supported_pt_extensions":
            return set(_folder_paths.supported_pt_extensions)
        case "folder_names_and_paths":
            return MappingProxyType(
                {
                    name: (tuple(entry[0]), frozenset(entry[1]))
                    for name, entry in _folder_paths.folder_names_and_paths.items()
                }
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_output_directory() -> str:
    """Return the folder ComfyUI's save nodes write to."""
    return _folder_paths.get_output_directory()


def get_input_directory() -> str:
    """Return the folder of uploaded files that ComfyUI's load nodes list."""
    return _folder_paths.get_input_directory()


def get_temp_directory() -> str:
    """Return the folder for temporary files, which ComfyUI's preview nodes also write to.

    ComfyUI empties it, so a file written there does not stay.
    """
    return _folder_paths.get_temp_directory()


def get_user_directory() -> str:
    """Return the user folder, which holds a folder for each user (`default` for a single user).

    A user's folder holds their settings and saved workflows.
    """
    return _folder_paths.get_user_directory()


def get_system_user_directory(name: str = "system") -> str:
    """Return `__<name>` in the user folder: a folder ComfyUI's user-data endpoints refuse; not created.

    Raises `ValueError` unless `name` is letters, digits and underscores, not starting with an
    underscore.
    """
    return _folder_paths.get_system_user_directory(name=name)


def get_input_subfolders() -> list[str]:
    """Return every folder below the input folder, relative to it with `/` separators, sorted.

    Linked folders are not listed or entered; a missing input folder gives `[]`.
    """
    return _folder_paths.get_input_subfolders()


def get_folder_paths(folder_name: str) -> list[str]:
    """Return the directories of a model folder such as `"loras"`, as a new list.

    `"unet"` and `"clip"` name `"diffusion_models"` and `"text_encoders"`. Raises `KeyError`
    for a folder that does not exist.
    """
    return _folder_paths.get_folder_paths(folder_name=folder_name)


def get_full_path(folder_name: str, filename: str) -> str | None:
    """Return the path of `filename` in the first of the folder's directories that has it, else None.

    Only files count. `filename` cannot name a file above the directories: a leading `/` and
    `..` above the directory are dropped. A link inside a directory is followed. An unknown
    folder gives None, not `KeyError`.
    """
    try:
        return _folder_paths.get_full_path(folder_name=folder_name, filename=filename)
    except ValueError:
        # On Windows ComfyUI cannot make a path on another drive relative to a directory.
        return None


def annotated_filepath(name: str) -> tuple[str, str | None]:
    """Split a mark off a name: `"a.png [output]"` gives `("a.png", the output folder)`.

    A name ending in `[output]`, `[input]` or `[temp]` is marked: the mark and the one character
    before it, normally a space, are removed. Any other name gives `(name, None)`.
    """
    return _folder_paths.annotated_filepath(name=name)


def get_annotated_filepath(name: str, default_dir: str | None = None) -> str:
    """Return the path of a file name as load nodes receive it, such as `"a.png [output]"`.

    An unmarked name is in `default_dir`, or else the input folder. Raises `ValueError` if the
    name leads outside that folder.
    """
    return _folder_paths.get_annotated_filepath(name=name, default_dir=default_dir)


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0
) -> tuple[str, str, int, str, str]:
    """Return where ComfyUI's save nodes would write the next file named `filename_prefix`.

    Returns `(folder, filename, counter, subfolder, filename_prefix)`. `counter` is the number
    ComfyUI's save nodes would give the next file, as in their `<filename>_00001_.png`: one
    more than the highest in the folder, or 1. A prefix that ends in a separator gets 1 each
    time, which can name a file that exists.
    `%width%`, `%height%`, `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%` and `%second%` are
    filled in, also in the returned `filename_prefix`; `%date:...%` is not. Creates the folder if
    missing; raises `Exception` if the prefix leads outside `output_dir`.
    """
    return _folder_paths.get_save_image_path(
        filename_prefix=filename_prefix,
        output_dir=output_dir,
        image_width=image_width,
        image_height=image_height,
    )


def add_model_folder_path(
    folder_name: str, full_folder_path: str | os.PathLike[str], is_default: bool = False
) -> None:
    """Add a directory to a model folder, for every node in this ComfyUI.

    A directory added as the pack loads, in its import or `on_load` or in a task or
    `asyncio.to_thread` call they start, is taken out again if ComfyUI disables the pack or
    takes it away, and one it moved goes back behind the directory it followed. Any other
    stays until ComfyUI restarts: one added from a node or a route, or on a thread started
    another way, which ComfyUI cannot tell is the pack's. `is_default` puts it first,
    moving it if it is already there; otherwise adding it again does nothing. A folder name that
    is new gets no extensions, which lists every file. Nothing is created on disk. `"unet"` and
    `"clip"` name `"diffusion_models"` and `"text_encoders"`. A path object is stored as its
    string. Raises `TypeError` for a folder name that is not a string or a directory that is not
    a string or path: ComfyUI's table would keep the value, and its model lists would fail.
    """
    from .. import _packs

    _packs._registered(_add_directory(folder_name, full_folder_path, is_default))


def _add_directory(
    folder_name: str, full_folder_path: str | os.PathLike[str], is_default: bool
) -> Callable[[], None]:
    """Add the directory through ComfyUI and return a function that takes the addition back."""
    directory = (
        os.fspath(full_folder_path)
        if isinstance(full_folder_path, os.PathLike)
        else full_folder_path
    )
    if not isinstance(folder_name, str) or not isinstance(directory, str):
        raise TypeError(
            "folder_name must be a string and full_folder_path a string or path, "
            f"not {folder_name!r} and {full_folder_path!r}"
        )
    table = _folder_paths.folder_names_and_paths
    before = {name: list(entry[0]) for name, entry in table.items()}
    _folder_paths.add_model_folder_path(
        folder_name=folder_name,
        full_folder_path=directory,
        is_default=is_default,
    )
    added = [
        (name, added_directory, name not in before)
        for name, entry in table.items()
        for added_directory in entry[0]
        if added_directory not in before.get(name, ())
    ]
    # (folder, the directory now first, the one it followed) where the call only moved one.
    moved = [
        (name, now[0], order[order.index(now[0]) - 1])
        for name, order in before.items()
        for now in [table.get(name, ([],))[0]]
        if now and order and now != order and now[0] != order[0] and now[0] in order
        if not any(a[0] == name for a in added)
    ]
    return partial(_remove_directories, added, moved)


def _remove_directories(
    added: list[tuple[str, str, bool]], moved: list[tuple[str, str, str]]
) -> None:
    """Take out the directories an `add_model_folder_path` call added, and put a directory it
    only moved to the front back behind the one it followed, if both are still listed."""
    table = _folder_paths.folder_names_and_paths
    for name, directory, followed in moved:
        order = table[name][0] if name in table else []
        if directory in order and followed in order:
            order.remove(directory)
            order.insert(order.index(followed) + 1, directory)
    for name, directory, created in added:
        entry = table.get(name)
        if entry is not None and directory in entry[0]:
            entry[0].remove(directory)
            if created and not entry[0]:
                del table[name]
        # ComfyUI's cached list of a folder's files notices a directory added, not one removed.
        _folder_paths.filename_list_cache.pop(name, None)


def filter_files_content_types(
    files: list[str], content_types: list[Literal["image", "video", "audio", "model"]]
) -> list[str]:
    """Return the names in `files` whose type, guessed from the extension, is in `content_types`.

    The type is guessed with Python's `mimetypes` from this machine's MIME table, so it can differ
    between machines, except that `.webp` is always `"image"` and `.fbx` always `"model"`.
    `"model"` means 3D models. Order is kept.
    """
    return _folder_paths.filter_files_content_types(
        files=files, content_types=content_types
    )


def filter_files_extensions(
    files: Collection[str], extensions: Collection[str]
) -> list[str]:
    """Return the names in `files` whose lowercased extension, with its dot, is in `extensions`, sorted.

    All of them if `extensions` is empty.
    """
    return _folder_paths.filter_files_extensions(files=files, extensions=extensions)


def recursive_search(
    directory: str, excluded_dir_names: list[str] | None = None
) -> tuple[list[str], dict[str, float]]:
    """Return `(files relative to directory, {directory and each folder below it: modified time})`.

    Folder keys are paths starting with `directory` as given, not relative ones. Files are in no
    set order and use this machine's separator. Follows links and skips folders named in
    `excluded_dir_names` at any depth; `([], {})` if `directory` is not a directory.
    """
    return _folder_paths.recursive_search(
        directory=directory, excluded_dir_names=excluded_dir_names
    )


def is_within_directory(directory: str, target: str) -> bool:
    """Return whether `target` is `directory` or inside it, after resolving links on both."""
    return _folder_paths.is_within_directory(directory=directory, target=target)


__all__ = [
    "add_model_folder_path",
    "annotated_filepath",
    "base_path",
    "filter_files_content_types",
    "filter_files_extensions",
    "folder_names_and_paths",
    "get_annotated_filepath",
    "get_folder_paths",
    "get_full_path",
    "get_input_directory",
    "get_input_subfolders",
    "get_output_directory",
    "get_save_image_path",
    "get_system_user_directory",
    "get_temp_directory",
    "get_user_directory",
    "is_within_directory",
    "models_dir",
    "recursive_search",
    "supported_pt_extensions",
]
