"""ComfyUI's folders: where inputs, outputs, temporary files and model files are.

The calls that return paths in the user's folders (input, output, user, models and the base
folder) are provided for compatibility with packs written against `folder_paths`. ComfyUI
decides what each answers: on the user's own machine the user's folder, but a host may give a
pack folders of its own for one run, or a path the pack cannot open. A pack that has to run
everywhere takes the user's files as node inputs, returns its results as node outputs, and writes
temporary files to `get_temp_directory()`.

A pack that runs in its own process gets the output, input, temp and user folders that process
was started with.
"""

import os
from collections.abc import Collection
from typing import Literal

import folder_paths as _folder_paths

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting;
# only `map_legacy`, which takes one, is called by position.

base_path: str
"""ComfyUI's base folder: `--base-directory`, or the folder ComfyUI is installed in."""
models_dir: str
"""The folder holding the model folders: `--models-directory`, or `models` in `base_path`."""
output_directory: str
"""The same path as `get_output_directory()`."""
input_directory: str
"""The same path as `get_input_directory()`."""
temp_directory: str
"""The same path as `get_temp_directory()`."""
supported_pt_extensions: set[str]
"""The weight file extensions of folders such as `checkpoints` and `loras`; a new set on every read."""
folder_names_and_paths: dict[str, tuple[list[str], set[str]]]
"""ComfyUI's own table of model folders: `{name: ([directories], extensions)}`.

`extensions` is a set, or a list for `configs` and `diffusers`. Most folders share one extension
set, so adding to one folder's set adds to all of them. Changes to the table are seen by every node
in this ComfyUI, and nothing removes them.
"""


def __getattr__(name: str):
    match name:
        case "base_path":
            return _folder_paths.base_path
        case "models_dir":
            return _folder_paths.models_dir
        case "output_directory":
            return _folder_paths.get_output_directory()
        case "input_directory":
            return _folder_paths.get_input_directory()
        case "temp_directory":
            return _folder_paths.get_temp_directory()
        case "supported_pt_extensions":
            return set(_folder_paths.supported_pt_extensions)
        case "folder_names_and_paths":
            return _folder_paths.folder_names_and_paths
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


def get_directory_by_type(type_name: str) -> str | None:
    """Return the output, input or temp folder for `"output"`, `"input"` or `"temp"`; None otherwise."""
    return _folder_paths.get_directory_by_type(type_name=type_name)


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


def get_filename_list(folder_name: str) -> list[str]:
    """Return the files a model dropdown offers for this folder, sorted, relative to their directory.

    Only files with one of the folder's extensions are listed, or every file if it has none.
    A name in a subfolder uses this machine's separator: `\\` on Windows, `/` elsewhere. Raises
    `KeyError` for a folder that does not exist.
    """
    return _folder_paths.get_filename_list(folder_name=folder_name)


def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    """Scan a model folder now: `(files, {each directory scanned: modified time}, scan time)`.

    The scan time is a `time.perf_counter()` value. Raises `KeyError` for a folder that does
    not exist.
    """
    return _folder_paths.get_filename_list_(folder_name=folder_name)


def clear_filename_list_cache(folder_name: str) -> None:
    """Make the next `get_filename_list(folder_name)` scan the disk again.

    ComfyUI rescans by itself when a directory's modified time changes; call this after a
    change it cannot see, such as new extensions for the folder. While ComfyUI builds its node
    list for `/object_info`, a folder already listed for it is not scanned again.
    """
    _folder_paths.filename_list_cache.pop(_folder_paths.map_legacy(folder_name), None)


def get_full_path(folder_name: str, filename: str) -> str | None:
    """Return the path of `filename` in the first of the folder's directories that has it, else None.

    Only files count. `filename` cannot leave the directories: a leading `/` and `..` above the
    directory are dropped. An unknown folder gives None, not `KeyError`.
    """
    return _folder_paths.get_full_path(folder_name=folder_name, filename=filename)


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    """Return the path as `get_full_path` does, but raise `FileNotFoundError` if there is none."""
    return _folder_paths.get_full_path_or_raise(
        folder_name=folder_name, filename=filename
    )


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


def exists_annotated_filepath(name: str) -> bool:
    """Return whether the path `get_annotated_filepath(name)` gives exists; False if it leads outside."""
    return _folder_paths.exists_annotated_filepath(name=name)


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0
) -> tuple[str, str, int, str, str]:
    """Return where ComfyUI's save nodes would write the next file named `filename_prefix`.

    Returns `(folder, filename, counter, subfolder, filename_prefix)`. `counter` is one more than
    the highest N among the folder's files named `<filename>_N` where N ends at the end of the
    name, a `.` or a `_` (as in ComfyUI's own `<filename>_00001_.png`), or 1.
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
    """Add a directory to a model folder, for every node in this ComfyUI; nothing removes it.

    `is_default` puts it first, moving it if it is already there; otherwise adding it again does
    nothing. A folder name not in the table gets an entry with no extensions, which lists every
    file. Nothing is created on disk. `"unet"` and `"clip"` name `"diffusion_models"` and
    `"text_encoders"`. A path object is stored as its string. Raises `TypeError` for a folder name
    that is not a string or a directory that is not a string or path: ComfyUI's table would keep
    the value, and its model lists would fail.
    """
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
    _folder_paths.add_model_folder_path(
        folder_name=folder_name,
        full_folder_path=directory,
        is_default=is_default,
    )


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
    "clear_filename_list_cache",
    "exists_annotated_filepath",
    "filter_files_content_types",
    "filter_files_extensions",
    "folder_names_and_paths",
    "get_annotated_filepath",
    "get_directory_by_type",
    "get_filename_list",
    "get_filename_list_",
    "get_folder_paths",
    "get_full_path",
    "get_full_path_or_raise",
    "get_input_directory",
    "get_input_subfolders",
    "get_output_directory",
    "get_save_image_path",
    "get_system_user_directory",
    "get_temp_directory",
    "get_user_directory",
    "input_directory",
    "is_within_directory",
    "models_dir",
    "output_directory",
    "recursive_search",
    "supported_pt_extensions",
    "temp_directory",
]
