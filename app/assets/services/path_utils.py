import os
from pathlib import Path
from typing import Literal

import folder_paths
from app.assets.helpers import normalize_tags


_NON_MODEL_FOLDER_NAMES = frozenset({"custom_nodes"})


def get_comfy_models_folders() -> list[tuple[str, list[str]]]:
    """Build list of (folder_name, base_paths[]) for all model locations.

    Includes every category registered in folder_names_and_paths,
    regardless of whether its paths are under the main models_dir,
    but excludes non-model entries like custom_nodes.
    """
    targets: list[tuple[str, list[str]]] = []
    for name, values in folder_paths.folder_names_and_paths.items():
        if name in _NON_MODEL_FOLDER_NAMES:
            continue
        paths, _exts = values[0], values[1]
        if paths:
            targets.append((name, paths))
    return targets


def resolve_destination_from_tags(tags: list[str]) -> tuple[str, list[str]]:
    """Validates and maps tags -> (base_dir, subdirs_for_fs)"""
    if not tags:
        raise ValueError("tags must not be empty")
    root = tags[0].lower()
    if root == "models":
        if len(tags) < 2:
            raise ValueError("at least two tags required for model asset")
        try:
            bases = folder_paths.folder_names_and_paths[tags[1]][0]
        except KeyError:
            raise ValueError(f"unknown model category '{tags[1]}'")
        if not bases:
            raise ValueError(f"no base path configured for category '{tags[1]}'")
        base_dir = os.path.abspath(bases[0])
        raw_subdirs = tags[2:]
    elif root == "input":
        base_dir = os.path.abspath(folder_paths.get_input_directory())
        raw_subdirs = tags[1:]
    elif root == "output":
        base_dir = os.path.abspath(folder_paths.get_output_directory())
        raw_subdirs = tags[1:]
    else:
        raise ValueError(f"unknown root tag '{tags[0]}'; expected 'models', 'input', or 'output'")
    _sep_chars = frozenset(("/", "\\", os.sep))
    for i in raw_subdirs:
        if i in (".", "..") or _sep_chars & set(i):
            raise ValueError("invalid path component in tags")

    return base_dir, raw_subdirs if raw_subdirs else []


def validate_path_within_base(candidate: str, base: str) -> None:
    cand_abs = Path(os.path.abspath(candidate))
    base_abs = Path(os.path.abspath(base))
    if not cand_abs.is_relative_to(base_abs):
        raise ValueError("destination escapes base directory")


def compute_relative_filename(file_path: str) -> str | None:
    """
    Return the model's path relative to the last well-known folder (the model category),
    using forward slashes, eg:
      /.../models/checkpoints/flux/123/flux.safetensors -> "flux/123/flux.safetensors"
      /.../models/text_encoders/clip_g.safetensors -> "clip_g.safetensors"

    For non-model paths, returns None.
    """
    try:
        root_category, rel_path = get_asset_category_and_relative_path(file_path)
    except ValueError:
        return None

    p = Path(rel_path)
    parts = [seg for seg in p.parts if seg not in (".", "..", p.anchor)]
    if not parts:
        return None

    if root_category == "models":
        # parts[0] is the category ("checkpoints", "vae", etc) – drop it
        inside = parts[1:] if len(parts) > 1 else [parts[0]]
        return "/".join(inside)
    return "/".join(parts)  # input/output: keep all parts


def get_asset_category_and_relative_path(
    file_path: str,
) -> tuple[Literal["input", "output", "temp", "models"], str]:
    """Determine which root category a file path belongs to.

    Categories:
      - 'input': under folder_paths.get_input_directory()
      - 'output': under folder_paths.get_output_directory()
      - 'temp': under folder_paths.get_temp_directory()
      - 'models': under any base path from get_comfy_models_folders()

    Returns:
        (root_category, relative_path_inside_that_root)

    Raises:
        ValueError: path does not belong to any known root.
    """
    fp_abs = os.path.abspath(file_path)

    def _check_is_within(child: str, parent: str) -> bool:
        return Path(child).is_relative_to(parent)

    def _compute_relative(child: str, parent: str) -> str:
        # Normalize relative path, stripping any leading ".." components
        # by anchoring to root (os.sep) then computing relpath back from it.
        return os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )

    # 1) input
    input_base = os.path.abspath(folder_paths.get_input_directory())
    if _check_is_within(fp_abs, input_base):
        return "input", _compute_relative(fp_abs, input_base)

    # 2) output
    output_base = os.path.abspath(folder_paths.get_output_directory())
    if _check_is_within(fp_abs, output_base):
        return "output", _compute_relative(fp_abs, output_base)

    # 3) temp
    temp_base = os.path.abspath(folder_paths.get_temp_directory())
    if _check_is_within(fp_abs, temp_base):
        return "temp", _compute_relative(fp_abs, temp_base)

    # 4) models (check deepest matching base to avoid ambiguity)
    best: tuple[int, str, str] | None = None  # (base_len, bucket, rel_inside_bucket)
    for bucket, bases in get_comfy_models_folders():
        for b in bases:
            base_abs = os.path.abspath(b)
            if not _check_is_within(fp_abs, base_abs):
                continue
            cand = (len(base_abs), bucket, _compute_relative(fp_abs, base_abs))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is not None:
        _, bucket, rel_inside = best
        combined = os.path.join(bucket, rel_inside)
        return "models", os.path.relpath(os.path.join(os.sep, combined), os.sep)

    raise ValueError(
        f"Path is not within input, output, temp, or configured model bases: {file_path}"
    )


def get_name_and_tags_from_asset_path(file_path: str) -> tuple[str, list[str]]:
    """Return (name, tags) derived from a filesystem path.

    - name: base filename with extension
    - tags: [root_category] + parent folder names in order

    Raises:
        ValueError: path does not belong to any known root.
    """
    root_category, some_path = get_asset_category_and_relative_path(file_path)
    p = Path(some_path)
    parent_parts = [
        part for part in p.parent.parts if part not in (".", "..", p.anchor)
    ]
    return p.name, list(dict.fromkeys(normalize_tags([root_category, *parent_parts])))
