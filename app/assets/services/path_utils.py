import os
from pathlib import Path
from typing import Literal

import folder_paths


_NON_MODEL_FOLDER_NAMES = frozenset({"configs", "custom_nodes"})
_KNOWN_SUBFOLDER_TAGS = frozenset({"3d", "pasted", "painter", "threed", "webcam"})


def get_comfy_models_folders() -> list[tuple[str, list[str], set[str]]]:
    """Build list of (folder_name, base_paths[], extensions) for all model locations.

    Includes every category registered in folder_names_and_paths,
    regardless of whether its paths are under the main models_dir,
    but excludes non-model entries like configs and custom_nodes.

    An empty extensions set means the category accepts any extension,
    matching folder_paths.filter_files_extensions semantics.
    """
    targets: list[tuple[str, list[str], set[str]]] = []
    for name, values in folder_paths.folder_names_and_paths.items():
        if name in _NON_MODEL_FOLDER_NAMES:
            continue
        paths, exts = values[0], values[1]
        if paths:
            targets.append((name, paths, set(exts)))
    return targets


def resolve_destination_from_tags(tags: list[str]) -> tuple[str, list[str]]:
    """Validates and maps upload routing tags -> (base_dir, subdirs_for_fs).

    The request tags are only used to choose the write destination. Extra tags
    remain labels; they do not become path components or trusted classification.
    """
    destination_roles = [t for t in tags if t in {"input", "models", "output"}]
    if len(destination_roles) != 1:
        raise ValueError("uploads require exactly one destination role: input, models, or output")

    root = destination_roles[0]
    if root == "models":
        model_type_tags = [t for t in tags if t.startswith("model_type:")]
        if len(model_type_tags) != 1:
            raise ValueError("models uploads require exactly one model_type:<folder_name> tag")
        folder_name = model_type_tags[0].split(":", 1)[1]
        if not folder_name:
            raise ValueError("models uploads require exactly one model_type:<folder_name> tag")
        model_folder_paths = {
            name: paths for name, paths, _exts in get_comfy_models_folders()
        }
        try:
            bases = model_folder_paths[folder_name]
        except KeyError:
            raise ValueError(f"unknown model category '{folder_name}'")
        if not bases:
            raise ValueError(f"no base path configured for category '{folder_name}'")
        base_dir = os.path.abspath(bases[0])
    elif root == "input":
        base_dir = os.path.abspath(folder_paths.get_input_directory())
    else:
        base_dir = os.path.abspath(folder_paths.get_output_directory())

    return base_dir, []


def validate_path_within_base(candidate: str, base: str) -> None:
    cand_abs = Path(os.path.abspath(candidate))
    base_abs = Path(os.path.abspath(base))
    if not cand_abs.is_relative_to(base_abs):
        raise ValueError("destination escapes base directory")


def _compute_relative_path(child: str, parent: str) -> str:
    rel = os.path.relpath(os.path.abspath(child), os.path.abspath(parent))
    if rel == ".":
        return ""
    return rel.replace(os.sep, "/")


def _is_relative_to(child: str, parent: str) -> bool:
    return Path(os.path.abspath(child)).is_relative_to(os.path.abspath(parent))


def compute_asset_response_paths(file_path: str) -> tuple[str, str | None] | None:
    """Return (logical_path, display_name) for a file path.

    ``logical_path`` is the internal namespaced storage locator (e.g.
    ``models/checkpoints/foo/bar.safetensors``); ``display_name`` is the
    human-facing label below that namespace, served on Asset responses. These
    are storage locators, not model-loader namespaces. Registered model-folder
    membership is represented by backend tags such as
    ``model_type:<folder_name>``; these paths only use known storage roots.
    """
    fp_abs = os.path.abspath(file_path)
    candidates: list[tuple[int, int, str, str]] = []

    for order, (namespace, base) in enumerate(
        (
            ("input", folder_paths.get_input_directory()),
            ("output", folder_paths.get_output_directory()),
            ("temp", folder_paths.get_temp_directory()),
            ("models", getattr(folder_paths, "models_dir", "")),
        )
    ):
        if not base:
            continue
        base_abs = os.path.abspath(base)
        if _is_relative_to(fp_abs, base_abs):
            candidates.append((len(base_abs), -order, namespace, base_abs))

    if not candidates:
        return None

    _base_len, _order, namespace, base = max(candidates)
    rel = _compute_relative_path(fp_abs, base)
    public_path = f"{namespace}/{rel}" if rel else namespace
    return public_path, rel or None


def compute_display_name(file_path: str) -> str | None:
    """Return the asset's `display_name`, or None for unknown paths."""
    result = compute_asset_response_paths(file_path)
    return result[1] if result else None


def compute_logical_path(file_path: str) -> str | None:
    """Return the internal namespaced storage locator, or None for unknown paths."""
    result = compute_asset_response_paths(file_path)
    return result[0] if result else None


def compute_loader_path(file_path: str) -> str | None:
    """
    Return the asset's in-root loader path: the path relative to the last
    well-known folder (the model category), using forward slashes, eg:
      /.../models/checkpoints/flux/123/flux.safetensors -> "flux/123/flux.safetensors"
      /.../models/text_encoders/clip_g.safetensors -> "clip_g.safetensors"

    This is the value model loaders consume (the model category is dropped). It
    is persisted as ``AssetReference.loader_path`` and served as the public
    Asset response `loader_path` field. The human-facing `display_name` comes
    from compute_asset_response_paths().

    For input/output/temp paths the full path relative to that root is returned.
    For paths outside any known root, returns None.
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
        rel = os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )
        return "" if rel == "." else rel.replace(os.sep, "/")

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
    ext = os.path.splitext(fp_abs)[1].lower()
    best: tuple[int, str, str] | None = None  # (base_len, bucket, rel_inside_bucket)
    for bucket, bases, extensions in get_comfy_models_folders():
        # A bucket only lists files within its extension set (empty set
        # accepts any extension), so a bucket that cannot load the file
        # must not contribute a loader path.
        if extensions and ext not in extensions:
            continue
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
        normalized = os.path.relpath(os.path.join(os.sep, combined), os.sep)
        return "models", normalized.replace(os.sep, "/")

    raise ValueError(
        f"Path is not within input, output, temp, or configured model bases: {file_path}"
    )


def get_backend_system_tags_from_path(path: str) -> list[str]:
    """Return trusted backend tags derived from current filesystem facts.

    The returned tags are only the backend-generated system tags: ``models``,
    ``model_type:<folder_name>``, ``input``, ``output``, and ``temp``. Model
    type tags are based on registered folder names, not path components.

    A ``model_type:<folder_name>`` tag is only emitted when the file's
    extension is accepted by that folder's registered extension set, so
    categories sharing a base directory tag only the files they can
    actually load. Files under a model base whose extension matches no
    category still get the ``models`` tag.
    """
    fp_abs = os.path.abspath(path)
    fp_path = Path(fp_abs)
    tags: list[str] = []

    def _add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    for role, base in (
        ("input", folder_paths.get_input_directory()),
        ("output", folder_paths.get_output_directory()),
        ("temp", folder_paths.get_temp_directory()),
    ):
        if fp_path.is_relative_to(os.path.abspath(base)):
            _add(role)

    ext = os.path.splitext(fp_abs)[1].lower()
    model_types: list[str] = []
    under_models_base = False
    for folder_name, bases, extensions in get_comfy_models_folders():
        for base in bases:
            if fp_path.is_relative_to(os.path.abspath(base)):
                under_models_base = True
                # Empty set accepts any extension, matching
                # folder_paths.filter_files_extensions semantics.
                if not extensions or ext in extensions:
                    model_types.append(folder_name)
                break

    if under_models_base:
        _add("models")
    for folder_name in model_types:
        _add(f"model_type:{folder_name}")

    if not tags:
        raise ValueError(
            f"Path is not within input, output, temp, or configured model bases: {path}"
        )
    return tags


def get_known_subfolder_tags(subfolder: str | None) -> list[str]:
    """Return tags for known UI/input subfolder names."""
    if subfolder in _KNOWN_SUBFOLDER_TAGS:
        return [subfolder]
    return []


def get_known_input_subfolder_tags_from_path(path: str) -> list[str]:
    """Return known input-layout tags for files in canonical input subfolders.

    These are compatibility tags for current UI-origin input directories such as
    ``pasted`` and ``webcam``. They are intentionally narrow: only files directly
    inside a known top-level input directory receive the matching tag.
    """
    fp_abs = os.path.abspath(path)
    input_base = os.path.abspath(folder_paths.get_input_directory())
    if not Path(fp_abs).is_relative_to(input_base):
        return []

    rel = os.path.relpath(fp_abs, input_base)
    parts = Path(rel).parts
    if len(parts) == 2:
        return get_known_subfolder_tags(parts[0])
    return []


def get_path_derived_tags_from_path(path: str) -> list[str]:
    """Return all backend-derived tags for an asset path."""
    tags = get_backend_system_tags_from_path(path)
    for tag in get_known_input_subfolder_tags_from_path(path):
        if tag not in tags:
            tags.append(tag)
    return tags


def get_name_and_tags_from_asset_path(file_path: str) -> tuple[str, list[str]]:
    """Return (name, tags) derived from a filesystem path.

    - name: base filename with extension
    - tags: backend-derived tags from root/model classification and known input
      subfolder layout conventions

    Raises:
        ValueError: path does not belong to any known root.
    """
    return Path(file_path).name, get_path_derived_tags_from_path(file_path)
