import logging
import os
from pathlib import Path
import re
import time

try:
    from comfy_config import config_parser
except ImportError:
    # Without pydantic-settings ComfyUI still starts, as its loader does; V4 packs are refused.
    config_parser = None


_ID_PATTERN = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?")
RESERVED_IDS = frozenset({"comfy-core", "my-pack"})
NODE_FIELDS = frozenset(name.lower() for name in (
    "title", "id", "type", "pos", "size", "flags", "order", "mode", "outputs", "inputs", "properties", "shape",
    "boxcolor", "color", "bgcolor", "showAdvanced", "widgets_values", "widgets_values_named", "extensions",
))
_SKIPPED_DIRS = frozenset({"node_modules", ".git", "__pycache__"})
_REGISTER_PACK_CALL = re.compile(rb"registerPack\s*(?:\?\.\s*)?\(\s*import\s*\.\s*meta\b")
_checked: tuple[frozenset[str], tuple[dict[str, tuple[str, str | None]], dict[str, str]]] | None = None


def calls_register_pack(module_path: str) -> bool:
    """Whether any .js or .mjs file in the folder calls registerPack(import.meta), which makes it a V4 pack."""
    for root, dirs, files in os.walk(module_path):
        dirs[:] = [d for d in dirs if d not in _SKIPPED_DIRS]
        for name in files:
            if not name.endswith((".js", ".mjs")):
                continue
            try:
                with open(os.path.join(root, name), "rb") as f:
                    if _REGISTER_PACK_CALL.search(f.read()):
                        return True
            except OSError:
                continue
    return False


def pack_identity(module_path: str) -> tuple[str, str | None]:
    """The (id, version) a pack's pyproject.toml declares; ValueError says why it has none."""
    toml_path = Path(module_path) / "pyproject.toml"
    if config_parser is None:
        raise ValueError("ComfyUI cannot read pyproject.toml files: pydantic-settings is not installed; run 'pip install -r requirements.txt'")
    if not toml_path.is_file():
        raise ValueError('it has no pyproject.toml; add one beside its __init__.py with [project] name = "<registry id>"')
    try:
        project = config_parser.load_pyproject_settings(toml_path).project
    except (ValueError, OSError) as e:
        raise ValueError(f"its pyproject.toml does not parse ({e}); fix the TOML so it gives [project] name")
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError('its pyproject.toml has no [project] name; add name = "<registry id>" under [project]')
    pack_id = name.strip().lower()
    if len(pack_id) > 100 or not _ID_PATTERN.fullmatch(pack_id):
        raise ValueError(f"[project] name '{name}' is not a registry id; change it to 1-100 letters, digits, '.', '_' or '-', starting and ending with a letter or digit")
    if pack_id in RESERVED_IDS:
        raise ValueError(f"[project] name '{name}' is reserved; change it to the pack's own registry id")
    if pack_id in NODE_FIELDS:
        raise ValueError(f"[project] name '{name}' is a field every node saves; change it to the pack's own registry id")
    version = project.get("version")
    return pack_id, version if isinstance(version, str) else None


def check_packs(module_paths: list[str], held_ids: dict[str, str]) -> tuple[dict[str, tuple[str, str | None]], dict[str, str]]:
    """Check every V4 pack among the folders the loader will import; held_ids maps ids already verified to their folder.

    Returns the verified packs as {module path: (id, version)} and the refused ones as {module path: reason}.
    """
    identities = {}
    refusals = {}
    for module_path in module_paths:
        if not os.path.isdir(module_path) or not calls_register_pack(module_path):
            continue
        try:
            identities[module_path] = pack_identity(module_path)
        except ValueError as e:
            refusals[module_path] = f"V4 pack {module_path} is not loaded: {e}."

    claimants = {}
    for module_path, (pack_id, _) in list(identities.items()):
        if pack_id in held_ids:
            del identities[module_path]
            refusals[module_path] = f"V4 pack {module_path} is not loaded: its id '{pack_id}' is already held by {held_ids[pack_id]}; remove or disable one of them."
        else:
            claimants.setdefault(pack_id, []).append(module_path)
    for pack_id, paths in claimants.items():
        if len(paths) > 1:
            for module_path in paths:
                del identities[module_path]
                refusals[module_path] = f"V4 pack {module_path} is not loaded: its id '{pack_id}' is claimed by more than one folder ({', '.join(sorted(paths))}); remove or disable all but one of them."
    return identities, refusals


def check_custom_nodes(module_paths: list[str]) -> tuple[dict[str, tuple[str, str | None]], dict[str, str]]:
    """check_packs for the custom-node folders ComfyUI will run, logged once per set of folders.

    The prestartup phase and the loader pass the same folders, so both get one result and skip the same packs. If the
    folders changed in between, only the new ones are checked: earlier results stand, and a new folder claiming an id
    verified earlier is refused.
    """
    global _checked
    folders = frozenset(p for p in module_paths if os.path.isdir(p) and os.path.basename(p) != "__pycache__")
    if _checked is None or _checked[0] != folders:
        earlier_folders, (earlier_identities, earlier_refusals) = _checked if _checked is not None else (frozenset(), ({}, {}))
        held_ids = {pack_id: module_path for module_path, (pack_id, _) in earlier_identities.items() if module_path in folders}
        time_before = time.perf_counter()
        identities, refusals = check_packs(sorted(folders - earlier_folders), held_ids)
        logging.info("Checked V4 packs in {:.3f} seconds: {} verified, {} refused".format(time.perf_counter() - time_before, len(identities), len(refusals)))
        for reason in refusals.values():
            logging.error(reason)
        identities.update((p, identity) for p, identity in earlier_identities.items() if p in folders)
        refusals.update((p, reason) for p, reason in earlier_refusals.items() if p in folders)
        _checked = (folders, (identities, refusals))
    return _checked[1]


def held_by(pack_id: str) -> str | None:
    """The folder of the V4 pack that check_custom_nodes verified with this id, if any."""
    identities = _checked[1][0] if _checked is not None else {}
    return next((path for path, (held, _) in identities.items() if held == pack_id), None)


def pack_records(web_dirs: dict[str, str], identities: dict[str, tuple[str, str | None]]) -> list[dict]:
    """One record for each served web folder that lies in a verified V4 pack."""
    records = []
    for segment, web_dir in web_dirs.items():
        web_dir = Path(os.path.abspath(web_dir))
        for module_path, (pack_id, version) in identities.items():
            if web_dir.is_relative_to(os.path.abspath(module_path)):
                records.append({"segment": segment, "id": pack_id, "version": version})
    return records
