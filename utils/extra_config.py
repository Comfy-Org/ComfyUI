from __future__ import annotations

import os
import yaml
import folder_paths
import logging

_SYSTEM_DIR_KEYS = frozenset({"output", "input", "temp", "user"})


def _resolve_base(raw: str, parent_base: str | None, yaml_dir: str) -> str:
    """Resolve a base_path value: expand vars/user, join onto parent_base or yaml_dir if relative."""
    raw = os.path.expandvars(os.path.expanduser(raw))
    if not os.path.isabs(raw):
        anchor = parent_base if parent_base else yaml_dir
        raw = os.path.abspath(os.path.join(anchor, raw))
    return os.path.normpath(raw)


def _add_model_paths(category: str, raw_value: str, base: str | None, yaml_dir: str, is_default: bool) -> None:
    """Split a (possibly multi-line) path value and register each path as a model folder."""
    for raw in str(raw_value).split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if base and not os.path.isabs(raw):
            full_path = os.path.join(base, raw)
        elif not os.path.isabs(raw):
            full_path = os.path.abspath(os.path.join(yaml_dir, raw))
        else:
            full_path = raw
        normalized = os.path.normpath(full_path)
        logging.info("Adding extra search path %s %s", category, normalized)
        folder_paths.add_model_folder_path(category, normalized, is_default)


def _implicit_scan(base: str, exclude: set[str], is_default: bool) -> None:
    """Auto-register base/<category>/ for known model categories that exist on disk.

    custom_nodes and system directory keys are always excluded from the scan.
    """
    skip = _SYSTEM_DIR_KEYS | {"custom_nodes"} | exclude
    for category in folder_paths.folder_names_and_paths:
        if category in skip:
            continue
        path = os.path.normpath(os.path.join(base, category))
        if os.path.isdir(path):
            logging.info("Adding extra search path %s %s", category, path)
            folder_paths.add_model_folder_path(category, path, is_default)


def load_extra_path_config(yaml_path: str, allow_system_dirs: bool = False) -> None:
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

    for _block_name, conf in config.items():
        if conf is None:
            continue

        # Pop block-level meta keys (preserved for flat backward-compat style)
        block_base = None
        if "base_path" in conf:
            block_base = _resolve_base(conf.pop("base_path"), None, yaml_dir)
        block_is_default = bool(conf.pop("is_default", False))

        has_models_block = False
        flat_model_keys: set[str] = set()

        for key, value in conf.items():
            if allow_system_dirs and key in _SYSTEM_DIR_KEYS:
                # System directory override → set_*_directory()
                path = _resolve_base(str(value).strip(), block_base, yaml_dir)
                logging.info("Setting %s directory to %s", key, path)
                getattr(folder_paths, f"set_{key}_directory")(path)

            elif key == "custom_nodes":
                _add_model_paths("custom_nodes", value, block_base, yaml_dir, block_is_default)

            elif key == "models" and isinstance(value, dict):
                # New nested style: models: { base_path, is_default, <categories> }
                has_models_block = True
                models_conf = dict(value)
                models_base = block_base
                if "base_path" in models_conf:
                    models_base = _resolve_base(models_conf.pop("base_path"), block_base, yaml_dir)
                models_is_default = bool(models_conf.pop("is_default", block_is_default))
                explicit: set[str] = set(models_conf.keys())
                for cat, raw in models_conf.items():
                    _add_model_paths(cat, raw, models_base, yaml_dir, models_is_default)
                if models_base:
                    _implicit_scan(models_base, explicit, models_is_default)

            else:
                # Flat model key — backward-compat style
                _add_model_paths(key, value, block_base, yaml_dir, block_is_default)
                flat_model_keys.add(key)

        # Flat-style implicit scan (only when no nested models: block)
        if block_base and not has_models_block:
            _implicit_scan(block_base, flat_model_keys, block_is_default)
