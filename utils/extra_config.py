import os
import yaml
import folder_paths
import logging

def _extra_paths(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, y))
                yield x, os.path.normpath(full_path), is_default

def load_extra_path_config(yaml_path):
    for x, normalized_path, is_default in _extra_paths(yaml_path):
        logging.info("Adding extra search path {} {}".format(x, normalized_path))
        folder_paths.add_model_folder_path(x, normalized_path, is_default)

def restore_extra_path_config(yaml_path):
    """Re-add extra search paths that a custom node dropped by replacing a folder's path list.
    Only appends missing paths, so the node's own paths keep their order."""
    for x, normalized_path, _is_default in _extra_paths(yaml_path):
        entry = folder_paths.folder_names_and_paths.get(folder_paths.map_legacy(x))
        if entry is not None and normalized_path not in entry[0]:
            logging.info("Restoring extra search path {} {} (a custom node replaced this folder's paths)".format(x, normalized_path))
            folder_paths.add_model_folder_path(x, normalized_path)
