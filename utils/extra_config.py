import os
import yaml
import folder_paths
import logging

def _iter_paths(value, folder_name, section, yaml_path):
    """Yield the configured paths for one folder entry.

    A newline-separated string is the documented form. A YAML list is the shape
    people reach for anyway, and it used to abort startup with
    `AttributeError: 'list' object has no attribute 'split'` — a traceback that
    named neither the file nor the key. Anything else is skipped with a warning
    rather than crashing the whole config load.
    """
    if isinstance(value, str):
        yield from value.split("\n")
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                yield from item.split("\n")
            else:
                logging.warning(
                    "Skipping entry in extra search path '%s.%s' in %s: expected a path "
                    "string, got %s",
                    section, folder_name, yaml_path, type(item).__name__,
                )
        return

    logging.warning(
        "Skipping extra search path '%s.%s' in %s: expected a path string or a list of "
        "path strings, got %s",
        section, folder_name, yaml_path, type(value).__name__,
    )


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        if not isinstance(conf, dict):
            # A section that is not a mapping used to raise
            # `TypeError: string indices must be integers` from the
            # `"base_path" in conf` test, which named neither the file nor the
            # section. An empty section is already tolerated above; say what is
            # wrong with this one and keep loading the rest.
            logging.warning(
                "Skipping extra search path section '%s' in %s: expected a mapping of "
                "folder name to path(s), got %s",
                c, yaml_path, type(conf).__name__,
            )
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
            for y in _iter_paths(conf[x], x, c, yaml_path):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, y))
                normalized_path = os.path.normpath(full_path)
                logging.info("Adding extra search path {} {}".format(x, normalized_path))
                folder_paths.add_model_folder_path(x, normalized_path, is_default)
