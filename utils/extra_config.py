import os
import yaml
import folder_paths
import logging

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        if "base_path" in c:
            base_path = os.path.expandvars(os.path.expanduser(conf))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
                continue
        if "is_default" in conf:
            continue
        if len(conf) == 0:
            continue
        full_path = conf
        if base_path:
            full_path = os.path.join(base_path, full_path)
        elif not os.path.isabs(full_path):
            full_path = os.path.abspath(os.path.join(yaml_dir, conf))
        normalized_path = os.path.normpath(full_path)
        logging.info("Adding extra search path {} {}".format(c, normalized_path))
        folder_paths.add_model_folder_path(c, normalized_path, False)