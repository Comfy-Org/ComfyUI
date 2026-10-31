import os
import yaml
import folder_paths
import logging

from .json_util import merge_json_recursive

default_server_config = {
    'internal': {
        'modelsDownload': {
            'allowedSources': [
                'https://civitai.com/',
                'https://huggingface.co/'
            ],
            'allowedSuffixes': [
                '.safetensors',
                '.sft'
            ],
            'whitelistedUrls': [
                'https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt',
                'https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth?download=true',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            ]
        }
    }
}

def load_extra_path_config(yaml_path):
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
                normalized_path = os.path.normpath(full_path)
                logging.info("Adding extra search path {} {}".format(x, normalized_path))
                folder_paths.add_model_folder_path(x, normalized_path, is_default)

def load_server_config(component=None):
    """
    Load and returns the server configuration.
    ensure default configuration is present
    if a component is specified returns this sub configuration

    Warning: Current merge_json_recursive concatenate arrays and so there is no way to remove default allowed sources for instance
    """
    config_path = 'config.yaml'
    config = dict()
    try:
        with open(config_path, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        pass  # Default config could be empty

    config = merge_json_recursive(default_server_config, config)
    if component is not None:
        return config.get(component)
    return config
