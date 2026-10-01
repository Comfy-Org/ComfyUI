# models.yaml lists the models the Models sidebar tab can install, grouped by folder inside ComfyUI's models directory:
#   vae:
#     - name: SDXL VAE
#       url: https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
#       filename: sdxl_vae.safetensors
# Models are handled as {name: {"name", "folder", "url", "filename"}}.

import yaml


def parse_catalog(text):
    folders = yaml.safe_load(text) or {}
    return {model["name"]: model | {"folder": folder} for folder, models in folders.items() for model in models}


def dump_catalog(catalog):
    folders = {}
    for model in catalog.values():
        folders.setdefault(model["folder"], []).append({key: value for key, value in model.items() if key != "folder"})
    return yaml.safe_dump(folders, sort_keys=False, allow_unicode=True)
