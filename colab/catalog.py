# models.yaml lists the models the Download Models sidebar tab can install, grouped by folder inside ComfyUI's models directory.
# A model is downloaded from its url, or copied from its path (e.g. a mounted Google Drive), filename defaults to the
# path's file name:
#   vae:
#     - name: SDXL VAE
#       url: https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
#       filename: sdxl_vae.safetensors
#   loras:
#     - name: My LoRA
#       path: /content/drive/MyDrive/ComfyUI/models/loras/my_lora.safetensors
# Models are handled as {name: {"name", "folder", "url" or "path", "filename"}}.

import os
import shutil

import yaml


def parse_catalog(text):
    folders = yaml.safe_load(text) or {}
    catalog = {}
    for folder, models in folders.items():
        for model in models:
            if "path" in model:
                model.setdefault("filename", os.path.basename(model["path"]))
            catalog[model["name"]] = model | {"folder": folder}
    return catalog


def dump_catalog(catalog):
    folders = {}
    for model in catalog.values():
        folders.setdefault(model["folder"], []).append({key: value for key, value in model.items() if key != "folder"})
    return yaml.safe_dump(folders, sort_keys=False, allow_unicode=True)


# Copies to a .part file first so an interrupted copy isn't mistaken for an installed model.
def copy_model(source, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(source, path + ".part")
    os.replace(path + ".part", path)
