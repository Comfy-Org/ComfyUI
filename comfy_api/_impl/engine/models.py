"""Core engine extension: ComfyUI's model folders and loaders."""
_DTYPES = ("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")


def _path(folder, name):
    import folder_paths

    return folder_paths.get_full_path_or_raise(folder, name)


def _dtype(name):
    import torch

    if name is None:
        return None
    name = str(name).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(_DTYPES)}")
    return getattr(torch, name)


class EngineOps:
    def list(self, folder):
        import folder_paths

        return folder_paths.get_filename_list(folder)

    def read(self, folder, name):
        import comfy.utils

        sd, metadata = comfy.utils.load_torch_file(_path(folder, name), safe_load=True, return_metadata=True)
        return {"sd": sd, "metadata": dict(metadata or {})}

    def load_checkpoint(self, name, options=None):
        import comfy.sd
        import folder_paths

        out = comfy.sd.load_checkpoint_guess_config(
            _path("checkpoints", name), output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=dict(options or {}))
        return tuple(out[:3])

    def load_diffusion_model(self, name, options=None):
        import comfy.sd

        options = dict(options or {})
        if "dtype" in options:
            options["dtype"] = _dtype(options["dtype"])
        return comfy.sd.load_diffusion_model(_path("diffusion_models", name), model_options=options)

    def load_vae(self, name, device=None, dtype=None):
        import comfy.sd
        import comfy.utils
        import torch

        sd, metadata = comfy.utils.load_torch_file(_path("vae", name), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=torch.device(device) if device else None,
                           dtype=_dtype(dtype))
        vae.throw_exception_if_invalid()
        return vae
