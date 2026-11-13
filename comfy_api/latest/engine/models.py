"""ComfyUI's own engine extensions for ``eng.model``, ``eng.clip``, ``eng.vae``, ``eng.models``, ``eng.sampler`` and the ``info`` rows: the engine's models, text encoders, VAEs and samplers."""
from comfy_api.latest import _runtime


def _patched(model, patches, strength_patch, strength_model):
    patched = model.clone()
    # A patch value is a tuple to the engine, however it arrived.
    patched.add_patches({key: tuple(value) for key, value in patches.items()}, strength_patch, strength_model)
    return patched


def _hook(setter):
    def method(self, model, fn):
        patched = model.clone()
        getattr(patched, setter)(fn)
        return patched
    method.__name__ = setter
    return method


class ModelOps:
    """The ``eng.model`` rows (a MODEL is ComfyUI's ModelPatcher)."""

    def clone(self, model):
        return model.clone()

    def add_patches(self, model, patches, strength_patch=1.0, strength_model=1.0):
        return _patched(model, patches, strength_patch, strength_model)

    def load_lora(self, model, clip, lora, strength_model=1.0, strength_clip=1.0, lora_metadata=None):
        import comfy.sd
        if lora_metadata:  # only when given: ComfyUI versions before lora_metadata take five arguments
            return tuple(comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                                       lora_metadata=dict(lora_metadata)))
        return tuple(comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip))

    def read_weight(self, model, key):
        # a plain tensor, as documented: an unpatched weight is an nn.Parameter, which would cross as a handle
        return model.patch_weight_to_device(key, return_weight=True).detach()

    def latent_process_in(self, model, latent):
        return model.model.latent_format.process_in(latent)

    def latent_process_out(self, model, latent):
        return model.model.latent_format.process_out(latent)

    # The model's own options, as plain data: what sampling functions and patches see.
    def options(self, model):
        from comfy_api.latest._engine import hook_view

        return hook_view(model.model_options)

    def set_option(self, model, path, value):
        from comfy_api.latest._engine import hook_view

        if not path or not all(isinstance(p, str) and p for p in path):
            raise ValueError("path is a list of option keys, such as ['transformer_options', 'my_key']")
        if hook_view(value) is None and value is not None:
            raise TypeError("an option value is plain data or tensors")
        patched = model.clone()
        node = patched.model_options
        for key in path[:-1]:
            child = node.get(key)
            node[key] = dict(child) if isinstance(child, dict) else {}  # copy: the original model keeps its own
            node = node[key]
        node[path[-1]] = value
        return patched

    # Pack functions around sampling stages and model calls. They get the engine's next step as an
    # EngineFunction and engine values as Opaque, wherever the pack runs.
    def add_wrapper(self, model, kind, fn):
        from comfy.patcher_extension import WrappersMP
        from comfy_api.latest._engine import pack_function

        kinds = {name.lower(): value for name, value in vars(WrappersMP).items() if name.isupper()}
        if kind not in kinds.values():
            raise ValueError(f"wrapper kind must be one of {sorted(kinds.values())}, got {kind!r}")
        patched = model.clone()
        patched.add_wrapper_with_key(kind, None, pack_function(fn, engine_args=(0,)))
        return patched

    def set_unet_wrapper(self, model, fn):
        """A wrapper already on the model runs inside this one: ``apply_model`` goes through it."""
        from comfy_api.latest._engine import pack_function

        run = pack_function(fn, engine_args=(0,))
        patched = model.clone()
        old = patched.model_options.get("model_function_wrapper")
        if old is None:
            wrapper = run
        else:
            def wrapper(apply_model, args):
                def inner(x, t, **c):
                    return old(apply_model, {"input": x, "timestep": t, "c": c, "cond_or_uncond": args.get("cond_or_uncond")})
                return run(inner, args)
        patched.set_model_unet_function_wrapper(wrapper)
        return patched

    # Model sampling: sigmas and timesteps, as samplers compute them.
    def percent_to_sigma(self, model, percent):
        return float(model.get_model_object("model_sampling").percent_to_sigma(percent))

    def timestep(self, model, sigma):
        return model.get_model_object("model_sampling").timestep(sigma)

    def sigma(self, model, timestep):
        return model.get_model_object("model_sampling").sigma(timestep)

    # Functions that run during sampling. The dispatcher has already wrapped ``fn`` so it sees
    # tensors and plain data only.
    set_cfg = _hook("set_model_sampler_cfg_function")
    set_post_cfg = _hook("set_model_sampler_post_cfg_function")
    set_pre_cfg = _hook("set_model_sampler_pre_cfg_function")
    set_denoise_mask = _hook("set_model_denoise_mask_function")
    set_attn1_patch = _hook("set_model_attn1_patch")
    set_attn2_patch = _hook("set_model_attn2_patch")
    set_attn1_output_patch = _hook("set_model_attn1_output_patch")
    set_attn2_output_patch = _hook("set_model_attn2_output_patch")
    set_input_block_patch = _hook("set_model_input_block_patch")
    set_input_block_patch_after_skip = _hook("set_model_input_block_patch_after_skip")
    set_output_block_patch = _hook("set_model_output_block_patch")
    set_middle_block_after_patch = _hook("set_model_middle_block_after_patch")

    def set_attention(self, model, backend):
        from comfy.ldm.modules.attention import get_attention_function, REGISTERED_ATTENTION_FUNCTIONS

        function = get_attention_function(backend, default=None)
        if function is None:
            names = ", ".join(["optimized", *sorted(REGISTERED_ATTENTION_FUNCTIONS)])
            raise ValueError(f"unknown attention backend {backend!r}; this ComfyUI has: {names}")
        patched = model.clone()
        patched.set_model_optimized_attention(function)
        return patched

    def torch_compile(self, model, backend="inductor", mode=None, fullgraph=False, dynamic=None, keys=None, options=None):
        from comfy_api.torch_helpers import set_torch_compile_wrapper

        patched = model.clone()
        set_torch_compile_wrapper(patched, backend=backend, options=options, mode=mode, fullgraph=fullgraph,
                                  dynamic=dynamic, keys=list(keys) if keys else ["diffusion_model"])
        return patched


class ClipOps:
    """The ``eng.clip`` rows (a CLIP is ComfyUI's text encoder)."""

    def clone(self, clip):
        return clip.clone()

    def tokenize(self, clip, text, **kwargs):
        return clip.tokenize(text, **kwargs)

    def encode_tokens(self, clip, tokens, **kwargs):
        return clip.encode_from_tokens_scheduled(tokens, **kwargs)

    def encode_from_tokens(self, clip, tokens, return_pooled=False, return_dict=False):
        return clip.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)

    def add_patches(self, clip, patches, strength_patch=1.0, strength_model=1.0):
        patched = clip.clone()
        patched.add_patches({key: tuple(value) for key, value in patches.items()}, strength_patch, strength_model)
        return patched

    def read_weight(self, clip, key):
        # a plain tensor, as documented: an unpatched weight is an nn.Parameter, which would cross as a handle
        return clip.patcher.patch_weight_to_device(key, return_weight=True).detach()


class VaeOps:
    """The ``eng.vae`` rows (a VAE is ComfyUI's comfy.sd.VAE)."""

    def decode(self, vae, samples):
        return vae.decode(samples)

    def decode_tiled(self, vae, samples, **kwargs):
        return vae.decode_tiled(samples, **kwargs)

    def encode(self, vae, pixels):
        return vae.encode(pixels)

    def encode_tiled(self, vae, pixels, **kwargs):
        return vae.encode_tiled(pixels, **kwargs)

    def from_state_dict(self, sd):
        import comfy.sd

        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return vae


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


class ModelsOps:
    """ComfyUI's model folders and loaders."""

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


def _plain(value):
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _sampling(m):
    sampling = getattr(getattr(m, "model", None), "model_sampling", None)
    if sampling is None:
        return {"sigma_min": None, "sigma_max": None}
    return {"sigma_min": float(sampling.sigma_min), "sigma_max": float(sampling.sigma_max)}


_INFO_FIELDS = {
    "MODEL": lambda m: {
        "model_type": getattr(getattr(m.model, "model_type", None), "name", None),
        "config_class": type(getattr(m.model, "model_config", None)).__name__,
        "unet_config": _plain(getattr(getattr(m.model, "model_config", None), "unet_config", {})),
        "latent_scale_factor": getattr(getattr(m.model, "latent_format", None), "scale_factor", None),
        "latent_channels": getattr(getattr(m.model, "latent_format", None), "latent_channels", None),
        "zsnr": bool(getattr(getattr(m.model, "model_sampling", None), "zsnr", False)),
        "dtype": str(m.model_dtype()).removeprefix("torch.") if hasattr(m, "model_dtype") else None,
        **_sampling(m),
    },
    "VAE": lambda v: {
        "latent_channels": getattr(v, "latent_channels", None),
        "dtype": str(getattr(v, "vae_dtype", None)),
        "spacial_compression_decode": v.spacial_compression_decode(),
        "spacial_compression_encode": v.spacial_compression_encode(),
        "temporal_compression_decode": v.temporal_compression_decode(),
        "downscale_ratio": _plain(getattr(v, "downscale_ratio", None)) if isinstance(getattr(v, "downscale_ratio", None), (int, float)) else None,
        "upscale_ratio": _plain(getattr(v, "upscale_ratio", None)) if isinstance(getattr(v, "upscale_ratio", None), (int, float)) else None,
    },
    "UPSCALE_MODEL": lambda u: {"scale": _plain(getattr(u, "scale", None)), "architecture": type(getattr(u, "model", None)).__name__},
    "CLIP": lambda c: {"layer_idx": getattr(c, "layer_idx", None)},
}


class InfoOps:
    """The ``info`` and ``weight_keys`` rows (``eng.model.info``, ``eng.vae.info``,
``eng.clip.info``...), read-only facts about engine objects."""

    def info(self, obj):
        kind = _runtime.handle_kind(obj)
        fields = _INFO_FIELDS.get(kind)
        if fields is None:
            raise _runtime.CallNotSupported(f"info doesn't cover {kind}")
        return fields(obj)

    def weight_keys(self, obj):
        patcher = getattr(obj, "patcher", obj)
        return sorted(patcher.model_state_dict().keys())


class SamplerOps:
    """The ``eng.sampler`` rows ComfyUI implements itself."""

    def custom(self, fn, options=None, model_sampling=False):
        """A SAMPLER whose loop is ``fn(model, x, sigmas, extra_args, callback, disable, **options)``:
        ``model`` (the denoiser) and ``callback`` (previews, progress) are EngineFunctions it calls;
        a SAMPLER among ``options`` is one too; ``model_sampling=True`` adds the model's sampling
        facts as the ``model_sampling`` keyword."""
        import comfy.samplers
        from comfy_api.latest import _engine

        if options is not None and not isinstance(options, dict):
            raise TypeError("options is a dict of the sampler's own keyword arguments")
        options = {k: _engine.sampler_function(v) if isinstance(v, comfy.samplers.Sampler) and hasattr(v, "sampler_function") else v
                   for k, v in (options or {}).items()}
        run = _engine.pack_function(fn, engine_args=(0, "model", "callback"))

        def sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            _engine.check_sigmas(sigmas)
            if model_sampling:
                kwargs["model_sampling"] = _engine.model_sampling_view(model.inner_model.inner_model.model_sampling)
            return run(model, x, sigmas, extra_args or {}, callback=callback, disable=disable, **kwargs)
        sampler_function.__name__ = getattr(fn, "__name__", "pack_sampler")
        return comfy.samplers.KSAMPLER(sampler_function, extra_options=options)


EXTENSIONS = {"model": ModelOps, "clip": ClipOps, "vae": VaeOps, "models": ModelsOps, "info": InfoOps, "sampler": SamplerOps}
