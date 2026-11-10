"""Core engine extension: the ``eng.model`` rows (a MODEL is ComfyUI's ModelPatcher)."""


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


class EngineOps:
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
        from comfy_engine._engine import hook_view

        return hook_view(model.model_options)

    def set_option(self, model, path, value):
        from comfy_engine._engine import hook_view

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
        from comfy_engine._engine import pack_function

        kinds = {name.lower(): value for name, value in vars(WrappersMP).items() if name.isupper()}
        if kind not in kinds.values():
            raise ValueError(f"wrapper kind must be one of {sorted(kinds.values())}, got {kind!r}")
        patched = model.clone()
        patched.add_wrapper_with_key(kind, None, pack_function(fn, engine_args=(0,)))
        return patched

    def set_unet_wrapper(self, model, fn):
        """A wrapper already on the model runs inside this one: ``apply_model`` goes through it."""
        from comfy_engine._engine import pack_function

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
