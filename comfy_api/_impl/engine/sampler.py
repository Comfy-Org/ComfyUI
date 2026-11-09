"""Core engine extension: the ``eng.sampler`` rows ComfyUI implements itself."""


class EngineOps:
    def custom(self, fn, options=None, model_sampling=False):
        """A SAMPLER whose loop is ``fn(model, x, sigmas, extra_args, callback, disable, **options)``:
        ``model`` (the denoiser) and ``callback`` (previews, progress) are EngineFunctions it calls;
        a SAMPLER among ``options`` is one too; ``model_sampling=True`` adds the model's sampling
        facts as the ``model_sampling`` keyword."""
        import comfy.samplers
        from comfy_api._impl import _engine

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
