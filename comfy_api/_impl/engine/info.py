"""Core engine extension: the ``info`` and ``weight_keys`` rows (``eng.model.info``, ``eng.vae.info``,
``eng.clip.info``...), read-only facts about engine objects."""
from comfy_api._impl import _runtime


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


class EngineOps:
    def info(self, obj):
        kind = _runtime.handle_kind(obj)
        fields = _INFO_FIELDS.get(kind)
        if fields is None:
            raise _runtime.CallNotSupported(f"info doesn't cover {kind}")
        return fields(obj)

    def weight_keys(self, obj):
        patcher = getattr(obj, "patcher", obj)
        return sorted(patcher.model_state_dict().keys())
