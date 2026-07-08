import torch
import comfy.model_management
import comfy.samplers
import comfy.utils
import numpy as np
import logging
import comfy.nested_tensor

def prepare_noise_inner(latent_image, generator, noise_inds=None):
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    return torch.cat(noises, axis=0)

def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)

    if latent_image.is_nested:
        tensors = latent_image.unbind()
        noises = []
        for t in tensors:
            noises.append(prepare_noise_inner(t, generator, noise_inds))
        noises = comfy.nested_tensor.NestedTensor(noises)
    else:
        noises = prepare_noise_inner(latent_image, generator, noise_inds)

    return noises

def fix_empty_latent_channels(model, latent_image, downscale_ratio_spacial=None, downscale_ratio_temporal=None):
    if latent_image.is_nested:
        return latent_image
    latent_format = model.get_model_object("latent_format") #Resize the empty latent image so it has the right number of channels
    is_empty = torch.count_nonzero(latent_image) == 0
    if is_empty:
        if latent_format.latent_channels != latent_image.shape[1]:
            latent_image = comfy.utils.repeat_to_batch_size(latent_image, latent_format.latent_channels, dim=1)
        if downscale_ratio_spacial is not None:
            if downscale_ratio_spacial != latent_format.spacial_downscale_ratio:
                ratio = downscale_ratio_spacial / latent_format.spacial_downscale_ratio
                latent_image = comfy.utils.common_upscale(latent_image, round(latent_image.shape[-1] * ratio), round(latent_image.shape[-2] * ratio), "nearest-exact", crop="disabled")

    if latent_format.latent_dimensions == 3 and latent_image.ndim == 4:
        latent_image = latent_image.unsqueeze(2)

    if is_empty and downscale_ratio_temporal is not None:
        if downscale_ratio_temporal != latent_format.temporal_downscale_ratio:
            ratio = downscale_ratio_temporal / latent_format.temporal_downscale_ratio
            new_t = max(1, round(latent_image.shape[2] * ratio))
            latent_image = comfy.utils.repeat_to_batch_size(latent_image, new_t, dim=2)

    return latent_image

def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    logging.warning("Warning: comfy.sample.prepare_sampling isn't used anymore and can be removed")
    return model, positive, negative, noise_mask, []

def cleanup_additional_models(models):
    logging.warning("Warning: comfy.sample.cleanup_additional_models isn't used anymore and can be removed")

def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return samples

def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    samples = comfy.samplers.sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return samples
