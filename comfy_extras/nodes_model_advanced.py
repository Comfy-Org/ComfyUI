import comfy.sd
import comfy.model_sampling
import comfy.latent_formats
import nodes
import torch
import node_helpers


class LCM(comfy.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input

class ModelSamplingDiscreteDistilled(comfy.model_sampling.ModelSamplingDiscrete):
    original_timesteps = 50

    def __init__(self, model_config=None, zsnr=None):
        super().__init__(model_config, zsnr=zsnr)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return (dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(((timestep.float().to(self.log_sigmas.device) - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)


class ModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"],),
                              "zsnr": ("BOOLEAN", {"default": False, "advanced": True}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch"

    def patch(self, model, sampling, zsnr):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == "lcm":
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled
        elif sampling == "x0":
            sampling_type = comfy.model_sampling.X0
        elif sampling == "img_to_img":
            sampling_type = comfy.model_sampling.IMG_TO_IMG
        elif sampling == "img_to_img_flow":
            sampling_type = comfy.model_sampling.IMG_TO_IMG_FLOW

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)

        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

class ModelSamplingStableCascade:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch/stable cascade"

    def patch(self, model, shift):
        m = model.clone()

        sampling_base = comfy.model_sampling.StableCascadeSampling
        sampling_type = comfy.model_sampling.EPS

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

class ModelSamplingSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch/stable diffusion"

    def patch(self, model, shift, multiplier=1000):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

class ModelSamplingAuraFlow(ModelSamplingSD3):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 1.73, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    FUNCTION = "patch_aura"
    CATEGORY = "model/patch"

    def patch_aura(self, model, shift):
        return self.patch(model, shift, multiplier=1.0)

class ModelSamplingFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01, "advanced": True}),
                              "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "advanced": True}),
                              "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch/flux"

    def patch(self, model, max_shift, base_shift, width, height):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )


class ModelSamplingContinuousEDM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"],),
                              "sigma_max": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1000.0, "step":0.001, "round": False, "advanced": True}),
                              "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1000.0, "step":0.001, "round": False, "advanced": True}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch"

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
        latent_format = None
        sigma_data = 1.0
        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "edm":
            sampling_type = comfy.model_sampling.EDM
            sigma_data = 0.5
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION
        elif sampling == "edm_playground_v2.5":
            sampling_type = comfy.model_sampling.EDM
            sigma_data = 0.5
            latent_format = comfy.latent_formats.SDXL_Playground_2_5()
        elif sampling == "cosmos_rflow":
            sampling_type = comfy.model_sampling.COSMOS_RFLOW
            sampling_base = comfy.model_sampling.ModelSamplingCosmosRFlow

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        if latent_format is not None:
            m.add_object_patch("latent_format", latent_format)
        return (m, )

class ModelSamplingContinuousV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["v_prediction"],),
                              "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step":0.001, "round": False, "advanced": True}),
                              "sigma_min": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1000.0, "step":0.001, "round": False, "advanced": True}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch"

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        sigma_data = 1.0
        if sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousV, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

class RescaleCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch"

    def patch(self, model, multiplier):
        def rescale_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

            #rescalecfg
            x_cfg = uncond + cond_scale * (cond - uncond)
            ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

        m = model.clone()
        m.set_model_sampler_cfg_function(rescale_cfg)
        return (m, )

class ModelNoiseScale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.01,
                                                       "tooltip": "Absolute training noise scale. For example HiDream-O1 base: 8.0, dev: 7.5."}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model/patch"

    def patch(self, model, noise_scale):
        m = model.clone()
        original = m.get_model_object("model_sampling")
        ms = type(original)(m.model.model_config)
        ms.set_parameters(shift=original.shift, multiplier=original.multiplier)
        ms.set_noise_scale(noise_scale)
        m.add_object_patch("model_sampling", ms)
        return (m, )


class ModelComputeDtype:
    SEARCH_ALIASES = ["model precision", "change dtype"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "dtype": (["default", "fp32", "fp16", "bf16"], {"advanced": True}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/debug"

    def patch(self, model, dtype):
        m = model.clone()
        m.set_model_compute_dtype(node_helpers.string_to_torch_dtype(dtype))
        return (m, )


NODE_CLASS_MAPPINGS = {
    "ModelSamplingDiscrete": ModelSamplingDiscrete,
    "ModelSamplingContinuousEDM": ModelSamplingContinuousEDM,
    "ModelSamplingContinuousV": ModelSamplingContinuousV,
    "ModelSamplingStableCascade": ModelSamplingStableCascade,
    "ModelSamplingSD3": ModelSamplingSD3,
    "ModelSamplingAuraFlow": ModelSamplingAuraFlow,
    "ModelSamplingFlux": ModelSamplingFlux,
    "ModelNoiseScale": ModelNoiseScale,
    "RescaleCFG": RescaleCFG,
    "ModelComputeDtype": ModelComputeDtype,
}
