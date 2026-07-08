"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import comfy.ldm.hunyuan3dv2_1
import comfy.ldm.hunyuan3dv2_1.hunyuandit
import torch
import logging
import comfy.ldm.lightricks.av_model
import comfy.ldm.lightricks.symmetric_patchifier
import comfy.context_windows
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from comfy.ldm.cascade.stage_c import StageC
from comfy.ldm.cascade.stage_b import StageB
from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
import comfy.ldm.genmo.joint_model.asymm_models_joint
import comfy.ldm.aura.mmdit
import comfy.ldm.pixart.pixartms
import comfy.ldm.hydit.models
import comfy.ldm.audio.dit
import comfy.ldm.audio.embedders
import comfy.ldm.flux.model
import comfy.ldm.lens.model
import comfy.ldm.lightricks.model
import comfy.ldm.hunyuan_video.model
import comfy.ldm.cosmos.model
import comfy.ldm.cosmos.predict2
import comfy.ldm.lumina.model
import comfy.ldm.wan.model
import comfy.ldm.wan.model_animate
import comfy.ldm.wan.ar_model
import comfy.ldm.wan.model_wandancer
import comfy.ldm.hunyuan3d.model
import comfy.ldm.triposplat.model
import comfy.ldm.hidream.model
import comfy.ldm.chroma.model
import comfy.ldm.chroma_radiance.model
import comfy.ldm.pixeldit.model
import comfy.ldm.pixeldit.pid
import comfy.ldm.ace.model
import comfy.ldm.omnigen.omnigen2
import comfy.ldm.boogu.model
import comfy.ldm.qwen_image.model
import comfy.ldm.ideogram4.model
import comfy.ldm.krea2.model
import comfy.ldm.kandinsky5.model
import comfy.ldm.anima.model
import comfy.ldm.ace.ace_step15
import comfy.ldm.cogvideo.model
import comfy.ldm.rt_detr.rtdetr_v4
import comfy.ldm.ernie.model
import comfy.ldm.sam3.detector
import comfy.ldm.hidream_o1.model
from comfy.ldm.hidream_o1.conditioning import build_extra_conds
import comfy.ldm.depth_anything_3.model

import comfy.model_management
import comfy.patcher_extension
import comfy.conds
import comfy.ops
from enum import Enum
from . import utils
import comfy.latent_formats
import comfy.model_sampling
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8
    IMG_TO_IMG = 9
    FLOW_COSMOS = 10
    IMG_TO_IMG_FLOW = 11
    V_PREDICTION_DDPM = 12


def model_sampling(model_config, model_type):
    s = comfy.model_sampling.ModelSamplingDiscrete

    if model_type == ModelType.EPS:
        c = comfy.model_sampling.EPS
    elif model_type == ModelType.V_PREDICTION:
        c = comfy.model_sampling.V_PREDICTION
    elif model_type == ModelType.V_PREDICTION_EDM:
        c = comfy.model_sampling.V_PREDICTION
        s = comfy.model_sampling.ModelSamplingContinuousEDM
    elif model_type == ModelType.FLOW:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingDiscreteFlow
    elif model_type == ModelType.STABLE_CASCADE:
        c = comfy.model_sampling.EPS
        s = comfy.model_sampling.StableCascadeSampling
    elif model_type == ModelType.EDM:
        c = comfy.model_sampling.EDM
        s = comfy.model_sampling.ModelSamplingContinuousEDM
    elif model_type == ModelType.V_PREDICTION_CONTINUOUS:
        c = comfy.model_sampling.V_PREDICTION
        s = comfy.model_sampling.ModelSamplingContinuousV
    elif model_type == ModelType.FLUX:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingFlux
    elif model_type == ModelType.IMG_TO_IMG:
        c = comfy.model_sampling.IMG_TO_IMG
    elif model_type == ModelType.FLOW_COSMOS:
        c = comfy.model_sampling.COSMOS_RFLOW
        s = comfy.model_sampling.ModelSamplingCosmosRFlow
    elif model_type == ModelType.IMG_TO_IMG_FLOW:
        c = comfy.model_sampling.IMG_TO_IMG_FLOW
    elif model_type == ModelType.V_PREDICTION_DDPM:
        c = comfy.model_sampling.V_PREDICTION_DDPM

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


def convert_tensor(extra, dtype, device):
    if hasattr(extra, "dtype"):
        if extra.dtype != torch.int and extra.dtype != torch.long:
            extra = comfy.model_management.cast_to_device(extra, device, dtype)
        else:
            extra = comfy.model_management.cast_to_device(extra, device, None)
    return extra


class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher: 'ModelPatcher' = None

        if not unet_config.get("disable_unet_model_creation", False):
            if model_config.custom_operations is None:
                fp8 = model_config.optimizations.get("fp8", False)
                operations = comfy.ops.pick_operations(unet_config.get("dtype", None), self.manual_cast_dtype, fp8_optimizations=fp8, model_config=model_config)
            else:
                operations = model_config.custom_operations
            self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
            self.diffusion_model.eval()
            if comfy.model_management.force_channels_last():
                self.diffusion_model.to(memory_format=torch.channels_last)
                logging.debug("using channels last mode for diffusion model")
            logging.info("model weight dtype {}, manual cast: {}".format(self.get_dtype(), self.manual_cast_dtype))
            comfy.model_management.archive_model_dtypes(self.diffusion_model)

        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor
        self.memory_usage_factor_conds = ()
        self.memory_usage_shape_process = {}

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._apply_model,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        context = c_crossattn
        dtype = self.get_dtype_inference()

        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]

            if hasattr(extra, "dtype"):
                extra = convert_tensor(extra, dtype, device)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(convert_tensor(ext, dtype, device))
                extra = ex
            extra_conds[o] = extra

        t = self.process_timestep(t, x=x, **extra_conds)
        if "latent_shapes" in extra_conds:
            xc = utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))

        transformer_options = transformer_options.copy()
        transformer_options["prefetch_dynamic_vbars"] = (
            self.current_patcher is not None and self.current_patcher.is_dynamic()
        )

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
        if len(model_output) > 1 and not torch.is_tensor(model_output):
            model_output, _ = utils.pack_latents(model_output)

        return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

    def process_timestep(self, timestep, **kwargs):
        return timestep

    def get_dtype(self):
        return self.diffusion_model.dtype

    def get_dtype_inference(self):
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        return dtype

    def encode_adm(self, **kwargs):
        return None

    def concat_cond(self, **kwargs):
        if len(self.concat_keys) > 0:
            cond_concat = []
            denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
            concat_latent_image = kwargs.get("concat_latent_image", None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get("latent_image", None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)

            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = utils.common_upscale(concat_latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")
                if noise.ndim == 5:
                    if concat_latent_image.shape[-3] < noise.shape[-3]:
                        concat_latent_image = torch.nn.functional.pad(concat_latent_image, (0, 0, 0, 0, 0, noise.shape[-3] - concat_latent_image.shape[-3]), "constant", 0)
                    else:
                        concat_latent_image = concat_latent_image[:, :, :noise.shape[-3]]

            concat_latent_image = utils.resize_to_batch_size(concat_latent_image, noise.shape[0])

            if denoise_mask is not None:
                if len(denoise_mask.shape) == len(noise.shape):
                    denoise_mask = denoise_mask[:, :1]

                num_dim = noise.ndim - 2
                denoise_mask = denoise_mask.reshape((-1, 1) + tuple(denoise_mask.shape[-num_dim:]))
                if denoise_mask.shape[-2:] != noise.shape[-2:]:
                    denoise_mask = utils.common_upscale(denoise_mask, noise.shape[-1], noise.shape[-2], "bilinear", "center")
                denoise_mask = utils.resize_to_batch_size(denoise_mask.round(), noise.shape[0])

            for ck in self.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask.to(device))
                    elif ck == "masked_image":
                        cond_concat.append(concat_latent_image.to(device))  # NOTE: the latent_image should be masked by the mask in pixel space
                    elif ck == "mask_inverted":
                        cond_concat.append(1.0 - denoise_mask.to(device))
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(self.blank_inpaint_image_like(noise))
                    elif ck == "mask_inverted":
                        cond_concat.append(torch.zeros_like(noise)[:, :1])
                if ck == "concat_image":
                    if concat_latent_image is not None:
                        cond_concat.append(concat_latent_image.to(device))
                    else:
                        cond_concat.append(torch.zeros_like(noise))
            data = torch.cat(cond_concat, dim=1)
            return data
        return None

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        """Override in subclasses to handle model-specific cond slicing for context windows.
        Return a sliced cond object, or None to fall through to default handling.
        Use comfy.context_windows.slice_cond() for common cases."""
        return None

    def extra_conds(self, **kwargs):
        out = {}
        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(concat_cond)

        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out['crossattn_controlnet'] = comfy.conds.CONDCrossAttn(cross_attn_cnet)

        c_concat = kwargs.get("noise_concat", None)
        if c_concat is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(c_concat)

        return out

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, unet_state_dict, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        extra_sds = []
        if clip_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_state_dict_for_saving(clip_state_dict))
        if vae_state_dict is not None:
            extra_sds.append(self.model_config.process_vae_state_dict_for_saving(vae_state_dict))
        if clip_vision_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_vision_state_dict_for_saving(clip_vision_state_dict))
        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(unet_state_dict)
        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        for sd in extra_sds:
            unet_state_dict.update(sd)

        return unet_state_dict

    def set_inpaint(self):
        self.concat_keys = ("mask", "masked_image")
        def blank_inpaint_image_like(latent_image):
            blank_image = torch.ones_like(latent_image)
            # these are the values for "zero" in pixel space translated to latent space
            blank_image[:,0] *= 0.8223
            blank_image[:,1] *= -0.6876
            blank_image[:,2] *= 0.6364
            blank_image[:,3] *= 0.1380
            return blank_image
        self.blank_inpaint_image_like = blank_inpaint_image_like

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return self.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)), noise, latent_image)

    def memory_required(self, input_shape, cond_shapes={}):
        input_shapes = [input_shape]
        for c in self.memory_usage_factor_conds:
            shape = cond_shapes.get(c, None)
            if shape is not None:
                if c in self.memory_usage_shape_process:
                    out = []
                    for s in shape:
                        out.append(self.memory_usage_shape_process[c](s))
                    shape = out

                if len(shape) > 0:
                    input_shapes += shape

        if comfy.model_management.xformers_enabled() or comfy.model_management.pytorch_attention_flash_attention():
            dtype = self.get_dtype_inference()
            #TODO: this needs to be tweaked
            area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))
            return (area * comfy.model_management.dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (1024 * 1024)
        else:
            #TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
            area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))
            return (area * 0.15 * self.memory_usage_factor) * (1024 * 1024)

    def extra_conds_shapes(self, **kwargs):
        return {}


def unclip_adm(unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0, seed=None):
    adm_inputs = []
    weights = []
    noise_aug = []
    for unclip_cond in unclip_conditioning:
        for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
            weight = unclip_cond["strength"]
            noise_augment = unclip_cond["noise_augmentation"]
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), noise_level=torch.tensor([noise_level], device=device), seed=seed)
            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            weights.append(weight)
            noise_aug.append(noise_augment)
            adm_inputs.append(adm_out)

    if len(noise_aug) > 1:
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
        adm_out = torch.cat((c_adm, noise_level_emb), 1)

    return adm_out

class SD21UNCLIP(BaseModel):
    def __init__(self, model_config, noise_aug_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

    def encode_adm(self, **kwargs):
        unclip_conditioning = kwargs.get("unclip_conditioning", None)
        device = kwargs["device"]
        if unclip_conditioning is None:
            return torch.zeros((1, self.adm_channels), device=device)
        else:
            return unclip_adm(unclip_conditioning, device, self.noise_augmentor, kwargs.get("unclip_noise_augment_merge", 0.05), kwargs.get("seed", 0) - 10)

def sdxl_pooled(args, noise_augmentor):
    if "unclip_conditioning" in args:
        return unclip_adm(args.get("unclip_conditioning", None), args["device"], noise_augmentor, seed=args.get("seed", 0) - 10)[:,:1280]
    else:
        return args["pooled_output"]

class SDXLRefiner(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)

class SDXL(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SVD_img2vid(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)

    def encode_adm(self, **kwargs):
        fps_id = kwargs.get("fps", 6) - 1
        motion_bucket_id = kwargs.get("motion_bucket_id", 127)
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.Tensor([fps_id])))
        out.append(self.embedder(torch.Tensor([motion_bucket_id])))
        out.append(self.embedder(torch.Tensor([augmentation])))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat

    def extra_conds(self, **kwargs):
        out = {}
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = utils.common_upscale(latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        latent_image = utils.resize_to_batch_size(latent_image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)

        if "time_conditioning" in kwargs:
            out["time_context"] = comfy.conds.CONDCrossAttn(kwargs["time_conditioning"])

        out['num_video_frames'] = comfy.conds.CONDConstant(noise.shape[0])
        return out

class SV3D_u(SVD_img2vid):
    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat

class SV3D_p(SVD_img2vid):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder_512 = Timestep(512)

    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)
        elevation = kwargs.get("elevation", 0) #elevation and azimuth are in degrees here
        azimuth = kwargs.get("azimuth", 0)
        noise = kwargs.get("noise", None)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))
        out.append(self.embedder_512(torch.deg2rad(torch.fmod(torch.flatten(90 - torch.Tensor([elevation])), 360.0))))
        out.append(self.embedder_512(torch.deg2rad(torch.fmod(torch.flatten(torch.Tensor([azimuth])), 360.0))))

        out = list(map(lambda a: utils.resize_to_batch_size(a, noise.shape[0]), out))
        return torch.cat(out, dim=1)


class Stable_Zero123(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, cc_projection_weight=None, cc_projection_bias=None):
        super().__init__(model_config, model_type, device=device)
        self.cc_projection = comfy.ops.manual_cast.Linear(cc_projection_weight.shape[1], cc_projection_weight.shape[0], dtype=self.get_dtype(), device=device)
        self.cc_projection.weight = torch.nn.Parameter(cc_projection_weight.clone())
        self.cc_projection.bias = torch.nn.Parameter(cc_projection_bias.clone())

    def extra_conds(self, **kwargs):
        out = {}

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = utils.common_upscale(latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        latent_image = utils.resize_to_batch_size(latent_image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if cross_attn.shape[-1] != 768:
                cross_attn = self.cc_projection(cross_attn)
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out

class SD_X4Upscaler(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = ImageConcatWithNoiseAugmentation(noise_schedule_config={"linear_start": 0.0001, "linear_end": 0.02}, max_noise_level=350)

    def extra_conds(self, **kwargs):
        out = {}

        image = kwargs.get("concat_image", None)
        noise = kwargs.get("noise", None)
        noise_augment = kwargs.get("noise_augmentation", 0.0)
        device = kwargs["device"]
        seed = kwargs["seed"] - 10

        noise_level = round((self.noise_augmentor.max_noise_level) * noise_augment)

        if image is None:
            image = torch.zeros_like(noise)[:,:3]

        if image.shape[1:] != noise.shape[1:]:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        noise_level = torch.tensor([noise_level], device=device)
        if noise_augment > 0:
            image, noise_level = self.noise_augmentor(image.to(device), noise_level=noise_level, seed=seed)

        image = utils.resize_to_batch_size(image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(image)
        out['y'] = comfy.conds.CONDRegular(noise_level)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out

class IP2P:
    def concat_cond(self, **kwargs):
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)
        else:
            image = image.to(device=device)

        if image.shape[1:] != noise.shape[1:]:
            image = utils.common_upscale(image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = utils.resize_to_batch_size(image, noise.shape[0])
        return self.process_ip2p_image_in(image)


class SD15_instructpix2pix(IP2P, BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.process_ip2p_image_in = lambda image: image


class SDXL_instructpix2pix(IP2P, SDXL):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        if model_type == ModelType.V_PREDICTION_EDM:
            self.process_ip2p_image_in = lambda image: comfy.latent_formats.SDXL().process_in(image) #cosxl ip2p
        else:
            self.process_ip2p_image_in = lambda image: image #diffusers ip2p

class Lotus(BaseModel):
    def extra_conds(self, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        device = kwargs["device"]
        task_emb = torch.tensor([1, 0]).float().to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)]).unsqueeze(0)
        out['y'] = comfy.conds.CONDRegular(task_emb)
        return out

    def __init__(self, model_config, model_type=ModelType.IMG_TO_IMG, device=None):
        super().__init__(model_config, model_type, device=device)

class StableCascade_C(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageC)

    def extra_conds(self, **kwargs):
        out = {}
        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out['clip_text_pooled'] = comfy.conds.CONDRegular(clip_text_pooled)

        if "unclip_conditioning" in kwargs:
            embeds = []
            for unclip_cond in kwargs["unclip_conditioning"]:
                weight = unclip_cond["strength"]
                embeds.append(unclip_cond["clip_vision_output"].image_embeds.unsqueeze(0) * weight)
            clip_img = torch.cat(embeds, dim=1)
        else:
            clip_img = torch.zeros((1, 1, 768))
        out["clip_img"] = comfy.conds.CONDRegular(clip_img)
        out["sca"] = comfy.conds.CONDRegular(torch.zeros((1,)))
        out["crp"] = comfy.conds.CONDRegular(torch.zeros((1,)))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['clip_text'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out


class StableCascade_B(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageB)

    def extra_conds(self, **kwargs):
        out = {}
        noise = kwargs.get("noise", None)

        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out['clip'] = comfy.conds.CONDRegular(clip_text_pooled)

        #size of prior doesn't really matter if zeros because it gets resized but I still want it to get batched
        prior = kwargs.get("stable_cascade_prior", torch.zeros((1, 16, (noise.shape[2] * 4) // 42, (noise.shape[3] * 4) // 42), dtype=noise.dtype, layout=noise.layout, device=noise.device))

        out["effnet"] = comfy.conds.CONDRegular(prior.to(device=noise.device))
        out["sca"] = comfy.conds.CONDRegular(torch.zeros((1,)))
        return out


class SD3(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=OpenAISignatureMMDITWrapper)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out


class AuraFlow(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.aura.mmdit.MMDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out


class StableAudio1(BaseModel):
    def __init__(self, model_config, seconds_start_embedder_weights, seconds_total_embedder_weights, model_type=ModelType.V_PREDICTION_CONTINUOUS, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.audio.dit.AudioDiffusionTransformer)
        self.seconds_start_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_total_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_start_embedder.load_state_dict(seconds_start_embedder_weights)
        self.seconds_total_embedder.load_state_dict(seconds_total_embedder_weights)

    def extra_conds(self, **kwargs):
        out = {}

        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        seconds_start = kwargs.get("seconds_start", 0)
        seconds_total = kwargs.get("seconds_total", int(noise.shape[-1] / 21.53))

        seconds_start_embed = self.seconds_start_embedder([seconds_start])[0].to(device)
        seconds_total_embed = self.seconds_total_embedder([seconds_total])[0].to(device)

        global_embed = torch.cat([seconds_start_embed, seconds_total_embed], dim=-1).reshape((1, -1))
        out['global_embed'] = comfy.conds.CONDRegular(global_embed)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            cross_attn = torch.cat([cross_attn.to(device), seconds_start_embed.repeat((cross_attn.shape[0], 1, 1)), seconds_total_embed.repeat((cross_attn.shape[0], 1, 1))], dim=1)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

    def state_dict_for_saving(self, unet_state_dict, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        sd = super().state_dict_for_saving(unet_state_dict, clip_state_dict=clip_state_dict, vae_state_dict=vae_state_dict, clip_vision_state_dict=clip_vision_state_dict)
        d = {"conditioner.conditioners.seconds_start.": self.seconds_start_embedder.state_dict(), "conditioner.conditioners.seconds_total.": self.seconds_total_embedder.state_dict()}
        for k in d:
            s = d[k]
            for l in s:
                sd["{}{}".format(k, l)] = s[l]
        return sd

class StableAudio3(BaseModel):
    def __init__(self, model_config, seconds_total_embedder_weights, padding_embedding=None, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.audio.dit.AudioDiffusionTransformer)
        self.seconds_total_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=384, fourier_features_type=model_config.unet_config["timestep_features_type"])
        self.seconds_total_embedder.load_state_dict(seconds_total_embedder_weights)
        if padding_embedding is not None:
            self.padding_embedding = torch.nn.Parameter(padding_embedding, requires_grad=False)
        else:
            self.padding_embedding = None

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        image = kwargs.get("concat_latent_image", None)

        if image is None:
            shape_image = list(noise.shape)
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            image = self.process_latent_in(image)
            # TODO: scale if not match
            image = utils.resize_to_batch_size(image, noise.shape[0])

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :1]
        else:
            if mask.shape[1] != 1:
                mask = torch.mean(mask, dim=1, keepdim=True)
            mask = 1.0 - mask
            # TODO: scale if not match
            mask = utils.resize_to_batch_size(mask, noise.shape[0])

        return torch.cat((mask, image), dim=1)

    def extra_conds(self, **kwargs):
        out = {}

        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out['local_add_cond'] = comfy.conds.CONDNoiseShape(concat_cond)

        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        seconds_total = kwargs.get("seconds_total", int(noise.shape[-1] / 10.7666))
        seconds_total_embed = self.seconds_total_embedder([seconds_total])[0].to(device)

        global_embed = seconds_total_embed.reshape((1, -1))
        out['global_embed'] = comfy.conds.CONDRegular(global_embed)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            cross_attn = cross_attn.to(device)
            if self.padding_embedding is not None:
                pe = self.padding_embedding.to(device=device, dtype=cross_attn.dtype)
                max_text_tokens = self.model_config.unet_config.get("max_text_tokens", 256)
                n_text = cross_attn.shape[1]
                if n_text < max_text_tokens:
                    pad = pe.view(1, 1, -1).expand(cross_attn.shape[0], max_text_tokens - n_text, -1)
                    cross_attn = torch.cat([cross_attn, pad], dim=1)
            cross_attn = torch.cat([cross_attn, seconds_total_embed.repeat((cross_attn.shape[0], 1, 1))], dim=1)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        return out

    def state_dict_for_saving(self, unet_state_dict, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        sd = super().state_dict_for_saving(unet_state_dict, clip_state_dict=clip_state_dict, vae_state_dict=vae_state_dict, clip_vision_state_dict=clip_vision_state_dict)

        d = {"conditioner.conditioners.seconds_total.": self.seconds_total_embedder.state_dict()}

        for k in d:
            s = d[k]
            for l in s:
                sd["{}{}".format(k, l)] = s[l]

        if self.padding_embedding is not None:
            sd["conditioner.conditioners.prompt.padding_embedding"] = self.padding_embedding.data
        return sd


class HunyuanDiT(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hydit.models.HunYuanDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['text_embedding_mask'] = comfy.conds.CONDRegular(attention_mask)

        conditioning_mt5xl = kwargs.get("conditioning_mt5xl", None)
        if conditioning_mt5xl is not None:
            out['encoder_hidden_states_t5'] = comfy.conds.CONDRegular(conditioning_mt5xl)

        attention_mask_mt5xl = kwargs.get("attention_mask_mt5xl", None)
        if attention_mask_mt5xl is not None:
            out['text_embedding_mask_t5'] = comfy.conds.CONDRegular(attention_mask_mt5xl)

        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out['image_meta_size'] = comfy.conds.CONDRegular(torch.FloatTensor([[height, width, target_height, target_width, 0, 0]]))
        return out

class PixArt(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.pixart.pixartms.PixArtMS)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        width = kwargs.get("width", None)
        height = kwargs.get("height", None)
        if width is not None and height is not None:
            out["c_size"] = comfy.conds.CONDRegular(torch.FloatTensor([[height, width]]))
            out["c_ar"] = comfy.conds.CONDRegular(torch.FloatTensor([[kwargs.get("aspect_ratio", height/width)]]))

        return out

class Flux(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None, unet_model=comfy.ldm.flux.model.Flux):
        super().__init__(model_config, model_type, device=device, unet_model=unet_model)
        self.memory_usage_factor_conds = ("ref_latents",)

    def concat_cond(self, **kwargs):
        try:
            #Handle Flux control loras dynamically changing the img_in weight.
            num_channels = self.diffusion_model.img_in.weight.shape[1] // (self.diffusion_model.patch_size * self.diffusion_model.patch_size)
        except:
            #Some cases like tensorrt might not have the weights accessible
            num_channels = self.model_config.unet_config["in_channels"]

        out_channels = self.model_config.unet_config["out_channels"]

        if num_channels <= out_channels:
            return None

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = utils.resize_to_batch_size(image, noise.shape[0])
        image = self.process_latent_in(image)
        if num_channels <= out_channels * 2:
            return image

        #inpaint model
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.ones_like(noise)[:, :1]

        mask = torch.mean(mask, dim=1, keepdim=True)
        mask = utils.common_upscale(mask.to(device), noise.shape[-1] * 8, noise.shape[-2] * 8, "bilinear", "center")
        mask = mask.view(mask.shape[0], mask.shape[2] // 8, 8, mask.shape[3] // 8, 8).permute(0, 2, 4, 1, 3).reshape(mask.shape[0], -1, mask.shape[2] // 8, mask.shape[3] // 8)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return torch.cat((image, mask), dim=1)

    def encode_adm(self, **kwargs):
        return kwargs.get("pooled_output", None)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        # upscale the attention mask, since now we
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            shape = kwargs["noise"].shape
            mask_ref_size = kwargs.get("attention_mask_img_shape", None)
            if mask_ref_size is not None:
                # the model will pad to the patch size, and then divide
                # essentially dividing and rounding up
                (h_tok, w_tok) = (math.ceil(shape[2] / self.diffusion_model.patch_size), math.ceil(shape[3] / self.diffusion_model.patch_size))
                attention_mask = utils.upscale_dit_mask(attention_mask, mask_ref_size, (h_tok, w_tok))
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()[2:]), ref_latents))])
        return out

class LongCatImage(Flux):
    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        rope_opts = transformer_options.get("rope_options", {})
        rope_opts = dict(rope_opts)
        pe_len = float(c_crossattn.shape[1]) if c_crossattn is not None else 512.0
        rope_opts.setdefault("shift_t", 1.0)
        rope_opts.setdefault("shift_y", pe_len)
        rope_opts.setdefault("shift_x", pe_len)
        transformer_options["rope_options"] = rope_opts
        return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out.pop('guidance', None)
        return out

class Flux2(Flux):
    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            target_text_len = 512
            if cross_attn.shape[1] < target_text_len:
                cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, target_text_len - cross_attn.shape[1], 0))
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out


class Lens(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(
            model_config, model_type, device=device,
            unet_model=comfy.ldm.lens.model.LensTransformer2DModel,
        )

    def encode_adm(self, **kwargs):
        return None  # Lens has no pooled/ADM conditioning.

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        return out

class GenmoMochi(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.genmo.joint_model.asymm_models_joint.AsymmDiTJoint)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
            out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class LTXV(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.lightricks.model.LTXVModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        out['frame_rate'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", 25))

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)

        keyframe_idxs = kwargs.get("keyframe_idxs", None)
        if keyframe_idxs is not None:
            out['keyframe_idxs'] = comfy.conds.CONDRegular(keyframe_idxs)

        guide_attention_entries = kwargs.get("guide_attention_entries", None)
        if guide_attention_entries is not None:
            out['guide_attention_entries'] = comfy.conds.CONDConstant(guide_attention_entries)

        return out

    def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
        if denoise_mask is None:
            return timestep
        return self.diffusion_model.patchifier.patchify(((denoise_mask) * timestep.view([timestep.shape[0]] + [1] * (denoise_mask.ndim - 1)))[:, :1])[0]

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image

class LTXAV(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.lightricks.av_model.LTXAVModel) #TODO

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        device = kwargs["device"]

        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if hasattr(self.diffusion_model, "preprocess_text_embeds"):
                cross_attn = self.diffusion_model.preprocess_text_embeds(cross_attn.to(device=device, dtype=self.get_dtype_inference()), unprocessed=kwargs.get("unprocessed_ltxav_embeds", False))
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        out['frame_rate'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", 25))

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))

        audio_denoise_mask = None
        if denoise_mask is not None and "latent_shapes" in kwargs:
            denoise_mask = utils.unpack_latents(denoise_mask, kwargs["latent_shapes"])
            if len(denoise_mask) > 1:
                audio_denoise_mask = denoise_mask[1]
            denoise_mask = denoise_mask[0]

        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)

        if audio_denoise_mask is not None:
            out["audio_denoise_mask"] = comfy.conds.CONDRegular(audio_denoise_mask)

        keyframe_idxs = kwargs.get("keyframe_idxs", None)
        if keyframe_idxs is not None:
            out['keyframe_idxs'] = comfy.conds.CONDRegular(keyframe_idxs)

        latent_shapes = kwargs.get("latent_shapes", None)
        if latent_shapes is not None:
            out['latent_shapes'] = comfy.conds.CONDConstant(latent_shapes)

        guide_attention_entries = kwargs.get("guide_attention_entries", None)
        if guide_attention_entries is not None:
            out['guide_attention_entries'] = comfy.conds.CONDConstant(guide_attention_entries)

        ref_audio = kwargs.get("ref_audio", None)
        if ref_audio is not None:
            out['ref_audio'] = comfy.conds.CONDConstant(ref_audio)

        return out

    def process_timestep(self, timestep, x, denoise_mask=None, audio_denoise_mask=None, **kwargs):
        v_timestep = timestep
        a_timestep = timestep

        if denoise_mask is not None:
            v_timestep = self.diffusion_model.patchifier.patchify(((denoise_mask) * timestep.view([timestep.shape[0]] + [1] * (denoise_mask.ndim - 1)))[:, :1])[0]
        if audio_denoise_mask is not None:
            a_timestep = self.diffusion_model.a_patchifier.patchify(((audio_denoise_mask) * timestep.view([timestep.shape[0]] + [1] * (audio_denoise_mask.ndim - 1)))[:, :1, :, :1])[0]

        return v_timestep, a_timestep

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image

    def map_context_window_to_modalities(self, primary_indices, latent_shapes, dim):
        result = [primary_indices]
        if len(latent_shapes) < 2:
            return result

        video_total = latent_shapes[0][dim]

        for i in range(1, len(latent_shapes)):
            mod_total = latent_shapes[i][dim]
            # Map each primary index to its proportional range of modality indices and
            # concatenate in order. Preserves wrapped/strided geometry so the modality
            # attends to the same temporal regions as the primary window.
            mod_indices = []
            seen = set()
            for v_idx in primary_indices:
                a_start = min(int(round(v_idx * mod_total / video_total)), mod_total - 1)
                a_end = min(int(round((v_idx + 1) * mod_total / video_total)), mod_total)
                if a_end <= a_start:
                    a_end = a_start + 1
                for a in range(a_start, a_end):
                    if a not in seen:
                        seen.add(a)
                        mod_indices.append(a)
            result.append(mod_indices)

        return result

    @staticmethod
    def _get_guide_entries(conds):
        for cond_list in conds:
            if cond_list is None:
                continue
            for cond_dict in cond_list:
                model_conds = cond_dict.get('model_conds', {})
                entries = model_conds.get('guide_attention_entries')
                if entries is not None and hasattr(entries, 'cond') and entries.cond:
                    return entries.cond
        return None

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        # Audio denoise mask — slice using audio modality window
        if cond_key == "audio_denoise_mask" and hasattr(window, 'modality_windows') and window.modality_windows:
            audio_window = window.modality_windows.get(1)
            if audio_window is not None and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                sliced = audio_window.get_tensor(cond_value.cond, device, dim=2)
                return cond_value._copy_with(sliced)

        # Video denoise mask — split into video + guide portions, slice each
        if cond_key == "denoise_mask" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
            cond_tensor = cond_value.cond
            guide_count = cond_tensor.size(window.dim) - x_in.size(window.dim)
            if guide_count > 0:
                T_video = x_in.size(window.dim)
                video_mask = cond_tensor.narrow(window.dim, 0, T_video)
                guide_mask = cond_tensor.narrow(window.dim, T_video, guide_count)
                sliced_video = window.get_tensor(video_mask, device, retain_index_list=retain_index_list)
                suffix_indices = window.guide_frames_indices
                if suffix_indices:
                    idx = tuple([slice(None)] * window.dim + [suffix_indices])
                    sliced_guide = guide_mask[idx].to(device)
                    return cond_value._copy_with(torch.cat([sliced_video, sliced_guide], dim=window.dim))
                else:
                    return cond_value._copy_with(sliced_video)

        # Keyframe indices — regenerate pixel coords for window, select guide positions
        if cond_key == "keyframe_idxs":
            kf_local_pos = window.guide_kf_local_positions
            if not kf_local_pos:
                return cond_value._copy_with(cond_value.cond[:, :, :0, :])  # empty
            H, W = x_in.shape[3], x_in.shape[4]
            window_len = len(window.index_list)
            # account for causal_window_fix anchor in coord space size
            anchor_idx = getattr(window, 'causal_anchor_index', None)
            if anchor_idx is not None and anchor_idx >= 0:
                window_len += 1
            patchifier = self.diffusion_model.patchifier
            latent_coords = patchifier.get_latent_coords(window_len, H, W, 1, cond_value.cond.device)
            scale_factors = self.diffusion_model.vae_scale_factors
            pixel_coords = comfy.ldm.lightricks.symmetric_patchifier.latent_to_pixel_coords(
                latent_coords,
                scale_factors,
                causal_fix=self.diffusion_model.causal_temporal_positioning)
            tokens = []
            for pos in kf_local_pos:
                tokens.extend(range(pos * H * W, (pos + 1) * H * W))
            pixel_coords = pixel_coords[:, :, tokens, :]

            # Adjust spatial end positions for dilated (downscaled) guides.
            # Each guide entry may have a different downscale factor; expand the
            # per-entry factor to cover all tokens belonging to that entry.
            downscale_factors = window.guide_downscale_factors
            overlap_info = window.guide_overlap_info
            if downscale_factors:
                per_token_factor = []
                for (entry_idx, overlap_count), dsf in zip(overlap_info, downscale_factors):
                    per_token_factor.extend([dsf] * (overlap_count * H * W))
                factor_tensor = torch.tensor(per_token_factor, device=pixel_coords.device, dtype=pixel_coords.dtype)
                spatial_end_offset = (factor_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - 1) * torch.tensor(
                    scale_factors[1:], device=pixel_coords.device, dtype=pixel_coords.dtype,
                ).view(1, -1, 1, 1)
                pixel_coords[:, 1:, :, 1:] += spatial_end_offset

            B = cond_value.cond.shape[0]
            if B > 1:
                pixel_coords = pixel_coords.expand(B, -1, -1, -1)
            return cond_value._copy_with(pixel_coords)

        # Guide attention entries — adjust per-guide counts based on window overlap
        if cond_key == "guide_attention_entries":
            overlap_info = window.guide_overlap_info
            H, W = x_in.shape[3], x_in.shape[4]
            new_entries = []
            for entry_idx, overlap_count in overlap_info:
                e = cond_value.cond[entry_idx]
                new_entries.append({**e,
                    "pre_filter_count": overlap_count * H * W,
                    "latent_shape": [overlap_count, H, W]})
            return cond_value._copy_with(new_entries)

        return None

class HunyuanVideo(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan_video.model.HunyuanVideo)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 6.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        guiding_frame_index = kwargs.get("guiding_frame_index", None)
        if guiding_frame_index is not None:
            out['guiding_frame_index'] = comfy.conds.CONDRegular(torch.FloatTensor([guiding_frame_index]))

        ref_latent = kwargs.get("ref_latent", None)
        if ref_latent is not None:
            out['ref_latent'] = comfy.conds.CONDRegular(self.process_latent_in(ref_latent))

        return out

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return latent_image

class HunyuanVideoI2V(HunyuanVideo):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)
        self.concat_keys = ("concat_image", "mask_inverted")

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return super().scale_latent_inpaint(latent_image=latent_image, **kwargs)

class HunyuanVideoSkyreelsI2V(HunyuanVideo):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)
        self.concat_keys = ("concat_image",)

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return super().scale_latent_inpaint(latent_image=latent_image, **kwargs)

class CosmosVideo(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EDM, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.cosmos.model.GeneralDIT)
        self.image_to_video = image_to_video
        if self.image_to_video:
            self.concat_keys = ("mask_inverted",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        out['fps'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", None))
        return out

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1))
        sigma_noise_augmentation = 0 #TODO
        if sigma_noise_augmentation != 0:
            latent_image = latent_image + noise
        latent_image = self.model_sampling.calculate_input(torch.tensor([sigma_noise_augmentation], device=latent_image.device, dtype=latent_image.dtype), latent_image)
        return latent_image * ((sigma ** 2 + self.model_sampling.sigma_data ** 2) ** 0.5)

class CosmosPredict2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW_COSMOS, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.cosmos.predict2.MiniTrainDIT)
        self.image_to_video = image_to_video
        if self.image_to_video:
            self.concat_keys = ("mask_inverted",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)

        out['fps'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", None))
        return out

    def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
        if denoise_mask is None:
            return timestep
        if denoise_mask.ndim <= 4:
            return timestep
        condition_video_mask_B_1_T_1_1 = denoise_mask.mean(dim=[1, 3, 4], keepdim=True)
        c_noise_B_1_T_1_1 = 0.0 * (1.0 - condition_video_mask_B_1_T_1_1) + timestep.reshape(timestep.shape[0], 1, 1, 1, 1) * condition_video_mask_B_1_T_1_1
        out = c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4])
        return out

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1))
        sigma_noise_augmentation = 0 #TODO
        if sigma_noise_augmentation != 0:
            latent_image = latent_image + noise
        latent_image = self.model_sampling.calculate_input(torch.tensor([sigma_noise_augmentation], device=latent_image.device, dtype=latent_image.dtype), latent_image)
        sigma = (sigma / (sigma + 1))
        return latent_image / (1.0 - sigma)

class Anima(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.anima.model.Anima)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        t5xxl_ids = kwargs.get("t5xxl_ids", None)
        t5xxl_weights = kwargs.get("t5xxl_weights", None)
        device = kwargs["device"]
        if cross_attn is not None:
            if t5xxl_ids is not None:
                if t5xxl_weights is not None:
                    t5xxl_weights = t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)
                t5xxl_ids = t5xxl_ids.unsqueeze(0)

                if torch.is_inference_mode_enabled():  # if not we are training
                    cross_attn = self.diffusion_model.preprocess_text_embeds(cross_attn.to(device=device, dtype=self.get_dtype_inference()), t5xxl_ids.to(device=device), t5xxl_weights=t5xxl_weights.to(device=device, dtype=self.get_dtype_inference()))
                else:
                    out['t5xxl_ids'] = comfy.conds.CONDRegular(t5xxl_ids)
                    out['t5xxl_weights'] = comfy.conds.CONDRegular(t5xxl_weights)

            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class Lumina2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.lumina.model.NextDiT)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
            out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
            if 'num_tokens' not in out:
                out['num_tokens'] = comfy.conds.CONDConstant(cross_attn.shape[1])

        clip_text_pooled = kwargs.get("pooled_output", None)  # NewBie
        if clip_text_pooled is not None:
            out['clip_text_pooled'] = comfy.conds.CONDRegular(clip_text_pooled)

        clip_vision_outputs = kwargs.get("clip_vision_outputs", list(map(lambda a: a.get("clip_vision_output"), kwargs.get("unclip_conditioning", [{}]))))  # Z Image omni
        if clip_vision_outputs is not None and len(clip_vision_outputs) > 0:
            sigfeats = []
            for clip_vision_output in clip_vision_outputs:
                if clip_vision_output is not None:
                    image_size = clip_vision_output.image_sizes[0]
                    shape = clip_vision_output.last_hidden_state.shape
                    sigfeats.append(clip_vision_output.last_hidden_state.reshape(shape[0], image_size[1] // 16, image_size[2] // 16, shape[-1]))
            if len(sigfeats) > 0:
                out['siglip_feats'] = comfy.conds.CONDList(sigfeats)

        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

        ref_contexts = kwargs.get("reference_latents_text_embeds", None)
        if ref_contexts is not None:
            out['ref_contexts'] = comfy.conds.CONDList(ref_contexts)

        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()[2:]), ref_latents))])
        return out

class ZImagePixelSpace(Lumina2):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        BaseModel.__init__(self, model_config, model_type, device=device, unet_model=comfy.ldm.lumina.model.NextDiTPixelSpace)
        self.memory_usage_factor_conds = ("ref_latents",)


class PixelDiTT2I(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device,
                         unet_model=comfy.ldm.pixeldit.model.PixDiT_T2I)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out["attention_mask"] = comfy.conds.CONDRegular(attention_mask)
        return out


class PiD(PixelDiTT2I):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        BaseModel.__init__(self, model_config, model_type, device=device,
                           unet_model=comfy.ldm.pixeldit.pid.PidNet)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        lq_latent = kwargs.get("lq_latent", None)
        if lq_latent is not None:
            out["lq_latent"] = comfy.conds.CONDRegular(lq_latent)
        degrade_sigma = kwargs.get("degrade_sigma", None)
        if degrade_sigma is not None:
            out["degrade_sigma"] = comfy.conds.CONDRegular(degrade_sigma)
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "lq_latent" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
            lq = cond_value.cond
            dim = window.dim
            if dim >= lq.ndim:
                return None
            lq_proj = self.diffusion_model.lq_proj
            ratio = lq_proj.sr_scale * lq_proj.latent_spatial_down_factor
            # Map x window indices -> lq indices (deduplicated, sorted, in-bounds).
            lq_size = lq.size(dim)
            lq_indices = sorted({i // ratio for i in window.index_list if 0 <= i // ratio < lq_size})
            if not lq_indices:
                return None
            idx = tuple([slice(None)] * dim + [lq_indices])
            return cond_value._copy_with(lq[idx].to(device))
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)


class WAN21(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.WanModel)
        self.image_to_video = image_to_video

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        extra_channels = self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        if image is None:
            shape_image = list(noise.shape)
            shape_image[1] = extra_channels
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            latent_dim = self.latent_format.latent_channels
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            for i in range(0, image.shape[1], latent_dim):
                image[:, i: i + latent_dim] = self.process_latent_in(image[:, i: i + latent_dim])
            image = utils.resize_to_batch_size(image, noise.shape[0])

        if extra_channels != image.shape[1] + 4:
            if not self.image_to_video or extra_channels == image.shape[1]:
                return image

        if image.shape[1] > (extra_channels - 4):
            image = image[:, :(extra_channels - 4)]

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :4]
        else:
            if mask.shape[1] != 4:
                mask = torch.mean(mask, dim=1, keepdim=True)
            mask = 1.0 - mask
            mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            if mask.shape[-3] < noise.shape[-3]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
            if mask.shape[1] == 1:
                mask = mask.repeat(1, 4, 1, 1, 1)
            mask = utils.resize_to_batch_size(mask, noise.shape[0])

        concat_mask_index = kwargs.get("concat_mask_index", 0)
        if concat_mask_index != 0:
            return torch.cat((image[:, :concat_mask_index], mask, image[:, concat_mask_index:]), dim=1)
        else:
            return torch.cat((mask, image), dim=1)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        clip_vision_output = kwargs.get("clip_vision_output", None)
        if clip_vision_output is not None:
            out['clip_fea'] = comfy.conds.CONDRegular(clip_vision_output.penultimate_hidden_states)

        time_dim_concat = kwargs.get("time_dim_concat", None)
        if time_dim_concat is not None:
            out['time_dim_concat'] = comfy.conds.CONDRegular(self.process_latent_in(time_dim_concat))

        reference_latents = kwargs.get("reference_latents", None)
        if reference_latents is not None:
            out['reference_latent'] = comfy.conds.CONDRegular(self.process_latent_in(reference_latents[-1])[:, :, 0])

        # In-context reference conditioning (Bernini)
        context_latents = kwargs.get("context_latents", None)
        if context_latents is not None:
            out['context_latents'] = comfy.conds.CONDList([self.process_latent_in(l) for l in context_latents])

        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        # In-context cond slicing (Bernini)
        if cond_key == "context_latents" and isinstance(getattr(cond_value, "cond", None), list):
            dim = window.dim
            out = []
            for lat in cond_value.cond:
                if lat.ndim > dim and lat.shape[dim] > 1 and lat.shape[dim] == x_in.shape[dim]:
                    out.append(window.get_tensor(lat, device, dim=dim, retain_index_list=retain_index_list))
                else:
                    out.append(lat.to(device))
            return cond_value._copy_with(out)
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)


class WAN21_CausalAR(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device,
                                    unet_model=comfy.ldm.wan.ar_model.CausalWanModel)
        self.image_to_video = False


class WAN21_Vace(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.VaceWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        noise = kwargs.get("noise", None)
        noise_shape = list(noise.shape)
        vace_frames = kwargs.get("vace_frames", None)
        if vace_frames is None:
            noise_shape[1] = 32
            vace_frames = [torch.zeros(noise_shape, device=noise.device, dtype=noise.dtype)]

        mask = kwargs.get("vace_mask", None)
        if mask is None:
            noise_shape[1] = 64
            mask = [torch.ones(noise_shape, device=noise.device, dtype=noise.dtype)] * len(vace_frames)

        vace_frames_out = []
        for j in range(len(vace_frames)):
            vf = vace_frames[j].to(device=noise.device, dtype=noise.dtype, copy=True)
            for i in range(0, vf.shape[1], 16):
                vf[:, i:i + 16] = self.process_latent_in(vf[:, i:i + 16])
            vf = torch.cat([vf, mask[j].to(device=noise.device, dtype=noise.dtype)], dim=1)
            vace_frames_out.append(vf)

        vace_frames = torch.stack(vace_frames_out, dim=1)
        out['vace_context'] = comfy.conds.CONDRegular(vace_frames)

        vace_strength = kwargs.get("vace_strength", [1.0] * len(vace_frames_out))
        out['vace_strength'] = comfy.conds.CONDConstant(vace_strength)
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "vace_context":
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=3, retain_index_list=retain_index_list)
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

class WAN21_Camera(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.CameraWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        camera_conditions = kwargs.get("camera_conditions", None)
        if camera_conditions is not None:
            out['camera_conditions'] = comfy.conds.CONDRegular(camera_conditions)
        return out

class WAN21_HuMo(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.HumoWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        noise = kwargs.get("noise", None)

        audio_embed = kwargs.get("audio_embed", None)
        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)

        if "c_concat" not in out:  # 1.7B model
            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                out['reference_latent'] = comfy.conds.CONDRegular(self.process_latent_in(reference_latents[-1]))
        else:
            noise_shape = list(noise.shape)
            noise_shape[1] += 4
            concat_latent = torch.zeros(noise_shape, device=noise.device, dtype=noise.dtype)
            zero_vae_values_first = torch.tensor([0.8660, -0.4326, -0.0017, -0.4884, -0.5283, 0.9207, -0.9896, 0.4433, -0.5543, -0.0113, 0.5753, -0.6000, -0.8346, -0.3497, -0.1926, -0.6938]).view(1, 16, 1, 1, 1)
            zero_vae_values_second = torch.tensor([1.0869, -1.2370, 0.0206, -0.4357, -0.6411, 2.0307, -1.5972, 1.2659, -0.8595, -0.4654, 0.9638, -1.6330, -1.4310, -0.1098, -0.3856, -1.4583]).view(1, 16, 1, 1, 1)
            zero_vae_values = torch.tensor([0.8642, -1.8583, 0.1577, 0.1350, -0.3641, 2.5863, -1.9670, 1.6065, -1.0475, -0.8678, 1.1734, -1.8138, -1.5933, -0.7721, -0.3289, -1.3745]).view(1, 16, 1, 1, 1)
            concat_latent[:, 4:] = zero_vae_values
            concat_latent[:, 4:, :1] = zero_vae_values_first
            concat_latent[:, 4:, 1:2] = zero_vae_values_second
            out['c_concat'] = comfy.conds.CONDNoiseShape(concat_latent)
            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                ref_latent = self.process_latent_in(reference_latents[-1])
                ref_latent_shape = list(ref_latent.shape)
                ref_latent_shape[1] += 4 + ref_latent_shape[1]
                ref_latent_full = torch.zeros(ref_latent_shape, device=ref_latent.device, dtype=ref_latent.dtype)
                ref_latent_full[:, 20:] = ref_latent
                ref_latent_full[:, 16:20] = 1.0
                out['reference_latent'] = comfy.conds.CONDRegular(ref_latent_full)

        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "audio_embed":
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=1)
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

class WAN22_Animate(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model_animate.AnimateWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        face_video_pixels = kwargs.get("face_video_pixels", None)
        if face_video_pixels is not None:
            out['face_pixel_values'] = comfy.conds.CONDRegular(face_video_pixels)

        pose_latents = kwargs.get("pose_video_latent", None)
        if pose_latents is not None:
            out['pose_latents'] = comfy.conds.CONDRegular(self.process_latent_in(pose_latents))
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "face_pixel_values":
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=2, temporal_scale=4, temporal_offset=1)
        if cond_key == "pose_latents":
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=2, temporal_offset=1)
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

class WAN22_S2V(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.WanModel_S2V)
        self.memory_usage_factor_conds = ("reference_latent", "reference_motion")
        self.memory_usage_shape_process = {"reference_motion": lambda shape: [shape[0], shape[1], 1.5, shape[-2], shape[-1]]}

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        audio_embed = kwargs.get("audio_embed", None)
        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)

        reference_latents = kwargs.get("reference_latents", None)
        if reference_latents is not None:
            out['reference_latent'] = comfy.conds.CONDRegular(self.process_latent_in(reference_latents[-1]))

        reference_motion = kwargs.get("reference_motion", None)
        if reference_motion is not None:
            out['reference_motion'] = comfy.conds.CONDRegular(self.process_latent_in(reference_motion))

        control_video = kwargs.get("control_video", None)
        if control_video is not None:
            out['control_video'] = comfy.conds.CONDRegular(self.process_latent_in(control_video))
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['reference_latent'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])

        reference_motion = kwargs.get("reference_motion", None)
        if reference_motion is not None:
            out['reference_motion'] = reference_motion.shape
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key == "audio_embed":
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=1)
        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

class WAN22(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.WanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        denoise_mask = kwargs.get("denoise_mask", None)
        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)
        return out

    def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
        if denoise_mask is None:
            return timestep
        temp_ts = (torch.mean(denoise_mask[:, :, :, :, :], dim=(1, 3, 4), keepdim=True) * timestep.view([timestep.shape[0]] + [1] * (denoise_mask.ndim - 1))).reshape(timestep.shape[0], -1)
        return temp_ts

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image

class WAN21_FlowRVS(WAN21):
    def __init__(self, model_config, model_type=ModelType.IMG_TO_IMG_FLOW, image_to_video=False, device=None):
        model_config.unet_config["model_type"] = "t2v"
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.WanModel)
        self.image_to_video = image_to_video

class WAN21_SCAIL(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.SCAILWanModel)
        self.memory_usage_factor_conds = ("reference_latent", "pose_latents")
        self.memory_usage_shape_process = {"pose_latents": lambda shape: [shape[0], shape[1], 1.5, shape[-2], shape[-1]]}
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        reference_latents = kwargs.get("reference_latents", None)
        if reference_latents is not None:
            # SCAIL-2 multi-reference: reference_latents[0] is the primary ref, [1:] are additional
            # references. Stack as [additional..., primary] so the primary stays adjacent to the video.
            ordered = list(reference_latents[1:]) + list(reference_latents[:1])
            stacked = []
            for lat in ordered:
                lat = self.process_latent_in(lat)
                stacked.append(torch.cat([lat, torch.ones_like(lat[:, :4])], dim=1))
            out['reference_latent'] = comfy.conds.CONDRegular(torch.cat(stacked, dim=2))

        pose_latents = kwargs.get("pose_video_latent", None)
        if pose_latents is not None:
            pose_latents = self.process_latent_in(pose_latents)
            pose_mask = torch.ones_like(pose_latents[:, :4])
            pose_latents = torch.cat([pose_latents, pose_mask], dim=1)
            out['pose_latents'] = comfy.conds.CONDRegular(pose_latents)

        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['reference_latent'] = list([1, 20, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])

        pose_latents = kwargs.get("pose_video_latent", None)
        if pose_latents is not None:
            out['pose_latents'] = [pose_latents.shape[0], 20, *pose_latents.shape[2:]]

        return out

class WAN21_SCAIL2(WAN21_SCAIL):
    """SCAIL-2: SCAIL-Preview + an additive binary multi-identity mask stream."""

    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.SCAIL2WanModel)
        self.memory_usage_factor_conds = ("reference_latent", "pose_latents", "ref_mask_latents", "sam_latents")
        self.memory_usage_shape_process = {
            "pose_latents": lambda shape: [shape[0], shape[1], 1.5, shape[-2], shape[-1]],
            "sam_latents":  lambda shape: [shape[0], shape[1], 1.5, shape[-2], shape[-1]],
        }
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        driving_mask_28ch = kwargs.get("driving_mask_28ch", None)
        if driving_mask_28ch is not None:
            out['sam_latents'] = comfy.conds.CONDRegular(driving_mask_28ch.movedim(1, 2).contiguous())

        # ref_mask_28ch holds one identity mask per stacked reference frame (additional refs first, then the primary ref), followed by zeros over the video frames.
        ref_mask_28ch = kwargs.get("ref_mask_28ch", None)
        if ref_mask_28ch is not None:
            out['ref_mask_latents'] = comfy.conds.CONDRegular(ref_mask_28ch.movedim(1, 2).contiguous())

        ref_mask_flag = kwargs.get("ref_mask_flag", None)
        if ref_mask_flag is not None:
            out['ref_mask_flag'] = comfy.conds.CONDConstant(ref_mask_flag)

        return out

    def extra_conds_shapes(self, **kwargs):
        out = super().extra_conds_shapes(**kwargs)
        driving_mask_28ch = kwargs.get("driving_mask_28ch", None)
        if driving_mask_28ch is not None:
            s = driving_mask_28ch.shape
            out['sam_latents'] = [s[0], 28, s[1], s[3], s[4]]
        ref_mask_28ch = kwargs.get("ref_mask_28ch", None)
        if ref_mask_28ch is not None:
            s = ref_mask_28ch.shape
            out['ref_mask_latents'] = [s[0], 28, s[1], s[3], s[4]]
        return out

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        if cond_key in ("sam_latents", "pose_latents"):
            # Return sliced view omitting retain_index_list
            return comfy.context_windows.slice_cond(cond_value, window, x_in, device, temporal_dim=2, temporal_offset=0)
        if cond_key == "ref_mask_latents" and hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
            # The ref mask is N leading ref frames padded with frames of zeros, so just grab the first frames for all windows
            full_ref_mask = cond_value.cond
            video_frame_count = x_in.shape[2]
            ref_frame_count = full_ref_mask.shape[2] - video_frame_count
            if ref_frame_count < 1:
                return None
            window_length = len(window.index_list)

            # Account for the causal anchor frame if it exists
            anchor_index = getattr(window, "causal_anchor_index", None)
            if anchor_index is not None and anchor_index >= 0:
                window_length += 1

            window_ref_mask = full_ref_mask[:, :, :window_length + ref_frame_count].to(device)
            return cond_value._copy_with(window_ref_mask)

        return super().resize_cond_for_context_window(cond_key, cond_value, window, x_in, device, retain_index_list=retain_index_list)

    def concat_cond(self, **kwargs):
        # The 4 extra channels are the history_mask (1 at clean-anchor frames).
        noise = kwargs.get("noise", None)
        extra_channels = self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
        if extra_channels != 4:
            return super().concat_cond(**kwargs)

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            return torch.zeros_like(noise)[:, :4]

        device = kwargs["device"]
        if mask.shape[1] != 4:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = 1.0 - mask
        mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        if mask.shape[-3] < noise.shape[-3]:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 4, 1, 1, 1)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return mask

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        # Hold anchor constant across all sigmas instead of base sigma*noise + (1-sigma)*latent_image.
        return latent_image


class WAN22_WanDancer(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=True, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model_wandancer.WanDancerModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        audio_embed = kwargs.get("audio_embed", None)
        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)

        clip_vision_output_ref = kwargs.get("clip_vision_output_ref", None)
        if clip_vision_output_ref is not None:
            out['clip_fea_ref'] = comfy.conds.CONDRegular(clip_vision_output_ref.penultimate_hidden_states)

        fps = kwargs.get("fps", None)
        if fps is not None:
            out['fps'] = comfy.conds.CONDRegular(torch.FloatTensor([fps]))

        audio_inject_scale = kwargs.get("audio_inject_scale", None)
        if audio_inject_scale is not None:
            out['audio_inject_scale'] = comfy.conds.CONDRegular(torch.FloatTensor([audio_inject_scale]))
        return out

class Hunyuan3Dv2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan3d.model.Hunyuan3Dv2)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 5.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class Hunyuan3Dv2_1(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan3dv2_1.hunyuandit.HunYuanDiTPlain)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 5.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class TripoSplat(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.triposplat.model.LatentSeqMMFlowModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None) # DINOv3 token sequence -> cross-attention context.
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        ref_latents = kwargs.get("reference_latents", None) # Flux2 VAE image latent -> additive second conditioning.
        if ref_latents is not None:
            out['ref_latents'] = comfy.conds.CONDList(list(ref_latents))
        latent_shapes = kwargs.get("latent_shapes", None) # {latent, camera} nested latent
        if latent_shapes is not None:
            out['latent_shapes'] = comfy.conds.CONDConstant(latent_shapes)
        return out


class HiDream(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hidream.model.HiDreamImageTransformer2DModel)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        conditioning_llama3 = kwargs.get("conditioning_llama3", None)
        if conditioning_llama3 is not None:
            out['encoder_hidden_states_llama3'] = comfy.conds.CONDRegular(conditioning_llama3)
        image_cond = kwargs.get("concat_latent_image", None)
        if image_cond is not None:
            out['image_cond'] = comfy.conds.CONDNoiseShape(self.process_latent_in(image_cond))
        return out

class HiDreamO1(BaseModel):
    """HiDream-O1-Image: pixel-space DiT (no VAE). Refs from HiDreamO1ReferenceImages and tokens from the stub TE flow through
    extra_conds; the heavy preprocessing lives in comfy.ldm.hidream_o1.conditioning."""
    PATCH_SIZE = 32

    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hidream_o1.model.HiDreamO1Transformer)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        text_input_ids = kwargs.get("text_input_ids", None)
        noise = kwargs.get("noise", None)
        if text_input_ids is None or noise is None:
            return out

        # handle area conds
        area = kwargs.get("area", None)
        if area is not None:
            crop_h = min(noise.shape[-2] - area[2], area[0])
            crop_w = min(noise.shape[-1] - area[3], area[1])
            noise = torch.empty((noise.shape[0], 3, crop_h, crop_w), dtype=noise.dtype, device=noise.device)

        conds = build_extra_conds(
            text_input_ids, noise,
            ref_images=kwargs.get("reference_latents", None),
            target_patch_size=self.PATCH_SIZE,
        )
        for k, v in conds.items():
            # ar_len is a Python int (precomputed to avoid a GPU sync in forward).
            cls = comfy.conds.CONDConstant if k == "ar_len" else comfy.conds.CONDRegular
            out[k] = cls(v)
        return out

class Chroma(Flux):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None, unet_model=comfy.ldm.chroma.model.Chroma):
        super().__init__(model_config, model_type, device=device, unet_model=unet_model)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        guidance = kwargs.get("guidance", 0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class ChromaRadiance(Chroma):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.chroma_radiance.model.ChromaRadiance)

class ACEStep(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ace.model.ACEStepTransformer2DModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        noise = kwargs.get("noise", None)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        conditioning_lyrics = kwargs.get("conditioning_lyrics", None)
        if cross_attn is not None:
            out['lyric_token_idx'] = comfy.conds.CONDRegular(conditioning_lyrics)
        out['speaker_embeds'] = comfy.conds.CONDRegular(torch.zeros(noise.shape[0], 512, device=noise.device, dtype=noise.dtype))
        out['lyrics_strength'] = comfy.conds.CONDConstant(kwargs.get("lyrics_strength", 1.0))
        return out

class ACEStep15(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ace.ace_step15.AceStepConditionGenerationModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        device = kwargs["device"]
        noise = kwargs["noise"]

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if torch.count_nonzero(cross_attn) == 0:
                out['replace_with_null_embeds'] = comfy.conds.CONDConstant(True)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        conditioning_lyrics = kwargs.get("conditioning_lyrics", None)
        if cross_attn is not None:
            out['lyric_embed'] = comfy.conds.CONDRegular(conditioning_lyrics)

        refer_audio = kwargs.get("reference_audio_timbre_latents", None)
        if refer_audio is None or len(refer_audio) == 0:
            refer_audio = comfy.ldm.ace.ace_step15.get_silence_latent(noise.shape[2], device)
            pass_audio_codes = True
        else:
            refer_audio = refer_audio[-1][:, :, :noise.shape[2]]
            out['is_covers'] = comfy.conds.CONDConstant(True)
            pass_audio_codes = False

        if pass_audio_codes:
            audio_codes = kwargs.get("audio_codes", None)
            if audio_codes is not None:
                out['audio_codes'] = comfy.conds.CONDRegular(torch.tensor(audio_codes, device=device))
                refer_audio = refer_audio[:, :, :750]
            else:
                out['is_covers'] = comfy.conds.CONDConstant(False)

        if refer_audio.shape[2] < noise.shape[2]:
            pad = comfy.ldm.ace.ace_step15.get_silence_latent(noise.shape[2], device)
            refer_audio = torch.cat([refer_audio.to(pad), pad[:, :, refer_audio.shape[2]:]], dim=2)

        out['refer_audio'] = comfy.conds.CONDRegular(refer_audio)
        return out

class Omnigen2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.omnigen.omnigen2.OmniGen2Transformer2DModel)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
            out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])
        return out

class Boogu(Omnigen2):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super(Omnigen2, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.boogu.model.BooguTransformer2DModel)
        self.memory_usage_factor_conds = ("ref_latents",)

class QwenImage(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.qwen_image.model.QwenImageTransformer2DModel)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])
        return out

class Ideogram4(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class Krea2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.krea2.model.SingleStreamDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class HunyuanImage21(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan_video.model.HunyuanVideo)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        conditioning_byt5small = kwargs.get("conditioning_byt5small", None)
        if conditioning_byt5small is not None:
            out['txt_byt5'] = comfy.conds.CONDRegular(conditioning_byt5small)

        guidance = kwargs.get("guidance", 6.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        return out

class HunyuanImage21Refiner(HunyuanImage21):
    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        image = kwargs.get("concat_latent_image", None)
        noise_augmentation = kwargs.get("noise_augmentation", 0.0)
        device = kwargs["device"]

        if image is None:
            shape_image = list(noise.shape)
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            image = self.process_latent_in(image)
            image = utils.resize_to_batch_size(image, noise.shape[0])
            if noise_augmentation > 0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(kwargs.get("seed", 0) - 10)
                noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device="cpu").to(image.device)
                image = noise_augmentation * noise + min(1.0 - noise_augmentation, 0.75) * image
            else:
                image = 0.75 * image
        return image

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out['disable_time_r'] = comfy.conds.CONDConstant(True)
        return out

class HunyuanVideo15(HunyuanVideo):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        extra_channels = self.diffusion_model.img_in.proj.weight.shape[1] - noise.shape[1] - 1 #noise 32 img cond 32 + mask 1
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        if image is None:
            shape_image = list(noise.shape)
            shape_image[1] = extra_channels
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            latent_dim = self.latent_format.latent_channels
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            for i in range(0, image.shape[1], latent_dim):
                image[:, i: i + latent_dim] = self.process_latent_in(image[:, i: i + latent_dim])
            image = utils.resize_to_batch_size(image, noise.shape[0])

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :1]
        else:
            mask = 1.0 - mask
            mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            if mask.shape[-3] < noise.shape[-3]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
            mask = utils.resize_to_batch_size(mask, noise.shape[0])

        return torch.cat((image, mask), dim=1)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        conditioning_byt5small = kwargs.get("conditioning_byt5small", None)
        if conditioning_byt5small is not None:
            out['txt_byt5'] = comfy.conds.CONDRegular(conditioning_byt5small)

        guidance = kwargs.get("guidance", 6.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        clip_vision_output = kwargs.get("clip_vision_output", None)
        if clip_vision_output is not None:
            out['clip_fea'] = comfy.conds.CONDRegular(clip_vision_output.last_hidden_state)

        return out

class HunyuanVideo15_SR_Distilled(HunyuanVideo15):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        image = kwargs.get("concat_latent_image", None)
        noise_augmentation = kwargs.get("noise_augmentation", 0.0)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros([noise.shape[0], noise.shape[1] * 2 + 2, noise.shape[-3], noise.shape[-2], noise.shape[-1]], device=comfy.model_management.intermediate_device())
        else:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            #image = self.process_latent_in(image) # scaling wasn't applied in reference code
            image = utils.resize_to_batch_size(image, noise.shape[0])
            lq_image_slice = slice(noise.shape[1] + 1, 2 * noise.shape[1] + 1)
            if noise_augmentation > 0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(kwargs.get("seed", 0) - 10)
                noise = torch.randn(image[:, lq_image_slice].shape, generator=generator, dtype=image.dtype, device="cpu").to(image.device)
                image[:, lq_image_slice] = noise_augmentation * noise + min(1.0 - noise_augmentation, 0.75) * image[:, lq_image_slice]
            else:
                image[:, lq_image_slice] = 0.75 * image[:, lq_image_slice]
        return image

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out['disable_time_r'] = comfy.conds.CONDConstant(False)
        return out

class Kandinsky5(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.kandinsky5.model.Kandinsky5)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        device = kwargs["device"]
        image = torch.zeros_like(noise)

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :1]
        else:
            mask = 1.0 - mask
            mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            if mask.shape[-3] < noise.shape[-3]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
            mask = utils.resize_to_batch_size(mask, noise.shape[0])

        return torch.cat((image, mask), dim=1)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        time_dim_replace = kwargs.get("time_dim_replace", None)
        if time_dim_replace is not None:
            out['time_dim_replace'] = comfy.conds.CONDRegular(self.process_latent_in(time_dim_replace))

        return out

class Kandinsky5Image(Kandinsky5):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)

    def concat_cond(self, **kwargs):
        return None

class RT_DETR_v4(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.rt_detr.rtdetr_v4.RTv4)


class DepthAnything3(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device,
                         unet_model=comfy.ldm.depth_anything_3.model.DepthAnything3Net)

class ErnieImage(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ernie.model.ErnieImageModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class SAM3(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.sam3.detector.SAM3Model)

class CogVideoX(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION_DDPM, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.cogvideo.model.CogVideoXTransformer3DModel)
        self.image_to_video = image_to_video

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        # Detect extra channels needed (e.g. 32 - 16 = 16 for ref latent)
        extra_channels = self.diffusion_model.in_channels - noise.shape[1]
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        if image is None:
            shape = list(noise.shape)
            shape[1] = extra_channels
            return torch.zeros(shape, dtype=noise.dtype, layout=noise.layout, device=noise.device)

        latent_dim = self.latent_format.latent_channels
        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        if noise.ndim == 5 and image.ndim == 5:
            if image.shape[-3] < noise.shape[-3]:
                image = torch.nn.functional.pad(image, (0, 0, 0, 0, 0, noise.shape[-3] - image.shape[-3]), "constant", 0)
            elif image.shape[-3] > noise.shape[-3]:
                image = image[:, :, :noise.shape[-3]]

        for i in range(0, image.shape[1], latent_dim):
            image[:, i:i + latent_dim] = self.process_latent_in(image[:, i:i + latent_dim])
        image = utils.resize_to_batch_size(image, noise.shape[0])

        if image.shape[1] > extra_channels:
            image = image[:, :extra_channels]
        elif image.shape[1] < extra_channels:
            repeats = extra_channels // image.shape[1]
            remainder = extra_channels % image.shape[1]
            parts = [image] * repeats
            if remainder > 0:
                parts.append(image[:, :remainder])
            image = torch.cat(parts, dim=1)

        return image

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        # OFS embedding (CogVideoX 1.5 I2V), default 2.0 as used by SparkVSR
        if self.diffusion_model.ofs_proj_dim is not None:
            ofs = kwargs.get("ofs", None)
            if ofs is None:
                noise = kwargs.get("noise", None)
                ofs = torch.full((noise.shape[0],), 2.0, device=noise.device, dtype=noise.dtype)
            out['ofs'] = comfy.conds.CONDRegular(ofs)
        return out
