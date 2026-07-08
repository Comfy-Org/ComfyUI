import torch
from . import model_base
from . import utils

from . import sd1_clip
from . import sdxl_clip
import comfy.text_encoders.sd2_clip
import comfy.text_encoders.sd3_clip
import comfy.text_encoders.sa_t5
import comfy.text_encoders.sa3
import comfy.text_encoders.aura_t5
import comfy.text_encoders.pixart_t5
import comfy.text_encoders.hydit
import comfy.text_encoders.flux
import comfy.text_encoders.genmo
import comfy.text_encoders.lt
import comfy.text_encoders.hunyuan_video
import comfy.text_encoders.cosmos
import comfy.text_encoders.lumina2
import comfy.text_encoders.wan
import comfy.text_encoders.ace
import comfy.text_encoders.omnigen2
import comfy.text_encoders.qwen_image
import comfy.text_encoders.hunyuan_image
import comfy.text_encoders.kandinsky5
import comfy.text_encoders.z_image
import comfy.text_encoders.ideogram4
import comfy.text_encoders.boogu
import comfy.text_encoders.krea2
import comfy.text_encoders.anima
import comfy.text_encoders.ace15
import comfy.text_encoders.longcat_image
import comfy.text_encoders.ernie
import comfy.text_encoders.cogvideo
import comfy.text_encoders.hidream_o1
import comfy.text_encoders.pixeldit

from . import supported_models_base
from . import latent_formats

from . import diffusers_convert
import comfy.model_management

class SD15(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent_formats.SD15
    memory_usage_factor = 1.0

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
            if ids.dtype == torch.float32:
                state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict:
                state_dict.pop(p)

        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)

class SD20(supported_models_base.BASE):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
        "attn_precision": torch.float32,
    }

    latent_format = latent_formats.SD15
    memory_usage_factor = 1.0

    def model_type(self, state_dict, prefix=""):
        if self.unet_config["in_channels"] == 4: #SD2.0 inpainting models are not v prediction
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict.get(k, None)
            if out is not None and torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
                return model_base.ModelType.V_PREDICTION
        return model_base.ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_h." #SD2 in sgm format
        replace_prefix["cond_stage_model.model."] = "clip_h."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_h.", "clip_h.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.sd2_clip.SD2Tokenizer, comfy.text_encoders.sd2_clip.SD2ClipModel)

class SD21UnclipL(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 1536,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 768}


class SD21UnclipH(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2048,
        "use_temporal_attention": False,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1024}

class SDXLRefiner(supported_models_base.BASE):
    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL
    memory_usage_factor = 1.0

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.SDXLRefiner(self, device=device)

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_g."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix["clip_g"] = "conditioner.embedders.0.model"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLRefinerClipModel)

class SDXL(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL

    memory_usage_factor = 0.8

    def model_type(self, state_dict, prefix=""):
        if 'edm_mean' in state_dict and 'edm_std' in state_dict: #Playground V2.5
            self.latent_format = latent_formats.SDXL_Playground_2_5()
            self.sampling_settings["sigma_data"] = 0.5
            self.sampling_settings["sigma_max"] = 80.0
            self.sampling_settings["sigma_min"] = 0.002
            return model_base.ModelType.EDM
        elif "edm_vpred.sigma_max" in state_dict:
            self.sampling_settings["sigma_max"] = float(state_dict["edm_vpred.sigma_max"].item())
            if "edm_vpred.sigma_min" in state_dict:
                self.sampling_settings["sigma_min"] = float(state_dict["edm_vpred.sigma_min"].item())
            return model_base.ModelType.V_PREDICTION_EDM
        elif "v_pred" in state_dict:
            if "ztsnr" in state_dict: #Some zsnr anime checkpoints
                self.sampling_settings["zsnr"] = True
            return model_base.ModelType.V_PREDICTION
        else:
            return model_base.ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"
        replace_prefix["conditioner.embedders.1.model."] = "clip_g."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

        state_dict = utils.state_dict_key_replace(state_dict, keys_to_replace)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix["clip_g"] = "conditioner.embedders.1.model"
        replace_prefix["clip_l"] = "conditioner.embedders.0"
        state_dict_g = utils.state_dict_prefix_replace(state_dict_g, replace_prefix)
        return state_dict_g

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLClipModel)

class SSD1B(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class Segmind_Vega(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class KOALA_700M(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 5],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class KOALA_1B(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 6],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

class SVD_img2vid(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 768,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }

    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
        "attn_precision": torch.float32,
    }

    clip_vision_prefix = "conditioner.embedders.0.open_clip.model.visual."

    latent_format = latent_formats.SD15

    sampling_settings = {"sigma_max": 700.0, "sigma_min": 0.002}

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SVD_img2vid(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None

class SV3D_u(SVD_img2vid):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 256,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }

    vae_key_prefix = ["conditioner.embedders.1.encoder."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SV3D_u(self, device=device)
        return out

class SV3D_p(SV3D_u):
    unet_config = {
        "model_channels": 320,
        "in_channels": 8,
        "use_linear_in_transformer": True,
        "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
        "context_dim": 1024,
        "adm_in_channels": 1280,
        "use_temporal_attention": True,
        "use_temporal_resblock": True
    }


    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SV3D_p(self, device=device)
        return out

class Stable_Zero123(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    required_keys = {
        "cc_projection.weight": None,
        "cc_projection.bias": None,
    }

    clip_vision_prefix = "cond_stage_model.model.visual."

    latent_format = latent_formats.SD15

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Stable_Zero123(self, device=device, cc_projection_weight=state_dict["cc_projection.weight"], cc_projection_bias=state_dict["cc_projection.bias"])
        return out

    def clip_target(self, state_dict={}):
        return None

class SD_X4Upscaler(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 256,
        'in_channels': 7,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "disable_self_attentions": [True, True, True, False],
        "num_classes": 1000,
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent_formats.SD_X4

    sampling_settings = {
        "linear_start": 0.0001,
        "linear_end": 0.02,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SD_X4Upscaler(self, device=device)
        return out

class Stable_Cascade_C(supported_models_base.BASE):
    unet_config = {
        "stable_cascade_stage": 'c',
    }

    unet_extra_config = {}

    latent_format = latent_formats.SC_Prior
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 2.0,
    }

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoder."]
    clip_vision_prefix = "clip_l_vision."

    def process_unet_state_dict(self, state_dict):
        key_list = list(state_dict.keys())
        for y in ["weight", "bias"]:
            suffix = "in_proj_{}".format(y)
            keys = filter(lambda a: a.endswith(suffix), key_list)
            for k_from in keys:
                weights = state_dict.pop(k_from)
                prefix = k_from[:-(len(suffix) + 1)]
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["to_q", "to_k", "to_v"]
                    k_to = "{}.{}.{}".format(prefix, p[x], y)
                    state_dict[k_to] = weights[shape_from*x:shape_from*(x + 1)]
        return state_dict

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.state_dict_prefix_replace(state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True)
        if "clip_g.text_projection" in state_dict:
            state_dict["clip_g.transformer.text_projection.weight"] = state_dict.pop("clip_g.text_projection").transpose(0, 1)
        return state_dict

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.StableCascade_C(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(sdxl_clip.StableCascadeTokenizer, sdxl_clip.StableCascadeClipModel)

class Stable_Cascade_B(Stable_Cascade_C):
    unet_config = {
        "stable_cascade_stage": 'b',
    }

    unet_extra_config = {}

    latent_format = latent_formats.SC_B
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    sampling_settings = {
        "shift": 1.0,
    }

    clip_vision_prefix = None

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.StableCascade_B(self, device=device)
        return out

class SD15_instructpix2pix(SD15):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.SD15_instructpix2pix(self, device=device)

class SDXL_instructpix2pix(SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
        "in_channels": 8,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.SDXL_instructpix2pix(self, model_type=self.model_type(state_dict, prefix), device=device)

class LotusD(SD20):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "use_temporal_attention": False,
        "adm_in_channels": 4,
        "in_channels": 4,
    }

    unet_extra_config = {
        "num_classes": 'sequential',
        "num_head_channels": 64,
    }

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Lotus(self, device=device)

class SD3(supported_models_base.BASE):
    unet_config = {
        "in_channels": 16,
        "pos_embed_scaling_factor": None,
    }

    sampling_settings = {
        "shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.SD3

    memory_usage_factor = 1.6

    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.SD3(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        clip_l = False
        clip_g = False
        t5 = False
        pref = self.text_encoder_key_prefix[0]
        if "{}clip_l.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:
            clip_l = True
        if "{}clip_g.transformer.text_model.final_layer_norm.weight".format(pref) in state_dict:
            clip_g = True
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        if "dtype_t5" in t5_detect:
            t5 = True

        return supported_models_base.ClipTarget(comfy.text_encoders.sd3_clip.SD3Tokenizer, comfy.text_encoders.sd3_clip.sd3_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, **t5_detect))

class StableAudio(supported_models_base.BASE):
    unet_config = {
        "audio_model": "dit1.0",
    }

    sampling_settings = {"sigma_max": 500.0, "sigma_min": 0.03}

    unet_extra_config = {}
    latent_format = latent_formats.StableAudio1

    text_encoder_key_prefix = ["text_encoders."]
    vae_key_prefix = ["pretransform.model."]

    def get_model(self, state_dict, prefix="", device=None):
        seconds_start_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_start.": ""}, filter_keys=True)
        seconds_total_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_total.": ""}, filter_keys=True)
        return model_base.StableAudio1(self, seconds_start_embedder_weights=seconds_start_sd, seconds_total_embedder_weights=seconds_total_sd, device=device)

    def process_unet_state_dict(self, state_dict):
        for k in list(state_dict.keys()):
            if k.endswith(".cross_attend_norm.beta") or k.endswith(".ff_norm.beta") or k.endswith(".pre_norm.beta"): #These weights are all zero
                state_dict.pop(k)
        return state_dict

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.sa_t5.SAT5Tokenizer, comfy.text_encoders.sa_t5.SAT5Model)

class StableAudio3(StableAudio):
    unet_config = {
        "audio_model": "dit1.0",
        "global_cond_shared_embed": True,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 2.0,
    }

    latent_format = latent_formats.StableAudio3

    memory_usage_factor = 7

    def get_model(self, state_dict, prefix="", device=None):
        seconds_total_sd = utils.state_dict_prefix_replace(state_dict, {"conditioner.conditioners.seconds_total.": ""}, filter_keys=True)
        padding_embedding = state_dict.get("conditioner.conditioners.prompt.padding_embedding", None)
        return model_base.StableAudio3(self, seconds_total_embedder_weights=seconds_total_sd, padding_embedding=padding_embedding, device=device)

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.sa3.SAT5GemmaTokenizer, comfy.text_encoders.sa3.SAT5GemmaModel)

class AuraFlow(supported_models_base.BASE):
    unet_config = {
        "cond_seq_dim": 2048,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.73,
    }

    unet_extra_config = {}
    latent_format = latent_formats.SDXL

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.AuraFlow(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.aura_t5.AuraT5Tokenizer, comfy.text_encoders.aura_t5.AuraT5Model)

class PixArtAlpha(supported_models_base.BASE):
    unet_config = {
        "image_model": "pixart_alpha",
    }

    sampling_settings = {
        "beta_schedule" : "sqrt_linear",
        "linear_start"  : 0.0001,
        "linear_end"    : 0.02,
        "timesteps"     : 1000,
    }

    unet_extra_config = {}
    latent_format = latent_formats.SD15

    memory_usage_factor = 0.5

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.PixArt(self, device=device)
        return out.eval()

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.pixart_t5.PixArtTokenizer, comfy.text_encoders.pixart_t5.PixArtT5XXL)

class PixArtSigma(PixArtAlpha):
    unet_config = {
        "image_model": "pixart_sigma",
    }
    latent_format = latent_formats.SDXL

class HunyuanDiT(supported_models_base.BASE):
    unet_config = {
        "image_model": "hydit",
    }

    unet_extra_config = {
        "attn_precision": torch.float32,
    }

    sampling_settings = {
        "linear_start": 0.00085,
        "linear_end": 0.018,
    }

    latent_format = latent_formats.SDXL

    memory_usage_factor = 1.3

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanDiT(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.hydit.HyditTokenizer, comfy.text_encoders.hydit.HyditModel)

class HunyuanDiT1(HunyuanDiT):
    unet_config = {
        "image_model": "hydit1",
    }

    unet_extra_config = {}

    sampling_settings = {
        "linear_start" : 0.00085,
        "linear_end" : 0.03,
    }

class Flux(supported_models_base.BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    memory_usage_factor = 3.1 # TODO: debug why flux mem usage is so weird on windows.

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def process_unet_state_dict(self, state_dict):
        out_sd = {}
        for k in list(state_dict.keys()):
            key_out = k
            if key_out.endswith("_norm.scale"):
                key_out = "{}.weight".format(key_out[:-len(".scale")])
            out_sd[key_out] = state_dict[k]
        return out_sd

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))

class FluxInpaint(Flux):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
        "in_channels": 96,
    }

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

class FluxSchnell(Flux):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": False,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.0,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux(self, model_type=model_base.ModelType.FLOW, device=device)
        return out

class Flux2(Flux):
    unet_config = {
        "image_model": "flux2",
    }

    sampling_settings = {
        "shift": 2.02,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux2

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.memory_usage_factor * (2.0 * 2.0) * (unet_config['hidden_size'] / 2604)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))
        if len(detect) > 0:
            detect["model_type"] = "qwen3_4b"
            return supported_models_base.ClipTarget(comfy.text_encoders.flux.KleinTokenizer, comfy.text_encoders.flux.klein_te(**detect))

        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_8b.transformer.".format(pref))
        if len(detect) > 0:
            detect["model_type"] = "qwen3_8b"
            return supported_models_base.ClipTarget(comfy.text_encoders.flux.KleinTokenizer8B, comfy.text_encoders.flux.klein_te(**detect))

        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}mistral3_24b.transformer.".format(pref))
        if len(detect) > 0:
            if "{}mistral3_24b.transformer.model.layers.39.post_attention_layernorm.weight".format(pref) not in state_dict:
                detect["pruned"] = True
            return supported_models_base.ClipTarget(comfy.text_encoders.flux.Flux2Tokenizer, comfy.text_encoders.flux.flux2_te(**detect))

        return None


class Lens(supported_models_base.BASE):
    """Microsoft Lens (3.8B dual-stream MMDiT, GPT-OSS-20B text features, Flux2 VAE)."""

    unet_config = {
        "image_model": "lens",
    }

    sampling_settings = {
        "shift": 1.829, # Default mu for 1440x1440 (and any seq_len > 4300
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux2

    memory_usage_factor = 4.0

    supported_inference_dtypes = [torch.bfloat16, torch.float32] # fp16 causes NaNs

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Lens(self, model_type=model_base.ModelType.FLUX, device=device)

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        for hint in ("gpt_oss.transformer.", ""):
            full_prefix = "{}{}".format(pref, hint)
            if "{}layers.0.self_attn.sinks".format(full_prefix) in state_dict:
                detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, full_prefix)
                return supported_models_base.ClipTarget(
                    comfy.text_encoders.gpt_oss.LensTokenizer,
                    comfy.text_encoders.gpt_oss.lens_te(**detect),
                )
        return supported_models_base.ClipTarget(
            comfy.text_encoders.gpt_oss.LensTokenizer,
            comfy.text_encoders.gpt_oss.lens_te(),
        )


class GenmoMochi(supported_models_base.BASE):
    unet_config = {
        "image_model": "mochi_preview",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 6.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Mochi

    memory_usage_factor = 2.0 #TODO

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.GenmoMochi(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.genmo.MochiT5Tokenizer, comfy.text_encoders.genmo.mochi_te(**t5_detect))

class LTXV(supported_models_base.BASE):
    unet_config = {
        "image_model": "ltxv",
    }

    sampling_settings = {
        "shift": 2.37,
    }

    unet_extra_config = {}
    latent_format = latent_formats.LTXV

    memory_usage_factor = 5.5 # TODO: img2vid is about 2x vs txt2vid

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = (unet_config.get("cross_attention_dim", 2048) / 2048) * 5.5

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.LTXV(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.lt.LTXVT5Tokenizer, comfy.text_encoders.lt.ltxv_te(**t5_detect))

class LTXAV(LTXV):
    unet_config = {
        "image_model": "ltxav",
    }

    latent_format = latent_formats.LTXAV

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 0.077  # TODO

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.LTXAV(self, device=device)
        return out

class HunyuanVideo(supported_models_base.BASE):
    unet_config = {
        "image_model": "hunyuan_video",
    }

    sampling_settings = {
        "shift": 7.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.HunyuanVideo

    memory_usage_factor = 1.8 #TODO

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanVideo(self, device=device)
        return out

    def process_unet_state_dict(self, state_dict):
        out_sd = {}
        for k in list(state_dict.keys()):
            key_out = k
            key_out = key_out.replace("txt_in.t_embedder.mlp.0.", "txt_in.t_embedder.in_layer.").replace("txt_in.t_embedder.mlp.2.", "txt_in.t_embedder.out_layer.")
            key_out = key_out.replace("txt_in.c_embedder.linear_1.", "txt_in.c_embedder.in_layer.").replace("txt_in.c_embedder.linear_2.", "txt_in.c_embedder.out_layer.")
            key_out = key_out.replace("_mod.linear.", "_mod.lin.").replace("_attn_qkv.", "_attn.qkv.")
            key_out = key_out.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")
            key_out = key_out.replace("_attn_q_norm.weight", "_attn.norm.query_norm.weight").replace("_attn_k_norm.weight", "_attn.norm.key_norm.weight")
            key_out = key_out.replace(".q_norm.weight", ".norm.query_norm.weight").replace(".k_norm.weight", ".norm.key_norm.weight")
            key_out = key_out.replace("_attn_proj.", "_attn.proj.")
            key_out = key_out.replace(".modulation.linear.", ".modulation.lin.")
            key_out = key_out.replace("_in.mlp.2.", "_in.out_layer.").replace("_in.mlp.0.", "_in.in_layer.")
            if key_out.endswith(".scale"):
                key_out = "{}.weight".format(key_out[:-len(".scale")])
            out_sd[key_out] = state_dict[k]
        return out_sd

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}llama.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideoTokenizer, comfy.text_encoders.hunyuan_video.hunyuan_video_clip(**hunyuan_detect))

class HunyuanVideoI2V(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "in_channels": 33,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanVideoI2V(self, device=device)
        return out

class HunyuanVideoSkyreelsI2V(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "in_channels": 32,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanVideoSkyreelsI2V(self, device=device)
        return out

class CosmosT2V(supported_models_base.BASE):
    unet_config = {
        "image_model": "cosmos",
        "in_channels": 16,
    }

    sampling_settings = {
        "sigma_data": 0.5,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Cosmos1CV8x8x8

    memory_usage_factor = 1.6 #TODO

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32] #TODO

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.CosmosVideo(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.cosmos.CosmosT5Tokenizer, comfy.text_encoders.cosmos.te(**t5_detect))

class CosmosI2V(CosmosT2V):
    unet_config = {
        "image_model": "cosmos",
        "in_channels": 17,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.CosmosVideo(self, image_to_video=True, device=device)
        return out

class CosmosT2IPredict2(supported_models_base.BASE):
    unet_config = {
        "image_model": "cosmos_predict2",
        "in_channels": 16,
    }

    sampling_settings = {
        "sigma_data": 1.0,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = (unet_config.get("model_channels", 2048) / 2048) * 0.95

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.CosmosPredict2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.cosmos.CosmosT5Tokenizer, comfy.text_encoders.cosmos.te(**t5_detect))

class Anima(supported_models_base.BASE):
    unet_config = {
        "image_model": "anima",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Anima(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_06b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.anima.AnimaTokenizer, comfy.text_encoders.anima.te(**detect))

    def set_inference_dtype(self, dtype, manual_cast_dtype, **kwargs):
        self.memory_usage_factor = (self.unet_config.get("model_channels", 2048) / 2048) * 0.95
        if dtype is torch.float16:
            self.memory_usage_factor *= 1.4
        return super().set_inference_dtype(dtype, manual_cast_dtype, **kwargs)

class CosmosI2VPredict2(CosmosT2IPredict2):
    unet_config = {
        "image_model": "cosmos_predict2",
        "in_channels": 17,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.CosmosPredict2(self, image_to_video=True, device=device)
        return out

class Lumina2(supported_models_base.BASE):
    unet_config = {
        "image_model": "lumina2",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 6.0,
    }

    memory_usage_factor = 1.4

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Lumina2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}gemma2_2b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.lumina2.LuminaTokenizer, comfy.text_encoders.lumina2.te(**hunyuan_detect))

class ZImage(Lumina2):
    unet_config = {
        "image_model": "lumina2",
        "dim": 3840,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        if comfy.model_management.extended_fp16_support() and unet_config.get("allow_fp16", False):
            self.supported_inference_dtypes = self.supported_inference_dtypes.copy()
            self.supported_inference_dtypes.insert(1, torch.float16)

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.z_image.ZImageTokenizer, comfy.text_encoders.z_image.te(**hunyuan_detect))

class ZImagePixelSpace(ZImage):
    unet_config = {
        "image_model": "zimage_pixel",
    }

    # Pixel-space model: no spatial compression, operates on raw RGB patches.
    latent_format = latent_formats.ZImagePixelSpace

    # Much lower memory than latent-space models (no VAE, small patches).
    memory_usage_factor = 0.03 # TODO: figure out the optimal value for this.

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.ZImagePixelSpace(self, device=device)

class PixelDiTT2I(supported_models_base.BASE):
    unet_config = {
        "image_model": "pixeldit_t2i",
    }

    unet_extra_config = {}

    sampling_settings = {
        "shift": 4.0,  # 1024px stage 3 default; 2.0 for 512px
    }

    latent_format = latent_formats.PixelDiTPixel
    memory_usage_factor = 0.04
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.PixelDiTT2I(self, device=device)

    def process_unet_state_dict(self, state_dict):
        # pixel_dim from pixel_embedder.proj.weight = (pixel_dim, in_channels); p2 derived per-weight from total // (6 * pixel_dim).
        pixel_dim = next(v for k, v in state_dict.items() if k.endswith("pixel_embedder.proj.weight")).shape[0]

        out = {}
        marker = ".adaLN_modulation.0."
        for k, v in state_dict.items():
            if k.startswith("_repa_projector") or k.startswith("net_ema."):
                continue
            if k.startswith("core."):
                k = k[len("core."):]
            elif k.startswith("net."):
                k = k[len("net."):]
            if "pixel_blocks." in k and marker in k:
                # Split into msa (chunks 0-2) and mlp (chunks 3-5) for the two-Linear PiTBlock to reduce peak VRAM
                p2 = v.shape[0] // (6 * pixel_dim)
                trail = v.shape[1:]  # () for bias, (in_dim,) for weight
                vv = v.view(p2, 6, pixel_dim, *trail)
                base, suffix = k.split(marker)
                out[f"{base}.adaLN_modulation_msa.{suffix}"] = vv[:, 0:3].reshape(3 * p2 * pixel_dim, *trail).contiguous()
                out[f"{base}.adaLN_modulation_mlp.{suffix}"] = vv[:, 3:6].reshape(3 * p2 * pixel_dim, *trail).contiguous()
            else:
                out[k] = v
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(
            comfy.text_encoders.pixeldit.PixelDiTGemma2Tokenizer,
            comfy.text_encoders.pixeldit.PixelDiTGemma2TE,
        )

class PiD(PixelDiTT2I):
    unet_config = {
        "image_model": "pid",
    }

    sampling_settings = {
        "shift": 1.5, # close approximation of the original distill 4 steps [0.999, 0.866, 0.634, 0.342, 0]
    }

    memory_usage_factor = 0.04

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.PiD(self, device=device)

class WAN21_T2V(supported_models_base.BASE):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
    }

    sampling_settings = {
        "shift": 8.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    memory_usage_factor = 0.9

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2222

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.wan.WanT5Tokenizer, comfy.text_encoders.wan.te(**t5_detect))

class WAN21_CausalAR_T2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
        "causal_ar": True,
    }

    sampling_settings = {
        "shift": 5.0,
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.unet_config.pop("causal_ar", None)

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.WAN21_CausalAR(self, device=device)


class WAN21_I2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "i2v",
        "in_dim": 36,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, image_to_video=True, device=device)
        return out

class WAN21_FunControl2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "i2v",
        "in_dim": 48,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, image_to_video=False, device=device)
        return out

class WAN21_Camera(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "camera",
        "in_dim": 32,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)
        return out

class WAN22_Camera(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "camera_2.2",
        "in_dim": 36,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)
        return out

class WAN21_Vace(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "vace",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 1.2 * self.memory_usage_factor

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Vace(self, image_to_video=False, device=device)
        return out

class WAN21_HuMo(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "humo",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_HuMo(self, image_to_video=False, device=device)
        return out

class WAN22_S2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "s2v",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22_S2V(self, device=device)
        return out

class WAN22_Animate(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "animate",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22_Animate(self, device=device)
        return out

class WAN22_T2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
        "out_dim": 48,
    }

    latent_format = latent_formats.Wan22

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22(self, image_to_video=True, device=device)
        return out

class WAN21_FlowRVS(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "flow_rvs",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_FlowRVS(self, image_to_video=True, device=device)
        return out

class WAN21_SCAIL(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "scail",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_SCAIL(self, image_to_video=False, device=device)
        return out


class WAN21_SCAIL2(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "scail2",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_SCAIL2(self, image_to_video=False, device=device)
        return out

class WAN22_WanDancer(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "wandancer",
        "in_dim": 36,
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 1.8

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22_WanDancer(self, image_to_video=True, device=device)
        return out

    def process_unet_state_dict(self, state_dict):
        out_sd = {}
        for k in list(state_dict.keys()):
            # split music_encoder in_proj into q_proj, k_proj, v_proj
            if "music_encoder" in k and "self_attn.in_proj" in k:
                suffix = "weight" if k.endswith("weight") else "bias"
                tensor = state_dict[k]
                d = tensor.shape[0] // 3
                prefix = k.replace(f"in_proj_{suffix}", "")
                out_sd[f"{prefix}q_proj.{suffix}"] = tensor[:d]
                out_sd[f"{prefix}k_proj.{suffix}"] = tensor[d:2*d]
                out_sd[f"{prefix}v_proj.{suffix}"] = tensor[2*d:]
            else:
                out_sd[k] = state_dict[k]
        return out_sd

class Hunyuan3Dv2(supported_models_base.BASE):
    unet_config = {
        "image_model": "hunyuan3d2",
    }

    unet_extra_config = {}

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.0,
    }

    memory_usage_factor = 3.5

    clip_vision_prefix = "conditioner.main_image_encoder.model."
    vae_key_prefix = ["vae."]

    latent_format = latent_formats.Hunyuan3Dv2

    def process_unet_state_dict(self, state_dict):
        out_sd = {}
        for k in list(state_dict.keys()):
            key_out = k
            if key_out.endswith(".scale"):
                key_out = "{}.weight".format(key_out[:-len(".scale")])
            out_sd[key_out] = state_dict[k]
        return out_sd

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Hunyuan3Dv2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None

class Hunyuan3Dv2_1(Hunyuan3Dv2):
    unet_config = {
        "image_model": "hunyuan3d2_1",
    }

    latent_format = latent_formats.Hunyuan3Dv2_1

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Hunyuan3Dv2_1(self, device = device)
        return out

class Hunyuan3Dv2mini(Hunyuan3Dv2):
    unet_config = {
        "image_model": "hunyuan3d2",
        "depth": 8,
    }

    latent_format = latent_formats.Hunyuan3Dv2mini

class TripoSplat(supported_models_base.BASE):
    # Image -> 3D gaussian splat flow denoiser
    unet_config = {
        "image_model": "triposplat",
    }

    unet_extra_config = {}

    sampling_settings = {
        "shift": 3.0,
    }

    memory_usage_factor = 0.6

    latent_format = latent_formats.TripoSplat

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.TripoSplat(self, device=device)

    def clip_target(self, state_dict={}):
        return None

class HiDream(supported_models_base.BASE):
    unet_config = {
        "image_model": "hidream",
    }

    sampling_settings = {
        "shift": 3.0,
    }

    sampling_settings = {
    }

    # memory_usage_factor = 1.2 # TODO

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HiDream(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None #  TODO

class HiDreamO1(supported_models_base.BASE):
    unet_config = {
        "image_model": "hidream_o1",
    }

    sampling_settings = {
        "shift": 3.0,
        "noise_scale": 8.0,
    }

    latent_format = latent_formats.HiDreamO1Pixel
    memory_usage_factor = 0.033
    # fp16 not supported: LM MLP down_proj activations fp16 overflow, causing NaNs
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    optimizations = {"fp8": False}

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.HiDreamO1(self, device=device)

    def process_unet_state_dict(self, state_dict):
        # Drop unused Qwen3-VL deepstack merger weights; upstream discards them at inference.
        for key in list(state_dict.keys()):
            if "visual.deepstack_merger_list" in key:
                del state_dict[key]
        return state_dict

    def process_vae_state_dict(self, state_dict):
        # Pixel-space model: inject sentinel so VAE construction picks PixelspaceConversionVAE.
        return {"pixel_space_vae": torch.tensor(1.0)}

    def process_clip_state_dict(self, state_dict):
        # Tokenizer-only TE: inject sentinel so load_state_dict_guess_config triggers CLIP init.
        return {"_hidream_o1_te_sentinel": torch.zeros(1)}

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(
            comfy.text_encoders.hidream_o1.HiDreamO1Tokenizer,
            comfy.text_encoders.hidream_o1.HiDreamO1TE,
        )

class Chroma(supported_models_base.BASE):
    unet_config = {
        "image_model": "chroma",
    }

    unet_extra_config = {
    }

    sampling_settings = {
        "multiplier": 1.0,
    }

    latent_format = comfy.latent_formats.Flux

    memory_usage_factor = 3.2

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def process_unet_state_dict(self, state_dict):
        out_sd = {}
        for k in list(state_dict.keys()):
            key_out = k
            if key_out.endswith(".scale"):
                key_out = "{}.weight".format(key_out[:-len(".scale")])
            out_sd[key_out] = state_dict[k]
        return out_sd

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Chroma(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.pixart_t5.PixArtTokenizer, comfy.text_encoders.pixart_t5.pixart_te(**t5_detect))

class ChromaRadiance(Chroma):
    unet_config = {
        "image_model": "chroma_radiance",
    }

    latent_format = comfy.latent_formats.ChromaRadiance

    # Pixel-space model, no spatial compression for model input.
    memory_usage_factor = 0.044

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.ChromaRadiance(self, device=device)

class ACEStep(supported_models_base.BASE):
    unet_config = {
        "audio_model": "ace",
    }

    unet_extra_config = {
    }

    sampling_settings = {
        "shift": 3.0,
    }

    latent_format = comfy.latent_formats.ACEAudio

    memory_usage_factor = 0.5

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.ACEStep(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.ace.AceT5Tokenizer, comfy.text_encoders.ace.AceT5Model)

class Omnigen2(supported_models_base.BASE):
    unet_config = {
        "image_model": "omnigen2",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 2.6,
    }

    memory_usage_factor = 1.95 #TODO

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        if comfy.model_management.extended_fp16_support():
            self.supported_inference_dtypes = [torch.float16] + self.supported_inference_dtypes

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Omnigen2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_3b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.omnigen2.Omnigen2Tokenizer, comfy.text_encoders.omnigen2.te(**hunyuan_detect))

class Boogu(Omnigen2):
    unet_config = {
        "image_model": "boogu",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.16,
    }

    memory_usage_factor = 2.15

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Boogu(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_8b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.boogu.BooguTokenizer, comfy.text_encoders.boogu.te(**hunyuan_detect))

class Ideogram4(supported_models_base.BASE):
    unet_config = {
        "image_model": "ideogram4",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.0,
    }

    memory_usage_factor = 11.6

    unet_extra_config = {
        "num_attention_heads": 18,
        "attention_head_dim": 256,
        "intermediate_size": 12288,
        "adaln_dim": 512,
        "llm_features_dim": 53248,
        "rope_theta": 5000000,
        "mrope_section": [24, 20, 20],
        "norm_eps": 1e-5,
    }
    latent_format = latent_formats.Flux2

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Ideogram4(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_8b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.ideogram4.Ideogram4Tokenizer, comfy.text_encoders.ideogram4.te(**hunyuan_detect))


class Krea2(supported_models_base.BASE):
    unet_config = {
        "image_model": "krea2",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.15,
    }

    memory_usage_factor = 2.2

    latent_format = latent_formats.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Krea2(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3vl_4b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.krea2.Krea2Tokenizer, comfy.text_encoders.krea2.te(**hunyuan_detect))

class QwenImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "qwen_image",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.15,
    }

    memory_usage_factor = 1.8 #TODO

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.QwenImage(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.qwen_image.QwenImageTokenizer, comfy.text_encoders.qwen_image.te(**hunyuan_detect))

class HunyuanImage21(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "vec_in_dim": None,
    }

    sampling_settings = {
        "shift": 5.0,
    }

    latent_format = latent_formats.HunyuanImage21

    memory_usage_factor = 8.7

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanImage21(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_image.HunyuanImageTokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))

class HunyuanImage21Refiner(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "patch_size": [1, 1, 1],
        "vec_in_dim": None,
    }

    sampling_settings = {
        "shift": 4.0,
    }

    latent_format = latent_formats.HunyuanImage21Refiner

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanImage21Refiner(self, device=device)
        return out

class HunyuanVideo15(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "vision_in_dim": 1152,
    }

    sampling_settings = {
        "shift": 7.0,
    }
    memory_usage_factor = 4.0 #TODO
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    latent_format = latent_formats.HunyuanVideo15

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanVideo15(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))


class HunyuanVideo15_SR_Distilled(HunyuanVideo):
    unet_config = {
        "image_model": "hunyuan_video",
        "vision_in_dim": 1152,
        "in_channels": 98,
    }

    sampling_settings = {
        "shift": 2.0,
    }
    memory_usage_factor = 4.0 #TODO
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    latent_format = latent_formats.HunyuanVideo15

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.HunyuanVideo15_SR_Distilled(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer, comfy.text_encoders.hunyuan_image.te(**hunyuan_detect))


class Kandinsky5(supported_models_base.BASE):
    unet_config = {
        "image_model": "kandinsky5",
    }

    sampling_settings = {
        "shift": 10.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.HunyuanVideo

    memory_usage_factor = 1.25 #TODO

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Kandinsky5(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.kandinsky5.Kandinsky5Tokenizer, comfy.text_encoders.kandinsky5.te(**hunyuan_detect))


class Kandinsky5Image(Kandinsky5):
    unet_config = {
        "image_model": "kandinsky5",
        "model_dim": 2560,
        "visual_embed_dim": 64,
    }

    sampling_settings = {
        "shift": 3.0,
    }

    latent_format = latent_formats.Flux
    memory_usage_factor = 1.25 #TODO

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Kandinsky5Image(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.kandinsky5.Kandinsky5TokenizerImage, comfy.text_encoders.kandinsky5.te(**hunyuan_detect))


class ACEStep15(supported_models_base.BASE):
    unet_config = {
        "audio_model": "ace1.5",
    }

    unet_extra_config = {
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    latent_format = comfy.latent_formats.ACEAudio15

    memory_usage_factor = 4.7

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.ACEStep15(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        detect_2b = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_2b.transformer.".format(pref))
        detect_4b = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen3_4b.transformer.".format(pref))
        if "dtype_llama" in detect_2b:
            detect = detect_2b
            detect["lm_model"] = "qwen3_2b"
        elif "dtype_llama" in detect_4b:
            detect = detect_4b
            detect["lm_model"] = "qwen3_4b"

        return supported_models_base.ClipTarget(comfy.text_encoders.ace15.ACE15Tokenizer, comfy.text_encoders.ace15.te(**detect))


class LongCatImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": False,
        "vec_in_dim": None,
        "context_in_dim": 3584,
        "txt_ids_dims": [1, 2],
    }

    sampling_settings = {
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux

    memory_usage_factor = 2.5

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.LongCatImage(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}qwen25_7b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.longcat_image.LongCatImageTokenizer, comfy.text_encoders.longcat_image.te(**hunyuan_detect))


class RT_DETR_v4(supported_models_base.BASE):
    unet_config = {
        "image_model": "RT_DETR_v4",
    }

    supported_inference_dtypes = [torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.RT_DETR_v4(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None


class DepthAnything3(supported_models_base.BASE):
    unet_config = {
        "image_model": "DepthAnything3",
    }

    # Mono path: no num_heads / num_head_channels needed.
    unet_extra_config = {}

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.DepthAnything3(self, device=device)

    def clip_target(self, state_dict={}):
        return None


class ErnieImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "ernie",
    }

    sampling_settings = {
        "multiplier": 1000.0,
        "shift": 3.0,
    }

    memory_usage_factor = 10.0

    unet_extra_config = {}
    latent_format = latent_formats.Flux2

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.ErnieImage(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect(state_dict, "{}ministral3_3b.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.ernie.ErnieTokenizer, comfy.text_encoders.ernie.te(**hunyuan_detect))


class SAM3(supported_models_base.BASE):
    unet_config = {"image_model": "SAM3"}
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    text_encoder_key_prefix = ["detector.backbone.language_backbone."]
    unet_extra_prefix = ""

    def process_clip_state_dict(self, state_dict):
        clip_keys = getattr(self, "_clip_stash", {})
        clip_keys = utils.state_dict_prefix_replace(clip_keys, {"detector.backbone.language_backbone.": "", "backbone.language_backbone.": ""}, filter_keys=True)
        clip_keys = utils.clip_text_transformers_convert(clip_keys, "encoder.", "sam3_clip.transformer.")
        return {k: v for k, v in clip_keys.items() if not k.startswith("resizer.")}

    def process_unet_state_dict(self, state_dict):
        self._clip_stash = {k: state_dict.pop(k) for k in list(state_dict.keys()) if "language_backbone" in k and "resizer" not in k}
        # SAM3.1: remap tracker.model.* -> tracker.*
        for k in list(state_dict.keys()):
            if k.startswith("tracker.model."):
                state_dict["tracker." + k[len("tracker.model."):]] = state_dict.pop(k)
        # SAM3.1: remove per-block freqs_cis buffers (computed dynamically)
        for k in [k for k in list(state_dict.keys()) if ".attn.freqs_cis" in k]:
            state_dict.pop(k)
        # Split fused QKV projections
        for k in [k for k in list(state_dict.keys()) if k.endswith((".in_proj_weight", ".in_proj_bias"))]:
            t = state_dict.pop(k)
            base, suffix = k.rsplit(".in_proj_", 1)
            s = ".weight" if suffix == "weight" else ".bias"
            d = t.shape[0] // 3
            state_dict[base + ".q_proj" + s] = t[:d]
            state_dict[base + ".k_proj" + s] = t[d:2*d]
            state_dict[base + ".v_proj" + s] = t[2*d:]
        # Remap tracker SAM decoder transformer key names to match sam.py TwoWayTransformer
        for k in list(state_dict.keys()):
            if "sam_mask_decoder.transformer." not in k:
                continue
            new_k = k.replace(".mlp.lin1.", ".mlp.0.").replace(".mlp.lin2.", ".mlp.2.").replace(".norm_final_attn.", ".norm_final.")
            if new_k != k:
                state_dict[new_k] = state_dict.pop(k)
        return state_dict

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.SAM3(self, device=device)

    def clip_target(self, state_dict={}):
        import comfy.text_encoders.sam3_clip
        return supported_models_base.ClipTarget(comfy.text_encoders.sam3_clip.SAM3TokenizerWrapper, comfy.text_encoders.sam3_clip.SAM3ClipModelWrapper)


class SAM31(SAM3):
    unet_config = {"image_model": "SAM31"}


class CogVideoX_T2V(supported_models_base.BASE):
    unet_config = {
        "image_model": "cogvideox",
    }

    sampling_settings = {
        "linear_start": 0.00085,
        "linear_end": 0.012,
        "beta_schedule": "linear",
        "zsnr": True,
    }

    unet_extra_config = {}
    latent_format = latent_formats.CogVideoX

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        # 2b-class (dim=1920, heads=30) uses scale_factor=1.15258426.
        # 5b-class (dim=3072, heads=48) — incl. CogVideoX-5b, 1.5-5B, and
        # Fun-V1.5 inpainting — uses scale_factor=0.7 per vae/config.json.
        if unet_config.get("num_attention_heads", 0) >= 48:
            self.latent_format = latent_formats.CogVideoX1_5
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        # CogVideoX 1.5 (patch_size_t=2) has different training base dimensions for RoPE
        if self.unet_config.get("patch_size_t") is not None:
            self.unet_config.setdefault("sample_height", 96)
            self.unet_config.setdefault("sample_width", 170)
            self.unet_config.setdefault("sample_frames", 81)
        out = model_base.CogVideoX(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return supported_models_base.ClipTarget(comfy.text_encoders.cogvideo.CogVideoXT5Tokenizer, comfy.text_encoders.sd3_clip.T5XXLModel)

class CogVideoX_I2V(CogVideoX_T2V):
    unet_config = {
        "image_model": "cogvideox",
        "in_channels": 32,
    }

    def get_model(self, state_dict, prefix="", device=None):
        if self.unet_config.get("patch_size_t") is not None:
            self.unet_config.setdefault("sample_height", 96)
            self.unet_config.setdefault("sample_width", 170)
            self.unet_config.setdefault("sample_frames", 81)
        out = model_base.CogVideoX(self, image_to_video=True, device=device)
        return out

class CogVideoX_Inpaint(CogVideoX_T2V):
    unet_config = {
        "image_model": "cogvideox",
        "in_channels": 48,
    }

    def get_model(self, state_dict, prefix="", device=None):
        if self.unet_config.get("patch_size_t") is not None:
            self.unet_config.setdefault("sample_height", 96)
            self.unet_config.setdefault("sample_width", 170)
            self.unet_config.setdefault("sample_frames", 81)
        out = model_base.CogVideoX(self, image_to_video=True, device=device)
        return out


models = [
    LotusD,
    Stable_Zero123,
    SD15_instructpix2pix,
    SD15,
    SD20,
    SD21UnclipL,
    SD21UnclipH,
    SDXL_instructpix2pix,
    SDXLRefiner,
    SDXL,
    SSD1B,
    KOALA_700M,
    KOALA_1B,
    Segmind_Vega,
    SD_X4Upscaler,
    Stable_Cascade_C,
    Stable_Cascade_B,
    SV3D_u,
    SV3D_p,
    SD3,
    StableAudio3,
    StableAudio,
    AuraFlow,
    PixArtAlpha,
    PixArtSigma,
    HunyuanDiT,
    HunyuanDiT1,
    FluxInpaint,
    Flux,
    LongCatImage,
    FluxSchnell,
    GenmoMochi,
    LTXV,
    LTXAV,
    HunyuanVideo15_SR_Distilled,
    HunyuanVideo15,
    HunyuanImage21Refiner,
    HunyuanImage21,
    HunyuanVideoSkyreelsI2V,
    HunyuanVideoI2V,
    HunyuanVideo,
    CosmosT2V,
    CosmosI2V,
    CosmosT2IPredict2,
    CosmosI2VPredict2,
    ZImagePixelSpace,
    ZImage,
    PiD,
    PixelDiTT2I,
    Lumina2,
    WAN22_T2V,
    WAN21_CausalAR_T2V,
    WAN21_T2V,
    WAN21_I2V,
    WAN21_FunControl2V,
    WAN21_Vace,
    WAN21_Camera,
    WAN22_Camera,
    WAN22_S2V,
    WAN21_HuMo,
    WAN22_Animate,
    WAN21_FlowRVS,
    WAN21_SCAIL,
    WAN21_SCAIL2,
    WAN22_WanDancer,
    Hunyuan3Dv2mini,
    Hunyuan3Dv2,
    Hunyuan3Dv2_1,
    TripoSplat,
    HiDream,
    HiDreamO1,
    Chroma,
    ChromaRadiance,
    ACEStep,
    ACEStep15,
    Omnigen2,
    Boogu,
    QwenImage,
    Ideogram4,
    Krea2,
    Flux2,
    Lens,
    Kandinsky5Image,
    Kandinsky5,
    Anima,
    RT_DETR_v4,
    ErnieImage,
    SAM3,
    SAM31,
    CogVideoX_Inpaint,
    CogVideoX_I2V,
    CogVideoX_T2V,
    SVD_img2vid,
    DepthAnything3,
]
