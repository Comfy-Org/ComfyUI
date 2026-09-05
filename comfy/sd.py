import json
import torch
from enum import Enum
import logging

from comfy import model_management
from comfy.utils import ProgressBar
from .ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
from .ldm.cascade.stage_a import StageA
from .ldm.cascade.stage_c_coder import StageC_coder
from .ldm.audio.autoencoder import AudioOobleckVAE
import comfy.ldm.genmo.vae.model
import comfy.ldm.lightricks.vae.causal_video_autoencoder
import comfy.ldm.lightricks.vae.na_diffusion_decoder
import comfy.ldm.lightricks.vae.audio_vae
import comfy.ldm.cosmos.vae
import comfy.ldm.wan.vae
import comfy.ldm.trellis2.vae
import comfy.ldm.wan.vae2_2
import comfy.ldm.hunyuan3d.vae
import comfy.ldm.seedvr.vae
import comfy.ldm.mage_flow.vae
import comfy.ldm.triposplat.vae
import comfy.ldm.ace.vae.music_dcae_pipeline
import comfy.ldm.cogvideo.vae
import comfy.ldm.hunyuan_video.vae
import comfy.ldm.mmaudio.vae.autoencoder
import comfy.ldm.audio.vae_sa3
import comfy.ldm.minimax_music.dav
import comfy.pixel_space_convert
import comfy.weight_adapter
import yaml
import math
import os
from tqdm import tqdm

import comfy.utils
import comfy.ops

from . import clip_vision
from . import gligen
from . import diffusers_convert
from . import model_detection

from . import sd1_clip
from . import sdxl_clip
import comfy.text_encoders.sd2_clip
import comfy.text_encoders.sd3_clip
import comfy.text_encoders.sa_t5
import comfy.text_encoders.aura_t5
import comfy.text_encoders.pixart_t5
import comfy.text_encoders.hydit
import comfy.text_encoders.flux
import comfy.text_encoders.long_clipl
import comfy.text_encoders.genmo
import comfy.text_encoders.lt
import comfy.text_encoders.hunyuan_video
import comfy.text_encoders.cosmos
import comfy.text_encoders.lumina2
import comfy.text_encoders.pixeldit
import comfy.text_encoders.wan
import comfy.text_encoders.hidream
import comfy.text_encoders.ace
import comfy.text_encoders.omnigen2
import comfy.text_encoders.qwen_image
import comfy.text_encoders.hunyuan_image
import comfy.text_encoders.z_image
import comfy.text_encoders.krea2
import comfy.text_encoders.mage_flow
import comfy.text_encoders.ideogram4
import comfy.text_encoders.ovis
import comfy.text_encoders.kandinsky5
import comfy.text_encoders.jina_clip_2
import comfy.text_encoders.newbie
import comfy.text_encoders.anima
import comfy.text_encoders.ace15
import comfy.text_encoders.longcat_image
import comfy.text_encoders.qwen35
import comfy.text_encoders.qwen3vl
import comfy.text_encoders.minimax
import comfy.text_encoders.minimax_music
import comfy.ldm.minimax.vae
import comfy.ldm.minimax.audio_vae
import comfy.text_encoders.boogu
import comfy.text_encoders.ernie
import comfy.text_encoders.gemma4
import comfy.text_encoders.cogvideo
import comfy.text_encoders.sa3
import comfy.text_encoders.gpt_oss
import comfy.text_encoders.joyimage

import comfy.model_patcher
import comfy.lora
import comfy.lora_convert
import comfy.hooks
import comfy.t2i_adapter.adapter
import comfy.taesd.taesd
import comfy.taesd.taehv
import comfy.latent_formats

import comfy.ldm.flux.redux

def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
        if lora_metadata:
            new_modelpatcher.set_attachments("lora_metadata", lora_metadata)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
        if lora_metadata:
            new_clip.patcher.set_attachments("lora_metadata", lora_metadata)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning("NOT LOADED {}".format(x))

    return (new_modelpatcher, new_clip)


def load_bypass_lora_for_models(model, clip, lora, strength_model, strength_clip):
    """
    Load LoRA in bypass mode without modifying base model weights.

    Instead of patching weights, this injects the LoRA computation into the
    forward pass: output = base_forward(x) + lora_path(x)

    Non-adapter patches (bias diff, weight diff, etc.) are applied as regular patches.

    This is useful for training and when model weights are offloaded.
    """
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    logging.debug(f"[BypassLoRA] key_map has {len(key_map)} entries")

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)

    logging.debug(f"[BypassLoRA] loaded has {len(loaded)} entries")

    # Separate adapters (for bypass) from other patches (for regular patching)
    bypass_patches = {}  # WeightAdapterBase instances -> bypass mode
    regular_patches = {}  # diff, set, bias patches -> regular weight patching

    for key, patch_data in loaded.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            bypass_patches[key] = patch_data
        else:
            regular_patches[key] = patch_data

    logging.debug(f"[BypassLoRA] {len(bypass_patches)} bypass adapters, {len(regular_patches)} regular patches")

    k = set()
    k1 = set()

    if model is not None:
        new_modelpatcher = model.clone()

        # Apply regular patches (bias diff, weight diff, etc.) via normal patching
        if regular_patches:
            patched_keys = new_modelpatcher.add_patches(regular_patches, strength_model)
            k.update(patched_keys)

        # Apply adapter patches via bypass injection
        manager = comfy.weight_adapter.BypassInjectionManager()
        model_sd_keys = set(new_modelpatcher.model.state_dict().keys())

        for key, adapter in bypass_patches.items():
            if key in model_sd_keys:
                manager.add_adapter(key, adapter, strength=strength_model)
                k.add(key)
            else:
                logging.warning(f"[BypassLoRA] Adapter key not in model state_dict: {key}")

        injections = manager.create_injections(new_modelpatcher.model)

        if manager.get_hook_count() > 0:
            new_modelpatcher.set_injections("bypass_lora", injections)
    else:
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()

        # Apply regular patches to clip
        if regular_patches:
            patched_keys = new_clip.add_patches(regular_patches, strength_clip)
            k1.update(patched_keys)

        # Apply adapter patches via bypass injection
        clip_manager = comfy.weight_adapter.BypassInjectionManager()
        clip_sd_keys = set(new_clip.cond_stage_model.state_dict().keys())

        for key, adapter in bypass_patches.items():
            if key in clip_sd_keys:
                clip_manager.add_adapter(key, adapter, strength=strength_clip)
                k1.add(key)

        clip_injections = clip_manager.create_injections(new_clip.cond_stage_model)
        if clip_manager.get_hook_count() > 0:
            new_clip.patcher.set_injections("bypass_lora", clip_injections)
    else:
        new_clip = None

    for x in loaded:
        if (x not in k) and (x not in k1):
            patch_data = loaded[x]
            patch_type = type(patch_data).__name__
            if isinstance(patch_data, tuple):
                patch_type = f"tuple({patch_data[0]})"
            logging.warning(f"NOT LOADED: {x} (type={patch_type})")

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False, tokenizer_data={}, parameters=0, state_dict=[], model_options={}, disable_dynamic=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", model_management.text_encoder_device())
        offload_device = model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management.text_encoder_dtype(load_device)

        params['dtype'] = dtype
        params['device'] = model_options.get("initial_device", model_management.text_encoder_initial_device(load_device, offload_device, parameters * model_management.dtype_size(dtype)))
        params['model_options'] = model_options

        self.cond_stage_model = clip(**(params))

        for dt in self.cond_stage_model.dtypes:
            if not model_management.supports_cast(load_device, dt):
                load_device = offload_device
                if params['device'] != offload_device:
                    self.cond_stage_model.to(offload_device)
                    logging.warning("Had to shift TE back.")

        model_management.archive_model_dtypes(self.cond_stage_model)

        self.tokenizer = tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        te_disable_dynamic = disable_dynamic or getattr(self.cond_stage_model, "disable_offload", False)
        ModelPatcher = comfy.model_patcher.ModelPatcher if te_disable_dynamic else comfy.model_patcher.CoreModelPatcher
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        #Match torch.float32 hardcode upcast in TE implemention
        self.patcher.set_model_compute_dtype(torch.float32)
        self.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.patcher.is_clip = True
        self.apply_hooks_to_conds = None
        if len(state_dict) > 0:
            if isinstance(state_dict, list):
                for c in state_dict:
                    m, u = self.load_sd(c)
                    if len(m) > 0:
                        logging.warning("clip missing: {}".format(m))

                    if len(u) > 0:
                        logging.debug("clip unexpected: {}".format(u))
            else:
                m, u = self.load_sd(state_dict, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))

        if params['device'] == load_device:
            model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.layer_idx = None
        self.use_clip_schedule = False
        logging.info("CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(load_device, offload_device, params['device'], dtype))
        self.tokenizer_options = {}

    def clone(self, disable_dynamic=False):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone(disable_dynamic=disable_dynamic)
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def set_tokenizer_option(self, option_name, value):
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
        if len(tokenizer_options) > 0:
            kwargs["tokenizer_options"] = tokenizer_options
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def add_hooks_to_dict(self, pooled_dict: dict[str]):
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds
        return pooled_dict

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict[str]={}, show_pbar=True):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
            cond = pooled_dict.pop("cond")
            # add/update any keys with the provided add_dict
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model(tokens)
            device = self.patcher.load_device
            self.cond_stage_model.set_clip_options({"execution_device": device})
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            with model_management.cuda_device_context(device):
                for scheduled_opts in scheduled_keyframes:
                    t_range = scheduled_opts[0]
                    # don't bother encoding any conds outside of start_percent and end_percent bounds
                    if "start_percent" in add_dict:
                        if t_range[1] < add_dict["start_percent"]:
                            continue
                    if "end_percent" in add_dict:
                        if t_range[0] > add_dict["end_percent"]:
                            continue
                    hooks_keyframes = scheduled_opts[1]
                    for hook, keyframe in hooks_keyframes:
                        hook.hook_keyframe._current_keyframe = keyframe
                    # apply appropriate hooks with values that match new hook_keyframe
                    self.patcher.patch_hooks(all_hooks)
                    # perform encoding as normal
                    o = self.cond_stage_model.encode_token_weights(tokens)
                    cond, pooled = o[:2]
                    pooled_dict = {"pooled_output": pooled}
                    # add clip_start_percent and clip_end_percent in pooled
                    pooled_dict["clip_start_percent"] = t_range[0]
                    pooled_dict["clip_end_percent"] = t_range[1]
                    # add/update any keys with the provided add_dict
                    pooled_dict.update(add_dict)
                    # add hooks stored on clip
                    self.add_hooks_to_dict(pooled_dict)
                    all_cond_pooled.append([cond, pooled_dict])
                    if show_pbar:
                        pbar.update(1)
                    model_management.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model(tokens)
        device = self.patcher.load_device
        self.cond_stage_model.set_clip_options({"execution_device": device})

        with model_management.cuda_device_context(device):
            o = self.cond_stage_model.encode_token_weights(tokens)

        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        else:
            can_assign = self.patcher.is_dynamic()
            self.cond_stage_model.can_assign_sd = can_assign

            # The CLIP models are a pretty complex web of wrappers and its
            # a bit of an API change to plumb this all the way through.
            # So spray paint the model with this flag that the loading
            # nn.Module can then inspect for itself.
            for m in self.cond_stage_model.modules():
                m.can_assign_sd = can_assign

            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def state_dict_for_saving(self):
        sd_clip = self.patcher.model_state_dict_for_saving()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self, tokens={}):
        memory_used = 0
        if hasattr(self.cond_stage_model, "memory_estimation_function"):
            memory_used = self.cond_stage_model.memory_estimation_function(tokens, device=self.patcher.load_device)
        model_management.load_models_gpu([self.patcher], memory_required=memory_used)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0, seed=None, presence_penalty=0.0):
        self.cond_stage_model.reset_clip_options()

        self.load_model(tokens)
        device = self.patcher.load_device
        self.cond_stage_model.set_clip_options({"layer": None})
        self.cond_stage_model.set_clip_options({"execution_device": device})

        with model_management.cuda_device_context(device):
            return self.cond_stage_model.generate(tokens, do_sample=do_sample, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, presence_penalty=presence_penalty)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def is_dynamic(self):
        return self.patcher.is_dynamic()

class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        is_seedvr2_vae = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight" in sd
        if not is_seedvr2_vae and 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
            sd = diffusers_convert.convert_vae_state_dict(sd)

        if model_management.is_amd():
            VAE_KL_MEM_RATIO = 2.73
        else:
            VAE_KL_MEM_RATIO = 1.0

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO #These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False
        self.not_video = False
        self.size = None

        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None
        self.crop_input = True
        self.handles_tiling = False
        self.format_encoded = None

        self.audio_sample_rate = 44100

        if config is None:
            if "dec_in_proj.weight" in sd and "decoder.model.0.weight_g" in sd:  # MiniMax Music3 DAV
                self.first_stage_model = comfy.ldm.minimax_music.dav.MiniMaxMusic3DAV(operations=comfy.ops.disable_weight_init)
                self.latent_channels = 128
                self.output_channels = 2
                self.upscale_ratio = 512
                self.downscale_ratio = 512
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float32]
                self.disable_offload = True
                self.memory_used_decode = lambda shape, dtype: (shape[-1] * 512 * 1400 + 800_000_000) * model_management.dtype_size(dtype)
                def _no_encode(*args, **kwargs):
                    raise RuntimeError("MiniMax Music3 DAV cannot encode audio")
                self.memory_used_encode = _no_encode
            elif "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                            encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': encoder_config},
                                                            decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder", 'params': decoder_config})
            elif "taesd_decoder.1.weight" in sd:
                if isinstance(metadata, dict) and "tae_latent_channels" in metadata:
                    self.latent_channels = metadata["tae_latent_channels"]
                else:
                    self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
                self.first_stage_model = comfy.taesd.taesd.TAESD(latent_channels=self.latent_channels)
            elif "vquantizer.codebook.weight" in sd: #VQGan: stage a of stable cascade
                self.first_stage_model = StageA()
                self.downscale_ratio = 4
                self.upscale_ratio = 4
                #TODO
                #self.memory_used_encode
                #self.memory_used_decode
                self.process_input = lambda image: image
                self.process_output = lambda image: image
            elif "backbone.1.0.block.0.1.num_batches_tracked" in sd: #effnet: encoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["encoder.{}".format(k)] = sd[k]
                sd = new_sd
            elif "blocks.11.num_batches_tracked" in sd: #previewer: decoder for stage c latent of stable cascade
                self.first_stage_model = StageC_coder()
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["previewer.{}".format(k)] = sd[k]
                sd = new_sd
            elif "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd: #combined effnet and previewer for stable cascade
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
            elif "shape_dec.blocks.1.16.to_subdiv.weight" in sd: # trellis2 shape vae (struct_dec + shape_dec)
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.memory_used_decode = lambda shape, dtype: (2500 * math.prod(shape[2:])) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (2500 * math.prod(shape[2:])) * model_management.dtype_size(dtype)
                self.first_stage_model = comfy.ldm.trellis2.vae.ShapeVae()
            elif "txt_dec.blocks.3.4.conv2.weight" in sd: # trellis2 texture vae
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.memory_used_decode = lambda shape, dtype: (2500 * math.prod(shape[2:])) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (2500 * math.prod(shape[2:])) * model_management.dtype_size(dtype)
                self.first_stage_model = comfy.ldm.trellis2.vae.TextureVae()
            elif "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight" in sd: # seedvr2
                self.first_stage_model = comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper()
                self.latent_channels = comfy.ldm.seedvr.vae.SEEDVR2_LATENT_CHANNELS
                self.latent_dim = 3
                self.disable_offload = True
                self.memory_used_decode = lambda shape, dtype: self.first_stage_model.comfy_memory_used_decode(shape)
                self.memory_used_encode = lambda shape, dtype: (max(shape[2], 5) * shape[3] * shape[4] * 64) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.handles_tiling = True
                self.format_encoded = self.first_stage_model.comfy_format_encoded
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.process_input = lambda image: image * 2.0 - 1.0
                self.crop_input = False
            elif "student.dconv_encoder.proj_out.weight" in sd:  # Mage-VAE (one-step diffusion codec, Flux2-anchored 128ch/16x latents)
                sd = comfy.utils.state_dict_prefix_replace(sd, {"student.dconv_encoder.": "dconv_encoder.", "pipeline.": "decoder_model."})
                # Drop the unused Flux2-VAE anchor encoder carried in the checkpoint.
                sd = {k: v for k, v in sd.items() if not k.startswith("decoder_model.y_embedder.encoder.") and not k.startswith("decoder_model.y_embedder.bottleneck.")}
                self.first_stage_model = comfy.ldm.mage_flow.vae.MageVAE()
                self.latent_channels = 128
                self.downscale_ratio = 16
                self.upscale_ratio = 16
                self.working_dtypes = [torch.bfloat16, torch.float32]
                self.memory_used_encode = lambda shape, dtype: (400 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * shape[3] * 16 * 16) * model_management.dtype_size(dtype)
            elif "decoder.conv_in_x_t.weight" in sd:  # lightricks LTX 2.4 diffusion VAE decoder
                vae_config = None
                if metadata is not None and "config" in metadata:
                    vae_config = json.loads(metadata["config"]).get("vae", None)
                self.first_stage_model = comfy.ldm.lightricks.vae.na_diffusion_decoder.CausalDiffusionVAE(config=vae_config)
                self.latent_channels = sd["decoder.conv_in.weight"].shape[1]
                self.latent_dim = 3
                self.disable_offload = True
                self.crop_input = False  # generic crop would narrow the frame axis by the 32x spatial ratio
                self.memory_used_decode = lambda shape, dtype: (1700 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (80 * max(shape[2], 7) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                self.upscale_index_formula = (8, 32, 32)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
                self.downscale_index_formula = (8, 32, 32)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.conv_in.weight" in sd:
                if sd['decoder.conv_in.weight'].shape[1] == 64:
                    ddconfig = {"block_out_channels": [128, 256, 512, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 32, "downsample_match_channel": True, "upsample_match_channel": True}
                    self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                    self.downscale_ratio = 32
                    self.upscale_ratio = 32
                    self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                    self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                encoder_config={'target': "comfy.ldm.hunyuan_video.vae.Encoder", 'params': ddconfig},
                                                                decoder_config={'target': "comfy.ldm.hunyuan_video.vae.Decoder", 'params': ddconfig})

                    self.memory_used_encode = lambda shape, dtype: (700 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: (700 * shape[2] * shape[3] * 32 * 32) * model_management.dtype_size(dtype)
                elif sd['decoder.conv_in.weight'].shape[1] == 32 and sd['decoder.conv_in.weight'].ndim == 5:
                    ddconfig = {"block_out_channels": [128, 256, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16, "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "refiner_vae": False}
                    self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                    self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                    self.upscale_index_formula = (4, 16, 16)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                    self.downscale_index_formula = (4, 16, 16)
                    self.latent_dim = 3
                    self.not_video = True
                    self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': ddconfig},
                                                                decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': ddconfig})

                    self.memory_used_encode = lambda shape, dtype: (2800 * shape[-2] * shape[-1]) * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: (2800 * shape[-3] * shape[-2] * shape[-1] * 16 * 16) * model_management.dtype_size(dtype)
                else:
                    #default SD1.x/SD2.x VAE parameters
                    ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

                    if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd: #Stable diffusion x4 upscaler VAE
                        ddconfig['ch_mult'] = [1, 2, 4]
                        self.downscale_ratio = 4
                        self.upscale_ratio = 4

                    self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                    if 'decoder.post_quant_conv.weight' in sd:
                        sd = comfy.utils.state_dict_prefix_replace(sd, {"decoder.post_quant_conv.": "post_quant_conv.", "encoder.quant_conv.": "quant_conv."})

                    if 'bn.running_mean' in sd:
                        ddconfig["batch_norm_latent"] = True
                        self.downscale_ratio *= 2
                        self.upscale_ratio *= 2
                        self.latent_channels *= 4
                        old_memory_used_decode = self.memory_used_decode
                        self.memory_used_decode = lambda shape, dtype: old_memory_used_decode(shape, dtype) *  4.0

                    decoder_ch = sd['decoder.conv_in.weight'].shape[0] // ddconfig['ch_mult'][-1]
                    if decoder_ch != ddconfig['ch']:
                        decoder_ddconfig = ddconfig.copy()
                        decoder_ddconfig['ch'] = decoder_ch
                    else:
                        decoder_ddconfig = None

                    if 'post_quant_conv.weight' in sd:
                        self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1], **({"decoder_ddconfig": decoder_ddconfig} if decoder_ddconfig is not None else {}))
                    else:
                        self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                    encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
                                                                    decoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Decoder", 'params': decoder_ddconfig if decoder_ddconfig is not None else ddconfig})
            elif "decoder.layers.1.layers.0.beta" in sd:
                config = {}
                param_key = None
                self.upscale_ratio = 2048
                self.downscale_ratio = 2048
                if "decoder.layers.2.layers.1.weight_v" in sd:
                    param_key = "decoder.layers.2.layers.1.weight_v"
                if "decoder.layers.2.layers.1.parametrizations.weight.original1" in sd:
                    param_key = "decoder.layers.2.layers.1.parametrizations.weight.original1"
                if param_key is not None:
                    if sd[param_key].shape[-1] == 12:
                        config["strides"] = [2, 4, 4, 6, 10]
                        self.audio_sample_rate = 48000
                        self.upscale_ratio = 1920
                        self.downscale_ratio = 1920

                self.first_stage_model = AudioOobleckVAE(**config)
                self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * 2048) * model_management.dtype_size(dtype)
                self.latent_channels = 64
                self.output_channels = 2
                self.pad_channel_value = "replicate"
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.disable_offload = True
            elif "blocks.2.blocks.3.stack.5.weight" in sd or "decoder.blocks.2.blocks.3.stack.5.weight" in sd or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd: #genmo mochi vae
                if "blocks.2.blocks.3.stack.5.weight" in sd:
                    sd = comfy.utils.state_dict_prefix_replace(sd, {"": "decoder."})
                if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                    sd = comfy.utils.state_dict_prefix_replace(sd, {"": "encoder."})
                self.first_stage_model = comfy.ldm.genmo.vae.model.VideoVAE()
                self.latent_channels = 12
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
                self.upscale_index_formula = (6, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
                self.downscale_index_formula = (6, 8, 8)
                self.working_dtypes = [torch.float16, torch.float32]
            elif "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd: #lightricks ltxv
                tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
                version = 0
                if tensor_conv1.shape[0] == 512:
                    version = 0
                elif tensor_conv1.shape[0] == 1024:
                    version = 1
                    if "encoder.down_blocks.1.conv.conv.bias" in sd:
                        version = 2
                vae_config = None
                if metadata is not None and "config" in metadata:
                    vae_config = json.loads(metadata["config"]).get("vae", None)
                self.first_stage_model = comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(version=version, config=vae_config)
                self.latent_channels = 128
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (1200 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (80 * max(shape[2], 7) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                self.upscale_index_formula = (8, 32, 32)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
                self.downscale_index_formula = (8, 32, 32)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.conv_in.conv.weight" in sd and sd['decoder.conv_in.conv.weight'].shape[1] == 32:
                ddconfig = {"block_out_channels": [128, 256, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16, "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True}
                ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                self.latent_channels = 32
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                self.latent_dim = 3
                self.not_video = False
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.EmptyRegularizer"},
                                                            encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': ddconfig},
                                                            decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': ddconfig})

                self.memory_used_encode = lambda shape, dtype: (1400 * 9 * shape[-2] * shape[-1]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (3600 * 4 * shape[-2] * shape[-1] * 16 * 16) * model_management.dtype_size(dtype)
            elif "decoder.conv_in.conv.weight" in sd and "decoder.mid_block.resnets.0.norm1.norm_layer.weight" in sd:  # CogVideoX VAE
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = sd["encoder.conv_out.conv.weight"].shape[0] // 2
                self.first_stage_model = comfy.ldm.cogvideo.vae.AutoencoderKLCogVideoX(latent_channels=self.latent_channels)
                self.memory_used_decode = lambda shape, dtype: (2800 * max(2, ((shape[2] - 1) * 4) + 1) * shape[3] * shape[4] * (8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (1400 * max(1, shape[2]) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            elif "decoder.conv_in.conv.weight" in sd:
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                ddconfig["conv3d"] = True
                ddconfig["time_compress"] = 4
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                #This is likely to significantly over-estimate with single image or low frame counts as the
                #implementation is able to completely skip caching. Rework if used as an image only VAE
                self.memory_used_decode = lambda shape, dtype: (2800 * min(8, ((shape[2] - 1) * 4) + 1) * shape[3] * shape[4] * (8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (1400 * min(9, shape[2]) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            elif "decoder.unpatcher3d.wavelets" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
                self.upscale_index_formula = (8, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
                self.downscale_index_formula = (8, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {'z_channels': 16, 'latent_channels': self.latent_channels, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3, 'channels': 128, 'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0, 'patch_size': 4, 'num_groups': 1, 'temporal_compression': 8, 'spacial_compression': 8}
                self.first_stage_model = comfy.ldm.cosmos.vae.CausalContinuousVideoTokenizer(**ddconfig)
                #TODO: these values are a bit off because this is not a standard VAE
                self.memory_used_decode = lambda shape, dtype: (50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.middle.0.residual.0.gamma" in sd:
                if "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd:  # Wan 2.2 VAE
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                    self.upscale_index_formula = (4, 16, 16)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                    self.downscale_index_formula = (4, 16, 16)
                    self.latent_dim = 3
                    self.latent_channels = 48
                    ddconfig = {"dim": 160, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "dropout": 0.0}
                    self.first_stage_model = comfy.ldm.wan.vae2_2.WanVAE(**ddconfig)
                    self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                    self.memory_used_encode = lambda shape, dtype: 3300 * shape[3] * shape[4] * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: 8000 * shape[3] * shape[4] * (16 * 16) * model_management.dtype_size(dtype)
                else:  # Wan 2.1 VAE
                    dim = sd["decoder.head.0.gamma"].shape[0]
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                    self.upscale_index_formula = (4, 8, 8)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                    self.downscale_index_formula = (4, 8, 8)
                    self.latent_dim = 3
                    self.latent_channels = 16
                    self.output_channels = sd["encoder.conv1.weight"].shape[1]
                    self.conv_out_channels = sd["decoder.head.2.weight"].shape[0]
                    self.pad_channel_value = 1.0
                    ddconfig = {"dim": dim, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "image_channels": self.output_channels, "conv_out_channels": self.conv_out_channels, "dropout": 0.0}
                    self.first_stage_model = comfy.ldm.wan.vae.WanVAE(**ddconfig)
                    self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                    self.memory_used_encode = lambda shape, dtype: (1500 if shape[2]<=4 else 6000) * shape[3] * shape[4] * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: (2200 if shape[2]<=4 else 7000) * shape[3] * shape[4] * (8*8) * model_management.dtype_size(dtype)


            # Hunyuan 3d v2 2.0 & 2.1
            elif "geo_decoder.cross_attn_decoder.ln_1.bias" in sd:

                self.latent_dim = 1

                def estimate_memory(shape, dtype, num_layers = 16, kv_cache_multiplier = 2):
                    batch, num_tokens, hidden_dim = shape
                    dtype_size = model_management.dtype_size(dtype)

                    total_mem = batch * num_tokens * hidden_dim * dtype_size * (1 + kv_cache_multiplier * num_layers)
                    return total_mem

                # better memory estimations
                self.memory_used_encode = lambda shape, dtype, num_layers = 8, kv_cache_multiplier = 0:\
                    estimate_memory(shape, dtype, num_layers, kv_cache_multiplier)

                self.memory_used_decode = lambda shape, dtype, num_layers = 16, kv_cache_multiplier = 2: \
                    estimate_memory(shape, dtype, num_layers, kv_cache_multiplier)

                self.first_stage_model = comfy.ldm.hunyuan3d.vae.ShapeVAE()
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]


            elif "vocoder.backbone.channel_layers.0.0.bias" in sd: #Ace Step Audio
                self.first_stage_model = comfy.ldm.ace.vae.music_dcae_pipeline.MusicDCAE(source_sample_rate=44100)
                self.memory_used_encode = lambda shape, dtype: (shape[2] * 330) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (shape[2] * shape[3] * 87000) * model_management.dtype_size(dtype)
                self.latent_channels = 8
                self.output_channels = 2
                self.pad_channel_value = "replicate"
                self.upscale_ratio = 4096
                self.downscale_ratio = 4096
                self.latent_dim = 2
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.disable_offload = True
                self.extra_1d_channel = 16
            elif "pixel_space_vae" in sd:
                self.first_stage_model = comfy.pixel_space_convert.PixelspaceConversionVAE()
                self.memory_used_encode = lambda shape, dtype: (1 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                self.downscale_ratio = 1
                self.upscale_ratio = 1
                self.latent_channels = 3
                self.latent_dim = 2
                self.output_channels = 3
                self.disable_offload = True
            elif "vocoder.activation_post.downsample.lowpass.filter" in sd: #MMAudio VAE
                sample_rate = 16000
                if sample_rate == 16000:
                    mode = '16k'
                else:
                    mode = '44k'

                self.first_stage_model = comfy.ldm.mmaudio.vae.autoencoder.AudioAutoencoder(mode=mode)
                self.memory_used_encode = lambda shape, dtype: (30 * shape[2]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (90 * shape[2] * 1411.2) * model_management.dtype_size(dtype)
                self.latent_channels = 20
                self.output_channels = 2
                self.upscale_ratio = 512 * (44100 / sample_rate)
                self.downscale_ratio = 512 * (44100 / sample_rate)
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float32]
                self.crop_input = False
            elif "decoder.22.bias" in sd: # taehv, taew and lighttae
                self.latent_channels = sd["decoder.1.weight"].shape[1]
                self.latent_dim = 3
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                if self.latent_channels == 24 and sd["decoder.22.bias"].shape[0] == 12: # MiniMax H3
                    self.first_stage_model = comfy.taesd.taehv.TAEHV(latent_channels=self.latent_channels, latent_format=None)
                    self.process_input = self.process_output = lambda image: image
                    self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
                    self.downscale_ratio = (lambda a: max(1, (a - 1) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
                    self.memory_used_encode = lambda shape, dtype: (400 * ((shape[-3] + 16) // 17) * shape[-2] * shape[-1] * model_management.dtype_size(dtype))
                    self.memory_used_decode = lambda shape, dtype: ((260 * 16 * 16 + shape[1] * shape[-3]) * shape[-2] * shape[-1] * model_management.dtype_size(dtype))
                elif self.latent_channels in [48, 128]: # Wan 2.2 and LTX2
                    self.first_stage_model = comfy.taesd.taehv.TAEHV(latent_channels=self.latent_channels, latent_format=None) # taehv doesn't need scaling
                    self.process_input = self.process_output = lambda image: image
                    self.process_output = lambda image: image
                    self.memory_used_decode = lambda shape, dtype: (1800 * (max(1, (shape[-3] ** 0.7 * 0.1)) * shape[-2] * shape[-1] * 16 * 16) * model_management.dtype_size(dtype))
                elif self.latent_channels == 32 and sd["decoder.22.bias"].shape[0] == 12: # lighttae_hv15
                    self.first_stage_model = comfy.taesd.taehv.TAEHV(latent_channels=self.latent_channels, latent_format=comfy.latent_formats.HunyuanVideo15)
                    self.memory_used_decode = lambda shape, dtype: (1200 * (max(1, (shape[-3] ** 0.7 * 0.05)) * shape[-2] * shape[-1] * 32 * 32) * model_management.dtype_size(dtype))
                else:
                    if sd["decoder.1.weight"].dtype == torch.float16: # taehv currently only available in float16, so assume it's not lighttaew2_1 as otherwise state dicts are identical
                        latent_format=comfy.latent_formats.HunyuanVideo
                    else:
                        latent_format=None # lighttaew2_1 doesn't need scaling
                    self.first_stage_model = comfy.taesd.taehv.TAEHV(latent_channels=self.latent_channels, latent_format=latent_format)
                    self.process_input = self.process_output = lambda image: image
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                    self.upscale_index_formula = (4, 8, 8)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                    self.downscale_index_formula = (4, 8, 8)
                    self.memory_used_encode = lambda shape, dtype: (700 * (max(1, (shape[-3] ** 0.66 * 0.11)) * shape[-2] * shape[-1]) * model_management.dtype_size(dtype))
                    self.memory_used_decode = lambda shape, dtype: (50 * (max(1, (shape[-3] ** 0.65 * 0.26)) * shape[-2] * shape[-1] * 32 * 32) * model_management.dtype_size(dtype))
            elif "vocoder.resblocks.0.convs1.0.weight" in sd or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd: # LTX Audio
                sd = comfy.utils.state_dict_prefix_replace(sd, {"audio_vae.": "autoencoder."})
                self.first_stage_model = comfy.ldm.lightricks.vae.audio_vae.AudioVAE(metadata=metadata)
                self.memory_used_encode = lambda shape, dtype: (shape[2] * 330) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (shape[2] * shape[3] * 87000) * model_management.dtype_size(dtype)
                self.latent_channels = self.first_stage_model.latent_channels
                self.audio_sample_rate_output = self.first_stage_model.output_sample_rate
                self.autoencoder = self.first_stage_model.autoencoder  # TODO: remove hack for ltxv custom nodes
                self.output_channels = 2
                self.pad_channel_value = "replicate"
                self.upscale_ratio = 4096
                self.downscale_ratio = 4096
                self.latent_dim = 2
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float32]
                self.disable_offload = True
                self.extra_1d_channel = 16
            elif "decoder.layers.3.transformers.0.pre_norm.alpha" in sd:  # Stable Audio 3 VAE
                if "decoder.layers.3.transformers.11.self_attn.to_out.weight" in sd:
                    config = {"channels": 256, "transformer_depths": 12, "sinusoidal_blocks": 8,
                              "sliding_window": [1, 1], "decoder_conv_mapping": False,
                              "chunk_size": 128, "chunk_midpoint_shift": False}
                    self.memory_used_encode = lambda shape, dtype: (1500 * shape[2]) * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: (1500 * shape[2] * 4096) * model_management.dtype_size(dtype)
                else:
                    config = {"channels": 128, "transformer_depths": 6, "sinusoidal_blocks": 0,
                              "sliding_window": None, "decoder_conv_mapping": True,
                              "chunk_size": 32, "chunk_midpoint_shift": True}
                    self.memory_used_encode = lambda shape, dtype: (72 * shape[2]) * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: (72 * shape[2] * 4096) * model_management.dtype_size(dtype)

                self.first_stage_model = comfy.ldm.audio.vae_sa3.SA3AudioVAE(**config)
                self.latent_channels = 256
                self.output_channels = 2
                self.upscale_ratio = 4096
                self.downscale_ratio = 4096
                self.latent_dim = 1
                self.audio_sample_rate = 44100
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                #This VAE has Parameters and Buffers the non-dynamic caster cannot handle
                #Force cast it for --disable-dynamic-vram users until there is a true core fix.
                if not comfy.memory_management.aimdo_enabled:
                    self.disable_offload = True
            elif "decoder.transformer_blocks.0.scale1" in sd and "encoder.down.5.block.0.conv1.weight" in sd:  # MiniMax H3 video VAE
                minimax_ops = comfy.ops.disable_weight_init
                minimax_quant = comfy.utils.detect_layer_quantization(sd, "")
                if minimax_quant is not None:  # int8+convrot quantized decoder
                    minimax_ops = comfy.ops.mixed_precision_ops(minimax_quant, dtype if dtype is not None else torch.float16)
                self.first_stage_model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(operations=minimax_ops)
                self.latent_channels = 24
                self.latent_dim = 3
                # frames 17k+5 <-> latents 5k+2, 16x spatial
                self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
                self.upscale_index_formula = (4, 16, 16)
                self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
                self.downscale_index_formula = (4, 16, 16)
                self.working_dtypes = [torch.float16, torch.float32]
                # the model tiles internally (256px spatial, 17-frame temporal chunks)
                self.handles_tiling = True
                # decode finalizes straight to [0, 1] while streaming chunks out
                self.process_output = lambda image: image
                # one decoded temporal chunk (with overlap) is all that ever sits in VRAM
                chunk_frames = (self.first_stage_model.tokens_chunk_size + self.first_stage_model.token_overlap) * self.first_stage_model.vae_ratio_t

                def estimate_encode_memory(frames, height, width, dtype):
                    fixed = 110_000_000 if frames == 1 else 1_300_000_000
                    elements_per_pixel = 7 if frames == 1 else 9.5
                    # only one clip of the input video is ever resident on the GPU
                    frames = min(frames, self.first_stage_model.clip_length)
                    return (elements_per_pixel * frames * height * width + fixed) * model_management.dtype_size(dtype) * 1.03

                def estimate_decode_memory(frames, height, width, dtype):
                    fixed = 110_000_000 if frames <= 22 else 270_000_000
                    frames = min(frames, chunk_frames + 2)
                    return (9.5 * frames * height * width + fixed) * model_management.dtype_size(dtype) * 1.03

                self.memory_used_encode = lambda shape, dtype: estimate_encode_memory(shape[2], shape[3], shape[4], dtype)
                self.memory_used_decode = lambda shape, dtype: estimate_decode_memory(self.upscale_ratio[0](shape[2]), shape[3] * self.upscale_ratio[1], shape[4] * self.upscale_ratio[2], dtype)
            elif "pre_block.attn.zero_k_bias" in sd:  # MiniMax H3 audio VAE (DAC encoder + BigVGAN decoder)
                self.first_stage_model = comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE()
                self.latent_channels = 32
                self.output_channels = 2
                self.pad_channel_value = "replicate"
                self.audio_sample_rate = 32000
                self.upscale_ratio = 800
                self.downscale_ratio = 800
                self.latent_dim = 2  # [B, 32, stereo 2, T]
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float32]
                # encode gets the waveform shape [B, 2, samples], decode the latent shape [B, 32, 2, T]
                def estimate_encode_memory(samples, dtype):
                    return (900 * samples + 105_000_000) * model_management.dtype_size(dtype) * 1.03

                def estimate_decode_memory(samples, dtype):
                    return max(42_000_000, 220 * samples + 20_000_000) * model_management.dtype_size(dtype) * 1.03

                self.memory_used_encode = lambda shape, dtype: estimate_encode_memory(shape[2], dtype)
                self.memory_used_decode = lambda shape, dtype: estimate_decode_memory(shape[-1] * self.upscale_ratio, dtype)
            elif "gs.base_offset_scale" in sd and "octree.out_proj.weight" in sd:  # TripoSplat octree gaussian decoder
                self.first_stage_model = comfy.ldm.triposplat.vae.OctreeGaussianDecoder()
                self.latent_channels = 16
                self.latent_dim = 1
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                # The generic VAE.encode/decode path isn't used: VAEDecodeTripoSplat calls the gaussian
                # decoder directly (structured GaussianSplat objects, not a tensor and reserves VRAM itself from num_gaussians.
                def _no_generic_io(*args, **kwargs):
                    raise RuntimeError("TripoSplat gaussian decoder: use the 'TripoSplat Decode' (VAEDecodeTripoSplat)")
                self.memory_used_encode = self.memory_used_decode = _no_generic_io
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        model_management.archive_model_dtypes(self.first_stage_model)
        self.output_device = model_management.intermediate_device()

        mp = comfy.model_patcher.CoreModelPatcher
        if self.disable_offload:
            mp = comfy.model_patcher.ModelPatcher
        self.patcher = mp(self.first_stage_model, load_device=self.device, offload_device=offload_device)

        m, u = self.first_stage_model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))
        self.model_size()

    def model_size(self):
        if self.size is not None:
            return self.size
        self.size = comfy.model_management.module_size(self.first_stage_model)
        return self.size

    def throw_exception_if_invalid(self):
        if self.first_stage_model is None:
            raise RuntimeError("ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your checkpoint does not contain a valid VAE.")

    def vae_encode_crop_pixels(self, pixels):
        if self.crop_input:
            downscale_ratio = self.spacial_compression_encode()

            dims = pixels.shape[1:-1]
            for d in range(len(dims)):
                x = (dims[d] // downscale_ratio) * downscale_ratio
                x_offset = (dims[d] % downscale_ratio) // 2
                if x != dims[d]:
                    pixels = pixels.narrow(d + 1, x_offset, x)

        if pixels.shape[-1] > self.output_channels:
            pixels = pixels[..., :self.output_channels]
        elif pixels.shape[-1] < self.output_channels:
            if self.pad_channel_value is not None:
                if isinstance(self.pad_channel_value, str):
                    mode = self.pad_channel_value
                    value = None
                else:
                    mode = "constant"
                    value = self.pad_channel_value

                pixels = torch.nn.functional.pad(pixels, (0, self.output_channels - pixels.shape[-1]), mode=mode, value=value)
        return pixels

    def vae_output_dtype(self):
        return model_management.intermediate_dtype()

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16, term_pbar_desc=None):
        steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = comfy.utils.ProgressBar(steps, term_desc=term_pbar_desc)

        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
        output = self.process_output(
            (comfy.utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = self.upscale_ratio, output_device=self.output_device, pbar = pbar) +
            comfy.utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = self.upscale_ratio, output_device=self.output_device, pbar = pbar) +
             comfy.utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = self.upscale_ratio, output_device=self.output_device, pbar = pbar))
            / 3.0)
        return output

    def decode_tiled_1d(self, samples, tile_x=256, overlap=32, term_pbar_desc=None):
        if samples.ndim == 3:
            decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
        else:
            og_shape = samples.shape
            samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
            decode_fn = lambda a: self.first_stage_model.decode(a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1])).to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())

        return self.process_output(comfy.utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device, term_pbar_desc=term_pbar_desc))

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8), term_pbar_desc=None):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
        return self.process_output(comfy.utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device, term_pbar_desc=term_pbar_desc))

    def _decode_tiled_owned(self, samples, **kwargs):
        out = self.first_stage_model.decode_tiled(samples.to(self.vae_dtype).to(self.device), **kwargs)
        return self.process_output(out.to(device=self.output_device, dtype=self.vae_output_dtype(), copy=True))

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64, term_pbar_desc=None):
        steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = comfy.utils.ProgressBar(steps, term_desc=term_pbar_desc)

        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
        samples = comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount = (1/self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples += comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
        samples /= 3.0
        return samples

    def encode_tiled_1d(self, samples, tile_x=256 * 2048, overlap=64 * 2048, term_pbar_desc=None):
        if self.latent_dim == 1:
            encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
            out_channels = self.latent_channels
            upscale_amount = 1 / self.downscale_ratio
        else:
            extra_channel_size = self.extra_1d_channel
            out_channels = self.latent_channels * extra_channel_size
            tile_x = tile_x // extra_channel_size
            overlap = overlap // extra_channel_size
            upscale_amount = 1 / self.downscale_ratio
            encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).reshape(1, out_channels, -1).to(dtype=self.vae_output_dtype())

        out = comfy.utils.tiled_scale_multidim(samples, encode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=self.output_device, term_pbar_desc=term_pbar_desc)
        if self.latent_dim == 1:
            return out
        else:
            return out.reshape(samples.shape[0], self.latent_channels, extra_channel_size, -1)

    def encode_tiled_3d(self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64), term_pbar_desc=None):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())
        return comfy.utils.tiled_scale_multidim(samples, encode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.downscale_ratio, out_channels=self.latent_channels, downscale=True, index_formulas=self.downscale_index_formula, output_device=self.output_device, term_pbar_desc=term_pbar_desc)

    def _encode_tiled_owned(self, pixel_samples, **kwargs):
        x = self.process_input(pixel_samples).to(self.vae_dtype).to(self.device)
        out = self.first_stage_model.encode_tiled(x, **kwargs)
        return out.to(device=self.output_device, dtype=self.vae_output_dtype())

    def _owned_tiled_args(self, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap
        if tile_t is not None:
            args["tile_t"] = tile_t
        if overlap_t is not None:
            args["overlap_t"] = overlap_t
        return args

    def decode(self, samples_in, vae_options={}, term_pbar_desc="VaeDecode"):
        self.throw_exception_if_invalid()
        pixel_samples = None
        do_tile = False
        if self.latent_dim == 2 and samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]

        with model_management.cuda_device_context(self.device):
            try:
                memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
                model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                free_memory = self.patcher.get_free_memory(self.device)
                batch_number = int(free_memory / memory_used)
                batch_number = max(1, batch_number)

                # Pre-allocate output for VAEs that support direct buffer writes
                preallocated = False
                if getattr(self.first_stage_model, 'comfy_has_chunked_io', False):
                    pixel_samples = torch.empty(self.first_stage_model.decode_output_shape(samples_in.shape), device=self.output_device, dtype=self.vae_output_dtype())
                    preallocated = True

                for x in tqdm(range(0, samples_in.shape[0], batch_number), desc=term_pbar_desc, disable=term_pbar_desc is None):
                    samples = samples_in[x:x + batch_number].to(device=self.device, dtype=self.vae_dtype)
                    if preallocated:
                        self.first_stage_model.decode(samples, output_buffer=pixel_samples[x:x+batch_number], **vae_options)
                    else:
                        out = self.first_stage_model.decode(samples, **vae_options).to(device=self.output_device, dtype=self.vae_output_dtype(), copy=True)
                        if pixel_samples is None:
                            pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device, dtype=self.vae_output_dtype())
                        pixel_samples[x:x+batch_number].copy_(out)
                        del out
                    self.process_output(pixel_samples[x:x+batch_number])
            except Exception as e:
                model_management.raise_non_oom(e)
                logging.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                #NOTE: We don't know what tensors were allocated to stack variables at the time of the
                #exception and the exception itself refs them all until we get out of this except block.
                #So we just set a flag for tiler fallback so that tensor gc can happen once the
                #exception is fully off the books.
                do_tile = True

            if do_tile:
                pixel_samples = None
                comfy.model_management.soft_empty_cache()
                dims = samples_in.ndim - 2
                if dims == 1 or self.extra_1d_channel is not None:
                    pixel_samples = self.decode_tiled_1d(samples_in, term_pbar_desc=term_pbar_desc)
                elif dims == 2:
                    if self.handles_tiling:
                        tile = 256 // self.spacial_compression_decode()
                        overlap = tile // 4
                        pixel_samples = self._decode_tiled_owned(samples_in, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        pixel_samples = self.decode_tiled_(samples_in, term_pbar_desc=term_pbar_desc)
                elif dims == 3:
                    tile = 256 // self.spacial_compression_decode()
                    overlap = tile // 4
                    if self.handles_tiling:
                        memory_used = self.memory_used_decode(self._tile_bounded_shape(samples_in.shape, tile, tile, None), self.vae_dtype)
                        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                        pixel_samples = self._decode_tiled_owned(samples_in, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        # Reserve as much as an untiled decode could use (capped by what the device can provide), then size the tiles to fill that reservation:
                        # shrink the temporal tile until one tile fits, then grow the spatial tile while it still fits.
                        budget = min(memory_used, int(model_management.get_total_memory(self.device) * 0.8))
                        model_management.load_models_gpu([self.patcher], memory_required=budget, force_full_load=self.disable_offload)
                        tile_t = samples_in.shape[2]
                        est = lambda tt, txy: self.memory_used_decode(self._tile_bounded_shape(samples_in.shape, txy, txy, tt), self.vae_dtype)
                        while tile_t > 2 and est(tile_t, tile) > budget:
                            tile_t = -(-tile_t // 2)
                        while tile * 2 <= max(samples_in.shape[3], samples_in.shape[4]) and est(tile_t, tile * 2) <= budget:
                            tile *= 2
                        overlap = tile // 4
                        pixel_samples = self.decode_tiled_3d(samples_in, tile_t=tile_t, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap), term_pbar_desc=term_pbar_desc)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1,-1)
        return pixel_samples

    def prepare_decode(self, sample_shape, memory_required=None):
        """For VAEs whose real decode entry point bypasses decode()"""
        if memory_required is None:
            memory_required = self.memory_used_decode(sample_shape, self.vae_dtype)
        memory_required = max(1, int(memory_required))
        model_management.load_models_gpu([self.patcher], memory_required=memory_required, force_full_load=self.disable_offload)
        free_memory = self.patcher.get_free_memory(self.device)
        return max(1, int(free_memory / memory_required))

    def _tile_bounded_shape(self, shape, tile_x, tile_y, tile_t):
        """Clamp a latent shape to one tile for memory estimates: peak memory of a tiled decode is per-tile. Only caller-provided tile dims are clamped."""
        s = list(shape)
        if len(s) == 5:
            if tile_t is not None:
                s[2] = min(s[2], tile_t)
            if tile_y is not None:
                s[3] = min(s[3], tile_y)
            if tile_x is not None:
                s[4] = min(s[4], tile_x)
        elif len(s) == 4 and self.extra_1d_channel is None:
            if tile_y is not None:
                s[2] = min(s[2], tile_y)
            if tile_x is not None:
                s[3] = min(s[3], tile_x)
        elif tile_x is not None:
            s[-1] = min(s[-1], tile_x)
        return tuple(s)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None, term_pbar_desc="VaeDecode"):
        self.throw_exception_if_invalid()
        memory_used = self.memory_used_decode(self._tile_bounded_shape(samples.shape, tile_x, tile_y, tile_t), self.vae_dtype)
        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
        dims = samples.ndim - 2
        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        with model_management.cuda_device_context(self.device):
            if self.handles_tiling and dims in (2, 3):
                output = self._decode_tiled_owned(samples, **self._owned_tiled_args(tile_x, tile_y, overlap, tile_t, overlap_t))
            elif dims == 1 or self.extra_1d_channel is not None:
                args.pop("tile_y")
                output = self.decode_tiled_1d(samples, **args, term_pbar_desc=term_pbar_desc)
            elif dims == 2:
                output = self.decode_tiled_(samples, **args, term_pbar_desc=term_pbar_desc)
            elif dims == 3:
                if overlap_t is None:
                    args["overlap"] = (1, overlap, overlap)
                else:
                    args["overlap"] = (max(1, overlap_t), overlap, overlap)
                if tile_t is not None:
                    args["tile_t"] = max(2, tile_t)

                output = self.decode_tiled_3d(samples, **args, term_pbar_desc=term_pbar_desc)
        return output.movedim(1, -1)

    def encode(self, pixel_samples, term_pbar_desc="VaeEncode"):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        do_tile = False
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            if not self.not_video:
                pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
            else:
                pixel_samples = pixel_samples.unsqueeze(2)

        with model_management.cuda_device_context(self.device):
            try:
                memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
                model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
                free_memory = self.patcher.get_free_memory(self.device)
                batch_number = int(free_memory / max(1, memory_used))
                batch_number = max(1, batch_number)
                samples = None
                for x in tqdm(range(0, pixel_samples.shape[0], batch_number), desc=term_pbar_desc, disable=term_pbar_desc is None):
                    pixels_in = self.process_input(pixel_samples[x:x + batch_number]).to(self.vae_dtype)
                    if getattr(self.first_stage_model, 'comfy_has_chunked_io', False):
                        out = self.first_stage_model.encode(pixels_in, device=self.device)
                    else:
                        pixels_in = pixels_in.to(self.device)
                        out = self.first_stage_model.encode(pixels_in)
                    out = out.to(self.output_device).to(dtype=self.vae_output_dtype())
                    if samples is None:
                        samples = torch.empty((pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device, dtype=self.vae_output_dtype())
                    samples[x:x + batch_number] = out

            except Exception as e:
                model_management.raise_non_oom(e)
                logging.warning("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
                #NOTE: We don't know what tensors were allocated to stack variables at the time of the
                #exception and the exception itself refs them all until we get out of this except block.
                #So we just set a flag for tiler fallback so that tensor gc can happen once the
                #exception is fully off the books.
                do_tile = True

            if do_tile:
                comfy.model_management.soft_empty_cache()
                if self.latent_dim == 3:
                    tile = 256
                    overlap = tile // 4
                    if self.handles_tiling:
                        samples = self._encode_tiled_owned(pixel_samples, tile_x=tile, tile_y=tile, overlap=overlap)
                    else:
                        samples = self.encode_tiled_3d(pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap), term_pbar_desc=term_pbar_desc)
                elif self.latent_dim == 1 or self.extra_1d_channel is not None:
                    samples = self.encode_tiled_1d(pixel_samples, term_pbar_desc=term_pbar_desc)
                else:
                    samples = self.encode_tiled_(pixel_samples, term_pbar_desc=term_pbar_desc)

        if self.format_encoded is not None:
            samples = self.format_encoded(samples)
        return samples

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None, term_pbar_desc="VaeEncode"):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        dims = self.latent_dim
        pixel_samples = pixel_samples.movedim(-1, 1)
        if dims == 3 and pixel_samples.ndim < 5:
            if not self.not_video:
                pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
            else:
                pixel_samples = pixel_samples.unsqueeze(2)

        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)  # TODO: calculate mem required for tile
        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)

        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        with model_management.cuda_device_context(self.device):
            if dims == 1:
                args.pop("tile_y")
                samples = self.encode_tiled_1d(pixel_samples, **args, term_pbar_desc=term_pbar_desc)
            elif dims == 2:
                samples = self.encode_tiled_(pixel_samples, **args, term_pbar_desc=term_pbar_desc)
            elif dims == 3:
                if self.handles_tiling:
                    samples = self._encode_tiled_owned(pixel_samples, **self._owned_tiled_args(tile_x, tile_y, overlap, tile_t, overlap_t))
                else:
                    if tile_t is not None:
                        tile_t_latent = max(2, self.downscale_ratio[0](tile_t))
                    else:
                        tile_t_latent = 9999
                    args["tile_t"] = self.upscale_ratio[0](tile_t_latent)

                    spatial_overlap = overlap if overlap is not None else 64
                    if overlap_t is None:
                        args["overlap"] = (1, spatial_overlap, spatial_overlap)
                    else:
                        args["overlap"] = (self.upscale_ratio[0](max(1, min(tile_t_latent // 2, self.downscale_ratio[0](overlap_t)))), spatial_overlap, spatial_overlap)
                    maximum = pixel_samples.shape[2]
                    maximum = self.upscale_ratio[0](self.downscale_ratio[0](maximum))

                    samples = self.encode_tiled_3d(pixel_samples[:,:,:maximum], **args, term_pbar_desc=term_pbar_desc)

        if self.format_encoded is not None:
            samples = self.format_encoded(samples)
        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()

    def spacial_compression_decode(self):
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio

    def spacial_compression_encode(self):
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio

    def temporal_compression_decode(self):
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None

    def is_dynamic(self):
        # A VAE built from a state dict with no detectable VAE weights returns early
        # from __init__ ("No VAE weights detected") before self.patcher is assigned.
        patcher = getattr(self, "patcher", None)
        return patcher is not None and patcher.is_dynamic()

class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)


def load_style_model(ckpt_path):
    model_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = comfy.t2i_adapter.adapter.StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8)
    elif "redux_down.weight" in keys:
        model = comfy.ldm.flux.redux.ReduxImageEncoder()
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)

class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16
    OMNIGEN2 = 17
    QWEN_IMAGE = 18
    HUNYUAN_IMAGE = 19
    HUNYUAN_VIDEO_15 = 20
    OVIS = 21
    KANDINSKY5 = 22
    KANDINSKY5_IMAGE = 23
    NEWBIE = 24
    FLUX2 = 25
    LONGCAT_IMAGE = 26
    COGVIDEOX = 27
    LENS = 28
    PIXELDIT = 29
    IDEOGRAM4 = 30
    BOOGU = 31
    KREA2 = 32
    JOYIMAGE = 33
    MAGE = 34
    MINIMAX = 35



def load_clip_model_patcher(ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
    clip = load_clip(ckpt_paths, embedding_directory, clip_type, model_options, disable_dynamic)
    return clip.patcher

def load_clip(ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
    clip_data = []
    for p in ckpt_paths:
        sd, metadata = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
        if model_options.get("custom_operations", None) is None:
            sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
        clip_data.append(sd)
    clip = load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options, disable_dynamic=disable_dynamic)
    clip.patcher.cached_patcher_init = (load_clip_model_patcher, (ckpt_paths, embedding_directory, clip_type, model_options))
    return clip


class TEModel(Enum):
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9
    QWEN25_3B = 10
    QWEN25_7B = 11
    BYT5_SMALL_GLYPH = 12
    GEMMA_3_4B = 13
    MISTRAL3_24B = 14
    MISTRAL3_24B_PRUNED_FLUX2 = 15
    QWEN3_4B = 16
    QWEN3_2B = 17
    GEMMA_3_12B = 18
    JINA_CLIP_2 = 19
    QWEN3_8B = 20
    QWEN3_06B = 21
    GEMMA_3_4B_VISION = 22
    QWEN35_08B = 23
    QWEN35_2B = 24
    QWEN35_4B = 25
    QWEN35_9B = 26
    QWEN35_27B = 27
    MINISTRAL_3_3B = 28
    GEMMA_4_E4B = 29
    GEMMA_4_E2B = 30
    GEMMA_4_31B = 31
    T5_GEMMA = 32
    GPT_OSS_20B = 33
    QWEN3VL_4B = 34
    QWEN3VL_8B = 35
    GEMMA_4_12B = 36
    QWEN3VL_32B = 37


def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "model.encoder.layers.0.mixer.Wqkv.weight" in sd:
        return TEModel.JINA_CLIP_2
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[0] == 10240:
            return TEModel.T5_XXL
        elif weight.shape[0] == 5120:
            return TEModel.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        weight = sd['encoder.block.0.layer.0.SelfAttention.k.weight']
        if weight.shape[0] == 384:
            return TEModel.BYT5_SMALL_GLYPH
        return TEModel.T5_BASE
    if "model.encoder.layers.0.pre_self_attn_layernorm.weight" in sd:
        return TEModel.T5_GEMMA
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        if 'model.layers.59.self_attn.q_norm.weight' in sd:
            return TEModel.GEMMA_4_31B
        # Gemma4 12B Unified: 48 layers, encoder-free; global layers drop v_proj (attention_k_eq_v).
        if 'model.layers.47.self_attn.q_norm.weight' in sd and 'model.layers.5.self_attn.v_proj.weight' not in sd:
            return TEModel.GEMMA_4_12B
        if 'model.layers.41.self_attn.q_norm.weight' in sd and 'model.layers.47.self_attn.q_norm.weight' not in sd:
            return TEModel.GEMMA_4_E4B
        if 'model.layers.34.self_attn.q_norm.weight' in sd and 'model.layers.41.self_attn.q_norm.weight' not in sd:
            return TEModel.GEMMA_4_E2B
        if 'model.layers.47.self_attn.q_norm.weight' in sd:
            return TEModel.GEMMA_3_12B
        if 'model.layers.0.self_attn.q_norm.weight' in sd:
            if 'vision_model.embeddings.patch_embedding.weight' in sd:
                return TEModel.GEMMA_3_4B_VISION
            else:
                return TEModel.GEMMA_3_4B
        return TEModel.GEMMA_2_2B
    # Must precede the Qwen2.5-7B k_proj.bias=512 check (GPT-OSS also has 8*64=512).
    if "layers.0.self_attn.sinks" in sd and "layers.0.mlp.experts.gate_up_proj.weight" in sd:
        return TEModel.GPT_OSS_20B
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        weight = sd['model.layers.0.self_attn.k_proj.bias']
        if weight.shape[0] == 256:
            return TEModel.QWEN25_3B
        if weight.shape[0] == 512:
            return TEModel.QWEN25_7B
    if "model.language_model.layers.0.linear_attn.A_log" in sd and "model.language_model.layers.0.input_layernorm.weight" in sd:
        weight = sd['model.language_model.layers.0.input_layernorm.weight']
        if weight.shape[0] == 1024:
            return TEModel.QWEN35_08B
        if weight.shape[0] == 2560:
            return TEModel.QWEN35_4B
        if weight.shape[0] == 4096:
            return TEModel.QWEN35_9B
        if weight.shape[0] == 5120:
            return TEModel.QWEN35_27B
        return TEModel.QWEN35_2B
    if "model.visual.deepstack_merger_list.0.norm.weight" in sd:  # DeepStack is unique to Qwen3-VL
        return TEModel.QWEN3VL_4B if sd["model.visual.merger.linear_fc2.weight"].shape[0] == 2560 else TEModel.QWEN3VL_8B
    if "visual.deepstack_merger_list.0.norm.weight" in sd and "model.layers.49.self_attn.q_proj.weight" in sd:
        # MiniMax H3 conditioning encoder: Qwen3-VL-32B, truncated to 50 layers
        return TEModel.QWEN3VL_32B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        weight = sd['model.layers.0.post_attention_layernorm.weight']
        if 'model.layers.0.self_attn.q_norm.weight' in sd:
            if weight.shape[0] == 2560:
                return TEModel.QWEN3_4B
            elif weight.shape[0] == 2048:
                return TEModel.QWEN3_2B
            elif weight.shape[0] == 4096:
                return TEModel.QWEN3_8B
            elif weight.shape[0] == 1024:
                return TEModel.QWEN3_06B
        if weight.shape[0] == 5120:
            if "model.layers.39.post_attention_layernorm.weight" in sd:
                return TEModel.MISTRAL3_24B
            else:
                return TEModel.MISTRAL3_24B_PRUNED_FLUX2
        if weight.shape[0] == 3072:
            return TEModel.MINISTRAL_3_3B

        return TEModel.LLAMA3_8
    return None


def t5xxl_detect(clip_data):
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"

    for sd in clip_data:
        if weight_name in sd or weight_name_old in sd:
            return comfy.text_encoders.sd3_clip.t5_xxl_detect(sd)

    return {}

def llama_detect(clip_data):
    weight_names = ["model.layers.0.self_attn.k_proj.weight", "model.layers.0.linear_attn.in_proj_a.weight"]

    for sd in clip_data:
        for weight_name in weight_names:
            if weight_name in sd:
                return comfy.text_encoders.hunyuan_video.llama_detect(sd)

    return {}

def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) #old models saved with the CLIPSave node
        if "lm_head.weight" in clip_data[i]:
            clip_data[i]["model.lm_head.weight"] = clip_data[i].pop("lm_head.weight") # prefix missing in some models

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if clip_type == CLIPType.MINIMAX and "model.audio_decoder.projection.weight" in clip_data[0]:
            tokenizer_data["tokenizer_json"] = clip_data[0].pop("tokenizer_json", None)
            quant = comfy.utils.detect_layer_quantization(clip_data[0], "")
            if quant is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = quant
            clip_target.params["projection_config"] = comfy.text_encoders.minimax_music.detect_merged_config(clip_data[0])
            clip_target.clip = comfy.text_encoders.minimax_music.MiniMaxMusic3TEModel
            clip_target.tokenizer = comfy.text_encoders.minimax_music.MiniMaxMusic3Tokenizer
        elif te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = comfy.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=True, t5=False)
                clip_target.tokenizer = comfy.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = comfy.text_encoders.hidream.hidream_clip(clip_l=False, clip_g=True, t5=False, llama=False, dtype_t5=None, dtype_llama=None)
                clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = comfy.text_encoders.sd2_clip.SD2ClipModel
            clip_target.tokenizer = comfy.text_encoders.sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = comfy.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = comfy.text_encoders.lt.ltxv_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART or clip_type == CLIPType.CHROMA:
                clip_target.clip = comfy.text_encoders.pixart_t5.pixart_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.pixart_t5.PixArtTokenizer
            elif clip_type == CLIPType.WAN:
                clip_target.clip = comfy.text_encoders.wan.te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.wan.WanT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = comfy.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=True, llama=False, dtype_llama=None)
                clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer
            elif clip_type == CLIPType.COGVIDEOX:
                clip_target.clip = comfy.text_encoders.cogvideo.cogvideo_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.cogvideo.CogVideoXTokenizer
            else: #CLIPType.MOCHI
                clip_target.clip = comfy.text_encoders.genmo.mochi_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = comfy.text_encoders.cosmos.te(**t5xxl_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = comfy.text_encoders.aura_t5.AuraT5Model
            clip_target.tokenizer = comfy.text_encoders.aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            if clip_type == CLIPType.ACE or "spiece_model" in clip_data[0]:
                clip_target.clip = comfy.text_encoders.ace.AceT5Model
                clip_target.tokenizer = comfy.text_encoders.ace.AceT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            else:
                clip_target.clip = comfy.text_encoders.sa_t5.SAT5Model
                clip_target.tokenizer = comfy.text_encoders.sa_t5.SAT5Tokenizer
        elif te_model == TEModel.T5_GEMMA:
            clip_target.clip = comfy.text_encoders.sa3.SAT5GemmaModel
            clip_target.tokenizer = comfy.text_encoders.sa3.SAT5GemmaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model in (TEModel.GEMMA_4_E4B, TEModel.GEMMA_4_E2B, TEModel.GEMMA_4_31B, TEModel.GEMMA_4_12B):
            if te_model == TEModel.GEMMA_4_12B and "text_embedding_projection.video_aggregate_embed.weight" in clip_data[0]:
                clip_target.clip = comfy.text_encoders.lt.ltxav_te(
                    **llama_detect(clip_data),
                    **comfy.text_encoders.lt.sd_detect(clip_data),
                    text_encoder_model=comfy.text_encoders.gemma4.gemma4_text_encoder_model(comfy.text_encoders.gemma4.Gemma4_12B),
                    text_encoder_key="gemma4",
                )
                clip_target.tokenizer = comfy.text_encoders.lt.ltxav_gemma4_tokenizer(comfy.text_encoders.gemma4.Gemma4_12B.tokenizer)
            else:
                variant = {TEModel.GEMMA_4_E4B: comfy.text_encoders.gemma4.Gemma4_E4B,
                           TEModel.GEMMA_4_E2B: comfy.text_encoders.gemma4.Gemma4_E2B,
                           TEModel.GEMMA_4_31B: comfy.text_encoders.gemma4.Gemma4_31B,
                           TEModel.GEMMA_4_12B: comfy.text_encoders.gemma4.Gemma4_12B}[te_model]
                clip_target.clip = comfy.text_encoders.gemma4.gemma4_te(**llama_detect(clip_data), model_class=variant)
                clip_target.tokenizer = variant.tokenizer
            tokenizer_data["tokenizer_json"] = clip_data[0].get("tokenizer_json", None)
        elif te_model == TEModel.GEMMA_2_2B:
            if clip_type == CLIPType.PIXELDIT:
                clip_target.clip = comfy.text_encoders.pixeldit.pixeldit_te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.pixeldit.PixelDiTGemma2Tokenizer
            else:
                clip_target.clip = comfy.text_encoders.lumina2.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.GEMMA_3_4B:
            clip_target.clip = comfy.text_encoders.lumina2.te(**llama_detect(clip_data), model_type="gemma3_4b")
            clip_target.tokenizer = comfy.text_encoders.lumina2.NTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.GEMMA_3_4B_VISION:
            clip_target.clip = comfy.text_encoders.lumina2.te(**llama_detect(clip_data), model_type="gemma3_4b_vision")
            clip_target.tokenizer = comfy.text_encoders.lumina2.NTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.GEMMA_3_12B:
            clip_target.clip = comfy.text_encoders.lt.gemma3_te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.lt.Gemma3_12BTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.LLAMA3_8:
            clip_target.clip = comfy.text_encoders.hidream.hidream_clip(**llama_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=False, llama=True, dtype_t5=None)
            clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer
        elif te_model == TEModel.QWEN25_3B:
            clip_target.clip = comfy.text_encoders.omnigen2.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.omnigen2.Omnigen2Tokenizer
        elif te_model == TEModel.QWEN25_7B:
            if clip_type == CLIPType.HUNYUAN_IMAGE:
                clip_target.clip = comfy.text_encoders.hunyuan_image.te(byt5=False, **llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.hunyuan_image.HunyuanImageTokenizer
            elif clip_type == CLIPType.LONGCAT_IMAGE:
                clip_target.clip = comfy.text_encoders.longcat_image.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.longcat_image.LongCatImageTokenizer
            else:
                clip_target.clip = comfy.text_encoders.qwen_image.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.qwen_image.QwenImageTokenizer
        elif te_model == TEModel.MISTRAL3_24B or te_model == TEModel.MISTRAL3_24B_PRUNED_FLUX2:
            clip_target.clip = comfy.text_encoders.flux.flux2_te(**llama_detect(clip_data), pruned=te_model == TEModel.MISTRAL3_24B_PRUNED_FLUX2)
            clip_target.tokenizer = comfy.text_encoders.flux.Flux2Tokenizer
            tokenizer_data["tekken_model"] = clip_data[0].get("tekken_model", None)
        elif te_model == TEModel.GPT_OSS_20B:
            clip_target.clip = comfy.text_encoders.gpt_oss.lens_te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.gpt_oss.LensTokenizer
            tokenizer_data["tokenizer_json"] = clip_data[0].get("tokenizer_json", None)
        elif te_model == TEModel.QWEN3_4B:
            if clip_type == CLIPType.FLUX or clip_type == CLIPType.FLUX2:
                clip_target.clip = comfy.text_encoders.flux.klein_te(**llama_detect(clip_data), model_type="qwen3_4b")
                clip_target.tokenizer = comfy.text_encoders.flux.KleinTokenizer
            else:
                clip_target.clip = comfy.text_encoders.z_image.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.z_image.ZImageTokenizer
        elif te_model == TEModel.QWEN3_2B:
            clip_target.clip = comfy.text_encoders.ovis.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.ovis.OvisTokenizer
        elif te_model == TEModel.QWEN3_8B:
            if clip_type == CLIPType.IDEOGRAM4:
                clip_target.clip = comfy.text_encoders.ideogram4.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.ideogram4.Ideogram4Tokenizer
            else:
                clip_target.clip = comfy.text_encoders.flux.klein_te(**llama_detect(clip_data), model_type="qwen3_8b")
                clip_target.tokenizer = comfy.text_encoders.flux.KleinTokenizer8B
        elif te_model == TEModel.JINA_CLIP_2:
            clip_target.clip = comfy.text_encoders.jina_clip_2.JinaClip2TextModelWrapper
            clip_target.tokenizer = comfy.text_encoders.jina_clip_2.JinaClip2TokenizerWrapper
        elif te_model in (TEModel.QWEN35_08B, TEModel.QWEN35_2B, TEModel.QWEN35_4B, TEModel.QWEN35_9B, TEModel.QWEN35_27B):
            clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
            qwen35_type = {TEModel.QWEN35_08B: "qwen35_08b", TEModel.QWEN35_2B: "qwen35_2b", TEModel.QWEN35_4B: "qwen35_4b", TEModel.QWEN35_9B: "qwen35_9b", TEModel.QWEN35_27B: "qwen35_27b"}[te_model]
            clip_target.clip = comfy.text_encoders.qwen35.te(**llama_detect(clip_data), model_type=qwen35_type)
            clip_target.tokenizer = comfy.text_encoders.qwen35.tokenizer(model_type=qwen35_type)
        elif te_model in (TEModel.QWEN3VL_4B, TEModel.QWEN3VL_8B):
            if clip_type == CLIPType.IDEOGRAM4 and te_model == TEModel.QWEN3VL_8B:  # Ideogram4 reuses the full Qwen3-VL-8B (13-layer tap for conditioning + multimodal generate).
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                clip_target.clip = comfy.text_encoders.ideogram4.te_qwen3vl(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.ideogram4.Ideogram4Qwen3VLTokenizer
            elif clip_type == CLIPType.BOOGU and te_model == TEModel.QWEN3VL_8B:  # Boogu-Image: full Qwen3-VL-8B, last hidden state, no-think template.
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                clip_target.clip = comfy.text_encoders.boogu.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.boogu.BooguTokenizer
            elif clip_type == CLIPType.KREA2 and te_model == TEModel.QWEN3VL_4B:  # Krea2: full Qwen3-VL-4B (12-layer tap for conditioning + multimodal generate).
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                clip_target.clip = comfy.text_encoders.krea2.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.krea2.Krea2Tokenizer
            elif clip_type == CLIPType.MAGE and te_model == TEModel.QWEN3VL_4B:  # Mage-Flow: full Qwen3-VL-4B, last hidden state, Qwen-Image-style templates.
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                clip_target.clip = comfy.text_encoders.mage_flow.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.mage_flow.MageFlowTokenizer
            elif clip_type == CLIPType.JOYIMAGE and te_model == TEModel.QWEN3VL_8B:  # JoyImageEdit: full Qwen3-VL-8B, edit-conditioning template + drop_idx.
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                clip_target.clip = comfy.text_encoders.joyimage.te(**llama_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.joyimage.JoyImageTokenizer
            elif clip_type in (CLIPType.FLUX, CLIPType.FLUX2):  # Flux2 Klein reuses the Qwen3-VL LM (3-layer tap -> 12288); visual unused.
                klein_model_type = "qwen3_8b" if te_model == TEModel.QWEN3VL_8B else "qwen3_4b"
                clip_target.clip = comfy.text_encoders.flux.klein_te(**llama_detect(clip_data), model_type=klein_model_type)
                clip_target.tokenizer = comfy.text_encoders.flux.KleinTokenizer8B if te_model == TEModel.QWEN3VL_8B else comfy.text_encoders.flux.KleinTokenizer
            else:
                clip_data[0] = comfy.utils.state_dict_prefix_replace(clip_data[0], {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
                qwen3vl_type = {TEModel.QWEN3VL_4B: "qwen3vl_4b", TEModel.QWEN3VL_8B: "qwen3vl_8b"}[te_model]
                clip_target.clip = comfy.text_encoders.qwen3vl.te(**llama_detect(clip_data), model_type=qwen3vl_type)
                clip_target.tokenizer = comfy.text_encoders.qwen3vl.tokenizer(model_type=qwen3vl_type)
        elif te_model == TEModel.QWEN3VL_32B:
            clip_target.clip = comfy.text_encoders.minimax.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer
        elif te_model == TEModel.QWEN3_06B:
            clip_target.clip = comfy.text_encoders.anima.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.anima.AnimaTokenizer
        elif te_model == TEModel.MINISTRAL_3_3B:
            clip_target.clip = comfy.text_encoders.ernie.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.ernie.ErnieTokenizer
            tokenizer_data["tekken_model"] = clip_data[0].get("tekken_model", None)
        else:
            # clip_l
            if clip_type == CLIPType.SD3:
                clip_target.clip = comfy.text_encoders.sd3_clip.sd3_clip(clip_l=True, clip_g=False, t5=False)
                clip_target.tokenizer = comfy.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = comfy.text_encoders.hidream.hidream_clip(clip_l=True, clip_g=False, t5=False, llama=False, dtype_t5=None, dtype_llama=None)
                clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = comfy.text_encoders.sd3_clip.sd3_clip(clip_l=TEModel.CLIP_L in te_models, clip_g=TEModel.CLIP_G in te_models, t5=TEModel.T5_XXL in te_models, **t5xxl_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = comfy.text_encoders.hydit.HyditModel
            clip_target.tokenizer = comfy.text_encoders.hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            clip_target.clip = comfy.text_encoders.flux.flux_clip(**t5xxl_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = comfy.text_encoders.hunyuan_video.hunyuan_video_clip(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.hunyuan_video.HunyuanVideoTokenizer
        elif clip_type == CLIPType.HIDREAM:
            # Detect
            hidream_dualclip_classes = []
            for hidream_te in clip_data:
                te_model = detect_te_model(hidream_te)
                hidream_dualclip_classes.append(te_model)

            clip_l = TEModel.CLIP_L in hidream_dualclip_classes
            clip_g = TEModel.CLIP_G in hidream_dualclip_classes
            t5 = TEModel.T5_XXL in hidream_dualclip_classes
            llama = TEModel.LLAMA3_8 in hidream_dualclip_classes

            # Initialize t5xxl_detect and llama_detect kwargs if needed
            t5_kwargs = t5xxl_detect(clip_data) if t5 else {}
            llama_kwargs = llama_detect(clip_data) if llama else {}

            clip_target.clip = comfy.text_encoders.hidream.hidream_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, **t5_kwargs, **llama_kwargs)
            clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer
        elif clip_type == CLIPType.HUNYUAN_IMAGE:
            clip_target.clip = comfy.text_encoders.hunyuan_image.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.hunyuan_image.HunyuanImageTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO_15:
            clip_target.clip = comfy.text_encoders.hunyuan_image.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer
        elif clip_type == CLIPType.KANDINSKY5:
            clip_target.clip = comfy.text_encoders.kandinsky5.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.kandinsky5.Kandinsky5Tokenizer
        elif clip_type == CLIPType.KANDINSKY5_IMAGE:
            clip_target.clip = comfy.text_encoders.kandinsky5.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.kandinsky5.Kandinsky5TokenizerImage
        elif clip_type == CLIPType.LTXV:
            te_models = [detect_te_model(sd) for sd in clip_data]
            gemma4_models = {
                TEModel.GEMMA_4_E4B: comfy.text_encoders.gemma4.Gemma4_E4B,
                TEModel.GEMMA_4_E2B: comfy.text_encoders.gemma4.Gemma4_E2B,
                TEModel.GEMMA_4_31B: comfy.text_encoders.gemma4.Gemma4_31B,
                TEModel.GEMMA_4_12B: comfy.text_encoders.gemma4.Gemma4_12B,
            }
            gemma4_type = next((model for model in te_models if model in gemma4_models), None)
            if gemma4_type is None:
                clip_target.clip = comfy.text_encoders.lt.ltxav_te(**llama_detect(clip_data), **comfy.text_encoders.lt.sd_detect(clip_data))
                clip_target.tokenizer = comfy.text_encoders.lt.LTXAVGemmaTokenizer
                gemma_sd = clip_data[te_models.index(TEModel.GEMMA_3_12B)] if TEModel.GEMMA_3_12B in te_models else clip_data[0]
                tokenizer_data["spiece_model"] = gemma_sd.get("spiece_model", None)
            else:
                variant = gemma4_models[gemma4_type]
                clip_target.clip = comfy.text_encoders.lt.ltxav_te(
                    **llama_detect(clip_data),
                    **comfy.text_encoders.lt.sd_detect(clip_data),
                    text_encoder_model=comfy.text_encoders.gemma4.gemma4_text_encoder_model(variant),
                    text_encoder_key="gemma4",
                )
                clip_target.tokenizer = comfy.text_encoders.lt.ltxav_gemma4_tokenizer(variant.tokenizer)
                gemma_sd = clip_data[te_models.index(gemma4_type)]
                tokenizer_data["tokenizer_json"] = gemma_sd.get("tokenizer_json", None)
        elif clip_type == CLIPType.NEWBIE:
            clip_target.clip = comfy.text_encoders.newbie.te(**llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.newbie.NewBieTokenizer
            if "model.layers.0.self_attn.q_norm.weight" in clip_data[0]:
                clip_data_gemma = clip_data[0]
                clip_data_jina = clip_data[1]
            else:
                clip_data_gemma = clip_data[1]
                clip_data_jina = clip_data[0]
            tokenizer_data["gemma_spiece_model"] = clip_data_gemma.get("spiece_model", None)
            tokenizer_data["jina_spiece_model"] = clip_data_jina.get("spiece_model", None)
        elif clip_type == CLIPType.ACE:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            if TEModel.QWEN3_4B in te_models:
                model_type = "qwen3_4b"
            else:
                model_type = "qwen3_2b"
            clip_target.clip = comfy.text_encoders.ace15.te(lm_model=model_type, **llama_detect(clip_data))
            clip_target.tokenizer = comfy.text_encoders.ace15.ACE15Tokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = comfy.text_encoders.sd3_clip.sd3_clip(**t5xxl_detect(clip_data))
        clip_target.tokenizer = comfy.text_encoders.sd3_clip.SD3Tokenizer
    elif len(clip_data) == 4:
        clip_target.clip = comfy.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data), **llama_detect(clip_data))
        clip_target.tokenizer = comfy.text_encoders.hidream.HiDreamTokenizer

    parameters = 0
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, state_dict=clip_data, model_options=model_options, disable_dynamic=disable_dynamic)
    return clip

def load_gligen(ckpt_path):
    data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model = gligen.load_gligen(data)
    if model_management.should_use_fp16():
        model = model.half()
    return comfy.model_patcher.CoreModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device())

def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
    return ""

def load_checkpoint(config_path=None, ckpt_path=None, output_vae=True, output_clip=True, embedding_directory=None, state_dict=None, config=None):
    logging.warning("Warning: The load checkpoint with config function is deprecated and will eventually be removed, please use the other one.")
    model, clip, vae, _ = load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, output_clipvision=False, embedding_directory=embedding_directory, output_model=True)
    #TODO: this function is a mess and should be removed eventually
    if config is None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            m = model.clone()
            class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.V_PREDICTION):
                pass
            m.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
            model = m

    layer_idx = clip_config.get("params", {}).get("layer_idx", None)
    if layer_idx is not None:
        clip.clip_layer(layer_idx)

    return (model, clip, vae)

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(ckpt_path, model_detection_error_hint(ckpt_path, sd)))
    if out[0] is not None:
        out[0].cached_patcher_init = (load_checkpoint_guess_config, (ckpt_path, False, False, False, embedding_directory, output_model, model_options, te_model_options), 0)
    # Register reload factories for the CLIP and VAE produced by the same checkpoint so
    # ModelPatcher.deepclone_multigpu can spawn per-device copies (Select{CLIP,VAE}Device,
    # MultiGPU work-units, etc.) without falling back to copy.deepcopy of an
    # already-loaded module.
    if out[1] is not None and getattr(out[1], "patcher", None) is not None:
        out[1].patcher.cached_patcher_init = (load_checkpoint_clip_patcher, (ckpt_path, embedding_directory, model_options, te_model_options))
    if out[2] is not None and getattr(out[2], "patcher", None) is not None:
        out[2].patcher.cached_patcher_init = (load_checkpoint_vae_patcher, (ckpt_path, embedding_directory, model_options, te_model_options))
    return out


def load_checkpoint_clip_patcher(ckpt_path, embedding_directory=None, model_options={}, te_model_options={}, disable_dynamic=False):
    """Reload only the CLIP patcher from a checkpoint. Used as the cached_patcher_init
    factory for the CLIP returned by load_checkpoint_guess_config."""
    _, clip, _, _ = load_checkpoint_guess_config(
        ckpt_path,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=False,
        model_options=model_options,
        te_model_options=te_model_options,
        disable_dynamic=disable_dynamic,
    )
    return clip.patcher


def load_checkpoint_vae_patcher(ckpt_path, embedding_directory=None, model_options={}, te_model_options={}, disable_dynamic=False):
    """Reload only the VAE patcher from a checkpoint. Used as the cached_patcher_init
    factory for the VAE returned by load_checkpoint_guess_config."""
    _, _, vae, _ = load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=False,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=False,
        model_options=model_options,
        te_model_options=te_model_options,
        disable_dynamic=disable_dynamic,
    )
    return vae.patcher

def load_checkpoint_guess_config_model_only(ckpt_path, embedding_directory=None, model_options={}, te_model_options={}, disable_dynamic=False):
    model, *_ = load_checkpoint_guess_config(ckpt_path, False, False, False,
            embedding_directory=embedding_directory,
            model_options=model_options,
            te_model_options=te_model_options,
            disable_dynamic=disable_dynamic)
    return model

def load_checkpoint_guess_config_clip_only(ckpt_path, embedding_directory=None, model_options={}, te_model_options={}, disable_dynamic=False):
    _, clip, *_ = load_checkpoint_guess_config(ckpt_path, False, True, False,
            embedding_directory=embedding_directory, output_model=False,
            model_options=model_options,
            te_model_options=te_model_options,
            disable_dynamic=disable_dynamic)
    return clip.patcher

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, metadata=None, disable_dynamic=False):
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_options.get("load_device", model_management.get_torch_device())

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, diffusion_model_prefix, metadata=metadata)

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
        diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
        if diffusion_model is None:
            return None
        return (diffusion_model, None, VAE(sd={}), None)  # The VAE object is there to throw an exception if it's actually used'

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        ModelPatcher = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
        offload_device = model_options.get("offload_device", model_management.unet_offload_device())
        model_patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        model.load_model_weights(sd, diffusion_model_prefix, assign=model_patcher.is_dynamic())

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae_device = model_options.get("load_device", None)
        vae = VAE(sd=vae_sd, metadata=metadata, device=vae_device)

    if output_clip:
        if te_model_options.get("custom_operations", None) is None:
            scaled_fp8_list = []
            for k in list(sd.keys()):  # Convert scaled fp8 to mixed ops
                if k.endswith(".scaled_fp8"):
                    scaled_fp8_list.append(k[:-len("scaled_fp8")])

            if len(scaled_fp8_list) > 0:
                out_sd = {}
                for k in sd:
                    skip = False
                    for pref in scaled_fp8_list:
                        skip = skip or k.startswith(pref)
                    if not skip:
                        out_sd[k] = sd[k]

                for pref in scaled_fp8_list:
                    quant_sd, qmetadata = comfy.utils.convert_old_quants(sd, pref, metadata={})
                    for k in quant_sd:
                        out_sd[k] = quant_sd[k]
                    sd = out_sd

        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, state_dict=clip_sd, model_options=te_model_options, disable_dynamic=disable_dynamic)
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)


def load_diffusion_model_state_dict(sd, model_options={}, metadata=None, disable_dynamic=False):
    """
    Loads a UNet diffusion model from a state dictionary, supporting both diffusers and regular formats.

    Args:
        sd (dict): State dictionary containing model weights and configuration
        model_options (dict, optional): Additional options for model loading. Supports:
            - dtype: Override model data type
            - custom_operations: Custom model operations
            - fp8_optimizations: Enable FP8 optimizations

    Returns:
        ModelPatcher: A wrapped model instance that handles device management and weight loading.
        Returns None if the model configuration cannot be detected.

    The function:
    1. Detects and handles different model formats (regular, diffusers, mmdit)
    2. Configures model dtype based on parameters and device capabilities
    3. Handles weight conversion and device placement
    4. Manages model optimization settings
    5. Loads weights and returns a device-managed model instance
    """
    dtype = model_options.get("dtype", None)

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

    #Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
        if custom_operations is None:
            sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_options.get("load_device", model_management.get_torch_device())
    model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None: #diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else: #diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_options.get("offload_device", model_management.unet_offload_device())
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    ModelPatcher = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    model_patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(new_sd, "", assign=model_patcher.is_dynamic())
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return model_patcher

def load_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    model = load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    model.cached_patcher_init = (load_diffusion_model, (unet_path, model_options))
    return model


def load_vae_patcher(vae_path, metadata=None, device=None, disable_dynamic=False):
    """Reload a disk-backed VAE from ``vae_path`` and return its patcher.

    Used as the ``cached_patcher_init`` factory on ``VAE.patcher`` so
    :meth:`comfy.model_patcher.ModelPatcher.deepclone_multigpu` can produce a
    fresh, untainted VAE patcher (no inherited per-device load state, no
    in-place quantization fallout) for multigpu work-units and the
    SelectVAEDevice node. The optional ``device`` matches the source loader's
    VAE initialization path; the deepclone's ``load_device`` still controls
    where the cloned patcher is targeted.
    """
    if metadata is None:
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    else:
        sd = comfy.utils.load_torch_file(vae_path)
    vae = VAE(sd=sd, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()
    return vae.patcher

def load_unet(unet_path, dtype=None):
    logging.warning("The load_unet function has been deprecated and will be removed please switch to: load_diffusion_model")
    return load_diffusion_model(unet_path, model_options={"dtype": dtype})

def load_unet_state_dict(sd, dtype=None):
    logging.warning("The load_unet_state_dict function has been deprecated and will be removed please switch to: load_diffusion_model_state_dict")
    return load_diffusion_model_state_dict(sd, model_options={"dtype": dtype})

def save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.state_dict_for_saving()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    if metadata is None:
        metadata = {}

    model_management.load_models_gpu(load_models)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    sd = model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)
