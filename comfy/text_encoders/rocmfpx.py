import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import safetensors
import torch

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.sd1_clip
import comfy.utils
from comfy.text_encoders import minimax, qwen3vl, qwen_image21
from comfy.text_encoders.rocmfpx_runtime import FpxSession


class TokenBackbone(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.num_layers = 36
        self.session = None

    def forward(self, tokens):
        if self.session is None:
            raise RuntimeError("ROCmFPX encoding must run through its CLIP adapter")
        return torch.cat([torch.from_numpy(self.session.encode_tokens(row)) for row in tokens])


class QwenImageClip(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, layer="last", textmodel_json_config={},
                         special_tokens={"pad": 151643}, model_class=TokenBackbone,
                         layer_norm_hidden_state=False, model_options=model_options)
        self.image_spans = []

    def forward(self, tokens):
        if self.layer != "last":
            raise ValueError("ROCmFPX Qwen Image conditioning uses the final decoder layer")
        if any(not isinstance(token, int) for row in tokens for token in row):
            raise ValueError("The ROCmFPX Qwen Image encoder currently supports text-to-image prompts")
        hidden = self.transformer(tokens)
        mask = torch.ones(hidden.shape[:2], dtype=torch.long)
        for i, row in enumerate(tokens):
            ended = False
            left_pad = bool(row and row[0] == 151643)
            for j, token in enumerate(row):
                if ended or (left_pad and token == 151643):
                    mask[i, j] = 0
                else:
                    left_pad = False
                if not ended and token == 151643 and not left_pad:
                    ended = True
                    mask[i, j] = 0
        return hidden, None, {"attention_mask": mask}


class QwenImageTE(qwen_image21.QwenImage21TEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype,
                         model_options={**model_options, "qwen3vl_8b_class": QwenImageClip})

    def set_fpx_session(self, session):
        self.qwen3vl_8b.transformer.session = session


class ImageLanguageBackbone(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.session = None

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, position_ids=None, embeds_info=None,
                visual_pos_masks=None, deepstack_embeds=None):
        if self.session is None:
            raise RuntimeError("ROCmFPX encoding must run through its CLIP adapter")
        if embeds.shape[0] != 1 or attention_mask is not None or intermediate_output is not None:
            raise ValueError("ROCmFPX H3 conditioning requires one causal final-layer sequence")
        sequence = embeds.shape[1]
        packed = np.zeros((sequence, 20480), dtype=np.float32)
        packed[:, :5120] = embeds[0].detach().cpu().numpy()
        positions = np.zeros((4, sequence), dtype=np.int32)
        if position_ids is None:
            positions[:3] = np.arange(sequence, dtype=np.int32)
        else:
            positions[:3] = position_ids.detach().cpu().numpy().astype(np.int32)
        if deepstack_embeds is not None:
            mask = visual_pos_masks[0].detach().cpu().numpy()
            for index, value in enumerate(deepstack_embeds):
                packed[mask, (index + 1) * 5120:(index + 2) * 5120] = value.detach().float().cpu().numpy()
        hidden = self.session.encode_embeddings(packed, positions)
        return torch.from_numpy(hidden), None


class H3VisionBackbone(minimax.MiniMaxQwen3VL):
    def __init__(self, config_dict, dtype, device, operations):
        torch.nn.Module.__init__(self)
        config = qwen3vl.QWEN3VL_CONFIGS[self.model_type](**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = ImageLanguageBackbone(config, dtype, device, operations)
        vision_config = {**qwen3vl.QWEN3VL_VISION_COMMON, **qwen3vl.QWEN3VL_VISION[self.model_type],
                         "out_hidden_size": config.hidden_size}
        self.visual = qwen3vl.Qwen3VLVisionModel(vision_config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype


class H3Clip(minimax.MiniMaxH3ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        comfy.sd1_clip.SDClipModel.__init__(self, device=device, dtype=dtype, layer="last",
                                          textmodel_json_config={}, special_tokens={"pad": 151643},
                                          model_class=H3VisionBackbone, layer_norm_hidden_state=False,
                                          enable_attention_masks=False, return_attention_masks=False,
                                          model_options=model_options)


class H3TE(minimax.MiniMaxH3TEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype,
                         model_options={**model_options, "qwen3vl_32b_class": H3Clip})

    def set_fpx_session(self, session):
        self.qwen3vl_32b.transformer.model.session = session


class FpxModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model_path, dll_directory, model_kind, load_device, offload_device):
        self.session = None
        super().__init__(torch.nn.Module(), load_device, offload_device)
        self.model_path = model_path
        self.dll_directory = dll_directory
        self.model_kind = model_kind
        # Reserve packed backend weights separately from the native companion.
        self.size = model_path.stat().st_size * 2

    def model_dtype(self):
        return torch.float32

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if extra_memory < 0:
            self.partially_unload(self.offload_device, -extra_memory)
            return 0
        if self.session is not None:
            return 0
        if extra_memory < self.size:
            return 0
        comfy.model_management.soft_empty_cache()
        if comfy.model_management.get_free_memory(device_to) < self.size:
            raise comfy.model_management.OOM_EXCEPTION("ROCmFPX requires enough free GPU memory for the complete text encoder")
        self.session = FpxSession(self.model_path, self.dll_directory, self.model_kind,
                                  device_index=device_to.index or 0)
        self.model.model_loaded_weight_memory = self.size
        self.model.device = device_to
        return self.size

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        freed = self.loaded_size()
        if self.session is not None:
            self.session.close()
            self.session = None
        self.model.model_loaded_weight_memory = 0
        self.model.device = device_to
        return freed

    def detach(self, unpatch_all=True):
        # Model management also detaches with False when reusing loaded weights.
        if unpatch_all:
            self.partially_unload(self.offload_device)
        return super().detach(unpatch_all)

    def __del__(self):
        if self.session is not None:
            self.session.close()
            self.session = None


class ROCmFPXCLIP(comfy.sd.CLIP):
    def __init__(self, model_path=None, dll_directory=None, model_kind=None, companion_path=None,
                 embedding_directory=None, no_init=False):
        if no_init:
            return
        if torch.version.hip is None:
            raise RuntimeError("ROCmFPX text encoders require a ROCm PyTorch installation")
        model_path = Path(model_path)
        dll_directory = Path(dll_directory)
        self.model_kind = model_kind
        self.active_session = None
        if not (dll_directory / "llama.dll").is_file():
            raise FileNotFoundError("Select the ROCmFPX directory containing llama.dll")
        state_dict = {}
        dtype = torch.float32
        if model_kind == "minimax_h3":
            with safetensors.safe_open(companion_path, framework="pt", device="cpu") as source:
                dtype = source.get_tensor("model.layers.0.input_layernorm.weight").dtype
                state_dict = {key: source.get_tensor(key) for key in source.keys()
                              if key.startswith(("visual.", "model.embed_tokens."))}
            state_dict, _ = comfy.utils.convert_old_quants(state_dict, model_prefix="", metadata={})
            target = SimpleNamespace(params={}, clip=H3TE, tokenizer=minimax.MiniMaxH3Tokenizer)
        elif model_kind == "qwen_image_21":
            target = SimpleNamespace(params={}, clip=QwenImageTE, tokenizer=qwen_image21.QwenImage21Tokenizer)
        else:
            raise ValueError(f"Unsupported ROCmFPX text encoder: {model_kind}")
        model_options = {"initial_device": torch.device("cpu"), "dtype": dtype}
        quant_metadata = comfy.utils.detect_layer_quantization(state_dict, "")
        if quant_metadata is not None:
            model_options["quantization_metadata"] = quant_metadata
        super().__init__(target=target, embedding_directory=embedding_directory,
                         state_dict=[state_dict] if state_dict else [],
                         model_options=model_options, disable_dynamic=True)
        self.fpx_patcher = FpxModelPatcher(model_path, dll_directory, model_kind,
                                         self.patcher.load_device, self.patcher.offload_device)

    def clone(self, disable_dynamic=False):
        result = type(self)(no_init=True)
        result.patcher = self.patcher.clone(disable_dynamic=disable_dynamic)
        result.cond_stage_model = self.cond_stage_model
        result.tokenizer = self.tokenizer
        result.layer_idx = self.layer_idx
        result.tokenizer_options = self.tokenizer_options.copy()
        result.use_clip_schedule = self.use_clip_schedule
        result.apply_hooks_to_conds = self.apply_hooks_to_conds
        result.model_kind = self.model_kind
        result.fpx_patcher = self.fpx_patcher
        result.active_session = None
        return result

    def _memory_required(self, tokens):
        key = "qwen3vl_32b" if self.model_kind == "minimax_h3" else "qwen3vl_8b"
        count = 0
        for row in tokens.get(key, []):
            length = 0
            for token, *_ in row:
                if isinstance(token, dict) and token["type"] == "image":
                    image = token["data"]
                    length += math.ceil(image.shape[-3] / 32) * math.ceil(image.shape[-2] / 32)
                else:
                    length += 1
            count = max(count, length)
        layers = 50 if self.model_kind == "minimax_h3" else 36
        # Contexts and transient graph buffers remain scoped to each encode.
        kv_bytes = 2 * layers * max(256, math.ceil(count / 256) * 256) * 8 * 128 * 4
        return kv_bytes + 2 * 1024 ** 3

    def load_model(self, tokens={}):
        device = self.patcher.load_device
        if device.type != "cuda" or device != self.fpx_patcher.load_device:
            raise RuntimeError("ROCmFPX requires the text encoder and its backend on the same ROCm GPU; CPU placement and device switching are unsupported")
        if self.active_session is None:
            comfy.model_management.load_models_gpu([self.patcher, self.fpx_patcher],
                                                    memory_required=self._memory_required(tokens), force_full_load=True)
            if self.fpx_patcher.session is None:
                raise comfy.model_management.OOM_EXCEPTION("ROCmFPX requires full GPU loading; disable --novram or free GPU memory")
        return self.patcher

    def state_dict_for_saving(self):
        raise NotImplementedError("ROCmFPX text encoders cannot be saved as ComfyUI checkpoints; retain the original GGUF and companion files")

    @contextlib.contextmanager
    def _encoding(self, tokens):
        if self.active_session is not None:
            yield
            return
        try:
            self.load_model(tokens)
            self.active_session = self.fpx_patcher.session
            self.cond_stage_model.set_fpx_session(self.active_session)
            yield
        except BaseException:
            self.fpx_patcher.detach()
            raise
        finally:
            self.cond_stage_model.set_fpx_session(None)
            self.active_session = None

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        with self._encoding(tokens):
            return super().encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        with self._encoding(tokens):
            return super().encode_from_tokens_scheduled(tokens, unprojected=unprojected,
                                                        add_dict=add_dict, show_pbar=show_pbar)
