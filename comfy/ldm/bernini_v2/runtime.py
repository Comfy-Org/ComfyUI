# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Comfy-managed runtime objects for the Bernini v2 semantic planner."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import Qwen2TokenizerFast

import comfy.model_management
import comfy.ops
import comfy.utils
from comfy.model_patcher import CoreModelPatcher
from comfy.text_encoders.llama import Llama2_, Qwen25_7BVLI_Config
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer

from .planner_model import DiffLossFM, MLPConnector


class PlannerAux(nn.Module):
    """Connector, VIT diffusion decoder, and learned mask-token table."""

    def __init__(
        self,
        *,
        num_mask_tokens: int = 4096,
        hidden_size: int = 3584,
        decoder_width: int = 4096,
        decoder_depth: int = 16,
        device=None,
        dtype=None,
        operations,
    ):
        super().__init__()
        self.connector = MLPConnector(
            in_dim=hidden_size,
            out_dim_for_gen=4096,
            out_dim_for_vit=hidden_size,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.vit_decoder = DiffLossFM(
            target_channels=hidden_size,
            z_channels=hidden_size,
            depth=decoder_depth,
            width=decoder_width,
            shift=2.0,
            extra_one_step=True,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mask_tokens = nn.Parameter(
            torch.empty(
                1, num_mask_tokens, hidden_size, device=device, dtype=dtype
            ),
            requires_grad=False,
        )


@dataclass
class BerniniV2PlannerRuntime:
    """A lightweight handle tracked by Comfy through its model patchers."""

    language_model: nn.Module
    vision_model: nn.Module
    aux: PlannerAux
    language_patcher: object
    vision_patcher: object
    aux_patcher: object
    tokenizer: object
    dtype: torch.dtype
    load_device: torch.device

    def get_models(self) -> list[object]:
        return [self.language_patcher, self.vision_patcher, self.aux_patcher]

    def load_vision(self) -> None:
        comfy.model_management.load_models_gpu([self.vision_patcher])

    def load_planner(self) -> None:
        comfy.model_management.load_models_gpu(
            [self.language_patcher, self.aux_patcher]
        )


def _load_assign(
    module: nn.Module, state_dict: dict[str, torch.Tensor], label: str
) -> None:
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={result.missing_keys[:12]}, "
            f"unexpected={result.unexpected_keys[:12]}"
        )


def _uses_native_quant(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.endswith(".comfy_quant") for key in state_dict)


def _module_init_device(
    state_dict: dict[str, torch.Tensor], offload_device: torch.device
) -> torch.device | str:
    """Initialize quant hooks on storage and plain assign-loaded weights on meta."""

    return offload_device if _uses_native_quant(state_dict) else "meta"


def _embedded_bytes(state_dict: dict[str, torch.Tensor], key: str) -> bytes:
    try:
        tensor = state_dict.pop(key)
    except KeyError as error:
        raise ValueError(
            f"standalone planner is missing embedded tensor {key!r}"
        ) from error
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(
            f"embedded tensor {key!r} must be a one-dimensional U8 tensor"
        )
    return tensor.contiguous().cpu().numpy().tobytes()


def _embedded_json(
    state_dict: dict[str, torch.Tensor], key: str
) -> dict[str, object]:
    try:
        payload = json.loads(_embedded_bytes(state_dict, key).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"embedded tensor {key!r} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"embedded tensor {key!r} must contain a JSON object")
    return payload


def _qwen_config(payload: dict[str, object]) -> dict[str, object]:
    allowed = {field.name for field in dataclasses.fields(Qwen25_7BVLI_Config)}
    return {key: value for key, value in payload.items() if key in allowed}


def _build_qwen_tokenizer(
    tokenizer_json: bytes, tokenizer_config: dict[str, object]
) -> Qwen2TokenizerFast:
    kwargs = dict(tokenizer_config)
    kwargs.pop("tokenizer_class", None)
    kwargs.pop("added_tokens_decoder", None)
    return Qwen2TokenizerFast(
        tokenizer_object=Tokenizer.from_str(tokenizer_json.decode("utf-8")),
        **kwargs,
    )


def load_planner_runtime(
    checkpoint: str | Path, *, dtype: torch.dtype = torch.bfloat16
) -> BerniniV2PlannerRuntime:
    """Load the complete Bernini planner from one standalone safetensors file."""

    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix != ".safetensors":
        raise ValueError(
            f"expected one standalone planner .safetensors file: {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    state_dict = load_file(str(checkpoint), device="cpu")
    raw_config = _embedded_json(state_dict, "config_json")
    tokenizer_json = _embedded_bytes(state_dict, "tokenizer_json")
    tokenizer_config = _embedded_json(state_dict, "tokenizer_config")
    if "scaled_fp8" in state_dict:
        state_dict, _ = comfy.utils.convert_old_quants(state_dict)

    config_obj = Qwen25_7BVLI_Config(**_qwen_config(raw_config))
    rope_scaling = raw_config.get("rope_scaling")
    rope_dims = (
        rope_scaling.get("mrope_section")
        if isinstance(rope_scaling, dict)
        else None
    )
    if rope_dims is not None and list(rope_dims) != list(config_obj.rope_dims):
        raise ValueError(
            f"checkpoint mRoPE sections {rope_dims} do not match ComfyUI "
            f"Qwen config {config_obj.rope_dims}"
        )

    mllm_state = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(("model.", "visual."))
    }
    aux_state = {
        key: value
        for key, value in state_dict.items()
        if key == "mask_tokens"
        or key.startswith(("connector.", "vit_decoder."))
    }
    recognized = set(mllm_state) | set(aux_state)
    unexpected = sorted(set(state_dict) - recognized)
    if unexpected:
        raise ValueError(
            f"standalone planner contains unexpected tensors: {unexpected[:12]}"
        )
    if not mllm_state or not aux_state:
        raise ValueError(
            "standalone planner is missing Qwen or Bernini auxiliary weights"
        )
    del state_dict

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    mllm_quantized = _uses_native_quant(mllm_state)
    operations = (
        comfy.ops.mixed_precision_ops({}, dtype)
        if mllm_quantized
        else comfy.ops.manual_cast
    )
    init_device = _module_init_device(mllm_state, offload_device)
    language_model = Llama2_(
        config_obj, device=init_device, dtype=dtype, ops=operations
    )
    vision_model = Qwen2VLVisionTransformer(
        hidden_size=1280,
        output_hidden_size=config_obj.hidden_size,
        intermediate_size=3420,
        num_heads=16,
        num_layers=32,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        device=init_device,
        dtype=dtype,
        ops=operations,
    )
    language_state = {
        key.removeprefix("model."): value
        for key, value in mllm_state.items()
        if key.startswith("model.")
    }
    vision_state = {
        key.removeprefix("visual."): value
        for key, value in mllm_state.items()
        if key.startswith("visual.")
    }
    _load_assign(language_model, language_state, "Qwen language model")
    _load_assign(vision_model, vision_state, "Qwen vision model")
    del language_state, vision_state, mllm_state

    aux_quantized = _uses_native_quant(aux_state)
    aux_operations = (
        comfy.ops.mixed_precision_ops({}, dtype)
        if aux_quantized
        else comfy.ops.manual_cast
    )
    aux_init_device = _module_init_device(aux_state, offload_device)
    aux = PlannerAux(
        device=aux_init_device, dtype=dtype, operations=aux_operations
    )
    _load_assign(aux, aux_state, "Bernini planner auxiliary")
    del aux_state

    language_patcher = CoreModelPatcher(
        language_model, load_device=load_device, offload_device=offload_device
    )
    vision_patcher = CoreModelPatcher(
        vision_model, load_device=load_device, offload_device=offload_device
    )
    aux_patcher = CoreModelPatcher(
        aux, load_device=load_device, offload_device=offload_device
    )
    tokenizer = _build_qwen_tokenizer(tokenizer_json, tokenizer_config)
    return BerniniV2PlannerRuntime(
        language_model=language_model,
        vision_model=vision_model,
        aux=aux,
        language_patcher=language_patcher,
        vision_patcher=vision_patcher,
        aux_patcher=aux_patcher,
        tokenizer=tokenizer,
        dtype=dtype,
        load_device=load_device,
    )
