"""Marigold V2 NF4 precision recipe using native Qwen and bypass LoRA operations."""

import json

import torch

import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.quant_nf4
import comfy.sd
import comfy.utils
from comfy.quant_ops import QuantizedTensor


def nf4_operations(offload_device):
    class MarigoldOperations(comfy.ops.disable_weight_init):
        @staticmethod
        def RMSNorm(*args, **kwargs):
            kwargs["dtype"] = torch.float32
            norm = comfy.ops.disable_weight_init.RMSNorm(*args, **kwargs)
            norm.weight_compute_dtype = torch.float32
            return norm

        class Linear(comfy.ops.disable_weight_init.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                torch.nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.has_bias = bias
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                weight = state_dict.pop(prefix + "weight")
                if weight.shape != (self.out_features, self.in_features):
                    raise ValueError(f"Incompatible Marigold weight: {prefix}weight {weight.shape}")
                self.weight = torch.nn.Parameter(weight.to(offload_device), requires_grad=False)
                if self.has_bias:
                    bias = state_dict.pop(prefix + "bias")
                    if bias.shape != (self.out_features,):
                        raise ValueError(f"Incompatible Marigold bias: {prefix}bias {bias.shape}")
                    self.bias = torch.nn.Parameter(bias.to(offload_device),
                                                   requires_grad=False)

            def forward(self, input):
                comfy.ops.run_every_op()
                input_dtype = input.dtype
                with comfy.ops.CastBiasWeightContext(self, input, dtype=torch.bfloat16,
                                                     bias_dtype=torch.bfloat16,
                                                     offloadable=True) as (weight, bias):
                    input = input.to(torch.bfloat16)
                    if isinstance(weight, QuantizedTensor):
                        return comfy.quant_nf4.NF4Layout.linear(input, weight, bias).to(input_dtype)
                    return torch.nn.functional.linear(input, weight, bias)

    return MarigoldOperations


def autocast_transformer(executor, *args, **kwargs):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return executor(*args, **kwargs)


def load_model(unet_path, lora_path):
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("Marigold NF4 loading requires a CUDA device")
    comfy.quant_nf4.backend()
    offload_device = comfy.model_management.unet_offload_device()
    state, metadata = comfy.utils.load_torch_file(unet_path, safe_load=True, return_metadata=True)
    if metadata is None or metadata.get("marigold_recipe") != "qwen-image-edit-2509-nf4-bf16-v1":
        raise ValueError("Select the prequantized qwen_image_edit_2509_nf4.safetensors Marigold checkpoint")
    layers = json.loads(metadata.get("_quantization_metadata", "{}")).get("layers", {})
    if len(layers) != 844:
        raise ValueError("Incomplete Marigold NF4 checkpoint: expected 844 quantized layers")
    comfy.quant_nf4.restore_weights(state, layers)
    source_keys = set(state)
    model = comfy.sd.load_diffusion_model_state_dict(
        state, model_options={"dtype": torch.bfloat16,
                              "custom_operations": nf4_operations(offload_device)},
        disable_dynamic=True)
    if model is None or not isinstance(model.model, comfy.model_base.QwenImage):
        raise ValueError("Marigold NF4 requires Qwen-Image-Edit-2509")
    native_keys = set(model.model.diffusion_model.state_dict())
    if source_keys != native_keys:
        raise ValueError(f"Marigold checkpoint mismatch: missing={sorted(native_keys - source_keys)}, unexpected={sorted(source_keys - native_keys)}")
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "marigold_bf16", autocast_transformer)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    validate_adapter(model.model.diffusion_model, lora)
    return comfy.sd.load_bypass_lora_for_models(model, None, lora, 1.0, 0.0)[0]


def validate_adapter(transformer, lora):
    targets = ("img_in", "txt_in", "norm_out.linear", "attn.to_q", "attn.to_k", "attn.to_v",
               "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
               "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2",
               "txt_mlp.net.0.proj", "txt_mlp.net.2")
    expected = set()
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(targets):
            continue
        down_key = f"diffusion_model.{name}.lora_A.weight"
        up_key = f"diffusion_model.{name}.lora_B.weight"
        expected.update((down_key, up_key))
        if down_key not in lora or up_key not in lora:
            raise ValueError(f"Incomplete Marigold adapter: {name}")
        down, up = lora[down_key], lora[up_key]
        if down.ndim != 2 or up.shape != (module.out_features, down.shape[0]) or down.shape[1] != module.in_features:
            raise ValueError(f"Incompatible Marigold adapter: {name}")
    if set(lora) != expected:
        raise ValueError("Expected a converted Marigold modality LoRA with separate A/B weights")
