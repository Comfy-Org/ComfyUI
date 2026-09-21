import torch
from types import SimpleNamespace

# Some Comfy modules assume a CUDA-capable runtime during import. Patch the CUDA
# entry points for CPU-only test environments so the Qwen LoRA regression can be
# exercised without taking a GPU dependency.
torch.cuda.current_device = lambda: torch.device("cpu")
torch.cuda.is_available = lambda: True
torch.cuda.get_device_capability = lambda device=None: (8, 0)

import comfy.lora
import comfy.model_base
from comfy.weight_adapter.lora import LoRAAdapter


def test_qwen_image_default_lora_format_loads():
    x = "diffusion_model.transformer_blocks.0.attn.to_q.weight"
    lora = {
        f"{x}.lora_B.default.weight": torch.randn(8, 4),
        f"{x}.lora_A.default.weight": torch.randn(4, 8),
    }

    loaded = LoRAAdapter.load(x, lora, 1.0, None, set())

    assert loaded is not None
    assert loaded.weights[0].shape == (8, 4)
    assert loaded.weights[1].shape == (4, 8)
    assert f"{x}.lora_B.default.weight" in loaded.loaded_keys
    assert f"{x}.lora_A.default.weight" in loaded.loaded_keys


def test_qwen_image_peft_prefix_maps_to_transformer_weight(monkeypatch):
    key = "diffusion_model.transformer_blocks.0.attn.to_k.weight"
    model = object.__new__(comfy.model_base.QwenImage)
    torch.nn.Module.__init__(model)
    model.state_dict = lambda: {key: torch.empty(8, 8)}
    model.model_config = SimpleNamespace(unet_config={})
    monkeypatch.setattr(comfy.utils, "unet_to_diffusers", lambda config: {})

    key_map = comfy.lora.model_lora_keys_unet(model)

    assert key_map["base_model.model.transformer_blocks.0.attn.to_k"] == key
