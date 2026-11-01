import torch
from types import SimpleNamespace

orig_current_device = torch.cuda.current_device
orig_is_available = torch.cuda.is_available
orig_get_device_capability = getattr(torch.cuda, "get_device_capability", None)

torch.cuda.current_device = lambda: torch.device("cpu")
torch.cuda.is_available = lambda: True
torch.cuda.get_device_capability = lambda device=None: (8, 0)

try:
    import comfy.lora
    import comfy.model_base
    from comfy.weight_adapter.lora import LoRAAdapter
finally:
    # Restore original CUDA functions after imports to avoid affecting other tests
    torch.cuda.current_device = orig_current_device
    torch.cuda.is_available = orig_is_available
    if orig_get_device_capability is not None:
        torch.cuda.get_device_capability = orig_get_device_capability
    elif hasattr(torch.cuda, "get_device_capability"):
        delattr(torch.cuda, "get_device_capability")


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
