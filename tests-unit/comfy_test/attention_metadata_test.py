import json
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.ops
import comfy.utils
from comfy.ldm.modules import attention
from comfy.ldm.minimax.model import Attention as MiniMaxAttention
from comfy.ldm.qwen_image21.model import block_causal_attention, prefix_cached_attention
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.supported_models_base import BASE


class SmallDiffusionModel(torch.nn.Module):
    def __init__(self, dtype, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.blocks = torch.nn.ModuleList([
            MiniMaxAttention(8, heads=2, head_dim=4, eps=1e-6, dtype=dtype, device=device, operations=operations)
            for _ in range(2)
        ])


def make_model(operations=comfy.ops.disable_weight_init):
    config = BASE({"dtype": torch.float32})
    config.custom_operations = operations
    return BaseModel(config, device=torch.device("cpu"), unet_model=SmallDiffusionModel)


def encode_config(config):
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def model_weights(model):
    return {k: torch.randn_like(v) for k, v in model.diffusion_model.state_dict().items() if not k.endswith(".comfy_attention.config")}


def make_preference(method):
    preference = attention.ComfyAttention()
    preference.load_state_dict({"config": encode_config({"attention": method})})
    return preference


@pytest.fixture
def preferred_backend(monkeypatch):
    backend = Mock(wraps=attention.attention_pytorch)
    monkeypatch.setattr(attention, "attention_comfy_kitchen_int8", backend)
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", lambda device: True)
    return backend


def test_metadata_checkpoint_round_trip(tmp_path, preferred_backend):
    model = make_model()
    weights = model_weights(model)
    metadata_key = "blocks.0.comfy_attention.config"
    config = {"attention": "comfy_kitchen_int8"}
    weights[metadata_key] = encode_config(config)
    model.load_model_weights(weights)

    assert model.diffusion_model.blocks[0].comfy_attention.config == config
    assert model.diffusion_model.blocks[0].comfy_attention.function is preferred_backend
    assert model.diffusion_model.blocks[1].comfy_attention.config is None
    saved = model.state_dict_for_saving(model.diffusion_model.state_dict())
    path = tmp_path / "model.safetensors"
    comfy.utils.save_torch_file(saved, str(path))
    loaded = comfy.utils.load_torch_file(str(path))
    prefix = "model.diffusion_model."
    assert json.loads(loaded[prefix + metadata_key].numpy().tobytes()) == config
    assert comfy.utils.detect_layer_quantization(loaded, prefix) is None

    restored = make_model()
    restored.load_model_weights(loaded, prefix)
    assert restored.diffusion_model.blocks[0].comfy_attention.config == config
    assert restored.diffusion_model.blocks[0].comfy_attention.function is preferred_backend
    assert restored.diffusion_model.blocks[1].comfy_attention.config is None


@pytest.mark.parametrize("assign", [False, True])
def test_child_module_loads_and_saves_without_base_model(preferred_backend, assign):
    parent = torch.nn.Module()
    parent.comfy_attention = attention.ComfyAttention()
    assert not parent.state_dict()
    config = {"attention": "comfy_kitchen_int8"}
    parent.load_state_dict({"comfy_attention.config": encode_config(config)}, assign=assign)
    parent.to(device="meta", dtype=torch.bfloat16)
    saved = parent.state_dict()
    assert saved["comfy_attention.config"].device == torch.device("cpu")
    assert saved["comfy_attention.config"].dtype == torch.uint8
    restored = torch.nn.Module()
    restored.comfy_attention = attention.ComfyAttention()
    restored.load_state_dict(saved, assign=assign)
    assert restored.comfy_attention.config == config
    assert restored.comfy_attention.function is preferred_backend
    restored.load_state_dict({})
    assert restored.comfy_attention.function is None
    assert not restored.state_dict()


def test_metadata_only_changes_target_module(preferred_backend):
    model = make_model()
    weights = model_weights(model)
    weights["blocks.0.comfy_attention.config"] = encode_config({"attention": "comfy_kitchen_int8"})
    model.load_model_weights(weights)
    x = torch.randn(3, 8)
    model.diffusion_model.blocks[0](x)
    model.diffusion_model.blocks[1](x)
    assert preferred_backend.call_count == 1

    model.load_model_weights(model_weights(model))
    assert model.diffusion_model.blocks[0].comfy_attention.function is None
    assert model.diffusion_model.blocks[0].comfy_attention.config is None


def test_model_patcher_saves_metadata_but_does_not_merge_it(preferred_backend):
    model = make_model()
    weights = model_weights(model)
    key = "blocks.0.comfy_attention.config"
    config = {"attention": "comfy_kitchen_int8"}
    weights[key] = encode_config(config)
    model.load_model_weights(weights)
    patcher = ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    saved = patcher.model_state_dict_for_saving()
    assert json.loads(saved["diffusion_model." + key].numpy().tobytes()) == config
    patches = patcher.get_key_patches("diffusion_model.")
    assert "diffusion_model." + key not in patches
    assert "diffusion_model.blocks.0.qkv_proj.weight" in patches


def test_attention_metadata_does_not_enable_weight_quantization():
    weights = {"blocks.0.comfy_attention.config": encode_config({"attention": "comfy_kitchen_int8"})}
    assert comfy.utils.detect_layer_quantization(weights, "") is None
    weights["blocks.0.qkv_proj.comfy_quant"] = encode_config({"format": "float8_e4m3fn"})
    assert comfy.utils.detect_layer_quantization(weights, "") == {"mixed_ops": True}


@pytest.mark.parametrize("method", [None, [], {}, "", "unknown_attention", "pytorch", "optimized", "registered_attention"])
def test_unsupported_attention_method_is_ignored(method, caplog, preferred_backend, monkeypatch):
    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "registered_attention", preferred_backend)
    model = make_model()
    weights = model_weights(model)
    weights["blocks.0.comfy_attention.config"] = encode_config({"attention": method})
    weights["blocks.1.comfy_attention.config"] = encode_config({"attention": "comfy_kitchen_int8"})
    model.load_model_weights(weights)
    assert model.diffusion_model.blocks[0].comfy_attention.function is None
    assert model.diffusion_model.blocks[1].comfy_attention.function is preferred_backend
    assert "blocks.0.comfy_attention.config" not in model.diffusion_model.state_dict()
    assert caplog.messages == [f"Ignoring unsupported attention method {method!r} for blocks.0.comfy_attention"]
    model.diffusion_model.blocks[0](torch.randn(3, 8))
    preferred_backend.assert_not_called()


@pytest.mark.parametrize("module_name", ["blocks.0.qkv_proj", "blocks.2"])
def test_invalid_metadata_target_is_ignored_without_decoding(module_name, caplog, monkeypatch):
    model = make_model()
    weights = model_weights(model)
    weights[f"{module_name}.comfy_attention.config"] = encode_config({"attention": "comfy_kitchen_int8"})
    decode = Mock(side_effect=AssertionError("non-attention metadata must not be decoded here"))
    monkeypatch.setattr(json, "loads", decode)
    model.load_model_weights(weights)
    assert caplog.messages == [f"unet unexpected: ['{module_name}.comfy_attention.config']"]
    decode.assert_not_called()
    assert all(block.comfy_attention.config is None for block in model.diffusion_model.blocks)
    model.load_model_weights(model_weights(model))


def test_weight_quantization_is_decoded_only_by_weight_loader(monkeypatch, caplog):
    model = make_model(comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32))
    weights = model_weights(make_model())
    prefix = "blocks.0.qkv_proj."
    weights[prefix + "weight"] = weights[prefix + "weight"].to(torch.float8_e4m3fn)
    weights[prefix + "weight_scale"] = torch.tensor(1.0)
    weights[prefix + "comfy_quant"] = encode_config({"format": "float8_e4m3fn"})
    decode = Mock(wraps=json.loads)
    monkeypatch.setattr(json, "loads", decode)
    model.load_model_weights(weights)
    decode.assert_called_once()
    assert model.diffusion_model.blocks[0].qkv_proj.quant_format == "float8_e4m3fn"
    assert not caplog.messages


def test_mixed_precision_model_ignores_invalid_attention_target(caplog):
    model = make_model(comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32))
    weights = model_weights(make_model())
    expected = weights["blocks.0.qkv_proj.weight"].clone()
    weights["blocks.0.qkv_proj.comfy_attention.config"] = encode_config({"attention": "comfy_kitchen_int8"})
    model.load_model_weights(weights)
    torch.testing.assert_close(model.diffusion_model.blocks[0].qkv_proj.weight, expected)
    assert caplog.messages == ["unet unexpected: ['blocks.0.qkv_proj.comfy_attention.config']"]
    assert all(block.comfy_attention.config is None for block in model.diffusion_model.blocks)


def test_missing_backend_preserves_original_attention():
    q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
    expected = attention.attention_basic(q, k, v, 2)
    actual = attention.attention_basic(q, k, v, 2, preferred_attention=make_preference("unavailable_method"))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("containers", [False, True])
def test_user_override_takes_precedence(preferred_backend, containers):
    q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
    expected = attention.attention_pytorch(q, k, v, 2)
    override = Mock(spec=lambda: None, side_effect=lambda original, *a, **kw: original(*a, **kw))
    tensors = (q, k, v)
    if containers:
        tensors = tuple(attention.AttentionTensorContainer(t) for t in tensors)
    actual = attention.attention_pytorch(
        *tensors, 2, preferred_attention=make_preference("comfy_kitchen_int8"),
        transformer_options={"optimized_attention_override": override},
    )
    torch.testing.assert_close(actual, expected)
    preferred_backend.assert_not_called()
    override.assert_called_once()
    if containers:
        assert all(t.tensor is None for t in tensors)


@pytest.mark.parametrize("supported", [False, True])
def test_kitchen_support_is_checked_only_on_load(monkeypatch, supported):
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    model = make_model()
    weights = model_weights(model)
    config = {"attention": "comfy_kitchen_int8"}
    for index in range(2):
        weights[f"blocks.{index}.comfy_attention.config"] = encode_config(config)
    primary_device = torch.device("xpu:1")
    monkeypatch.setattr(attention.model_management, "get_torch_device", lambda: primary_device)
    probe = Mock(return_value=supported)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", probe)
    backend = Mock(wraps=attention.attention_pytorch)
    monkeypatch.setattr(attention, "attention_comfy_kitchen_int8", backend)
    model.load_model_weights(weights)

    x = torch.randn(3, 8)
    for _ in range(2):
        for block in model.diffusion_model.blocks:
            assert block.comfy_attention.function is (backend if supported else None)
            block(x)
    assert [call.args for call in probe.call_args_list] == [(primary_device,), (primary_device,)]
    assert backend.call_count == (4 if supported else 0)

    saved = model.state_dict_for_saving(model.diffusion_model.state_dict())
    prefix = "model.diffusion_model."
    assert json.loads(saved[prefix + "blocks.0.comfy_attention.config"].numpy().tobytes()) == config
    probe.return_value = not supported
    model.load_model_weights(saved, prefix)
    assert probe.call_count == 4
    assert all(block.comfy_attention.function is (None if supported else backend)
               for block in model.diffusion_model.blocks)


def test_unavailable_kitchen_keeps_metadata_and_normal_attention(monkeypatch, caplog):
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
    probe = Mock(side_effect=AssertionError("unavailable Kitchen must not be probed again"))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", probe)
    preference = make_preference("comfy_kitchen_int8")
    assert preference.function is None
    assert json.loads(preference.state_dict()["config"].numpy().tobytes()) == {"attention": "comfy_kitchen_int8"}
    probe.assert_not_called()
    assert not caplog.messages


@pytest.mark.parametrize("containers", [False, True])
@pytest.mark.parametrize("skip_reshape", [False, True])
def test_kitchen_preserves_full_precision_with_mask_and_gqa(monkeypatch, containers, skip_reshape):
    probe = Mock(side_effect=AssertionError("attention calls must not probe device support"))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", probe)
    kernel = Mock(side_effect=AssertionError("INT8 kernel must not run for full precision"))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention", kernel)
    monkeypatch.setattr(attention.comfy_kitchen, "prequantize_int8_attention", kernel)
    q = torch.randn(1, 4, 3, 8)
    k, v = (torch.randn(1, 2, 5, 8) for _ in range(2))
    if not skip_reshape:
        q, k, v = (t.transpose(1, 2).flatten(2) for t in (q, k, v))
    options = {"mask": torch.zeros(3, 5), "enable_gqa": True, "scale": 0.2, "low_precision_attention": False,
               "skip_reshape": skip_reshape, "skip_output_reshape": True}
    expected = attention.attention_pytorch(q, k, v, 4, **options)
    tensors = (q, k, v)
    if containers:
        tensors = tuple(attention.AttentionTensorContainer(t) for t in tensors)
    actual = attention.attention_comfy_kitchen_int8(*tensors, 4, **options)
    torch.testing.assert_close(actual, expected)
    probe.assert_not_called()
    kernel.assert_not_called()
    if containers:
        assert all(t.tensor is None for t in tensors)


def test_supported_kitchen_device_uses_kernel(monkeypatch):
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", lambda device: True)
    preference = make_preference("comfy_kitchen_int8")
    probe = Mock(side_effect=AssertionError("attention calls must not probe device support"))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", probe)
    kernel = Mock(side_effect=lambda q, k, v, **kw: torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention", kernel)
    q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
    expected = attention.attention_pytorch(q, k, v, 2)
    actual = attention.attention_pytorch(q, k, v, 2, preferred_attention=preference)
    torch.testing.assert_close(actual, expected)
    kernel.assert_called_once()


def test_supported_kitchen_containers_use_prequantized_path(monkeypatch):
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", lambda device: True)
    preference = make_preference("comfy_kitchen_int8")
    probe = Mock(side_effect=AssertionError("attention calls must not probe device support"))
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_is_available", probe)
    quantize = Mock(side_effect=lambda q, k, v, **kw: torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw))
    attend = Mock(side_effect=lambda quantized: quantized)
    monkeypatch.setattr(attention.comfy_kitchen, "prequantize_int8_attention", quantize)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_from_prequantized", attend)
    q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
    expected = attention.attention_pytorch(q, k, v, 2)
    containers = tuple(attention.AttentionTensorContainer(t) for t in (q, k, v))
    actual = attention.attention_pytorch(*containers, 2, preferred_attention=preference)
    torch.testing.assert_close(actual, expected)
    assert all(t.tensor is None for t in containers)
    quantize.assert_called_once()
    attend.assert_called_once()




@pytest.mark.parametrize("cached", [False, True])
def test_qwen_image21_helpers_pass_preference(preferred_backend, cached):
    q, k, v = (torch.randn(1, 3, 2, 4) for _ in range(3))
    if cached:
        run_attention = prefix_cached_attention(k[:, :1], v[:, :1])
    else:
        run_attention = block_causal_attention([(0, 3, None)])
    run_attention(q, k, v, 2, preferred_attention=make_preference("comfy_kitchen_int8"))
    preferred_backend.assert_called_once()
