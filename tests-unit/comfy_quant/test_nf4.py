"""CUDA NF4 loading, native casting and packed-weight lifecycle regression tests."""
import json
import sys

import pytest
import torch

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.marigold
import comfy.ops
from comfy.quant_ops import QuantizedTensor

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 requires CUDA")


@requires_cuda
def test_packed_metadata_roundtrip():
    bnb = pytest.importorskip("bitsandbytes.functional")
    source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    expected, state = bnb.quantize_4bit(source, quant_type="nf4", compress_statistics=False)
    actual = restored_fixture(source).cuda()
    clone = actual.cpu().clone().cuda()
    assert actual._qdata.dtype == torch.uint8
    assert torch.equal(clone._qdata, expected)
    assert torch.equal(clone.params.scale, state.absmax)
    assert torch.equal(clone.params.code, state.code)
    assert clone.params.blocksize == 64
    assert clone.params.quant_dtype == torch.bfloat16
    assert torch.equal(clone.dequantize(), bnb.dequantize_4bit(expected, state))
    clone._qdata.zero_()
    assert torch.equal(actual._qdata, expected)


@pytest.mark.parametrize("prefix,dtype,quantized", [
    ("img_in.", torch.bfloat16, True),
    ("transformer_blocks.0.img_mod.1.", torch.float32, False),
    ("proj_out.", torch.bfloat16, False),
])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@requires_cuda
def test_layer_recipe_and_cpu_offload(prefix, dtype, quantized, input_dtype):
    bnb = pytest.importorskip("bitsandbytes.functional")
    op = comfy.marigold.nf4_operations(torch.device("cpu"))
    layer = op.Linear(128, 64)
    source = torch.randn(64, 128, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.bfloat16)
    if quantized:
        prepared = restored_fixture(source)
    elif prefix == "proj_out.":
        prepared = restored_fixture(source).cuda().dequantize().cpu()
    else:
        prepared = source.to(torch.float32)
    layer._load_from_state_dict({prefix + "weight": prepared, prefix + "bias": bias.to(dtype if not quantized else torch.float32)},
                               prefix, {}, True, [], [], [])
    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == dtype
    assert isinstance(layer.weight, QuantizedTensor) == quantized
    x = torch.randn(2, 8, 128, device="cuda", dtype=input_dtype)
    if prefix.startswith("transformer_blocks.0"):
        w = source.cuda()
    else:
        packed, state = bnb.quantize_4bit(source.cuda(), quant_type="nf4")
        w = bnb.dequantize_4bit(packed, state)
    expected = torch.nn.functional.linear(x.to(torch.bfloat16), w, bias.cuda())
    if quantized:
        expected = expected.to(input_dtype)
    with torch.inference_mode():
        first = layer(x)
        layer.cuda()
        second = layer(x)
        layer.cpu()
        third = layer(x)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert torch.equal(third, expected)


@requires_cuda
def test_rms_norm_precision_survives_offload():
    layer = comfy.ops.disable_weight_init.RMSNorm(128, device="cpu", dtype=torch.float32)
    layer.weight = torch.nn.Parameter(torch.randn(128))
    layer.comfy_cast_weights = True
    layer.weight_compute_dtype = torch.float32
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    actual = layer(x)
    expected = torch.nn.functional.rms_norm(x, (128,), layer.weight.cuda(), layer.eps)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    layer.weight_compute_dtype = None
    assert layer(x).dtype == torch.bfloat16


@torch.inference_mode()
@requires_cuda
def test_bypass_clone_switch_repeat_and_model_offload():
    from comfy.model_patcher import ModelPatcher
    from comfy.weight_adapter import BypassInjectionManager
    from comfy.weight_adapter.lora import LoRAAdapter

    op = comfy.marigold.nf4_operations(torch.device('cpu'))
    model = torch.nn.Module()
    model.img_in = op.Linear(128, 64)
    model.img_in.load_state_dict({'weight': restored_fixture(torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)),
                                 'bias': torch.randn(64, dtype=torch.bfloat16)})
    base = ModelPatcher(model, torch.device('cuda'), torch.device('cpu'))
    packed = model.img_in.weight._qdata.clone()
    scale = model.img_in.weight.params.scale.clone()
    clones = []
    adapters = []
    for _ in range(2):
        up, down = torch.randn(64, 4, dtype=torch.bfloat16), torch.randn(4, 128, dtype=torch.bfloat16)
        adapter = LoRAAdapter([], (up, down, None, None, None, None))
        manager = BypassInjectionManager()
        manager.add_adapter('img_in.weight', adapter, strength=1.0)
        clone = base.clone()
        clone.set_injections('bypass_lora', manager.create_injections(model))
        clone.add_callback(comfy.patcher_extension.CallbacksMP.ON_DETACH, manager.offload)
        clones.append(clone)
        adapters.append(adapter)
    assert not base.injections
    x = torch.randn(2, 8, 128, device='cuda', dtype=torch.bfloat16)
    outputs = []
    with torch.inference_mode():
        for index in (0, 1, 0):
            clone = clones[index]
            clone.patch_model(torch.device('cuda'))
            out = model.img_in(x)
            assert torch.equal(out, model.img_in(x))
            clone.partially_unload(torch.device('cpu'), 1 << 30)
            assert model.img_in.weight.device.type == 'cpu'
            assert torch.equal(out, model.img_in(x))
            clone.load(torch.device('cuda'), full_load=True)
            assert torch.equal(out, model.img_in(x))
            clone.detach()
            assert adapters[index].weights[0].device.type == 'cpu'
            outputs.append(out)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert torch.equal(model.img_in.weight._qdata, packed)
    assert torch.equal(model.img_in.weight.params.scale, scale)
    assert all(a.weights[0].dtype == torch.bfloat16 for a in adapters)
    assert base.model_size() >= packed.nbytes + scale.nbytes


def test_adapter_validation_before_mutation():
    model = torch.nn.Module()
    model.img_in = torch.nn.Linear(128, 64, dtype=torch.bfloat16)
    before = model.img_in.weight.detach().clone()
    lora = {'diffusion_model.img_in.lora_A.weight': torch.ones(4, 128),
            'diffusion_model.img_in.lora_B.weight': torch.ones(64, 4)}
    comfy.marigold.validate_adapter(model, lora)
    lora.pop('diffusion_model.img_in.lora_B.weight')
    with pytest.raises(ValueError, match='Incomplete'):
        comfy.marigold.validate_adapter(model, lora)
    lora['diffusion_model.img_in.lora_B.weight'] = torch.ones(63, 4)
    with pytest.raises(ValueError, match='Incompatible'):
        comfy.marigold.validate_adapter(model, lora)
    assert torch.equal(before, model.img_in.weight)


@requires_cuda
def test_single_vector_matches_reference_kernel():
    bnb = pytest.importorskip('bitsandbytes')
    op = comfy.marigold.nf4_operations(torch.device('cpu'))
    layer = op.Linear(256, 128)
    state = {'weight': torch.randn(128, 256, dtype=torch.bfloat16),
             'bias': torch.randn(128, dtype=torch.bfloat16)}
    layer.load_state_dict({'weight': restored_fixture(state['weight']), 'bias': state['bias']})
    reference = bnb.nn.Linear4bit(256, 128, compute_dtype=torch.bfloat16,
                                 compress_statistics=False, quant_type='nf4')
    reference.weight = bnb.nn.Params4bit(state['weight'], requires_grad=False,
                                        compress_statistics=False, quant_type='nf4')
    reference.bias = torch.nn.Parameter(state['bias'], requires_grad=False)
    reference.cuda()
    x = torch.randn(1, 256, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        assert torch.equal(layer(x), reference(x))


def packed_fixture(source=None):
    bnb = pytest.importorskip("bitsandbytes.functional")
    if source is None:
        source = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    packed, qstate = bnb.quantize_4bit(source.cuda(), blocksize=64,
                                     quant_type="nf4", compress_statistics=False)
    state = {"img_in.weight": packed.cpu(), "img_in.weight_scale": qstate.absmax.cpu(),
             "img_in.weight_code": qstate.code.cpu()}
    layers = {"img_in": dict(format="nf4", shape=list(source.shape), blocksize=64,
                             dtype="bfloat16", double_quant=False)}
    return state, layers



def restored_fixture(source):
    state, layers = packed_fixture(source)
    comfy.quant_nf4.restore_weights(state, layers)
    return state["img_in.weight"]


@requires_cuda
def test_runtime_quantization_is_unsupported(monkeypatch):
    source = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    weight = restored_fixture(source)
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported runtime quantization reached bitsandbytes")
    monkeypatch.setattr(comfy.quant_nf4.backend(), "quantize_4bit", forbidden)
    with pytest.raises(NotImplementedError, match="prequantized checkpoint and separate LoRA"):
        QuantizedTensor.from_float(source, "NF4Layout")
    with pytest.raises(NotImplementedError, match="prequantized checkpoint and separate LoRA"):
        weight.requantize_from_float(source)


@requires_cuda
def test_saved_nf4_restores_without_quantization(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    state, layers = packed_fixture()
    path = tmp_path / "backbone.safetensors"
    save_file(state, path, metadata={"_quantization_metadata": json.dumps({"layers": layers})})
    def forbidden(*args, **kwargs):
        pytest.fail("Loading a prequantized checkpoint called quantize_4bit")
    monkeypatch.setattr(comfy.quant_nf4.backend(), "quantize_4bit", forbidden)
    restored, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    comfy.quant_nf4.restore_weights(restored, json.loads(metadata["_quantization_metadata"])["layers"])
    weight = restored["img_in.weight"]
    assert weight.shape == (64, 128)
    assert torch.equal(weight._qdata, state["img_in.weight"])
    assert torch.equal(weight.params.scale, state["img_in.weight_scale"])
    assert torch.equal(weight.params.code, state["img_in.weight_code"])
    layer = comfy.marigold.nf4_operations(torch.device("cpu")).Linear(128, 64, bias=False)
    layer.load_state_dict({"weight": weight})
    x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    assert layer(x).shape == (1, 64)


@pytest.mark.parametrize("fault", ["missing_scale", "wrong_shape", "double_quant", "wrong_dtype", "wrong_code"])
@requires_cuda
def test_malformed_saved_nf4(fault):
    state, layers = packed_fixture()
    if fault == "missing_scale":
        state.pop("img_in.weight_scale")
    elif fault == "wrong_shape":
        layers["img_in"]["shape"] = [65, 128]
    elif fault == "double_quant":
        layers["img_in"]["double_quant"] = True
    elif fault == "wrong_dtype":
        state["img_in.weight"] = state["img_in.weight"].float()
    else:
        state["img_in.weight_code"] = torch.zeros(15)
    with pytest.raises(ValueError, match="Marigold NF4"):
        comfy.quant_nf4.restore_weights(state, layers)


def test_rejects_bf16_checkpoint(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(comfy.quant_nf4, "backend", lambda: None)
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda *args, **kwargs: ({}, {}))
    with pytest.raises(ValueError, match="prequantized"):
        comfy.marigold.load_model("backbone.safetensors", "lora.safetensors")


def test_missing_optional_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", None)
    with pytest.raises(RuntimeError, match="pip install bitsandbytes"):
        comfy.quant_nf4.backend()
