"""Prism structural, row-scaled storage, packing and schedule regression tests."""
import json
from pathlib import Path
import torch
import pytest
from comfy import ops, memory_management
from comfy.ldm.prism.model import Prism, apply_rotary, rotary_table
from comfy.ldm.prism.quantization import RowScaledFP8Ops
from comfy import model_detection, supported_models


def test_released_checkpoint_hierarchy(monkeypatch):
    monkeypatch.setattr(memory_management, 'aimdo_enabled', False)
    model = Prism(device='meta', dtype=torch.bfloat16, operations=ops.manual_cast)
    state = model.state_dict()
    assert len(state) == 3735
    assert state['fusion_blocks.0.video_block.self_attn.q.weight'].shape == (5120, 5120)
    assert state['fusion_blocks.0.audio_block.self_attn.q.weight'].shape == (1536, 1536)
    assert state['video_dit_2.blocks.39.self_attn.q.weight'].shape == (5120, 5120)
    assert len(model.fusion_blocks) == 30
    assert len(model.remaining_video_blocks) == 10
    assert len(model.get_dynamic_units()) == 170
    config = model_detection.model_config_from_unet(state, '')
    assert isinstance(config, supported_models.Prism)
    # Every key dereferenced by the signature is guarded.
    state.pop('fusion_blocks.0.audio_block.self_attn.q.weight')
    assert model_detection.detect_unet_config(state, '') is None


@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_row_scaled_fp8_linear_and_export(monkeypatch, lazy, dtype):
    monkeypatch.setattr(memory_management, 'aimdo_enabled', lazy)
    layer = RowScaledFP8Ops.Linear(256, 17, dtype=dtype, device='meta')
    gen = torch.Generator().manual_seed(12)
    q = torch.randn(17, 256, generator=gen).to(torch.float8_e4m3fn)
    bias = torch.randn(17, dtype=torch.bfloat16, generator=gen)
    layer.load_state_dict({'weight': q, 'bias': bias}, strict=True, assign=True)
    layer.prism_scale = torch.linspace(0.001, 2.0, 17).view(-1, 1)
    x = torch.randn(2, 8, 256, dtype=dtype, generator=gen)
    expected = torch.nn.functional.linear(x, (q.float() * layer.prism_scale).to(dtype), bias.to(dtype))
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert layer.weight.dtype == torch.float8_e4m3fn
    state = layer.state_dict()
    assert set(state) == {'weight', 'bias', 'weight.prism_scale'}
    assert state['weight'].data_ptr() == q.data_ptr()


def test_complex_rotary_reference():
    x = torch.randn(1, 8, 256, dtype=torch.bfloat16)
    table = rotary_table(128, 8).view(8, 1, 64)
    pairs = x.reshape(1, 8, 2, 64, 2).double()
    expected = torch.view_as_real(torch.view_as_complex(pairs) * table).flatten(2).to(x.dtype)
    torch.testing.assert_close(apply_rotary(x, table, 2), expected, rtol=0, atol=0)


def test_shared_workflow_uses_native_nodes():
    path = Path(__file__).resolve().parents[2] / 'docs/prism/prism_native_test_workflow.json'
    workflow = json.loads(path.read_text(encoding='utf-8'))
    nodes = {node['id']: node for node in workflow['nodes']}
    assert nodes[1]['type'] == 'UNETLoader'
    assert nodes[8]['type'] == 'KSampler'
    assert nodes[11]['type'] == 'VAELoader'
    assert nodes[12]['type'] == 'VAEDecodeAudio'
    assert nodes[7]['widgets_values'][2] == 81
    for _, source, slot, target, input_slot, kind in workflow['links']:
        assert nodes[source]['outputs'][slot]['type'] == kind
        assert nodes[target]['inputs'][input_slot]['type'] == kind


def test_native_int8_meta_loading_and_roundtrip(monkeypatch):
    from comfy.quant_ops import TensorWiseINT8Layout, QuantizedTensor
    import comfy.ldm.prism.model

    class TinyPrism(torch.nn.Module):
        def __init__(self, device=None, dtype=None, operations=None, **kwargs):
            super().__init__()
            self.dtype = dtype
            self.linear = operations.Linear(256, 17, bias=False, dtype=dtype, device=device)

    monkeypatch.setattr(comfy.ldm.prism.model, 'Prism', TinyPrism)
    weight = torch.randn(17, 256)
    q, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=256)
    conf = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}
    state = {'linear.weight': q, 'linear.weight_scale': params.scale,
             'linear.comfy_quant': torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}
    config = supported_models.Prism({'image_model': 'prism_mova'})
    config.quant_config = {'mixed_ops': True}
    config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
    model = config.get_model(state)
    # Invalid metadata must fail rather than silently ignoring rotation.
    bad = state.copy()
    bad['linear.weight_scale'] = torch.zeros_like(params.scale)
    with pytest.raises(ValueError, match='finite and positive'):
        model.load_model_weights(bad)
    model.load_model_weights(state.copy())
    layer = model.diffusion_model.linear
    assert isinstance(layer.weight, QuantizedTensor)
    assert layer.weight.device.type == 'cpu'
    assert layer.weight._params.convrot
    assert layer.weight._qdata.data_ptr() == q.data_ptr()
    exported = model.diffusion_model.state_dict()
    assert set(exported) == set(state)
    assert json.loads(bytes(exported['linear.comfy_quant'].tolist())) == conf
    torch.testing.assert_close(exported['linear.weight_scale'], params.scale, rtol=0, atol=0)
