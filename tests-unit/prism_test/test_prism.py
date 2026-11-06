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

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_chunked_rotary_matches_reference(dtype, device):
    """Chunk boundaries and multiple batches preserve the original complex math."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x = torch.randn(2, 517, 256, dtype=dtype, device=device)
    table = rotary_table(128, 517, torch.device(device)).view(517, 1, 64)
    pairs = x.reshape(2, 517, 2, 64, 2).double().contiguous()
    expected = torch.view_as_real(torch.view_as_complex(pairs) * table).flatten(2).to(dtype)
    actual = apply_rotary(x, table, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert actual.device == x.device


def test_rotary_noncontiguous_input():
    """Rotation accepts the transposed layout used by model activations."""
    x = torch.randn(2, 256, 517).transpose(1, 2)
    table = rotary_table(128, 517).view(517, 1, 64)
    pairs = x.reshape(2, 517, 2, 64, 2).double().contiguous()
    expected = torch.view_as_real(torch.view_as_complex(pairs) * table).flatten(2).to(x.dtype)
    torch.testing.assert_close(apply_rotary(x, table, 2), expected, rtol=0, atol=0)


def test_fp32_rotary_fallback(monkeypatch):
    """Backends without FP64 use FP32 for time and rotary embeddings."""
    from comfy.ldm.prism.model import timestep_embedding
    from comfy import model_management
    monkeypatch.setattr(model_management, 'supports_fp64', lambda device: False)
    table = rotary_table(128, 517).unsqueeze(1)
    assert table.dtype == torch.float32
    assert not table.is_complex()
    x = torch.randn(2, 517, 256, dtype=torch.bfloat16)
    pairs = x.reshape(2, 517, 2, 64, 2).float()
    expected = (table[..., 0] * pairs[..., :1] + table[..., 1] * pairs[..., 1:]).flatten(2).to(x.dtype)
    torch.testing.assert_close(apply_rotary(x, table, 2), expected, rtol=0.01, atol=0.015625)
    timestep = torch.tensor([0., 500., 1000.])
    phase = torch.outer(timestep, torch.pow(10000, -torch.arange(128).float() / 128))
    torch.testing.assert_close(timestep_embedding(256, timestep), torch.cat((phase.cos(), phase.sin()), dim=-1))


@pytest.mark.parametrize('audio,grid', [(False, (5, 3, 7)), (True, (517,))])
def test_forward_local_frequency_tables(audio, grid):
    """Device-local tables retain the released temporal/spatial frequency layout."""
    from comfy.ldm.prism.model import Tower
    tower = Tower(256, 2, 512, 0, audio, 'meta', torch.bfloat16, ops.manual_cast)
    assert not hasattr(tower, 'freqs')
    if audio:
        expected = torch.cat(rotary_table(128, 16384).chunk(3, dim=-1), dim=-1)[:grid[0]].reshape(grid[0], 1, -1)
    else:
        t, h, w = grid
        f = [rotary_table(d, 1024) for d in (44, 42, 42)]
        expected = torch.cat((f[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            f[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            f[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)), dim=-1).reshape(t*h*w, 1, -1)
    actual = tower.assemble_freqs(grid, torch.device('cpu'))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_default_attention_does_not_import_sparse(monkeypatch):
    """A multi-frame default call uses dense attention without loading Triton."""
    import sys
    from comfy.ldm.prism.model import SelfAttention
    monkeypatch.setitem(sys.modules, 'comfy.ldm.prism.sparse', None)
    attention = SelfAttention(256, 2, 1e-6, 'cpu', torch.float32, ops.manual_cast)
    for name, parameter in attention.named_parameters():
        torch.nn.init.constant_(parameter, 1.0 if 'norm_' in name else 0.01)
    x = torch.randn(1, 8, 256)
    table = rotary_table(128, 8).view(8, 1, 64)
    actual = attention(x, table, (2, 2, 2), True, {})
    expected = attention(x, table, (2, 2, 2), True, {'prism_attention': 'dense'})
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('grid', [(4, 4, 4), (5, 3, 7)])
def test_sparse_forward_cuda(grid):
    """Forward-only kernels agree with dense attention when all blocks are kept."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('triton')
    from comfy.ldm.prism.sparse import sparse_attention
    import math
    streams = [torch.randn(1, math.prod(grid), 256, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    actual = sparse_attention(*streams, 2, grid, sparsity=0.0, cdf_threshold=1.0)
    q, k, v = [x.reshape(1, -1, 2, 128).transpose(1, 2) for x in streams]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

@pytest.mark.parametrize('audio,grid', [(False, (5, 3, 7)), (True, (517,))])
def test_forward_local_fp32_frequency_tables(monkeypatch, audio, grid):
    """Real-matrix tables broadcast correctly for both non-FP64 tower layouts."""
    from comfy.ldm.prism.model import Tower
    from comfy import model_management
    monkeypatch.setattr(model_management, 'supports_fp64', lambda device: False)
    tower = Tower(256, 2, 512, 0, audio, 'meta', torch.bfloat16, ops.manual_cast)
    table = tower.assemble_freqs(grid, torch.device('cpu'))
    import math
    length = math.prod(grid)
    assert table.shape == (length, 1, 64, 2, 2)
    x = torch.randn(2, length, 256, dtype=torch.bfloat16)
    actual = apply_rotary(x, table, 2)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()


def test_timestep_embedding_fp64_reference():
    """Supported devices retain the original time-embedding rounding order."""
    from comfy.ldm.prism.model import timestep_embedding
    time = torch.tensor([0., 1., 500., 999., 1000.])
    freq = torch.pow(10000, -torch.arange(128, dtype=torch.float64) / 128)
    phase = torch.outer(time.double(), freq)
    expected = torch.cat((phase.cos(), phase.sin()), dim=-1).to(time.dtype)
    torch.testing.assert_close(timestep_embedding(256, time), expected, rtol=0, atol=0)


def test_explicit_sparse_tail_safe_attention(monkeypatch):
    """The explicit sparse option still replaces incomplete temporal-tail queries."""
    import sys
    import types
    from comfy.ldm.prism.model import SelfAttention
    calls = []
    def sparse(q, k, v, heads, grid, sparsity, cdf_threshold):
        calls.append((grid, sparsity, cdf_threshold))
        return torch.zeros_like(q)
    monkeypatch.setitem(sys.modules, 'comfy.ldm.prism.sparse', types.SimpleNamespace(sparse_attention=sparse))
    attention = SelfAttention(256, 2, 1e-6, 'cpu', torch.float32, ops.manual_cast)
    for name, parameter in attention.named_parameters():
        torch.nn.init.constant_(parameter, 1.0 if 'norm_' in name else 0.01)
    x = torch.randn(1, 20, 256)
    table = rotary_table(128, 20).view(20, 1, 64)
    grid = (5, 2, 2)
    actual = attention(x, table, grid, True, {'prism_attention': 'prism_sparse_tail_safe'})
    dense = attention(x, table, grid, True, {'prism_attention': 'dense'})
    assert calls == [(grid, 0.75, 0.2)]
    torch.testing.assert_close(actual[:, 16:], dense[:, 16:])
    torch.testing.assert_close(actual[:, :16], attention.o(torch.zeros_like(x[:, :16])))
