import json

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.ops as ops
from comfy_kitchen.tensor.w4a8_packed import (
    _pack_codes,
    default_w4a8_codebook,
    nested_w4a8_codebook,
    pack_w4a8_mma_weight,
)


def config(**values):
    return torch.tensor(list(json.dumps(values).encode()), dtype=torch.uint8)


@pytest.mark.parametrize('bits', [4, 5])
def test_nested_checkpoint_roundtrip(bits):
    n, k = 16, 256
    codes = torch.arange(n * k, dtype=torch.int32).reshape(n, k) % (1 << bits)
    scales = torch.ones((n, k // 16)).to(torch.float8_e4m3fn)
    stream, tier = pack_w4a8_mma_weight(_pack_codes(codes, bits), scales, lsb_tier=True)
    state = {
        'weight': stream, 'weight_lsb_tier': tier,
        'weight_s_channel': torch.ones(n), 'weight_codebook': nested_w4a8_codebook(),
        'comfy_quant': config(format='nested_w4a8_int8', downgrade=7, convrot_groupsize=256),
    }
    linear = ops.mixed_precision_ops().Linear(k, n, bias=False, device='cpu')
    linear.load_state_dict(state)
    saved = linear.state_dict()
    assert linear._downgrade == 7
    assert linear.weight._params.mma_packed
    assert set(saved) == set(state)
    for key in state:
        if key == 'comfy_quant':
            assert json.loads(bytes(saved[key].tolist())) == json.loads(bytes(state[key].tolist()))
        else:
            assert torch.equal(saved[key], state[key]), key


def test_w3_checkpoint_roundtrip():
    n, k = 16, 256
    state = {
        'weight': _pack_codes(torch.zeros((n, k), dtype=torch.int32), 3),
        'weight_s_rel': torch.ones((n, k // 16)).to(torch.float8_e4m3fn),
        'weight_s_channel': torch.ones(n), 'weight_codebook': default_w4a8_codebook(3),
        'comfy_quant': config(format='w3a8_int8', group_size=16, convrot_groupsize=256),
    }
    linear = ops.mixed_precision_ops().Linear(k, n, bias=False, device='cpu')
    linear.load_state_dict(state)
    saved = linear.state_dict()
    assert set(saved) == set(state)
    for key in state:
        assert torch.equal(saved[key].view(torch.uint8), state[key].view(torch.uint8)), key


@pytest.mark.parametrize('missing', ['weight_s_channel', 'weight_codebook', 'weight_lsb_tier'])
def test_nested_checkpoint_requires_all_components(missing):
    n, k = 16, 256
    scales = torch.ones((n, k // 16)).to(torch.float8_e4m3fn)
    stream, tier = pack_w4a8_mma_weight(_pack_codes(torch.zeros((n, k), dtype=torch.int32), 4), scales, lsb_tier=True)
    state = {
        'weight': stream, 'weight_lsb_tier': tier, 'weight_s_channel': torch.ones(n),
        'weight_codebook': nested_w4a8_codebook(),
        'comfy_quant': config(format='nested_w4a8_int8', downgrade=0),
    }
    del state[missing]
    linear = ops.mixed_precision_ops().Linear(k, n, bias=False, device='cpu')
    with pytest.raises(ValueError, match='Missing nested_w4a8_int8'):
        linear.load_state_dict(state)
