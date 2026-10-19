import asyncio
import copy
import gc
import struct

import pytest
import torch

from comfy_api.latest import _sdk
from comfy_api.latest._gguf_tensor import GGUFTensor, tensor_state


def gguf(path, data, shape, qtype, storage):
    # One real little-endian GGUF v3 tensor, no metadata. Offsets in the SDK
    # table are absolute; the container's tensor offset is relative to data.
    name = b'weight'
    header = struct.pack('<4sIQQQ', b'GGUF', 3, 1, 0, len(name)) + name
    header += struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *reversed(shape))
    header += struct.pack('<IQ', qtype, 0)
    offset = (len(header) + 31) // 32 * 32
    path.write_bytes(header + bytes(offset - len(header)) + data)
    return {'format': 'gguf-tensor-table/1', 'byteorder': 'little', 'tensors': [
        {'name': 'weight', 'offset': offset, 'tensor_type': qtype,
         'shape': list(shape), 'storage_shape': list(storage)}]}


@pytest.mark.parametrize('qtype,dtype,width', [(0, torch.float32, 4), (1, torch.float16, 2),
                                               (24, torch.int8, 1), (25, torch.int16, 2),
                                               (26, torch.int32, 4), (27, torch.int64, 8),
                                               (28, torch.float64, 8)])
def test_typed_storage_bits_and_lifetime(tmp_path, qtype, dtype, width):
    path = tmp_path / 'state.gguf'
    raw = torch.arange(6).to(dtype)
    data = raw.view(torch.uint8).numpy().tobytes()
    table = gguf(path, data, (2, 3), qtype, (2, 3))
    state = tensor_state(path, table, path.stat().st_size)
    tensor = state['weight']
    assert isinstance(tensor, GGUFTensor) and tensor.dtype is dtype
    assert tensor.shape == (2, 3)
    assert torch.equal(tensor.as_subclass(torch.Tensor).flatten(), raw)
    del state
    gc.collect()
    assert tensor.clone() is tensor and tensor.detach() is tensor
    # COW mappings must never modify the original asset.
    tensor.as_subclass(torch.Tensor).fill_(0)
    assert path.read_bytes()[-len(data):] == data


@pytest.mark.parametrize('qtype,width', [(8, 34), (12, 144), (13, 176), (14, 210)])
def test_quantized_storage_remains_packed(tmp_path, qtype, width):
    path = tmp_path / 'quant.gguf'
    block = 32 if qtype == 8 else 256
    data = bytes(range(width))
    table = gguf(path, data, (1, block), qtype, (1, width))
    tensor = tensor_state(path, table, path.stat().st_size)['weight']
    assert type(tensor) is GGUFTensor
    assert tensor.tensor_type == qtype
    assert tensor.shape == (1, block) and tensor.size() == (1, width)
    assert tensor.numel() == width and tensor.is_largest_weight
    assert tensor.as_subclass(torch.Tensor).numpy().tobytes() == data
    moved = tensor.to(dtype=torch.float32)
    assert moved.tensor_type == tensor.tensor_type and moved.tensor_shape == tensor.tensor_shape
    assert moved.patches == tensor.patches and moved.patches is not tensor.patches


@pytest.mark.parametrize('shape', [(4,), (2, 2)])
def test_bf16_load_rule_and_bits(tmp_path, shape):
    path = tmp_path / 'bf16.gguf'
    bits = torch.tensor([0, 16256, 49152, 32704], dtype=torch.uint16)
    data = bits.view(torch.uint8).numpy().tobytes()
    table = gguf(path, data, shape, 30, (shape[0], len(data) // shape[0]))
    tensor = tensor_state(path, table, path.stat().st_size)['weight']
    if len(shape) == 2:
        assert isinstance(tensor, GGUFTensor) and tensor.dtype is torch.uint8
        assert tensor.as_subclass(torch.Tensor).numpy().tobytes() == data
    else:
        assert type(tensor) is torch.Tensor and tensor.dtype is torch.float16
        expected = (bits.to(torch.int32) << 16).view(torch.float32).to(torch.float16)
        assert torch.equal(tensor.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize('case', ['duplicate', 'negative-offset', 'out-of-file', 'storage-size',
                                 'logical-shape', 'block', 'unknown-type', 'extra-field',
                                 'byteorder', 'overlap', 'bool-type'])
def test_untrusted_table_refusals(tmp_path, case):
    path = tmp_path / 'invalid.gguf'
    table = gguf(path, b'\0' * 16, (2, 2), 0, (2, 2))
    row = table['tensors'][0]
    if case == 'duplicate':
        table['tensors'].append(copy.deepcopy(row))
    if case == 'negative-offset':
        row['offset'] = -1
    if case == 'out-of-file':
        row['offset'] = path.stat().st_size
    if case == 'storage-size':
        row['storage_shape'] = [2, 1]
    if case == 'logical-shape':
        row['shape'] = [True, 2]
    if case == 'block':
        row['tensor_type'] = 12
    if case == 'unknown-type':
        row['tensor_type'] = 20
    if case == 'extra-field':
        row['path'] = '/elsewhere'
    if case == 'byteorder':
        table['byteorder'] = 'big'
    if case == 'bool-type':
        row['tensor_type'] = True
    if case == 'overlap':
        other = dict(row, name='other', offset=row['offset'] + 4)
        table['tensors'].append(other)
    with pytest.raises((ValueError, NotImplementedError)):
        tensor_state(path, table, path.stat().st_size)


def test_file_and_budget_refusals(tmp_path):
    path = tmp_path / 'bounded.gguf'
    table = gguf(path, b'\0' * 16, (2, 2), 0, (2, 2))
    with pytest.raises(ValueError, match='byte limit'):
        tensor_state(path, table, path.stat().st_size - 1)
    path.write_bytes(b'fail' + path.read_bytes()[4:])
    with pytest.raises(ValueError, match='magic'):
        tensor_state(path, table, path.stat().st_size)


@pytest.mark.parametrize('case', ['foreign', 'released', 'relabeled'])
def test_sdk_asset_ref_authority_precedes_mapping(tmp_path, case):
    async def run():
        refs = _sdk.InProcessRefResolver()
        foreign = _sdk.InProcessRefResolver()
        owner = foreign if case == 'foreign' else refs
        base = await owner.create('VALUE' if case == 'relabeled' else 'ASSET', str(tmp_path / 'absent'))
        source = _sdk.AssetRef(kind='ASSET', id=base.id)
        if case == 'released':
            await refs.release(source)
        plan = _sdk.ExecutionPlan(prompt_id='gguf', node_id='1', node_type='GGUF',
                                  node_module=__name__, inputs={})
        with _sdk.bind_runtime(refs, _sdk.InProcessCtxProvider().build(plan), _sdk.InProcessOps()):
            with pytest.raises((KeyError, TypeError, ValueError)):
                await _sdk._InProcessAssets().gguf_tensor_state(source, {}, max_bytes=1024)
        refs.clear()
        foreign.clear()
    asyncio.run(run())
