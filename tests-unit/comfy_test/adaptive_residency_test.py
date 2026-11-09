import dataclasses
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.model_prefetch as prefetch
import comfy.model_patcher as model_patcher
import comfy.ops as ops
from comfy.model_patcher import ModelPatcherDynamic
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_packed import _pack_codes, nested_w4a8_codebook, pack_w4a8_mma_weight, w4a8_mma_packed_numel, w4a8_mma_tier_numel


def test_block_signature_tracks_active_allocation(monkeypatch):
    native = prefetch.comfy_aimdo.model_vbar
    unpinned = []
    monkeypatch.setattr(native, 'vbar_fault', lambda alloc: b'unchanged-generation')
    monkeypatch.setattr(native, 'vbar_unpin', unpinned.append)
    adaptive = SimpleNamespace(_v='full')
    block = SimpleNamespace(_v_block=('core', [adaptive]))
    full = prefetch.block_fault(block)
    prefetch.cleanup_prefetched_modules(block, [])
    adaptive._v = 'compact'
    compact = prefetch.block_fault(block)
    assert full != compact
    prefetch.cleanup_prefetched_modules(block, [])
    assert unpinned == ['core', 'full', 'core', 'compact']


def test_failed_block_fault_rolls_back_pins(monkeypatch):
    native = prefetch.comfy_aimdo.model_vbar
    unpinned = []
    monkeypatch.setattr(native, 'vbar_fault', lambda alloc: b'core' if alloc == 'core' else None)
    monkeypatch.setattr(native, 'vbar_unpin', unpinned.append)
    block = SimpleNamespace(_v_block=('core', [SimpleNamespace(_v='full')]))
    assert prefetch.block_fault(block) is None
    assert unpinned == ['core']
    assert not hasattr(block, '_v_block_faulted')


def test_fault_exception_keeps_partial_pins_reachable(monkeypatch):
    native = prefetch.comfy_aimdo.model_vbar
    unpinned = []

    def fault(alloc):
        if alloc != 'core':
            raise RuntimeError('fault failed')
        return b'core'

    monkeypatch.setattr(native, 'vbar_fault', fault)
    monkeypatch.setattr(native, 'vbar_unpin', unpinned.append)
    block = SimpleNamespace(_v_block=('core', [SimpleNamespace(_v='full')]))
    with pytest.raises(RuntimeError, match='fault failed'):
        prefetch.block_fault(block)
    prefetch.cleanup_prefetched_modules(block, [])
    assert unpinned == ['core']


def test_cleanup_unpins_every_block_allocation_on_error(monkeypatch):
    unpinned = []

    def unpin(alloc):
        unpinned.append(alloc)
        if alloc == 'core':
            raise RuntimeError('unpin failed')

    monkeypatch.setattr(prefetch.comfy_aimdo.model_vbar, 'vbar_unpin', unpin)
    block = SimpleNamespace(_v_block_faulted=[('core', b'1'), ('compact', b'2')])
    with pytest.raises(RuntimeError, match='unpin failed'):
        prefetch.cleanup_prefetched_modules(block, [])
    assert unpinned == ['core', 'compact']
    assert not hasattr(block, '_v_block_faulted')


def test_accounting_and_eviction_priority():
    calls = []
    device = torch.device('cuda:0')
    patcher = object.__new__(ModelPatcherDynamic)
    patcher.load_device = device

    def vbar(name, size):
        def free(request):
            calls.append((name, request))
            return min(request, size)
        return SimpleNamespace(loaded_size=lambda: size, free_memory=free)

    patcher.model = SimpleNamespace(
        dynamic_vbars={device: vbar('core', 30), (device, 'compact'): vbar('compact', 20),
                       (device, 'full'): vbar('full', 10)},
        model_loaded_weight_memory=5,
    )
    assert patcher.loaded_size() == 65
    assert patcher.partially_unload(torch.device('cpu'), 25) == 25
    assert calls == [('full', 25), ('compact', 15)]


def nested_linear(rank, bits=4):
    n, k = 64, 512
    linear = ops.mixed_precision_ops().Linear(k, n, bias=False, device='cpu')
    params = AsymW4A8Int8Layout.Params(
        scale=torch.empty(0, dtype=torch.float8_e4m3fn), s_channel=torch.ones(n),
        codebook=nested_w4a8_codebook(), orig_dtype=torch.bfloat16, orig_shape=(n, k),
        group_size=16, convrot_groupsize=256, mma_packed=True,
        lsb_tier=torch.zeros(w4a8_mma_tier_numel(n, k), dtype=torch.uint8),
    )
    linear.weight = torch.nn.Parameter(QuantizedTensor(
        torch.zeros(w4a8_mma_packed_numel(n, k, bits - 1), dtype=torch.int8),
        'AsymW4A8Int8Layout', params), requires_grad=False)
    linear._downgrade = rank
    return linear


def test_layout_order_and_repromotion_after_compact_eviction(monkeypatch):
    created, prioritized = [], []

    class VBAR:
        def __init__(self, size, device, **kwargs):
            self.max_size, self.offset = size, 0
            self.base_addr = (len(created) + 1) << 40
            self.resident = 0
            created.append(self)

        def alloc(self, size):
            alloc = (self, self.base_addr + self.offset, size)
            self.offset += size
            assert self.offset <= self.max_size
            return alloc

        def prioritize(self):
            prioritized.append(self)

        def get_residency(self):
            return [self.resident] * ((self.max_size + (1 << 25) - 1) >> 25)

    monkeypatch.setattr(model_patcher.comfy_aimdo.model_vbar, 'ModelVBAR', VBAR)
    monkeypatch.setattr(model_patcher.comfy_aimdo.host_buffer, 'HostBuffer', lambda *a: SimpleNamespace(size=0))
    monkeypatch.setattr(model_patcher.comfy.model_management, 'pinned_hostbuf_size', lambda size: 0)
    model = torch.nn.Module()
    model.body = torch.nn.ModuleList([nested_linear(1), nested_linear(0, 5)])
    model.body.append(ops.manual_cast.Linear(512, 64, bias=False, device='cpu'))
    model.embed = ops.manual_cast.Embedding(64, 512, device='cpu')
    model.get_dynamic_vram__units = lambda: ([model.body], [model.embed])
    patcher = ModelPatcherDynamic(model, torch.device('cuda:0'), torch.device('cpu'))
    patcher.load(device_to=patcher.load_device)
    core, full, compact = created
    first, second = model.body[:2]
    assert prioritized == [full, compact, core]
    assert second._v_full[1] > first._v_full[1]
    assert second._v_compact[1] < first._v_compact[1]
    assert model.embed._v[0] is full and model.embed._v[1] > second._v_full[1]
    assert model.body[2]._v[0] is core
    offsets = [vbar.offset for vbar in created]
    for module in (first, second):
        module._v = module._v_compact
    compact.resident = 1
    patcher.load(device_to=patcher.load_device)
    assert first._v is first._v_compact and second._v is second._v_compact
    compact.resident = 0
    patcher.load(device_to=patcher.load_device)
    assert first._v is first._v_full and second._v is second._v_full
    assert [vbar.offset for vbar in created] == offsets


@pytest.mark.parametrize('with_pin', [False, True])
@pytest.mark.parametrize('compact_resident', [False, True])
def test_compact_transfer_omits_tier(monkeypatch, with_pin, compact_resident):
    module = nested_linear(0)
    module._v_full, module._v_compact = ('full', 0, 0), ('compact', 0, 0)
    module._v = module._v_full
    module._v_signature = b'old'
    module._pin_state = {'fast_disk': True}
    module.seed_key = 'body.weight'
    full_size = ops.comfy.memory_management.vram_aligned_size([module.weight, None])
    pin = torch.empty(full_size, dtype=torch.uint8)
    ops.comfy.model_management.cast_to_gathered([module.weight, None], pin)
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, 'vbar_fault',
                        lambda alloc: b'new' if compact_resident and alloc is module._v_compact else None)
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, 'vbar_signature_compare', lambda a, b: a is not None and a == b)
    monkeypatch.setattr(ops.comfy.pinned_memory, 'get_pin', lambda *a, **kw: pin if with_pin else None)
    monkeypatch.setattr(ops.comfy.pinned_memory, 'pin_memory', lambda *a, **kw: None)
    monkeypatch.setattr(ops.comfy.model_management, 'get_offload_stream', lambda device: None)
    compact_size = full_size - ops.comfy.memory_management.vram_aligned_size(module.weight._params.lsb_tier)
    dest = torch.empty(compact_size, dtype=torch.uint8)
    monkeypatch.setattr(ops.comfy_aimdo.torch, 'aimdo_to_tensor', lambda *a: dest)
    ops.cast_modules_with_vbar([module], None, torch.device('cpu'), None, False)
    assert module._v is module._v_compact
    assert module._prefetch['xfer_dest'].numel() == compact_size
    transferred = ops.comfy.memory_management.interpret_gathered_like(
        module._prefetch['cast_geometry'], module._prefetch['xfer_dest'])[0]
    assert transferred._params.lsb_tier is None
    assert module.weight._params.lsb_tier is not None
    assert torch.equal(transferred._qdata, module.weight._qdata)
    assert torch.equal(transferred._params.codebook, module.weight._params.codebook)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('compact', [False, True])
def test_combined_nested_weights_preserve_logical_order(monkeypatch, reverse, compact):
    linears = (nested_linear(0), nested_linear(1))
    weights = []
    for index, linear in enumerate(linears):
        n, k = linear.weight.shape
        codes = (torch.arange(n * k, dtype=torch.int32).reshape(n, k) + index * 7) % 16
        scales = torch.ones((n, k // 16)).to(torch.float8_e4m3fn)
        stream, tier = pack_w4a8_mma_weight(_pack_codes(codes, 4), scales, lsb_tier=True)
        params = dataclasses.replace(linear.weight._params, s_channel=torch.full((n,), 1.0 + index),
                                     lsb_tier=None if compact else tier)
        weight = QuantizedTensor(stream, 'AsymW4A8Int8Layout', params)
        linear.quant_format = 'nested_w4a8_int8'
        linear._prefetch = {'resident': True}
        weights.append(weight)
    expected = torch.cat([weight.dequantize() for weight in weights])
    order = [1, 0] if reverse else [0, 1]
    storage = torch.empty(ops.comfy.memory_management.vram_aligned_size([weights[i] for i in order]), dtype=torch.uint8)
    ops.comfy.model_management.cast_to_gathered([weights[i] for i in order], storage)
    views = ops.comfy.memory_management.interpret_gathered_like([weights[i] for i in order], storage)
    offset = 0
    vbar = object()
    for i, view in zip(order, views):
        size = ops.comfy.memory_management.vram_aligned_size(view)
        linears[i]._v = (vbar, offset, size)
        linears[i]._v_weight = view
        offset += size
    combined = ops.CombinedLinear(linears)
    monkeypatch.setattr(ops.comfy_aimdo.torch, 'aimdo_to_tensor', lambda *a: storage)
    combined.prepare(torch.device('cpu'))
    assert combined._v_weight is not None
    assert torch.equal(combined._v_weight.dequantize(), expected)
    assert all(linear._v_weight is None for linear in linears)
    combined.linears[0]._prefetch['resident'] = False
    assert combined.forward(torch.empty(1, 512)) is NotImplemented


def test_combined_declines_members_in_different_vbars(monkeypatch):
    linears = (nested_linear(0), nested_linear(1))
    for index, linear in enumerate(linears):
        linear.quant_format = 'nested_w4a8_int8'
        linear._v_weight = linear.weight
        linear._v = (object(), index * 1024, 1024)
    combined = ops.CombinedLinear(linears)
    monkeypatch.setattr(ops.comfy_aimdo.torch, 'aimdo_to_tensor', lambda *a: pytest.fail('must not touch mixed VBAR storage'))
    combined.prepare(torch.device('cpu'))
    assert combined._v_weight is None
    assert all(linear._v_weight is linear.weight for linear in linears)


def test_combined_member_downgrade_reloads_both_members(monkeypatch):
    linears = (nested_linear(0), nested_linear(1))
    combined = ops.CombinedLinear(linears)
    combined._v_weight = object()
    for index, linear in enumerate(linears):
        linear._v_full = ('full', index, 0)
        linear._v_compact = ('compact', index, 0)
        linear._v = linear._v_full
        linear._v_signature = b'old'
        linear._pin_state = {'fast_disk': True}
        linear.seed_key = str(index)
    faults = []

    def fault(alloc):
        faults.append(alloc)
        return b'old' if alloc is linears[1]._v_full else None

    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, 'vbar_fault', fault)
    monkeypatch.setattr(ops.comfy_aimdo.model_vbar, 'vbar_signature_compare', lambda a, b: a is not None and a == b)
    monkeypatch.setattr(ops.comfy.pinned_memory, 'get_pin', lambda *a, **kw: None)
    monkeypatch.setattr(ops.comfy.pinned_memory, 'pin_memory', lambda *a, **kw: None)
    monkeypatch.setattr(ops.comfy.model_management, 'get_offload_stream', lambda device: None)
    size = ops.comfy.memory_management.vram_aligned_size(linears[1].weight)
    monkeypatch.setattr(ops.comfy_aimdo.torch, 'aimdo_to_tensor', lambda *a: torch.empty(size, dtype=torch.uint8))
    ops.cast_modules_with_vbar(linears, None, torch.device('cpu'), None, False)
    assert combined._v_weight is None
    assert linears[0]._v is linears[0]._v_compact
    assert linears[1]._v is linears[1]._v_full
    assert not any(linear._prefetch['resident'] for linear in linears)
    assert faults == [linears[0]._v_full, linears[0]._v_compact, linears[1]._v_full]


def test_pin_modules_reconsiders_group_when_later_member_changes(monkeypatch):
    linears = (nested_linear(0), nested_linear(1))
    group = ops.CombinedLinear(linears)
    prepared = []
    for index, linear in enumerate(linears):
        linear._v_relayout = index == 0
        linear._pin_state = {'fast_disk': True}
    monkeypatch.setattr(group, 'prepare', prepared.append)
    monkeypatch.setattr(ops, 'cast_modules_with_vbar', lambda *a, **kw: (None, True))
    monkeypatch.setattr(ops, 'resolve_cast_module_with_vbar', lambda *a, **kw: None)
    monkeypatch.setattr(ops.comfy.model_management, 'sync_stream', lambda *a: None)
    prefetch.pin_modules(linears, torch.device('cpu'), torch.bfloat16)
    assert prepared == [torch.device('cpu')]
