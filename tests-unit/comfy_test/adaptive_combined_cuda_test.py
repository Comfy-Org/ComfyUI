import dataclasses
import warnings

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.memory_management as memory_management
import comfy.model_patcher as model_patcher
import comfy.model_prefetch as prefetch
import comfy.ops as ops
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_packed import _pack_codes, nested_w4a8_codebook, pack_w4a8_mma_weight


@pytest.mark.parametrize('bits', [4, 5])
def test_nested_combined_residency_and_graph_transitions(bits):
    if not memory_management.aimdo_enabled or not torch.cuda.is_available():
        pytest.skip('requires initialized DynamicVRAM and CUDA')

    torch.manual_seed(53)
    device = torch.device('cuda:0')
    linear_ops = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = linear_ops.Linear(512, 128, bias=False)
            self.b = linear_ops.Linear(512, 128, bias=False)
            for rank, linear in enumerate((self.a, self.b)):
                n, k = 128, 512
                codes = torch.randint(0, 1 << bits, (n, k), dtype=torch.int32)
                scales = torch.ones((n, k // 16)).to(torch.float8_e4m3fn)
                stream, tier = pack_w4a8_mma_weight(_pack_codes(codes, bits), scales, lsb_tier=True)
                params = AsymW4A8Int8Layout.Params(
                    scale=torch.empty(0, dtype=torch.float8_e4m3fn), s_channel=torch.full((n,), 0.02),
                    codebook=nested_w4a8_codebook(), orig_dtype=torch.bfloat16, orig_shape=(n, k),
                    mma_packed=True, lsb_tier=tier,
                )
                linear.weight = torch.nn.Parameter(QuantizedTensor(stream, 'AsymW4A8Int8Layout', params), requires_grad=False)
                linear.quant_format = 'nested_w4a8_int8'
                linear.layout_type = AsymW4A8Int8Layout
                linear.comfy_gemv_optimized = True
                linear._downgrade = rank
            self.combined = ops.CombinedLinear((self.a, self.b))

        def get_dynamic_vram__units(self):
            return [self], []

        def forward(self, x):
            result = self.combined.forward(x)
            return torch.cat((self.a(x), self.b(x)), dim=-1) if result is NotImplemented else result

    model = Projections()
    members = (model.a, model.b)
    patcher = model_patcher.ModelPatcherDynamic(model, device, torch.device('cpu'))
    x = torch.randn(3, 512, device=device, dtype=torch.bfloat16)
    full = [linear.weight.to(device=device) for linear in members]
    compact = [QuantizedTensor(weight._qdata, weight._layout_cls, dataclasses.replace(weight._params, lsb_tier=None)) for weight in full]

    def expected(weights):
        return torch.cat([weight.linear_input_act(x) for weight in weights], dim=-1)

    signatures = []

    def step(combined, reference):
        prefetch.pin_modules(members, device, x.dtype)
        try:
            assert (model.combined._v_weight is not None) == combined
            signatures.append(prefetch.block_fault(model))
            assert signatures[-1] is not None
            actual = model(x)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = model(x)
            graph.replay()
            torch.testing.assert_close(captured, reference, rtol=0, atol=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph.reset()
            del graph
        finally:
            prefetch.cleanup_prefetched_modules(model, members)
            for vbar in patcher._vbars():
                assert not any(flags & 2 for flags in vbar.get_residency())

    try:
        patcher.partially_load(device)
        step(True, expected(full))
        model.a._v = model.a._v_compact
        model.a._v_signature = None
        step(False, expected((compact[0], full[1])))
        model.b._v = model.b._v_compact
        model.b._v_signature = None
        step(True, expected(compact))
        assert signatures[0] != signatures[1] != signatures[2]
        patcher.partially_unload(torch.device('cpu'), 1 << 30)
        patcher.load(device_to=device)
        assert all(linear._v is linear._v_full for linear in members)
        step(True, expected(full))
    finally:
        torch.cuda.synchronize()
        patcher.unpatch_model(unpatch_weights=True)
