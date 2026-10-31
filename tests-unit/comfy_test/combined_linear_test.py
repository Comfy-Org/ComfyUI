import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.lora  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402


def test_combined_linear_declines_without_running_split_or_activation():
    linears = (comfy.ops.manual_cast.Linear(16, 24, bias=False), comfy.ops.manual_cast.Linear(16, 40, bias=False))
    combined = comfy.ops.CombinedLinear(linears)
    x = torch.randn(2, 16)
    original = x.clone()
    assert combined.forward(x) is NotImplemented
    assert comfy.ops.linear_input_act(combined, x, "model_owned_activation") is NotImplemented
    assert torch.equal(x, original)


def _gathered(layout, rows, k, **kwargs):
    weights = [QuantizedTensor.from_float(torch.randn(n, k), layout, **kwargs) for n in rows]
    gathered = torch.empty(comfy.memory_management.vram_aligned_size(weights), dtype=torch.uint8)
    views = comfy.memory_management.interpret_gathered_like(weights, gathered)
    for weight, view in zip(weights, views):
        inner_tensors, _ = weight.__tensor_flatten__()
        view.__tensor_flatten__()
        for attr in inner_tensors:
            getattr(view, attr).copy_(getattr(weight, attr))
    return weights, views, gathered


@pytest.mark.parametrize("layout, kwargs", [
    ("TensorWiseINT8Layout", {"per_channel": True, "convrot": True}),
    ("AsymW4A8Int8Layout", {}),
    ("AsymW4A8Int8Layout", {"bits": 6}),
])
def test_cat_gathered_concatenates_rows_in_place(layout, kwargs):
    torch.manual_seed(7)
    rows = (256, 128, 384)
    weights, views, gathered = _gathered(layout, rows, 512, **kwargs)
    combined = comfy.memory_management.cat_gathered(views, gathered)
    assert tuple(combined.shape) == (sum(rows), 512)
    assert combined._qdata.data_ptr() == gathered.data_ptr()
    assert torch.equal(combined.dequantize(), torch.cat([w.dequantize() for w in weights]))
    # Laid out like one gathered weight of the summed rows, so optimize_for_gemv's storage contract holds.
    inner_tensors, _ = combined.__tensor_flatten__()
    like = comfy.memory_management.interpret_gathered_like([combined], gathered)[0]
    like.__tensor_flatten__()
    for attr in inner_tensors:
        assert getattr(combined, attr).data_ptr() == getattr(like, attr).data_ptr()
    assert comfy.memory_management.vram_aligned_size(combined) <= gathered.numel()


def test_cat_gathered_declines_incompatible_weights():
    torch.manual_seed(7)
    # A tensor-wise scale cannot be shared across rows.
    weights, views, gathered = _gathered("TensorWiseINT8Layout", (128, 128), 512)
    assert comfy.memory_management.cat_gathered(views, gathered) is None
    assert all(torch.equal(v.dequantize(), w.dequantize()) for v, w in zip(views, weights))
    # Different K.
    a = _gathered("TensorWiseINT8Layout", (128,), 512, per_channel=True)[0][0]
    b = _gathered("TensorWiseINT8Layout", (128,), 256, per_channel=True)[0][0]
    gathered = torch.empty(comfy.memory_management.vram_aligned_size([a, b]), dtype=torch.uint8)
    views = comfy.memory_management.interpret_gathered_like([a, b], gathered)
    assert comfy.memory_management.cat_gathered(views, gathered) is None
    # Out of order.
    _, views, gathered = _gathered("TensorWiseINT8Layout", (128, 128), 512, per_channel=True)
    with pytest.raises(RuntimeError):
        comfy.memory_management.cat_gathered(views[::-1], gathered)


def test_int8_combined_on_hit_reloaded_on_partial_miss_including_lora():
    if not comfy.memory_management.aimdo_enabled or not torch.cuda.is_available():
        pytest.skip("requires initialized DynamicVRAM")

    torch.manual_seed(53)
    device = torch.device("cuda:0")
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = ops.Linear(512, 96, bias=False)
            self.b = ops.Linear(512, 160, bias=False)
            for layer in (self.a, self.b):
                weight = QuantizedTensor.from_float(
                    torch.randn(layer.out_features, 512, dtype=torch.bfloat16) * 0.03,
                    "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256,
                )
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                layer.quant_format = "int8_tensorwise"
                layer.layout_type = weight.layout_cls
                layer.comfy_gemv_optimized = True
            self.combined = comfy.ops.CombinedLinear((self.a, self.b))

        def get_dynamic_vram__units(self):
            return [self], []

        def forward(self, x):
            ab = self.combined.forward(x)
            return torch.cat((self.a(x), self.b(x)), dim=-1) if ab is NotImplemented else ab

    model = Projections()
    members = (model.a, model.b)
    canonical = [(s.weight._qdata.to(device), s.weight.params.scale.flatten().to(device)) for s in members]
    base = comfy.model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"))
    patches = comfy.lora.load_lora({
        "a.lora_up.weight": torch.randn(96, 4) * 0.1,
        "a.lora_down.weight": torch.randn(4, 512) * 0.1,
    }, {"a": "a.weight"})
    lora = base.clone()
    lora.add_patches(patches, strength_patch=0.75)
    x = torch.randn(3, 512, device=device, dtype=torch.bfloat16)

    def assert_unpinned():
        assert not any(flags & 2 for flags in model.a._v[0].get_residency())

    def step(combined, expected=None):
        comfy.model_prefetch.pin_modules(members, device, x.dtype)
        try:
            assert (model.combined._v_weight is not None) == combined
            out = model(x)
            if combined:
                assert all(s._v_weight is None for s in members)
                if expected:
                    # The combined weight is the members' canonical bytes, back to back.
                    weight = model.combined._v_weight
                    assert torch.equal(weight._qdata, torch.cat([q for q, _ in expected]))
                    assert torch.equal(weight.params.scale.flatten(), torch.cat([s for _, s in expected]))
            return out.clone()
        finally:
            comfy.model_prefetch.cleanup_prefetched_modules(model, members)
            assert_unpinned()

    try:
        base.partially_load(device)
        # An unprefetched member forward casts only itself and must not leave its peers pinned.
        split = torch.cat((model.a(x), model.b(x)), dim=-1)
        assert_unpinned()
        # The first pin after a load misses and runs split; the group combines on the next hit.
        results = [split, step(False), step(True, canonical), step(True, canonical)]
        # A member missing from a combined group reloads the group's canonical bytes in the same pin.
        model.b._v_signature = None
        results += [step(False), step(True, canonical)]
        assert all(torch.equal(results[0], r) for r in results[1:])

        # The baked patch is what gets combined.
        lora.partially_load(device)
        patched = [step(False), step(True)]
        assert torch.equal(patched[0], patched[1])
        assert not torch.equal(results[0], patched[0])

        base.partially_load(device)
        assert torch.equal(results[0], step(False))
        assert torch.equal(results[0], step(True, canonical))
    finally:
        torch.cuda.synchronize()
        base.unpatch_model(unpatch_weights=True)
