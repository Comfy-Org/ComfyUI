import pytest
import torch

from comfy.cli_args import args as cli_args

cli_args.cpu = True

import comfy.lora  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
from comfy.weight_adapter.bypass import (  # noqa: E402
    BypassInjectionManager,
    is_transposed_conv,
)
from comfy.weight_adapter.glora import GLoRAAdapter  # noqa: E402
from comfy.weight_adapter.loha import LoHaAdapter  # noqa: E402
from comfy.weight_adapter.lokr import LoKrAdapter  # noqa: E402
from comfy.weight_adapter.lora import LoRAAdapter  # noqa: E402
from comfy.weight_adapter.oft import OFTAdapter  # noqa: E402

DIM = 8
RANK = 2
KEY = "qkv.weight"
SLICES = {name: (0, i * DIM, DIM) for i, name in enumerate("qkv")}


class Fused(torch.nn.Module):
    """A fused qkv projection: one Linear(DIM, 3 * DIM) without bias."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(DIM, 3 * DIM, bias=False)


def make_adapter(kind, generator):
    """An adapter for one DIM x DIM slice, built through its load() from a synthetic state dict."""

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.3

    if kind == "lora":
        sd = {
            "a.lora_up.weight": rand(DIM, RANK),
            "a.lora_down.weight": rand(RANK, DIM),
        }
        return LoRAAdapter.load("a", sd, alpha=1.0, dora_scale=None)
    if kind == "loha":
        sd = {
            "a.hada_w1_a": rand(DIM, RANK),
            "a.hada_w1_b": rand(RANK, DIM),
            "a.hada_w2_a": rand(DIM, RANK),
            "a.hada_w2_b": rand(RANK, DIM),
        }
        return LoHaAdapter.load("a", sd, alpha=float(RANK), dora_scale=None)
    if kind == "lokr":
        sd = {"a.lokr_w1": rand(2, 2), "a.lokr_w2": rand(DIM // 2, DIM // 2)}
        return LoKrAdapter.load("a", sd, alpha=None, dora_scale=None)
    if kind == "oft":
        sd = {"a.oft_blocks": rand(2, DIM // 2, DIM // 2)}
        return OFTAdapter.load("a", sd, alpha=0.0, dora_scale=None)
    raise ValueError(kind)


def merged_weight(weight, patches):
    """Merge path reference: comfy.lora.calculate_weight over (strength, adapter, offset) patches."""
    patch_list = [
        (strength, adapter, 1.0, offset, None) for adapter, strength, offset in patches
    ]
    return comfy.lora.calculate_weight(patch_list, weight.clone(), KEY)


def bypass_output(model, patches, x):
    """Bypass path: register every patch on one manager, inject, run, eject."""
    manager = BypassInjectionManager()
    for adapter, strength, offset in patches:
        manager.add_adapter(KEY, adapter, strength=strength, offset=offset)
    (injection,) = manager.create_injections(model)
    injection.inject(None)
    out = model.qkv(x)
    injection.eject(None)
    return out


@pytest.fixture
def setup():
    generator = torch.Generator().manual_seed(0)
    model = Fused()
    with torch.no_grad():
        model.qkv.weight.copy_(torch.randn(3 * DIM, DIM, generator=generator))
    x = torch.randn(2, 5, DIM, generator=generator)
    return model, x, generator


@pytest.mark.parametrize("kind", ["lora", "loha", "lokr", "oft"])
def test_sliced_bypass_matches_merge(setup, kind):
    model, x, generator = setup
    # OFT's merge path scales its diff by strength twice; bypass scales once, so they agree at 1.0
    strength = 1.0 if kind == "oft" else 0.7
    patches = [(make_adapter(kind, generator), strength, SLICES[s]) for s in "qkv"]

    merged = torch.nn.functional.linear(
        x, merged_weight(model.qkv.weight.detach(), patches)
    )
    bypassed = bypass_output(model, patches, x)

    assert not torch.allclose(merged, model.qkv(x), atol=1e-3)
    torch.testing.assert_close(bypassed, merged, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["lora", "loha", "lokr", "oft"])
def test_sliced_bypass_matches_unfused_layers(setup, kind):
    model, x, generator = setup
    patches = {s: (make_adapter(kind, generator), 1.0, SLICES[s]) for s in "qkv"}

    bypassed = bypass_output(model, list(patches.values()), x)

    for s, (adapter, strength, (_, start, size)) in patches.items():
        unfused = Fused()
        unfused.qkv = torch.nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            unfused.qkv.weight.copy_(model.qkv.weight[start : start + size])
        expected = bypass_output(unfused, [(adapter, strength, None)], x)
        torch.testing.assert_close(
            bypassed[..., start : start + size], expected, atol=1e-4, rtol=1e-4
        )


def test_wrong_slice_differs(setup):
    model, x, generator = setup
    adapter = make_adapter("lora", generator)

    merged = torch.nn.functional.linear(
        x, merged_weight(model.qkv.weight.detach(), [(adapter, 1.0, SLICES["k"])])
    )
    bypassed = bypass_output(model, [(adapter, 1.0, SLICES["q"])], x)

    assert not torch.allclose(bypassed, merged, atol=1e-3)


def test_conv_slice_uses_channel_dim(setup):
    _, _, generator = setup
    model = torch.nn.Module()
    model.qkv = torch.nn.Conv2d(DIM, 3 * DIM, 1, bias=False)
    x = torch.randn(2, DIM, 3, 3, generator=generator)
    adapter = make_adapter("lora", generator)
    k = SLICES["k"]

    merged = merged_weight(model.qkv.weight.detach().flatten(1), [(adapter, 1.0, k)])
    expected = torch.nn.functional.conv2d(x, merged.view_as(model.qkv.weight))
    bypassed = bypass_output(model, [(adapter, 1.0, k)], x)

    torch.testing.assert_close(bypassed, expected, atol=1e-4, rtol=1e-4)


def test_chained_loads_stack_and_eject_to_original(setup):
    model, x, generator = setup
    base = model.qkv(x)
    patcher = comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu")
    )
    patches = [(make_adapter("lora", generator), 1.0, SLICES[s]) for s in "qv"]

    for adapter, strength, offset in patches:
        manager = BypassInjectionManager()
        manager.add_adapter(KEY, adapter, strength=strength, offset=offset)
        patcher.set_injections(
            comfy.sd._bypass_injection_name(patcher), manager.create_injections(model)
        )
    assert list(patcher.injections) == ["bypass_lora_0", "bypass_lora_1"]

    patcher.inject_model()
    stacked = model.qkv(x)
    patcher.eject_model()

    merged = torch.nn.functional.linear(
        x, merged_weight(model.qkv.weight.detach(), patches)
    )
    torch.testing.assert_close(stacked, merged, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(model.qkv(x), base)
    assert model.qkv.forward.__func__ is torch.nn.Linear.forward


def test_add_bypass_adapters_keeps_tuple_keys(setup):
    _, _, generator = setup
    q, k = make_adapter("lora", generator), make_adapter("lora", generator)
    bypass_patches = {
        (KEY, SLICES["q"]): q,
        (KEY, SLICES["k"]): k,
        (KEY, SLICES["v"], lambda w: w): make_adapter("lora", generator),
        "missing.weight": make_adapter("lora", generator),
    }
    manager, loaded = BypassInjectionManager(), set()

    comfy.sd._add_bypass_adapters(manager, bypass_patches, {KEY}, 0.5, loaded)

    assert manager.adapters == {"qkv": [(q, 0.5, SLICES["q"]), (k, 0.5, SLICES["k"])]}
    assert loaded == {(KEY, SLICES["q"]), (KEY, SLICES["k"])}


def test_rejects_unsupported_slices(setup):
    model, _, generator = setup
    manager = BypassInjectionManager()

    with pytest.raises(ValueError, match="dim 0"):
        manager.add_adapter(KEY, make_adapter("lora", generator), offset=(1, 0, DIM))
    with pytest.raises(ValueError, match="rewrites the module input"):
        manager.add_adapter(KEY, GLoRAAdapter(set(), (None,) * 6), offset=SLICES["q"])
    for bad in ((0, -1, 1), (0, 0, -1), (0, 0, 0)):
        with pytest.raises(ValueError, match="non-negative start"):
            manager.add_adapter(KEY, make_adapter("lora", generator), offset=bad)

    manager.add_adapter(
        KEY, make_adapter("lora", generator), offset=(0, 2 * DIM, 2 * DIM)
    )
    with pytest.raises(ValueError, match="exceeds"):
        manager.create_injections(model)


def test_rejects_slice_on_transposed_conv(setup):
    _, _, generator = setup
    model = torch.nn.Module()
    model.qkv = torch.nn.ConvTranspose2d(3 * DIM, DIM, 1, bias=False)
    manager = BypassInjectionManager()
    manager.add_adapter(KEY, make_adapter("lora", generator), offset=SLICES["k"])

    with pytest.raises(ValueError, match="transposed conv"):
        manager.create_injections(model)


def test_transposed_conv_check_by_class_name():
    class TransposeLinear(torch.nn.Linear):
        pass

    class QuantConvTranspose2d(torch.nn.Module):
        pass

    assert not is_transposed_conv(TransposeLinear(DIM, DIM))
    assert is_transposed_conv(QuantConvTranspose2d())
    assert not is_transposed_conv(torch.nn.Conv2d(DIM, DIM, 1))


def test_injection_name_skips_names_in_use(setup):
    model, _, _ = setup
    patcher = comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu")
    )
    for _ in range(2):
        patcher.set_injections(comfy.sd._bypass_injection_name(patcher), [])
    patcher.remove_injections("bypass_lora_0")

    name = comfy.sd._bypass_injection_name(patcher)

    assert name == "bypass_lora_0"
    patcher.set_injections(name, [])
    assert comfy.sd._bypass_injection_name(patcher) == "bypass_lora_2"
