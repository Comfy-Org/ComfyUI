"""Closed attention selection on real SDK refs and untrained ModelPatchers.

The available INT8 registry entry is an explicit CPU fixture, not an INT8 kernel
or trained-model parity test. Native ModelAttentionBackend is the reference.
"""
import asyncio

import pytest
import torch

from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import _model_transforms as mt
from comfy_api.latest import _sdk
from comfy_extras.nodes_model_advanced import ModelAttentionBackend


def patcher():
    result = ModelPatcher(torch.nn.Linear(4, 4), torch.device("cpu"), torch.device("cpu"))
    result.model_options["transformer_options"]["retained_option"] = {"value": 7}
    return result


def cpu_int8_fixture(*args, **kwargs):
    return attention.attention_pytorch(*args, **kwargs) + 0.25


def container_fixture(*args, **kwargs):
    return args, kwargs


cpu_int8_fixture.container_function = container_fixture


def inputs(masked):
    q = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 32
    k = q.flip(1)
    v = q + 0.5
    kwargs = {"mask": torch.tensor([[True, True, False, True]])} if masked else {}
    return (q, k, v, 2), kwargs


async def apply(model, **params):
    refs = _sdk.InProcessRefResolver()
    source = _sdk.ModelRef._wrap(await refs.create("MODEL", model))
    with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
        result = await source.patch("attention_impl", **params)
    assert isinstance(result, _sdk.ModelRef)
    assert result.id != source.id
    assert await refs.resolve(source) is model
    return await refs.resolve(result)


@pytest.mark.parametrize("allow_compile", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_available_selection_matches_native_clone_and_values(monkeypatch, allow_compile, masked):
    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8", cpu_int8_fixture)
    original = patcher()
    selected = asyncio.run(apply(original, mode="comfy_kitchen_int8", allow_compile=allow_compile))
    native = ModelAttentionBackend.execute(original, "comfy kitchen attention").result[0]
    args, kwargs = inputs(masked)
    got = selected.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs)
    expected = native.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs)
    assert torch.equal(got, expected)
    assert not torch.equal(got, attention.attention_pytorch(*args, **kwargs))
    assert selected is not original and selected.model is original.model
    assert selected.parent is original
    assert "optimized_attention_override" not in original.model_options["transformer_options"]
    assert selected.model_options["transformer_options"]["retained_option"] == {"value": 7}


@pytest.mark.parametrize("allow_compile", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_unavailable_selection_matches_native_pytorch_fallback(monkeypatch, caplog, allow_compile, masked):
    monkeypatch.delitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8", raising=False)
    original = patcher()
    selected = asyncio.run(apply(original, mode="comfy_kitchen_int8", allow_compile=allow_compile))
    native = ModelAttentionBackend.execute(original, "comfy kitchen attention").result[0]
    args, kwargs = inputs(masked)
    got = selected.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs)
    expected = native.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs)
    assert torch.equal(got, expected)
    assert selected is not original and selected.parent is original
    assert "optimized_attention_override" not in original.model_options["transformer_options"]
    assert "unavailable; using PyTorch attention" in caplog.text


@pytest.mark.parametrize("allow_compile", [False, True])
def test_explicit_pytorch_choice_keeps_native_values(allow_compile):
    original = patcher()
    selected = asyncio.run(apply(original, mode="pytorch", allow_compile=allow_compile))
    native = ModelAttentionBackend.execute(original, "pytorch attention").result[0]
    args, kwargs = inputs(False)
    assert torch.equal(
        selected.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs),
        native.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs),
    )


@pytest.mark.parametrize("allow_compile", [False, True])
def test_native_container_function_survives_selection(monkeypatch, allow_compile):
    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8", cpu_int8_fixture)
    selected = asyncio.run(apply(patcher(), mode="comfy_kitchen_int8", allow_compile=allow_compile))
    override = selected.model_options["transformer_options"]["optimized_attention_override"]
    assert override.container_function is container_fixture
    token = object()
    assert override.container_function(token, mask=None) == ((token,), {"mask": None})


def test_mode_is_described_through_existing_public_discovery():
    async def run():
        refs = _sdk.InProcessRefResolver()
        model = _sdk.ModelRef._wrap(await refs.create("MODEL", patcher()))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            return await model.transforms()
    descriptions = asyncio.run(run())
    spec = next(item for item in descriptions if item["name"] == "attention_impl")
    assert spec["params"]["mode"]["choices"] == [
        "pytorch", "basic", "split", "sub_quad", "xformers", "sage", "flash", "comfy_kitchen_int8",
    ]
    assert spec["params"]["allow_compile"]["default"] is False


@pytest.mark.parametrize("bad", ["comfy kitchen attention", "comfy_kitchen", "optimized", "guest.custom", None, 1])
def test_unknown_choice_refused_before_lookup(monkeypatch, bad):
    def lookup(*_args, **_kwargs):
        raise AssertionError("invalid mode reached the backend registry")
    monkeypatch.setattr(attention, "get_attention_function", lookup)
    with pytest.raises(mt.TransformError):
        asyncio.run(apply(patcher(), mode=bad))


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_compile_policy_remains_strict_boolean(bad):
    with pytest.raises(mt.TransformError, match="bool"):
        asyncio.run(apply(patcher(), mode="comfy_kitchen_int8", allow_compile=bad))


def test_guest_callback_still_refused_before_dispatch():
    with pytest.raises(TypeError, match="DATA"):
        asyncio.run(apply(patcher(), mode=lambda *_args: None))


@pytest.mark.parametrize("invalid", ["foreign", "released", "closed", "relabelled"])
def test_model_ownership_is_checked_before_backend_lookup(monkeypatch, invalid):
    def lookup(*_args, **_kwargs):
        raise AssertionError("unowned model reached the backend registry")
    monkeypatch.setattr(attention, "get_attention_function", lookup)
    async def run():
        refs = _sdk.InProcessRefResolver()
        owner = _sdk.InProcessRefResolver() if invalid == "foreign" else refs
        kind = "IMAGE" if invalid == "relabelled" else "MODEL"
        token = await owner.create(kind, patcher())
        model = _sdk.ModelRef(kind="MODEL", id=token.id)
        if invalid == "released":
            await refs.release(model)
        elif invalid == "closed":
            refs.clear()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises((KeyError, TypeError)):
                await model.patch("attention_impl", mode="comfy_kitchen_int8")
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["lookup", "clone", "install"])
def test_selection_errors_propagate_without_mutating_original(monkeypatch, stage):
    original = patcher()
    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8", cpu_int8_fixture)
    def fail(*_args, **_kwargs):
        raise RuntimeError("selected attention failure")
    if stage == "lookup":
        monkeypatch.setattr(attention, "get_attention_function", fail)
    elif stage == "clone":
        monkeypatch.setattr(original, "clone", fail)
    else:
        monkeypatch.setattr(ModelPatcher, "set_model_optimized_attention", fail)
    with pytest.raises(RuntimeError, match="selected attention failure"):
        ModelAttentionBackend.execute(original, "comfy kitchen attention")
    with pytest.raises(RuntimeError, match="selected attention failure"):
        asyncio.run(apply(original, mode="comfy_kitchen_int8", allow_compile=True))
    assert original.model_options["transformer_options"] == {"retained_option": {"value": 7}}


@pytest.mark.parametrize("mode,availability", [("sage", "SAGE_ATTENTION_IS_AVAILABLE"), ("flash", "FLASH_ATTENTION_IS_AVAILABLE")])
def test_other_unavailable_modes_keep_refusal(monkeypatch, mode, availability):
    monkeypatch.setattr(attention, availability, False)
    with pytest.raises(mt.TransformError):
        asyncio.run(apply(patcher(), mode=mode))


def test_kernel_error_is_not_silently_replaced_with_pytorch(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("kernel failure")
    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8", fail)
    selected = asyncio.run(apply(patcher(), mode="comfy_kitchen_int8", allow_compile=True))
    args, kwargs = inputs(False)
    with pytest.raises(RuntimeError, match="kernel failure"):
        selected.model_options["transformer_options"]["optimized_attention_override"](None, *args, **kwargs)
