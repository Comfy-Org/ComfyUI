import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from comfy_api.latest import _sdk


_FROM_WEIGHTS = object()


class Patchable:
    def __init__(self, weights, response=_FROM_WEIGHTS):
        self.weights = dict(weights)
        self.response = response
        self.applied = []
        self.prefixes = []
        self.clones = []

    def get_key_patches(self, prefix=None):
        self.prefixes.append(prefix)
        if self.response is not _FROM_WEIGHTS:
            return self.response
        return {
            key: (value,) for key, value in self.weights.items()
            if prefix is None or key.startswith(prefix)
        }

    def clone(self):
        result = Patchable(self.weights)
        self.clones.append(result)
        return result

    def add_patches(self, patches, donor_factor, receiver_factor):
        self.applied.append((patches, donor_factor, receiver_factor))
        for key, (value,) in patches.items():
            self.weights[key] = (
                self.weights.get(key, 0) * receiver_factor + value * donor_factor)


async def merge_case(kind, response=_FROM_WEIGHTS):
    key = "diffusion_model.weight" if kind == "MODEL" else "encoder.weight"
    refs = _sdk.InProcessRefResolver()
    ops = _sdk.InProcessOps()
    ref_type = _sdk.ModelRef if kind == "MODEL" else _sdk.ClipRef
    receiver = Patchable({key: 2, "encoder.position_ids": 3, "encoder.logit_scale": 4})
    donor = Patchable(
        {key: 10, "encoder.position_ids": 30, "encoder.logit_scale": 40}, response)
    first = ref_type._wrap(await refs.create(kind, receiver))
    second = ref_type._wrap(await refs.create(kind, donor))
    return SimpleNamespace(
        key=key, refs=refs, ops=ops, ref_type=ref_type, receiver=receiver,
        donor=donor, first=first, second=second,
    )


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("ratio", [0.25, -0.5, 1.5])
def test_merge_clones_the_receiver_and_preserves_both_inputs(kind, ratio):
    async def run():
        case = await merge_case(kind)
        before = (dict(case.receiver.weights), dict(case.donor.weights))
        with _sdk.bind_runtime(case.refs, None, case.ops):
            result_ref = await case.first.merge(case.second, ratio)
        result = await case.refs.resolve(result_ref)
        assert isinstance(result_ref, case.ref_type)
        assert result_ref.id not in {case.first.id, case.second.id}
        assert result is case.receiver.clones[0]
        assert result.weights[case.key] == 2 * ratio + 10 * (1 - ratio)
        assert result.weights["encoder.position_ids"] == 3
        assert result.weights["encoder.logit_scale"] == 4
        assert result.applied == [({case.key: (10,)}, 1 - ratio, ratio)]
        assert case.donor.prefixes == ["diffusion_model." if kind == "MODEL" else None]
        assert (case.receiver.weights, case.donor.weights) == before
        assert case.donor.clones == []

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("ratio,error", [
    (True, TypeError), (False, TypeError), (None, TypeError), ("0.5", TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError),
])
def test_invalid_ratios_are_refused_before_native_patching(kind, ratio, error):
    async def run():
        case = await merge_case(kind)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            with pytest.raises(error):
                await case.first.merge(case.second, ratio)
        assert case.receiver.clones == []
        assert case.donor.prefixes == []

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("operation", ["merge", "weighted_add"])
@pytest.mark.parametrize("invalid", ["other_type", "forged_kind", "foreign", "released", "receiver_kind"])
def test_wrong_or_unowned_refs_are_refused_before_cloning(kind, operation, invalid):
    async def run():
        case = await merge_case(kind)
        first, second = case.first, case.second
        error = TypeError
        other_kind = "CLIP" if kind == "MODEL" else "MODEL"
        other_type = _sdk.ClipRef if kind == "MODEL" else _sdk.ModelRef
        if invalid == "other_type":
            second = other_type._wrap(await case.refs.create(other_kind, case.donor))
        elif invalid == "forged_kind":
            handle = await case.refs.create(other_kind, case.donor)
            second = case.ref_type(kind=kind, id=handle.id)
        elif invalid == "foreign":
            foreign = _sdk.InProcessRefResolver()
            second = case.ref_type._wrap(await foreign.create(kind, case.donor))
            error = KeyError
        elif invalid == "released":
            await case.refs.release(second)
            error = KeyError
        else:
            first = case.ref_type(kind=other_kind, id=first.id)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            with pytest.raises(error):
                if operation == "merge":
                    await first.merge(second)
                else:
                    await first.weighted_add(second, strength_self=1, strength_other=-1)
        assert case.receiver.clones == []
        assert case.donor.prefixes == []

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("operation", ["merge", "weighted_add"])
@pytest.mark.parametrize("response", [
    None, [], {42: (1,)}, {"x" * 4097: (1,)},
    {f"weight.{index}": (1,) for index in range(65537)},
])
def test_malformed_or_oversized_patch_maps_are_refused_before_cloning(kind, operation, response):
    async def run():
        case = await merge_case(kind, response)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            with pytest.raises(ValueError, match="patches exceed bounds"):
                if operation == "merge":
                    await case.first.merge(case.second)
                else:
                    await case.first.weighted_add(case.second, strength_self=1, strength_other=-1)
        assert case.receiver.clones == []
        assert case.receiver.applied == []

    asyncio.run(run())


@pytest.mark.parametrize("response", [
    {"x" * 4096: (1,)},
    {f"weight.{index}": (1,) for index in range(65536)},
])
def test_patch_maps_at_the_declared_limits_remain_supported(response):
    async def run():
        case = await merge_case("CLIP", response)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            result = await case.first.merge(case.second)
        value = await case.refs.resolve(result)
        assert len(value.applied) == len(response)
        assert value is not case.receiver

    asyncio.run(run())


@pytest.mark.parametrize("class_name", ["ModelRef", "ClipRef"])
def test_public_declaration_matches_the_callable_merge_contract(class_name):
    ref_type = getattr(_sdk, class_name)
    signature = inspect.signature(ref_type.merge)
    hints = get_type_hints(ref_type.merge)
    assert tuple(signature.parameters) == ("self", "other", "ratio")
    assert hints == {"other": ref_type, "ratio": float, "return": ref_type}
    assert signature.parameters["ratio"].default == 0.5
    stub = ast.parse(Path(_sdk.__file__).with_name("_sdk_public.pyi").read_text())
    declaration = next(
        node for node in stub.body
        if isinstance(node, ast.ClassDef) and node.name == class_name)
    merge = next(
        node for node in declaration.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "merge")
    assert tuple(arg.arg for arg in merge.args.args) == ("self", "other", "ratio")
    assert merge.args.args[1].annotation.value == class_name
    assert merge.args.args[2].annotation.id == "float"
    assert merge.returns.value == class_name


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("strengths", [(1, -1), (0, 0), (0.25, 2.5), (-1.5, 0)])
def test_weighted_add_preserves_independent_native_factors(kind, strengths):
    async def run():
        case = await merge_case(kind)
        before = dict(case.receiver.weights), dict(case.donor.weights)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            ref = await case.first.weighted_add(
                case.second, strength_self=strengths[0], strength_other=strengths[1])
        result = await case.refs.resolve(ref)
        assert isinstance(ref, case.ref_type)
        assert ref.id not in {case.first.id, case.second.id}
        assert result.weights[case.key] == 2 * strengths[0] + 10 * strengths[1]
        assert result.applied == [({case.key: (10,)}, strengths[1], strengths[0])]
        assert (case.receiver.weights, case.donor.weights) == before

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("name", ["strength_self", "strength_other"])
@pytest.mark.parametrize("value,error", [
    (True, TypeError), (None, TypeError), ("0.5", TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError),
])
def test_weighted_add_validates_both_strengths_before_patching(kind, name, value, error):
    async def run():
        case = await merge_case(kind)
        strengths = {"strength_self": 1, "strength_other": -1, name: value}
        with _sdk.bind_runtime(case.refs, None, case.ops):
            with pytest.raises(error, match=name):
                await case.first.weighted_add(case.second, **strengths)
        assert case.receiver.clones == []
        assert case.donor.prefixes == []

    asyncio.run(run())


@pytest.mark.parametrize("class_name", ["ModelRef", "ClipRef"])
def test_weighted_add_requires_both_keyword_strengths_and_matches_stub(class_name):
    ref_type = getattr(_sdk, class_name)
    signature = inspect.signature(ref_type.weighted_add)
    for name in ("strength_self", "strength_other"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty
    stub = ast.parse(Path(_sdk.__file__).with_name("_sdk_public.pyi").read_text())
    declaration = next(node for node in stub.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in declaration.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "weighted_add")
    assert [arg.arg for arg in method.args.kwonlyargs] == ["strength_self", "strength_other"]
    assert method.args.kw_defaults == [None, None]

    async def run():
        case = await merge_case(ref_type.KIND)
        with _sdk.bind_runtime(case.refs, None, case.ops):
            with pytest.raises(TypeError):
                await case.first.weighted_add(case.second, 1, -1)
            for missing in ("strength_self", "strength_other"):
                with pytest.raises(TypeError):
                    await case.first.weighted_add(case.second, **{missing: 1})
        assert case.receiver.clones == []

    asyncio.run(run())
