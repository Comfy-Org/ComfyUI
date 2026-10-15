"""Native CPU patcher regressions with tiny, untrained tensors."""
import asyncio

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import lora
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_api.latest import _sdk
from comfy_extras.nodes_model_merging import CLIPMergeSimple, ModelMergeSimple


def native_value(kind, value, dtype):
    model = torch.nn.Module()
    scope = torch.nn.Module()
    scope.weight = torch.nn.Parameter(torch.full((2, 3), value, dtype=dtype))
    model.add_module("diffusion_model" if kind == "MODEL" else "encoder", scope)
    if kind == "CLIP":
        scope.register_buffer("position_ids", torch.tensor([value], dtype=torch.int64))
        scope.logit_scale = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
    model.register_buffer("outside", torch.tensor(value, dtype=dtype))
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    if kind == "MODEL":
        return patcher
    clip = CLIP(no_init=True)
    clip.patcher = patcher
    clip.cond_stage_model = model
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {"retained": True}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return clip


def patcher_for(kind, value):
    return value if kind == "MODEL" else value.patcher


def effective_weights(patcher):
    return {
        key: lora.calculate_weight(patcher.patches.get(key, []), weight.clone(), key)
        for key, weight in patcher.model.state_dict().items()
    }


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("ratio", [0.25, -0.5, 1.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("prepatched", [False, True])
def test_sdk_merge_matches_native_weights_and_preserves_input_patchers(kind, ratio, dtype, prepatched):
    receiver = native_value(kind, 2., dtype)
    donor = native_value(kind, 10., dtype)
    receiver_patcher = patcher_for(kind, receiver)
    donor_patcher = patcher_for(kind, donor)
    key = "diffusion_model.weight" if kind == "MODEL" else "encoder.weight"
    if prepatched:
        receiver_patcher.add_patches({key: (torch.full((2, 3), 4., dtype=dtype),)}, 0.5)
        donor_patcher.add_patches({key: (torch.full((2, 3), -2., dtype=dtype),)}, 0.5)
    before = [effective_weights(value) for value in (receiver_patcher, donor_patcher)]
    before_base = [
        {name: weight.clone() for name, weight in value.model.state_dict().items()}
        for value in (receiver_patcher, donor_patcher)
    ]

    async def run():
        refs = _sdk.InProcessRefResolver()
        ref_type = _sdk.ModelRef if kind == "MODEL" else _sdk.ClipRef
        first = ref_type._wrap(await refs.create(kind, receiver))
        second = ref_type._wrap(await refs.create(kind, donor))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await first.merge(second, ratio)
        assert isinstance(result, ref_type)
        assert result.id not in {first.id, second.id}
        return await refs.resolve(result)

    result = asyncio.run(run())
    native_merge = ModelMergeSimple() if kind == "MODEL" else CLIPMergeSimple()
    expected = native_merge.merge(receiver, donor, ratio)[0]
    result_patcher = patcher_for(kind, result)
    assert result is not receiver and result is not donor
    assert result_patcher is not receiver_patcher
    assert result_patcher.model is receiver_patcher.model
    assert result_patcher.load_device == result_patcher.offload_device == torch.device("cpu")
    assert result_patcher.patches.keys() == ({key} if kind == "MODEL" else {key, "outside"})
    actual_weights = effective_weights(result_patcher)
    expected_weights = effective_weights(patcher_for(kind, expected))
    for name, weight in actual_weights.items():
        torch.testing.assert_close(weight, expected_weights[name], rtol=0, atol=0)
        assert weight.dtype == before[0][name].dtype
        assert weight.device == torch.device("cpu")
    torch.testing.assert_close(
        actual_weights[key], before[0][key] * ratio + before[1][key] * (1 - ratio),
        rtol=0, atol=0)
    for value, original, original_base in zip(
        (receiver_patcher, donor_patcher), before, before_base,
    ):
        for name, weight in effective_weights(value).items():
            torch.testing.assert_close(weight, original[name], rtol=0, atol=0)
            torch.testing.assert_close(value.model.state_dict()[name], original_base[name], rtol=0, atol=0)
        assert len(value.patches.get(key, [])) == int(prepatched)
    if kind == "CLIP":
        assert result.tokenizer is receiver.tokenizer
        assert result.tokenizer_options == receiver.tokenizer_options
        assert result.tokenizer_options is not receiver.tokenizer_options
        for name in ("encoder.position_ids", "encoder.logit_scale"):
            torch.testing.assert_close(actual_weights[name], before[0][name], rtol=0, atol=0)
        torch.testing.assert_close(
            actual_weights["outside"], before[0]["outside"] * ratio + before[1]["outside"] * (1 - ratio),
            rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual_weights["outside"], before[0]["outside"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["MODEL", "CLIP"])
@pytest.mark.parametrize("strengths", [(1, -1), (2.75, -0.125), (0, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["overlap", "disjoint", "mismatched_shape"])
def test_independent_strengths_match_native_patch_stacks_and_missing_keys(kind, strengths, dtype, layout):
    receiver = native_value(kind, 1.3125, dtype)
    donor = native_value(kind, -2.21875, dtype)
    first, second = patcher_for(kind, receiver), patcher_for(kind, donor)
    prefix = "diffusion_model" if kind == "MODEL" else "encoder"
    key = prefix + ".weight"
    receiver_scope, donor_scope = first.model.get_submodule(prefix), second.model.get_submodule(prefix)
    receiver_scope.receiver_only = torch.nn.Parameter(torch.full((2, 3), 3.375, dtype=dtype))
    donor_scope.donor_only = torch.nn.Parameter(torch.full((2, 3), 7.625, dtype=dtype))
    if layout == "disjoint":
        del donor_scope.weight
    elif layout == "mismatched_shape":
        donor_scope.weight = torch.nn.Parameter(torch.full((2, 4), -2.21875, dtype=dtype))
    first.add_patches({key: (torch.full((2, 3), 0.3125, dtype=dtype),)}, 0.75)
    if layout == "overlap":
        second.add_patches({key: (torch.full((2, 3), -0.15625, dtype=dtype),)}, 0.625)
    before = [effective_weights(value) for value in (first, second)]
    before_base = [
        {name: weight.clone() for name, weight in value.model.state_dict().items()}
        for value in (first, second)
    ]

    async def run():
        refs = _sdk.InProcessRefResolver()
        ref_type = _sdk.ModelRef if kind == "MODEL" else _sdk.ClipRef
        a = ref_type._wrap(await refs.create(kind, receiver))
        b = ref_type._wrap(await refs.create(kind, donor))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            combined = await a.weighted_add(
                b, strength_self=strengths[0], strength_other=strengths[1])
        assert isinstance(combined, ref_type) and combined.id not in {a.id, b.id}
        return await refs.resolve(combined)

    actual = patcher_for(kind, asyncio.run(run()))
    expected_value = receiver.clone()
    patches = donor.get_key_patches("diffusion_model.") if kind == "MODEL" else donor.get_key_patches()
    for name, patch in patches.items():
        if kind == "CLIP" and name.endswith((".position_ids", ".logit_scale")):
            continue
        expected_value.add_patches({name: patch}, strengths[1], strengths[0])
    expected = patcher_for(kind, expected_value)
    assert actual is not first and actual is not second
    assert actual.model is first.model
    assert actual.patches.keys() == expected.patches.keys()
    weights = effective_weights(actual)
    for name, weight in effective_weights(expected).items():
        torch.testing.assert_close(weights[name], weight, rtol=0, atol=0)
        assert weights[name].dtype == dtype or not weights[name].is_floating_point()
        assert weights[name].device == torch.device("cpu")
    assert prefix + ".donor_only" not in weights
    torch.testing.assert_close(weights[prefix + ".receiver_only"], before[0][prefix + ".receiver_only"], rtol=0, atol=0)
    if layout == "disjoint":
        torch.testing.assert_close(weights[key], before[0][key], rtol=0, atol=0)
    for value, original, base in zip((first, second), before, before_base):
        for name, weight in effective_weights(value).items():
            torch.testing.assert_close(weight, original[name], rtol=0, atol=0)
            torch.testing.assert_close(value.model.state_dict()[name], base[name], rtol=0, atol=0)
