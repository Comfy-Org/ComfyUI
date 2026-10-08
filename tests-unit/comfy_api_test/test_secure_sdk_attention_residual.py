"""Canonical attention composition with pack-side residual math, no inference."""
import asyncio
import copy

import pytest
import torch

import comfy.ops
from comfy.ldm.modules.attention import BasicTransformerBlock, optimized_attention
from comfy_api.latest import _attention_residual, _node_closures, _sdk


class _Spatial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([
            BasicTransformerBlock(4, 2, 2, context_dim=4, dtype=torch.float32, operations=comfy.ops.manual_cast)])


class _Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_blocks = torch.nn.ModuleList([torch.nn.ModuleList([_Spatial()])])
        self.middle_block = torch.nn.ModuleList([torch.nn.Identity(), _Spatial()])
        self.output_blocks = torch.nn.ModuleList([torch.nn.ModuleList([_Spatial()])])


class _Patcher:
    def __init__(self, diffusion=None):
        self.diffusion = _Diffusion() if diffusion is None else diffusion
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def clone(self):
        clone = _Patcher(self.diffusion)
        clone.model_options = copy.deepcopy(self.model_options)
        return clone

    def set_model_attn2_replace(self, fn, *key):
        self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault("attn2", {})[key] = fn


def _args():
    query = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 30
    key = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 50
    value = key.flip(1)
    options = {"n_heads": 2, "original_shape": (1, 4, 16, 16), "cond_or_uncond": [0, 1], "sigmas": torch.tensor([3.5]), "transformer_index": 2,
               "private_host_object": object(), "ad_params": {"full_length": 8, "sub_idxs": [1, 3]}}
    return query, key, value, options


def test_residual_composes_after_existing_replacement_and_clones_original():
    original = _Patcher()
    original.set_model_attn2_replace(lambda query, *args: query + 10, "input", 0)
    seen = []
    def first(base, query, metadata, tensors, masks):
        seen.append(metadata)
        assert torch.equal(tensors[0], torch.tensor([2.]))
        return query * 2
    patched = _attention_residual.attach(original, first, {"tensors": [torch.tensor([2.])]})
    twice = _attention_residual.attach(patched, lambda base, *args: base / 2, {})
    args = _args()
    query = args[0]
    fn = twice.model_options["transformer_options"]["patches_replace"]["attn2"][("input", 0, 0)]
    assert torch.equal(fn(*args), (query + 10 + query * 2) * 1.5)
    assert set(seen[0]) == {"block", "block_index", "n_heads", "original_shape", "cond_or_uncond", "sigma", "transformer_index", "full_length", "temporal_indices"}
    assert seen[0]["sigma"] == 3.5 and seen[0]["temporal_indices"] == [1, 3]
    assert set(original.model_options["transformer_options"]["patches_replace"]["attn2"]) == {("input", 0)}
    assert set(patched.model_options["transformer_options"]["patches_replace"]["attn2"]) == {("input", 0), ("input", 0, 0), ("middle", 1, 0), ("output", 0, 0)}


def test_canonical_selected_attention_is_the_base_when_no_previous_patch():
    seen = []
    args = _args()
    expected = optimized_attention(*args[:3], args[3]["n_heads"])
    def program(base, query, metadata, tensors, masks):
        seen.append(base)
        return torch.full_like(base, 0.25)
    patched = _attention_residual.attach(_Patcher(), program, {})
    fn = patched.model_options["transformer_options"]["patches_replace"]["attn2"][("middle", 1, 0)]
    result = fn(*args)
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(result, expected + 0.25)


@pytest.mark.parametrize("program", [lambda base, *args: base[..., :1], lambda base, *args: base.double(), lambda base, *args: {"host": base}, lambda base, *args: torch.empty_like(base, device="meta")])
def test_bad_residuals_fail_closed(program):
    patched = _attention_residual.attach(_Patcher(), program, {})
    fn = patched.model_options["transformer_options"]["patches_replace"]["attn2"][("input", 0, 0)]
    with pytest.raises(TypeError, match="preserve shape, dtype, and device"):
        fn(*_args())


@pytest.mark.parametrize("captures", [
    {"tensors": [torch.empty(1)] * 513}, {"masks": [torch.empty(1)] * 33},
    {"tensors": [torch.empty(134217729, device="meta")]}, {"tensors": [object()]},
    {"tensors": [torch.empty(134217729, device="meta")[:1]]},
])
def test_capture_budget_denial_precedes_patcher_clone(captures):
    with pytest.raises((TypeError, ValueError)):
        _attention_residual.attach(_Patcher(), lambda *args: None, captures)


def test_inprocess_public_closure_contract_and_512_refs():
    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(prompt_id="attention", node_id="1", node_type="test"))
        original = _Patcher()
        model = _sdk.ModelRef._wrap(await refs.create("MODEL", original))
        tensor = _sdk.TensorRef._wrap(await refs.create("TENSOR", torch.ones(1)))
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain("cross_attention_residual", lambda base, query, metadata, tensors, masks: base * tensors[-1].sum(), captures={"tensors": [tensor] * 512})
            patched = await refs.resolve(await closure.attach_model(model))
        args = _args()
        expected = optimized_attention(*args[:3], 2)
        fn = patched.model_options["transformer_options"]["patches_replace"]["attn2"][("output", 0, 0)]
        torch.testing.assert_close(fn(*args), expected * 2)
    asyncio.run(run())


def test_capture_schema_has_no_arbitrary_host_object_or_model_slot():
    spec = _node_closures.get_kind("cross_attention_residual")
    with pytest.raises(Exception, match="has no capture"):
        spec.validate_captures({"model": _sdk.ModelRef(kind="MODEL", id="bogus")})
    with pytest.raises(Exception, match="TENSOR"):
        spec.validate_captures({"tensors": [_sdk.ModelRef(kind="MODEL", id="bogus")]})


def test_closed_metadata_does_not_accept_host_object_as_scalar():
    options = _args()[3]
    for name, value in (("n_heads", object()), ("cond_or_uncond", [object()]), ("original_shape", [object()]), ("sigmas", object()), ("transformer_index", object()), ("ad_params", {"sub_idxs": [object()]})):
        supplied = dict(options)
        supplied[name] = value
        with pytest.raises(ValueError):
            _attention_residual._metadata(supplied, ("input", 0, 0))


def test_three_condition_branches_are_preserved_but_other_branch_tags_are_denied():
    options = _args()[3]
    options["cond_or_uncond"] = [0, 1, 2]
    assert _attention_residual._metadata(options, ("input", 0, 0))["cond_or_uncond"] == [0, 1, 2]
    options["cond_or_uncond"] = [3]
    with pytest.raises(ValueError, match="branch"):
        _attention_residual._metadata(options, ("input", 0, 0))
