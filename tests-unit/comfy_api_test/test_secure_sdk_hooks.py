"""Closed HookGroup operations against Core and optional original local LoRA weights."""
import asyncio
import math
import os
from pathlib import Path

import pytest
import torch

import comfy.hooks
import comfy.utils
from comfy_api.latest import _sdk, sdk


@pytest.fixture(scope="module")
def original_hook_weights():
    name = os.environ.get("COMFY_TEST_LORA_FILE")
    if not name:
        pytest.skip("A local original LoRA fixture is required")
    import folder_paths
    path = Path(name)
    weights = comfy.utils.load_torch_file(str(path), safe_load=True)
    previous = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.folder_names_and_paths["loras"] = ([str(path.parent)], {".safetensors"})
    yield path, weights
    if previous is None:
        folder_paths.folder_names_and_paths.pop("loras", None)
    else:
        folder_paths.folder_names_and_paths["loras"] = previous


async def handle(refs, value, kind="OPAQUE"):
    cls = _sdk.AssetRef if kind == "ASSET" else _sdk.OpaqueRef
    return cls._wrap(await refs.create(kind, value))


def test_empty_uses_real_constructor_without_a_fabricated_subject():
    async def run():
        refs = _sdk.InProcessRefResolver()
        try:
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                output = await sdk.Hooks.empty()
            actual = await refs.resolve(output)
            assert isinstance(actual, comfy.hooks.HookGroup) and len(actual) == 0
            assert isinstance(output, _sdk.OpaqueRef) and output.kind == "OPAQUE"
        finally:
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("model,clip", [(.5, 1.), (0., 0.), (101., -101.),
                                     (1e6, 1e6), (float("inf"), float("-inf")),
                                     (float("nan"), .5)])
def test_original_weight_creation_matches_native(original_hook_weights, model, clip):
    async def run():
        path, weights = original_hook_weights
        native = comfy.hooks.create_hook_lora(weights, model, clip)
        refs = _sdk.InProcessRefResolver()
        try:
            asset = await handle(refs, path, "ASSET")
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                output = await sdk.Hooks.create_lora(asset, model, clip)
            actual = await refs.resolve(output)
            assert isinstance(actual, comfy.hooks.HookGroup)
            assert len(actual.hooks) == len(native.hooks) == 1
            left, right = actual.hooks[0], native.hooks[0]
            assert type(left) is type(right)
            for name in ("_strength_model", "_strength_clip"):
                a, b = getattr(left, name), getattr(right, name)
                assert a == b or (math.isnan(a) and math.isnan(b))
            assert left.need_weight_init == right.need_weight_init
            assert left.hook_scope == right.hook_scope and left.hook_id == right.hook_id
            assert left.weights.keys() == right.weights.keys()
            for name in right.weights:
                torch.testing.assert_close(left.weights[name], right.weights[name], rtol=0, atol=0)
        finally:
            refs.clear()
    asyncio.run(run())


def assert_native_clone(actual, source):
    assert actual is not source and len(actual.hooks) == len(source.hooks)
    for left, right in zip(actual.hooks, source.hooks):
        assert left is not right and type(left) is type(right)
        assert left.hook_ref is right.hook_ref
        assert left.hook_keyframe is right.hook_keyframe
        assert left.weights is right.weights and left.weights_clip is right.weights_clip
        assert left.hook_scope == right.hook_scope and left.hook_id == right.hook_id
        assert left.need_weight_init == right.need_weight_init


@pytest.mark.parametrize("combine", ["none", "same", "clone", "distinct"])
def test_clone_combine_preserves_native_identity_keyframes_and_dedup(original_hook_weights, combine):
    async def run():
        _, weights = original_hook_weights
        group = comfy.hooks.create_hook_lora(weights, .5, 1.)
        frames = comfy.hooks.HookKeyframeGroup()
        frames.add(comfy.hooks.HookKeyframe(.2, start_percent=.25, guarantee_steps=3))
        group.set_keyframes_on_hooks(frames)
        source_hook = group.hooks[0]
        other = {"none": None, "same": group, "clone": group.clone(),
                 "distinct": comfy.hooks.create_hook_lora(weights, -.5, .75)}[combine]
        expected = group.clone_and_combine(other)
        refs = _sdk.InProcessRefResolver()
        try:
            ref = await handle(refs, group)
            other_ref = None if other is None else await handle(refs, other)
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                cloned = await sdk.Hooks.clone(ref)
                combined = await sdk.Hooks.clone_and_combine(ref, other_ref)
            actual_clone = await refs.resolve(cloned)
            actual = await refs.resolve(combined)
            assert_native_clone(actual_clone, group)
            assert len(actual.hooks) == len(expected.hooks) == (2 if combine == "distinct" else 1)
            assert actual.hooks[0].hook_ref is source_hook.hook_ref
            assert actual.hooks[0].hook_keyframe is source_hook.hook_keyframe
            assert actual.hooks[0].weights is source_hook.weights
            assert group.hooks == [source_hook] and source_hook._strength_model == .5
            actual_clone.hooks[0]._strength_model = 17.
            assert source_hook._strength_model == .5
            assert len(actual.get_hooks_for_clip_schedule()) == len(expected.get_hooks_for_clip_schedule())
        finally:
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("case", ["wrong-kind", "relabelled", "wrong-object", "unknown", "released", "foreign"])
def test_authoritative_group_ref_validation(case):
    async def run():
        refs, foreign = _sdk.InProcessRefResolver(), _sdk.InProcessRefResolver()
        group = comfy.hooks.HookGroup()
        try:
            if case == "wrong-kind":
                ref = await refs.create("VALUE", group)
            elif case == "relabelled":
                existing = await refs.create("VALUE", group)
                ref = _sdk.Ref("OPAQUE", existing.id)
            elif case == "wrong-object":
                ref = await handle(refs, {"hooks": []})
            elif case == "unknown":
                ref = _sdk.Ref("OPAQUE", "not-in-this-table")
            elif case == "released":
                ref = await handle(refs, group)
                await refs.release(ref)
            else:
                ref = await handle(foreign, group)
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                with pytest.raises((TypeError, KeyError)):
                    await sdk.Hooks.clone(ref)
                valid = await handle(refs, group)
                with pytest.raises((TypeError, KeyError)):
                    await sdk.Hooks.clone_and_combine(valid, ref)
        finally:
            refs.clear()
            foreign.clear()
    asyncio.run(run())


@pytest.mark.parametrize("strength", [True, "0.5", None, [], {}])
@pytest.mark.parametrize("field", ["strength_model", "strength_clip"])
def test_creation_rejects_non_numeric_strengths(original_hook_weights, strength, field):
    async def run():
        refs = _sdk.InProcessRefResolver()
        try:
            asset = await handle(refs, original_hook_weights[0], "ASSET")
            params = {"strength_model": .5, "strength_clip": 1., field: strength}
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                with pytest.raises(TypeError, match="must be a number"):
                    await sdk.Hooks.create_lora(asset, **params)
        finally:
            refs.clear()
    asyncio.run(run())


def test_constructor_rejects_fake_subject_and_extra_operations():
    async def run():
        refs = _sdk.InProcessRefResolver()
        ops = _sdk.InProcessOps()
        try:
            ref = await handle(refs, comfy.hooks.HookGroup())
            with _sdk.bind_runtime(refs, None, ops):
                with pytest.raises(TypeError, match="has no subject"):
                    await ref.op("hooks.empty")
                with pytest.raises(_sdk.OpNotSupported):
                    await ref.op("hooks.__getattribute__", attribute="weights")
        finally:
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("case", ["outside", "symlink", "missing", "malformed", "not-path"])
def test_confined_safe_load_boundary(original_hook_weights, tmp_path, case):
    async def run():
        root = original_hook_weights[0].parent
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"invalid safetensors header")
        missing = root / "missing-hook-fixture.safetensors"
        refs = _sdk.InProcessRefResolver()
        # The writable temporary loras root is local to this negative test.
        import folder_paths
        previous = folder_paths.folder_names_and_paths["loras"]
        confined = tmp_path / "loras"
        confined.mkdir()
        malformed = confined / "malformed.safetensors"
        malformed.write_bytes(outside.read_bytes())
        link = confined / "escaped.safetensors"
        link.symlink_to(outside)
        folder_paths.folder_names_and_paths["loras"] = ([str(confined), str(root)], {".safetensors"})
        value = {"outside": outside, "symlink": link, "missing": missing,
                 "malformed": malformed, "not-path": {"path": str(root)}}[case]
        try:
            asset = await handle(refs, value, "ASSET")
            error = ValueError if case in ("outside", "symlink") else FileNotFoundError if case == "missing" else TypeError if case == "not-path" else Exception
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                with pytest.raises(error):
                    await sdk.Hooks.create_lora(asset, .5, 1.)
            assert len(refs._table) == 1
        finally:
            folder_paths.folder_names_and_paths["loras"] = previous
            refs.clear()
    asyncio.run(run())
