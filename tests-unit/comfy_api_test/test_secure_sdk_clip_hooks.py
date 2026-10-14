"""Typed CLIP Hook consumer guards and native no-op/error semantics."""
import asyncio
import inspect

import pytest

import comfy.hooks
from comfy.sd import CLIP
from comfy_extras.nodes_hooks import SetClipHooks
from comfy_api.latest import _sdk


async def clip_ref(refs, value):
    return _sdk.ClipRef._wrap(await refs.create("CLIP", value))


@pytest.mark.parametrize("apply_to_conds", [False, True])
@pytest.mark.parametrize("schedule_clip", [False, True])
def test_none_is_native_identity_without_a_new_reference(apply_to_conds, schedule_clip):
    async def run():
        # No model is constructed or encoded here. Full trained CLIP/LoRA
        # parity is tested with the original checkpoints in backend tests.
        value = CLIP(no_init=True)
        refs = _sdk.InProcessRefResolver()
        source = await clip_ref(refs, value)
        try:
            expected, = SetClipHooks().apply_hooks(
                value, schedule_clip=schedule_clip,
                apply_to_conds=apply_to_conds, hooks=None)
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                actual = await source.apply_hooks(
                    None, apply_to_conds=apply_to_conds, schedule_clip=schedule_clip)
            assert actual is source
            assert await refs.resolve(actual) is expected is value
            assert len(refs._table) == 1
        finally:
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("name", ["apply_to_conds", "schedule_clip"])
@pytest.mark.parametrize("bad", [1, "false", None, [], {}])
def test_flags_are_boolean_before_native_effects(name, bad):
    async def run():
        refs = _sdk.InProcessRefResolver()
        source = await clip_ref(refs, CLIP(no_init=True))
        try:
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()), pytest.raises(TypeError, match=name):
                await source.apply_hooks(None, **{name: bad})
            assert len(refs._table) == 1
        finally:
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("case", ["foreign", "relabelled", "released", "ended", "wrong-object"])
def test_current_reference_table_is_authoritative(case):
    async def run():
        refs, foreign = _sdk.InProcessRefResolver(), _sdk.InProcessRefResolver()
        value = CLIP(no_init=True)
        source = await clip_ref(foreign if case == "foreign" else refs,
                                {} if case == "wrong-object" else value)
        if case == "relabelled":
            original = await refs.create("OPAQUE", value)
            source = _sdk.ClipRef._wrap(_sdk.Ref("CLIP", original.id))
        elif case == "released":
            await refs.release(source)
        elif case == "ended":
            refs.clear()
        try:
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()), pytest.raises((TypeError, KeyError)):
                await source.apply_hooks(None)
        finally:
            refs.clear()
            foreign.clear()
    asyncio.run(run())


def test_native_failure_is_not_swallowed_or_published():
    async def run():
        value = CLIP(no_init=True)
        hooks = comfy.hooks.HookGroup()
        with pytest.raises(AttributeError) as native_error:
            SetClipHooks().apply_hooks(value, False, True, hooks)
        refs = _sdk.InProcessRefResolver()
        source = await clip_ref(refs, value)
        group = _sdk.OpaqueRef._wrap(await refs.create("OPAQUE", hooks))
        try:
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()), pytest.raises(AttributeError) as sdk_error:
                await source.apply_hooks(group)
            assert str(sdk_error.value) == str(native_error.value)
            assert len(refs._table) == 2
        finally:
            refs.clear()
    asyncio.run(run())


def test_public_flags_are_keyword_only_and_runtime_is_required():
    signature = inspect.signature(_sdk.ClipRef.apply_hooks)
    assert signature.parameters["hooks"].default is inspect.Parameter.empty
    for name, default in (("apply_to_conds", True), ("schedule_clip", False)):
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is default
    source = _sdk.ClipRef._wrap(_sdk.Ref("CLIP", "inactive-owner"))
    with pytest.raises(RuntimeError):
        asyncio.run(source.apply_hooks(None))
