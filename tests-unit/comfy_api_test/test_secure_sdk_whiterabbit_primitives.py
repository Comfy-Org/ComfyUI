from __future__ import annotations

import asyncio
import types

import pytest
import torch

from comfy_api.latest import _sdk


def _state(indices, is_skip_list=True):
    cls = type("InterpolationStateList", (), {})
    value = cls()
    value.frame_indices = indices
    value.is_skip_list = is_skip_list
    return value


def test_interpolation_states_projects_skip_and_keep_lists_as_data():
    async def run(value):
        refs = _sdk.InProcessRefResolver()
        token = await refs.create("INTERPOLATION_STATES", value)
        state = _sdk.InterpolationStatesRef._wrap(token)
        with _sdk.bind_runtime(
            refs, types.SimpleNamespace(), _sdk.InProcessOps(),
        ):
            return await state.skip_mask(5)

    assert asyncio.run(run(_state([1, 3]))) == [False, True, False, True, False]
    assert asyncio.run(run(_state([1, 3], False))) == [True, False, True, False, True]


def test_interpolation_states_never_invokes_foreign_behavior():
    touched = []

    class InterpolationStateList:
        def __init__(self):
            self.frame_indices = [0]
            self.is_skip_list = True

        def is_frame_skipped(self, _index):
            touched.append("method")
            raise AssertionError

        def __iter__(self):
            touched.append("iter")
            raise AssertionError

        def __repr__(self):
            touched.append("repr")
            raise AssertionError

    value = InterpolationStateList()
    assert _sdk._ref_type_for(value) == (
        _sdk.InterpolationStatesRef, "INTERPOLATION_STATES")

    async def run():
        refs = _sdk.InProcessRefResolver()
        state = _sdk.InterpolationStatesRef._wrap(
            await refs.create("INTERPOLATION_STATES", value))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            return await state.skip_mask(2)

    assert asyncio.run(run()) == [True, False]
    assert touched == []


def test_interpolation_states_rejects_malformed_or_unbounded_data():
    async def run(value, pair_count=2):
        refs = _sdk.InProcessRefResolver()
        state = _sdk.InterpolationStatesRef._wrap(
            await refs.create("INTERPOLATION_STATES", value))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            return await state.skip_mask(pair_count)

    with pytest.raises(TypeError, match="frame indices must be integers"):
        asyncio.run(run(_state([True])))
    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(run(_state([-1])))
    with pytest.raises(TypeError, match="pair_count must be an integer"):
        asyncio.run(run(_state([]), True))
    with pytest.raises(ValueError, match=r"\[1, 100000\]"):
        asyncio.run(run(_state([]), 100_001))


class _ScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))

    def forward(self, value):
        return torch.nn.functional.interpolate(
            value, scale_factor=2.0, mode="nearest")


class _Upscaler:
    def __init__(self):
        self.model = _ScaleModel()
        self.scale = 2

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def __call__(self, value):
        return self.model(value)


def test_upscale_model_uses_requested_initial_tile_and_oom_fallback(monkeypatch):
    import comfy.model_management
    import comfy.utils

    calls = []

    def tiled_scale(value, fn, *, tile_x, tile_y, **_kwargs):
        calls.append((tile_x, tile_y))
        if tile_x == 512:
            raise RuntimeError("synthetic oom")
        return fn(value)

    monkeypatch.setattr(comfy.model_management, "get_torch_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "raise_non_oom",
                        lambda _error: None)
    monkeypatch.setattr(comfy.model_management, "module_size",
                        lambda _model: 1)
    monkeypatch.setattr(comfy.model_management, "free_memory",
                        lambda *_args: None)
    monkeypatch.setattr(comfy.utils, "get_tiled_scale_steps",
                        lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(comfy.utils, "ProgressBar",
                        lambda _steps: object())
    monkeypatch.setattr(comfy.utils, "tiled_scale", tiled_scale)

    async def run():
        refs = _sdk.InProcessRefResolver()
        upscaler = _sdk.UpscaleModelRef._wrap(
            await refs.create("UPSCALE_MODEL", _Upscaler()))
        image = _sdk.ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 3, 3), dtype=torch.float32)))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await upscaler.upscale(
                image, tile_size=0, channels_last=True)
        return await refs.resolve(result)

    result = asyncio.run(run())
    assert calls == [(512, 512), (256, 256)]
    assert result.shape == (1, 4, 6, 3)
    assert result.dtype == torch.float32


def test_upscale_model_omitted_tile_keeps_the_existing_direct_path(monkeypatch):
    import comfy.model_management
    import comfy.utils

    monkeypatch.setattr(comfy.model_management, "get_torch_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(comfy.utils, "ProgressBar",
                        lambda _steps: types.SimpleNamespace(update=lambda _n: None))
    monkeypatch.setattr(
        comfy.utils, "tiled_scale",
        lambda *_args, **_kwargs: pytest.fail("tiled path must remain opt-in"))

    async def run():
        refs = _sdk.InProcessRefResolver()
        upscaler = _sdk.UpscaleModelRef._wrap(
            await refs.create("UPSCALE_MODEL", _Upscaler()))
        image = _sdk.ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 3, 3), dtype=torch.float32)))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await upscaler.upscale(image)
        return await refs.resolve(result)

    assert asyncio.run(run()).shape == (1, 4, 6, 3)
