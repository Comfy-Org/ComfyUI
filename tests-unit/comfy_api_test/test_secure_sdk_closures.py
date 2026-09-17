"""In-process half of the closed node-closure author contract (D21)."""
from __future__ import annotations

import asyncio

import pytest
import torch

from comfy_api.latest import _sdk


class _FakeModel:
    def __init__(self, parent=None):
        self.parent = parent
        self.post_cfg = None
        self.disable_cfg1 = None

    def clone(self):
        return _FakeModel(self)

    def set_model_sampler_post_cfg_function(
        self, function, disable_cfg1_optimization=False,
    ):
        self.post_cfg = function
        self.disable_cfg1 = bool(disable_cfg1_optimization)


def _context():
    return _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
        prompt_id="closure-core",
        node_id="1",
        node_type="closure-core",
    ))


def test_post_cfg_closure_clones_model_and_preserves_tensor_contract():
    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        original = _FakeModel()
        model = _sdk.ModelRef._wrap(await refs.create("MODEL", original))
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain(
                "post_cfg", lambda guided, *_args: guided * 1.5)
            patched_ref = await closure.attach_model(model)
            patched = await refs.resolve(patched_ref)
        guided = torch.full((1, 4, 2, 3), 2.0)
        result = patched.post_cfg({
            "denoised": guided,
            "cond_denoised": torch.ones_like(guided),
            "uncond_denoised": torch.zeros_like(guided),
            "input": torch.full_like(guided, 3.0),
            "sigma": torch.tensor([1.0]),
            "cond_scale": 7.5,
        })
        return original, patched, guided, result

    original, patched, guided, result = asyncio.run(run())
    assert patched is not original
    assert patched.parent is original
    assert patched.disable_cfg1 is True
    assert torch.equal(result, guided * 1.5)


def test_only_a_shipped_phase_can_be_retained():
    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            await context.closures.retain(
                "attention_couple", lambda value: value)

    with pytest.raises(
        Exception, match="unknown closure kind 'attention_couple'"
    ):
        asyncio.run(run())


def test_post_cfg_closure_cannot_change_shape_dtype_or_device():
    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        model = _sdk.ModelRef._wrap(
            await refs.create("MODEL", _FakeModel()))
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain(
                "post_cfg", lambda guided, *_args: guided[..., :1])
            patched = await refs.resolve(await closure.attach_model(model))
        guided = torch.ones((1, 4, 2, 3))
        return patched.post_cfg, guided

    callback, guided = asyncio.run(run())
    with pytest.raises(TypeError, match="preserve shape, dtype, and device"):
        callback({
            "denoised": guided,
            "cond_denoised": guided,
            "uncond_denoised": guided,
            "input": guided,
            "sigma": torch.tensor([1.0]),
            "cond_scale": 7.5,
        })


def test_model_sigma_closure_wraps_sampler_without_owning_model_calls():
    from comfy.samplers import KSAMPLER

    class Sampling:
        @staticmethod
        def percent_to_sigma(percent):
            return 10.0 * (1.0 - percent)

    class ModelCall:
        def __init__(self):
            self.inner_model = type("Guider", (), {
                "cfg": 4.0,
                "inner_model": type("Inner", (), {
                    "model_sampling": Sampling(),
                })(),
            })()
            self.seen = None

        def __call__(self, latent, sigma, **kwargs):
            self.seen = sigma
            return sigma

    def source_sampler(model, x, sigmas, *, marker):
        assert marker == "kept"
        return model(x, torch.tensor([5.0]))

    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        sampler = _sdk.SamplerRef._wrap(await refs.create(
            "SAMPLER", KSAMPLER(source_sampler, {"marker": "kept"})))
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain(
                "model_sigma",
                lambda sigma, sigmas, cfg, start_sigma, end_sigma:
                    sigma * 2.0
                    if end_sigma <= float(sigma.max()) <= start_sigma
                    else sigma,
            )
            wrapped = await refs.resolve(await closure.wrap_sampler(
                sampler, start_percent=0.1, end_percent=0.9))
        model = ModelCall()
        result = wrapped.sampler_function(
            model,
            torch.zeros((1, 4, 2, 3)),
            torch.tensor([9.0, 5.0, 1.0, 0.0]),
        )
        return model, result

    model, result = asyncio.run(run())
    assert torch.equal(result, torch.tensor([10.0]))
    assert torch.equal(model.seen, torch.tensor([10.0]))


def test_custom_sampler_closure_owns_the_loop_but_not_the_model_call():
    class Sampling:
        noise_scale = 1.0

    class ModelPatcher:
        @staticmethod
        def get_model_object(name):
            assert name == "model_sampling"
            return Sampling()

    class ModelCall:
        def __init__(self):
            self.inner_model = type("Inner", (), {
                "model_patcher": ModelPatcher(),
            })()
            self.seen = []

        def __call__(
            self, latent, sigma, denoise_mask=None, model_options=None,
            seed=None,
        ):
            self.seen.append((latent.clone(), sigma.clone(), seed))
            return latent + 2.0

    async def program(broker, latent, sigmas):
        schedule = await broker.schedule_parameters()
        assert schedule["parameterization"] == "sigma"
        denoised, uncond = await broker.denoise(latent, sigmas[0])
        assert uncond is None
        await broker.preview(
            0, latent, sigmas[0], sigmas[0], denoised)
        return denoised

    async def build():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain(
                "custom_sampler", program)
            sampler_ref = await closure.as_sampler()
            return await refs.resolve(sampler_ref)

    sampler = asyncio.run(build())
    model = ModelCall()
    latent = torch.zeros((1, 4, 2, 3))
    previews = []
    result = sampler.sampler_function(
        model,
        latent,
        torch.tensor([2.0, 1.0, 0.0]),
        extra_args={"seed": 7},
        callback=previews.append,
    )
    assert torch.equal(result, latent + 2.0)
    assert len(model.seen) == 1
    assert model.seen[0][2] == 7
    assert len(previews) == 1
    assert previews[0]["i"] == 0

    with pytest.raises(ValueError, match="floating-point"):
        sampler.sampler_function(
            ModelCall(), latent, torch.tensor([2, 1, 0]))
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        sampler.sampler_function(
            ModelCall(), latent, torch.tensor([2.0, 1.0, 0.0]),
            extra_args={"seed": -1},
        )

    async def bad_program(_broker, value, _sigmas):
        return value[..., :-1]

    async def build_bad():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain(
                "custom_sampler", bad_program)
            sampler_ref = await closure.as_sampler()
            return await refs.resolve(sampler_ref)

    bad_sampler = asyncio.run(build_bad())
    with pytest.raises(ValueError, match="temporary resize"):
        bad_sampler.sampler_function(
            ModelCall(), latent, torch.tensor([2.0, 1.0, 0.0]))
