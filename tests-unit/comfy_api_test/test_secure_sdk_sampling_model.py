"""Weightless scheduler models use the genuine Core sampling implementation."""
import asyncio

import pytest
import torch

import comfy.model_sampling
from comfy_api.latest import _sdk, sdk


def context():
    return _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
        prompt_id="sampling-unit", node_id="factory", node_type="SamplingProbe",
        node_module=__name__, input_mode="refs", inputs={}))


@pytest.mark.parametrize("shift", [-100.0, -1.0, -0.0, 1.0, 5.0, 100.0])
def test_public_factory_fresh_native_sampling_and_unknown_names(shift):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            ref = await sdk.ctx().models.create_sampling_model(
                "discrete_flow", {"shift": shift})
            assert isinstance(ref, sdk.ModelRef) and ref.kind == "MODEL"
            model = await refs.resolve(ref)
            first = model.get_model_object("model_sampling")
            second = model.get_model_object("model_sampling")
            assert type(first) is comfy.model_sampling.ModelSamplingDiscreteFlow
            assert first is not second and first.sigmas.data_ptr() != second.sigmas.data_ptr()
            expected = comfy.model_sampling.ModelSamplingDiscreteFlow()
            expected.set_parameters(shift=shift)
            assert first.shift == shift
            assert torch.equal(first.sigmas.view(torch.int32), expected.sigmas.view(torch.int32))
            assert model.get_model_object("unknown") is None
            assert model.get_model_object("diffusion_model") is None
            assert not hasattr(model, "model") and not hasattr(model, "model_options")
            first.set_parameters(shift=2.0)
            assert model.get_model_object("model_sampling").shift == shift
            await refs.release(ref)
            with pytest.raises(KeyError):
                await refs.resolve(ref)
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("parameters", [None, {}])
def test_default_shift_and_snapshot_of_parameters(parameters):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            ref = await sdk.ctx().models.create_sampling_model("discrete_flow", parameters)
            assert (await refs.resolve(ref)).get_model_object("model_sampling").shift == 1.0
            selected = {"shift": -5.0}
            ref = await sdk.ctx().models.create_sampling_model("discrete_flow", selected)
            selected["shift"] = 2.0
            assert (await refs.resolve(ref)).get_model_object("model_sampling").shift == -5.0
            refs.clear()
    asyncio.run(run())


@pytest.mark.parametrize("family,parameters,error", [
    ("trained", {}, ValueError), ("comfy.model_sampling.ModelSamplingDiscreteFlow", {}, ValueError),
    (None, {}, ValueError), ("discrete_flow", [], TypeError),
    ("discrete_flow", {"timesteps": 1000000}, ValueError),
    ("discrete_flow", {"shift": True}, TypeError),
    ("discrete_flow", {"shift": "1"}, TypeError),
    ("discrete_flow", {"shift": None}, TypeError),
    ("discrete_flow", {"shift": -100.01}, ValueError),
    ("discrete_flow", {"shift": 100.01}, ValueError),
    ("discrete_flow", {"shift": float("nan")}, ValueError),
    ("discrete_flow", {"shift": float("inf")}, ValueError),
    ("discrete_flow", {"shift": -float("inf")}, ValueError),
    ("discrete_flow", {"shift": 10 ** 500}, ValueError),
])
def test_factory_refuses_invalid_recipes_without_retaining_a_model(family, parameters, error):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            with pytest.raises(error):
                await sdk.ctx().models.create_sampling_model(family, parameters)
            assert not refs._table
            refs.clear()
    asyncio.run(run())


def test_factory_refuses_closed_execution():
    async def run():
        refs = _sdk.InProcessRefResolver()
        refs.clear()
        with _sdk.bind_runtime(refs, context(), _sdk.InProcessOps()):
            with pytest.raises(RuntimeError, match="closed"):
                await sdk.ctx().models.create_sampling_model("discrete_flow", {"shift": 1.0})
    asyncio.run(run())
