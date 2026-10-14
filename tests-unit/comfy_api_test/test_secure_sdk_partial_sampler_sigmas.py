"""Partial schedules pass the sigma gate; invalid tables still fail first."""
import asyncio

import pytest
import torch

from comfy_api.latest import _sdk


def sampler():
    async def build():
        refs = _sdk.InProcessRefResolver()
        context = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
            prompt_id="partial-sigma-contract", node_id="1", node_type="test"))

        async def unreachable(_broker, _latent, _sigmas):
            raise AssertionError("An absent model must never reach the closure")

        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            closure = await context.closures.retain("custom_sampler", unreachable)
            return await refs.resolve(await closure.as_sampler())

    return asyncio.run(build())


@pytest.mark.parametrize("sigmas", [
    torch.tensor([0.991, 0.920]),
    torch.tensor([0.991, 0.980, 0.920]),
    torch.tensor([1.0, 0.991]),
    torch.tensor([2.0, 1.0, 0.0]),
    torch.tensor([0.991, 0.991], dtype=torch.float64),
    torch.linspace(1.0, 0.25, 4097),
])
def test_partial_and_full_tables_reach_the_unchanged_model_authority_gate(sigmas):
    original = sigmas.clone()
    # No model is supplied: this checks schedule admission without fabricating
    # model inference. Real-model stage comparisons are run separately.
    with pytest.raises(TypeError, match="model does not expose model sampling"):
        sampler().sampler_function(None, torch.zeros(1, 4, 8, 8), sigmas)
    assert torch.equal(sigmas, original)


@pytest.mark.parametrize("sigmas", [
    torch.tensor([]),
    torch.tensor([0.920]),
    torch.ones(4098),
    torch.ones(2, 2),
    torch.tensor([1, 0]),
    torch.tensor([True, False]),
    torch.tensor([1.0, -0.1]),
    torch.tensor([0.920, 0.991]),
    torch.tensor([float("nan"), 0.920]),
    torch.tensor([float("inf"), 0.920]),
])
def test_invalid_tables_are_rejected_before_model_authority(sigmas):
    with pytest.raises(ValueError, match="custom-sampler sigmas"):
        sampler().sampler_function(None, torch.zeros(1, 4, 8, 8), sigmas)
