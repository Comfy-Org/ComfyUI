import asyncio
from types import SimpleNamespace

import pytest
import torch

from comfy_api.latest._sdk import (
    CondRef,
    ExecutionPlan,
    ImageRef,
    InProcessCtxProvider,
    InProcessOps,
    InProcessRefResolver,
    ModelRef,
    SigmasRef,
    bind_runtime,
)


def _plan():
    return ExecutionPlan(
        prompt_id="grounding",
        node_id="1",
        node_type="grounding-test",
        prompt={"1": {"class_type": "grounding-test"}},
        extra_pnginfo={},
    )


def test_sigmas_steps_is_bounded_scalar_metadata():
    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            sigmas = SigmasRef._wrap(await refs.create(
                "SIGMAS", torch.tensor([1.0, 0.5, 0.1, 0.0])))
            invalid = SigmasRef._wrap(await refs.create(
                "SIGMAS", torch.tensor([float("nan"), 0.0])))
            assert await sigmas.steps() == 3
            with pytest.raises(ValueError, match="finite"):
                await invalid.steps()

    asyncio.run(run())


def test_model_ground_image_delegates_to_official_sam3_and_bounds_results(
    monkeypatch,
):
    from comfy_extras.nodes_sam3 import SAM3_Detect

    diffusion_type = type("SAM3Model", (), {})
    diffusion_type.__module__ = "comfy.ldm.sam3.detector"
    diffusion = diffusion_type()
    base = SimpleNamespace(
        diffusion_model=diffusion,
        model_config=SimpleNamespace(unet_config={"image_model": "SAM31"}),
    )
    patcher = SimpleNamespace(model=base)
    calls = []

    def fake_execute(
        cls, model, image, conditioning=None, threshold=0.5,
        refine_iterations=2, individual_masks=False, **kwargs,
    ):
        calls.append({
            "model": model,
            "image": image,
            "conditioning": conditioning,
            "threshold": threshold,
            "refine_iterations": refine_iterations,
            "individual_masks": individual_masks,
        })
        return SimpleNamespace(result=(
            torch.stack([
                torch.full((4, 5), 1.0),
                torch.full((4, 5), 2.0),
            ]),
            [[
                {"x": 1, "y": 1, "width": 2, "height": 2, "score": 0.9},
                {"x": 2, "y": 1, "width": 2, "height": 2, "score": 0.8},
            ]],
        ))

    monkeypatch.setattr(SAM3_Detect, "execute", classmethod(fake_execute))

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            model = ModelRef._wrap(await refs.create("MODEL", patcher))
            image = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((1, 4, 5, 3))))
            conditioning_value = [[torch.zeros((1, 2, 3)), {}]]
            conditioning = CondRef._wrap(await refs.create(
                "CONDITIONING", conditioning_value))

            masks, boxes = await model.ground_image(
                image,
                conditioning,
                threshold=0.6,
                refine_iterations=1,
                individual_masks=True,
                max_detections=1,
            )
            assert torch.equal(
                await refs.resolve(masks), torch.full((1, 4, 5), 1.0))
            assert boxes == [[{
                "x": 1.0,
                "y": 1.0,
                "width": 2.0,
                "height": 2.0,
                "score": 0.9,
            }]]

            wrong = ModelRef._wrap(await refs.create(
                "MODEL", SimpleNamespace(model=SimpleNamespace())))
            with pytest.raises(TypeError, match="official SAM3"):
                await wrong.ground_image(image, conditioning)

    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0]["model"] is patcher
    assert calls[0]["threshold"] == 0.6
    assert calls[0]["refine_iterations"] == 1
    assert calls[0]["individual_masks"] is True
