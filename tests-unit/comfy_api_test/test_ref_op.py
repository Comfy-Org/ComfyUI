"""Any handle can reach the operation vocabulary, not just images."""
import asyncio

from comfy_api.latest import _sdk


def test_op_dispatches_from_a_non_image_ref():
    """A pack holding, say, a model handle can call a named operation on it and
    wrap that in its own typed accessor, without core growing a method."""

    class Ops(_sdk.InProcessOps):
        def __init__(self):
            super().__init__()
            self.seen = []

        async def apply(self, op, subject, params):
            self.seen.append((op, subject.kind, params))
            return "dispatched"

    async def run():
        refs = _sdk.InProcessRefResolver()
        ops = Ops()
        plan = _sdk.ExecutionPlan(
            prompt_id="p", node_id="1", node_type="T", tier="sandbox")
        ctx = _sdk.InProcessCtxProvider().build(plan)
        with _sdk.bind_runtime(refs, ctx, ops):
            handle = _sdk.LatentRef._wrap(await refs.create("LATENT", object()))
            result = await handle.op("vendor.thing", scale=2)
        assert result == "dispatched"
        assert ops.seen == [("vendor.thing", "LATENT", {"scale": 2})]

    asyncio.run(run())


def test_image_ref_still_narrows_the_same_dispatch():
    class Ops(_sdk.InProcessOps):
        async def apply(self, op, subject, params):
            return subject

    async def run():
        refs = _sdk.InProcessRefResolver()
        plan = _sdk.ExecutionPlan(
            prompt_id="p", node_id="1", node_type="T", tier="sandbox")
        ctx = _sdk.InProcessCtxProvider().build(plan)
        with _sdk.bind_runtime(refs, ctx, Ops()):
            import torch
            image = _sdk.ImageRef._wrap(
                await refs.create("IMAGE", torch.zeros((1, 2, 2, 3))))
            assert await image.op("image.rgb") is image

    asyncio.run(run())
