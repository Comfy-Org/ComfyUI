import asyncio
from types import SimpleNamespace

import pytest

from comfy_api.latest._sdk import (
    ExecutionPlan,
    InProcessCtxProvider,
    InProcessOps,
    InProcessRefResolver,
    OpaqueRef,
    bind_runtime,
)


def _plan():
    return ExecutionPlan(
        prompt_id="wanvideo",
        node_id="1",
        node_type="wanvideo-test",
        prompt={"1": {"class_type": "wanvideo-test"}},
        extra_pnginfo={},
    )


def test_wanvideo_projects_only_a_bounded_transformer_dimension():
    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            model = OpaqueRef._wrap(await refs.create(
                "OPAQUE",
                SimpleNamespace(model=SimpleNamespace(
                    diffusion_model=SimpleNamespace(dim=5120))),
            ))
            assert await context.integrations.call("wanvideo", "transformer_dim", model=model) == 5120

            missing = OpaqueRef._wrap(await refs.create(
                "OPAQUE", SimpleNamespace(model=SimpleNamespace())))
            with pytest.raises(ValueError, match="does not publish"):
                await context.integrations.call("wanvideo", "transformer_dim", model=missing)

            invalid = OpaqueRef._wrap(await refs.create(
                "OPAQUE",
                SimpleNamespace(model=SimpleNamespace(
                    diffusion_model=SimpleNamespace(dim=0))),
            ))
            with pytest.raises(ValueError, match="invalid"):
                await context.integrations.call("wanvideo", "transformer_dim", model=invalid)

    asyncio.run(run())
