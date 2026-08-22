"""Secure custom-node SDK seam — regression + POC.

Drives the real ``execution._async_map_node_over_list`` dispatch path with
nodes authored against the v0_0_3 ``sdk`` (refs + ctx), for both the sync and
async execute forms, and verifies:

  * output is correct (invert / scale) — no regression vs. today's in-process
    behavior;
  * the default execution backend is the in-process one (zero-overhead);
  * a registered overlay backend intercepts real node dispatch (the provider
    swap), while output stays correct.
"""
import asyncio

import torch

from comfy_api.latest import sdk
from comfy_api.latest._sdk import InProcessExecutionBackend
from comfy_api.v0_0_3 import io


class _InvertAsync(io.ComfyNode):
    # SDK asset node: receives an ImageRef, transforms via an engine-side op,
    # never touches a buffer.
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestInvertAsync", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, image):
        ctx = sdk.ctx()
        await ctx.progress.update(0.0, 1.0)
        out = await image.invert()   # operation on the asset
        await ctx.progress.update(1.0, 1.0)
        return io.NodeOutput(out)


class _ScaleSyncLegacy(io.ComfyNode):
    # Legacy (non-SDK) v3 node: sync execute, receives a raw tensor. Confirms
    # the sync dispatch branch + that non-SDK nodes are unaffected by the seam.
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestScaleSyncLegacy", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image):
        return io.NodeOutput(image * 0.5)


async def _run(node_cls, image):
    import execution

    results = await execution._async_map_node_over_list(
        prompt_id="p", unique_id="1", obj=node_cls,
        input_data_all={"image": [image]}, func=node_cls.FUNCTION, v3_data=None,
    )
    out = results[0]
    return out.result[0]


def _output_of(node_cls, image):
    return asyncio.run(_run(node_cls, image))


def test_default_backend_is_in_process():
    assert isinstance(sdk.providers.execution_backend, InProcessExecutionBackend)
    assert sdk.providers.overlay_active is False


def test_async_sdk_node_inverts_through_real_engine():
    img = torch.rand(1, 8, 8, 3)
    got = _output_of(_InvertAsync, img)
    assert torch.allclose(got, 1.0 - img)


def test_legacy_sync_node_scales_through_real_engine():
    img = torch.rand(1, 8, 8, 3)
    got = _output_of(_ScaleSyncLegacy, img)
    assert torch.allclose(got, img * 0.5)


def test_overlay_backend_intercepts_dispatch():
    calls = []

    class _FakeOverlayBackend:
        async def dispatch(self, plan, local_call):
            calls.append((plan.node_id, plan.node_type))
            return await local_call()  # delegate -> behavior preserved

    original = sdk.providers.execution_backend
    sdk.providers.register_execution_backend(_FakeOverlayBackend())
    try:
        img = torch.rand(1, 4, 4, 3)
        got = _output_of(_InvertAsync, img)
        assert torch.allclose(got, 1.0 - img)   # still correct
        assert len(calls) == 1                    # overlay saw the real dispatch
        assert calls[0][1] == "_InvertAsync"
    finally:
        sdk.providers.execution_backend = original


if __name__ == "__main__":
    # Runnable without pytest.
    test_default_backend_is_in_process()
    test_async_sdk_node_inverts_through_real_engine()
    test_legacy_sync_node_scales_through_real_engine()
    test_overlay_backend_intercepts_dispatch()
    print("PASS: all secure-SDK seam checks")
