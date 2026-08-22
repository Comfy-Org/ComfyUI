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


class _InvertWithUi(io.ComfyNode):
    # An SDK asset node that is ALSO an output node. Rebuilding its NodeOutput
    # to resolve refs must not discard what is not a result.
    SDK_REFS = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TestInvertWithUi", category="test",
            inputs=[io.Image.Input("image")], outputs=[io.Image.Output()],
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, image):
        out = await image.invert()
        return io.NodeOutput(out, ui={"text": ["hello"]})


async def _run_full(node_cls, image):
    import execution

    results = await execution._async_map_node_over_list(
        prompt_id="p", unique_id="1", obj=node_cls,
        input_data_all={"image": [image]}, func=node_cls.FUNCTION, v3_data=None,
    )
    return results[0]


async def _run(node_cls, image):
    out = await _run_full(node_cls, image)
    return out.result[0]


def test_sdk_node_keeps_its_ui_output():
    """Resolving output refs must preserve `ui`.

    `unwrap_outputs` rebuilds the NodeOutput to swap refs back for real
    objects. Rebuilding it from results alone dropped `ui`, which made every
    SDK_REFS node unable to be an output node: ComfyUI only emits the
    `executed` event that carries results to the frontend for nodes returning
    ui data, so such a node ran correctly and then displayed nothing.
    """
    img = torch.rand(1, 8, 8, 3)
    out = asyncio.run(_run_full(_InvertWithUi, img))
    assert torch.allclose(out.result[0], 1.0 - img), "pixels wrong"
    assert out.ui == {"text": ["hello"]}, f"ui was dropped: {out.ui!r}"


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
        async def dispatch(self, plan, local_call, runtime=None):
            calls.append((plan, runtime))
            return await local_call()  # delegate -> behavior preserved

    original = sdk.providers.execution_backend
    sdk.providers.register_execution_backend(_FakeOverlayBackend())
    try:
        img = torch.rand(1, 4, 4, 3)
        got = _output_of(_InvertAsync, img)
        assert torch.allclose(got, 1.0 - img)   # still correct
        assert len(calls) == 1                    # overlay saw the real dispatch
        plan, runtime = calls[0]
        assert plan.node_type == "_InvertAsync"
        # Work-unit payload: an out-of-process backend gets the module spec,
        # the ref-wrapped inputs, and the host runtime to broker against.
        assert plan.node_module == _InvertAsync.__module__
        assert isinstance(plan.inputs["image"], sdk.ImageRef)
        assert runtime is not None and runtime.refs is not None
    finally:
        sdk.providers.execution_backend = original


if __name__ == "__main__":
    # Runnable without pytest.
    test_default_backend_is_in_process()
    test_async_sdk_node_inverts_through_real_engine()
    test_legacy_sync_node_scales_through_real_engine()
    test_overlay_backend_intercepts_dispatch()
    print("PASS: all secure-SDK seam checks")
