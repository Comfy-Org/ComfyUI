"""POC custom node authored against the v0_0_3 custom-node SDK.

The node receives its image as an ``ImageRef`` (an asset handle), NOT a buffer,
and transforms it through an engine-side operation (``image.invert()``). It
never imports torch and never touches a raw tensor. In-process (OSS) the
operation runs on the trusted plane; under the overlay the same code runs in an
isolated guest and the op RPCs to the engine — unchanged."""
from __future__ import annotations

from comfy_api.v0_0_3 import ComfyExtension, io, sdk


class SandboxInvert(io.ComfyNode):
    # Opt in to the SDK asset model: execute() receives refs, not buffers.
    SDK_REFS = True

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SandboxInvert",
            display_name="Sandbox Invert (SDK POC)",
            category="poc",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, image) -> io.NodeOutput:  # image: sdk.ImageRef
        ctx = sdk.ctx()
        await ctx.progress.update(0.0, 1.0)
        out = await image.invert()  # engine-side op on the asset; no buffer here
        await ctx.progress.update(1.0, 1.0)
        return io.NodeOutput(out)  # returns an ImageRef; the engine resolves it


class PocExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SandboxInvert]


async def comfy_entrypoint() -> ComfyExtension:
    return PocExtension()
