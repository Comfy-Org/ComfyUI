"""POC custom node authored against the v0_0_3 custom-node SDK.

Demonstrates the SDK in real execution: the node uses ``ctx`` for progress and
round-trips its image through a ref (``ImageRef.from_tensor`` / ``.tensor()``).
In-process (OSS) that ref is zero-copy; under the overlay the same code runs in
an isolated guest with a shm/CUDA-IPC ref — unchanged."""
from __future__ import annotations

from comfy_api.v0_0_3 import ComfyExtension, io, sdk


class SandboxInvert(io.ComfyNode):
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
    async def execute(cls, image) -> io.NodeOutput:
        ctx = sdk.ctx()
        await ctx.progress.update(0.0, 1.0)

        # Ref round-trip: create a ref from the input, materialize it, invert.
        # In-process this is the real tensor (zero-copy); isolated it is a
        # shm/CUDA-IPC handle mapped into this process.
        ref = await sdk.ImageRef.from_tensor(image)
        t = await ref.tensor()
        out = 1.0 - t

        await ctx.progress.update(1.0, 1.0)
        return io.NodeOutput(out)


class PocExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SandboxInvert]


async def comfy_entrypoint() -> ComfyExtension:
    return PocExtension()
