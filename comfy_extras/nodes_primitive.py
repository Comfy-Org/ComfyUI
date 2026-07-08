import sys
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class String(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrimitiveString",
            search_aliases=["text", "string", "text box", "prompt"],
            display_name="Text String (DEPRECATED)",
            category="utilities/primitive",
            inputs=[io.String.Input("value")],
            outputs=[io.String.Output()],
            is_deprecated=True
        )

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)


class StringMultiline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrimitiveStringMultiline",
            search_aliases=["text", "string", "text multiline", "string multiline", "text box", "prompt"],
            display_name="Input Text",
            category="utilities/primitive",
            essentials_category="Basics",
            inputs=[io.String.Input("value", multiline=True)],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)


class Int(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrimitiveInt",
            display_name="Int",
            category="utilities/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize, control_after_generate=io.ControlAfterGenerate.fixed),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class Float(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrimitiveFloat",
            display_name="Float",
            category="utilities/primitive",
            inputs=[
                io.Float.Input("value", min=-sys.maxsize, max=sys.maxsize, step=0.1),
            ],
            outputs=[io.Float.Output()],
        )

    @classmethod
    def execute(cls, value: float) -> io.NodeOutput:
        return io.NodeOutput(value)


class Boolean(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrimitiveBoolean",
            display_name="Boolean",
            category="utilities/primitive",
            inputs=[
                io.Boolean.Input("value"),
            ],
            outputs=[io.Boolean.Output()],
        )

    @classmethod
    def execute(cls, value: bool) -> io.NodeOutput:
        return io.NodeOutput(value)


class PrimitivesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            String,
            StringMultiline,
            Int,
            Float,
            Boolean,
        ]

async def comfy_entrypoint() -> PrimitivesExtension:
    return PrimitivesExtension()
