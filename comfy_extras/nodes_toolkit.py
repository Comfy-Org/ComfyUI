from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class CreateList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template_autogrow = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input("input"),
            prefix="input",
        )
        return io.Schema(
            node_id="CreateList",
            display_name="Create List",
            category="utilities",
            is_input_list=True,
            search_aliases=["Image Iterator", "Text Iterator", "Iterator"],
            inputs=[io.Autogrow.Input("inputs", template=template_autogrow)],
            outputs=[
                io.AnyType.Output(
                    is_output_list=True,
                    display_name="list",
                ),
            ],
        )

    @classmethod
    def execute(cls, inputs: io.Autogrow.Type) -> io.NodeOutput:
        output_list = []
        for input in inputs.values():
            output_list += input
        return io.NodeOutput(output_list)


class GetItemFromList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetItemFromList",
            display_name="Get Item From List",
            category="utilities",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("list"),
                io.Int.Input("index", default=0),
            ],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, list, index) -> io.NodeOutput:
        return io.NodeOutput(list[index[0]])


class ToolkitExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CreateList,
            GetItemFromList,
        ]


async def comfy_entrypoint() -> ToolkitExtension:
    return ToolkitExtension()
