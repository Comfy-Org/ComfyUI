from comfy_api.latest import io


class ParentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="ParentNode", outputs=[io.String.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput("parent")


class ChildNode(ParentNode):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "ChildNode"
        schema.outputs[0].is_output_list = True
        return schema

    @classmethod
    def execute(cls):
        return io.NodeOutput(["child"])


def test_schema_cache_is_isolated_per_subclass():
    ParentNode.GET_SCHEMA()
    ChildNode.GET_SCHEMA()

    assert ParentNode.OUTPUT_IS_LIST == [False]
    assert ChildNode.OUTPUT_IS_LIST == [True]
