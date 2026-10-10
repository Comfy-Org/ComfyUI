from comfy_api.latest import io


class ParentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SchemaCacheParent",
            category="test",
            inputs=[],
            outputs=[io.Conditioning.Output(), io.Latent.Output()],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(None, None)


class ChildNode(ParentNode):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "SchemaCacheChild"
        schema.is_output_node = True
        schema.outputs.insert(0, io.Model.Output(display_name="model"))
        return schema


def test_subclass_does_not_inherit_parent_schema_cache():
    # schema the parent first, as node registration does
    assert ParentNode.RETURN_TYPES == ["CONDITIONING", "LATENT"]
    assert ParentNode.OUTPUT_NODE is False

    assert ChildNode.RETURN_TYPES == ["MODEL", "CONDITIONING", "LATENT"]
    assert ChildNode.RETURN_NAMES == ["model", "CONDITIONING", "LATENT"]
    assert ChildNode.OUTPUT_NODE is True
    assert ChildNode.SCHEMA.node_id == "SchemaCacheChild"

    assert ParentNode.RETURN_TYPES == ["CONDITIONING", "LATENT"]
    assert ParentNode.SCHEMA.node_id == "SchemaCacheParent"


def test_class_clone_keeps_schema_cache():
    ParentNode.GET_SCHEMA()
    clone = ParentNode.PREPARE_CLASS_CLONE(None)

    assert clone.__dict__["_RETURN_TYPES"] is ParentNode.RETURN_TYPES
    assert clone.__dict__["SCHEMA"] is ParentNode.SCHEMA
