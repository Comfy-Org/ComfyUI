import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io


pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("is_input_list", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
async def test_group_missing_fields_follow_execution_list_mode(is_input_list, lazy, empty):
    received = []

    class Group(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, is_input_list=is_input_list, inputs=[
                io.DynamicGroup.Input("rows", template=[
                    io.Float.Input("x"), io.Float.Input("optional", optional=True, default=1.0),
                ], max=3),
            ], outputs=[])

        @classmethod
        def execute(cls, rows):
            received.append(rows)
            return io.NodeOutput()

        @classmethod
        def check_lazy_status(cls, rows):
            received.append(rows)
            return []

    values = {} if empty else {"rows.0.x": 0.5, "rows.2.x": 0.8}
    inputs, _, metadata = execution.get_input_data(values, Group, "group")
    metadata["create_dynamic_tuple"] = lazy
    await execution._async_map_node_over_list(
        "test", "group", Group, inputs, "check_lazy_status" if lazy else "execute", v3_data=metadata,
    )
    expected = []
    if not empty:
        for index, value in enumerate([0.5, None, 0.8]):
            row = {"x": [value] if is_input_list else value, "optional": [None] if is_input_list else None}
            if lazy:
                row = {name: (value, f"rows.{index}.{name}") for name, value in row.items()}
            expected.append(row)
    assert received == [expected]


@pytest.mark.parametrize("values,input_name", [
    ({"rows.3.x": 0.5}, "rows.3.x"),
    ({"rows.bad.x": 0.5}, "rows.bad.x"),
    ({}, "rows"),
])
@pytest.mark.parametrize("downstream", [False, True])
async def test_group_validation_errors_identify_original_input(monkeypatch, values, input_name, downstream):
    class Group(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                io.DynamicGroup.Input("rows", template=[io.Float.Input("x")], min=1, max=3),
            ], outputs=[io.Float.Output()], is_output_node=not downstream)

        @classmethod
        def execute(cls, rows):
            raise AssertionError("Invalid group must not execute")

    class Sink(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[io.Float.Input("source")], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, source):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Group", Group)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Sink", Sink)
    prompt = {"group": {"class_type": "Group", "inputs": values}}
    if downstream:
        prompt["sink"] = {"class_type": "Sink", "inputs": {"source": ["group", 0]}}
    valid, _, _, errors = await execution.validate_prompt("test", prompt, None)
    assert not valid
    error, = errors["group"]["errors"]
    assert error["type"] == "invalid_dynamic_input"
    assert error["extra_info"] == {"input_name": input_name}
    assert input_name in error["details"]
