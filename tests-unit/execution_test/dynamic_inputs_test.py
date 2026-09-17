from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io


pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("input_spec,values,expected", [
    pytest.param(
        io.DynamicGroup.Input("rows", template=[io.Float.Input("x")]),
        {"rows.2.x": 0.5}, [{"x": None}, {"x": None}, {"x": 0.5}], id="sparse-group",
    ),
    pytest.param(
        io.DynamicGroup.Input("rows", template=[io.Float.Input("x")]),
        {}, [], id="empty-group",
    ),
    pytest.param(
        io.Autogrow.Input("rows", template=io.Autogrow.TemplatePrefix(io.Float.Input("x"), prefix="item", min=0)),
        {"rows.item0": 0.5}, {"item0": 0.5}, id="autogrow",
    ),
    pytest.param(
        io.Autogrow.Input("rows", template=io.Autogrow.TemplatePrefix(io.Float.Input("x"), prefix="item", min=0)),
        {}, {}, id="empty-autogrow",
    ),
    pytest.param(
        io.DynamicCombo.Input("rows", options=[io.DynamicCombo.Option("on", [io.Float.Input("x")])]),
        {"rows": "on", "rows.x": 0.5}, {"rows": "on", "x": 0.5}, id="combo",
    ),
    pytest.param(
        io.DynamicCombo.Input("rows", options=[io.DynamicCombo.Option("on", [
            io.DynamicGroup.Input("items", template=[io.Float.Input("x")]),
        ])]),
        {"rows": "on", "rows.items.0.x": 0.5}, {"rows": "on", "items": [{"x": 0.5}]}, id="nested-group",
    ),
])
@pytest.mark.parametrize("validation_result", [True, False, "Rows rejected"])
async def test_custom_validation_receives_only_requested_roots(monkeypatch, input_spec, values, expected, validation_result):
    received = []

    class Validator(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                input_spec,
                io.Float.Input("ignored"),
            ], outputs=[], is_output_node=True)

        @classmethod
        def validate_inputs(cls, rows):
            received.append(rows)
            return validation_result

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Validator", Validator)
    prompt = {"probe": {"class_type": "Validator", "inputs": {"ignored": 4.0, **values}}}
    valid, _, _, errors = await execution.validate_prompt("validation", prompt, None)

    assert received == [expected]
    assert valid is (validation_result is True)
    if validation_result is True:
        assert errors == {}
    else:
        assert {error["type"] for error in errors["probe"]["errors"]} == {"custom_validation_failed"}
        assert [error["extra_info"] for error in errors["probe"]["errors"]] == (
            [{"input_name": key} for key in values] or [{}]
        )


async def test_validation_omits_unrequested_dynamic_roots(monkeypatch):
    received = []

    class Validator(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                io.Float.Input("fixed"),
                io.DynamicGroup.Input("rows", template=[io.Float.Input("x")]),
            ], outputs=[], is_output_node=True)

        @classmethod
        def validate_inputs(cls, fixed):
            received.append(fixed)
            return True

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Validator", Validator)
    prompt = {"probe": {"class_type": "Validator", "inputs": {"fixed": 0.5, "rows.0.x": 9.0}}}
    valid, _, _, errors = await execution.validate_prompt("static", prompt, None)

    assert valid, errors
    assert received == [0.5]


async def test_kwargs_validator_retains_dynamic_and_static_inputs(monkeypatch):
    received = []

    class Validator(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                io.DynamicGroup.Input("rows", template=[io.Float.Input("x")]), io.Float.Input("fixed"),
            ], outputs=[], is_output_node=True)

        @classmethod
        def validate_inputs(cls, **kwargs):
            received.append(kwargs)
            return True

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Validator", Validator)
    prompt = {"probe": {"class_type": "Validator", "inputs": {"rows.0.x": 0.5, "fixed": 2.0}}}
    valid, _, _, errors = await execution.validate_prompt("kwargs", prompt, None)

    assert valid, errors
    assert received == [{"rows": [{"x": 0.5}], "fixed": 2.0}]


@pytest.mark.parametrize("values", [{}, {"value": 0.5}])
async def test_legacy_validation_rejection_is_recorded_even_without_inputs(monkeypatch, values):
    class Validator:
        @classmethod
        def INPUT_TYPES(cls):
            return {"optional": {"value": ("FLOAT", {})}}

        @classmethod
        def VALIDATE_INPUTS(cls, **kwargs):
            return "Rejected"

        RETURN_TYPES = ()
        OUTPUT_NODE = True

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Validator", Validator)
    prompt = {"probe": {"class_type": "Validator", "inputs": values}}
    valid, _, _, errors = await execution.validate_prompt("legacy", prompt, None)

    assert not valid
    assert errors["probe"]["errors"] == [{
        "type": "custom_validation_failed",
        "message": "Custom validation failed for node",
        "details": "value - Rejected" if values else "Rejected",
        "extra_info": {"input_name": "value"} if values else {},
    }]


@pytest.mark.parametrize("kind", ["group", "autogrow"])
@pytest.mark.parametrize("lazy,request_first", [(True, False), (True, True), (False, False)])
async def test_dynamic_lazy_inputs_execute_only_when_requested(monkeypatch, kind, lazy, request_first):
    executed = []
    if kind == "group":
        input_spec = io.DynamicGroup.Input("rows", template=[io.Float.Input("x", lazy=lazy)])
        keys = ["rows.0.x", "rows.1.x"]
    else:
        input_spec = io.Autogrow.Input("rows", template=io.Autogrow.TemplatePrefix(io.Float.Input("x", lazy=lazy), prefix="item"))
        keys = ["rows.item0", "rows.item1"]

    class Source(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[io.Float.Input("value")], outputs=[io.Float.Output()])

        @classmethod
        def execute(cls, value):
            executed.append(value)
            return io.NodeOutput(value)

    class Consumer(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[input_spec], outputs=[], is_output_node=True)

        @classmethod
        def check_lazy_status(cls, rows):
            value, key = rows[0]["x"] if kind == "group" else rows["item0"]
            return [key] if request_first and value is None else []

        @classmethod
        def execute(cls, rows):
            return io.NodeOutput(ui={"rows": [rows]})

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Source", Source)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Consumer", Consumer)
    prompt = {
        "first": {"class_type": "Source", "inputs": {"value": 2.0}},
        "second": {"class_type": "Source", "inputs": {"value": 3.0}},
        "consumer": {"class_type": "Consumer", "inputs": {keys[0]: ["first", 0], keys[1]: ["second", 0]}},
    }
    valid, _, outputs, errors = await execution.validate_prompt("lazy", prompt, None)
    assert valid, errors
    server = SimpleNamespace(client_id=None, send_sync=Mock())
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "lazy", execute_outputs=outputs)

    assert executor.success, executor.status_messages
    first = 2.0 if request_first or not lazy else None
    second = 3.0 if not lazy else None
    assert sorted(executed) == [value for value in [first, second] if value is not None]
    expected = [{"x": first}, {"x": second}] if kind == "group" else {"item0": first, "item1": second}
    assert executor.history_result["outputs"]["consumer"]["rows"] == [expected]
