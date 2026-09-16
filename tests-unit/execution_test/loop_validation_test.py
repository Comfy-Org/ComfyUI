import asyncio
from collections import Counter

import pytest

import nodes
from comfy_execution.validation import LoopValidationError, validate_loops
from comfy_extras.nodes_loop import EndLoop, StartLoop
from execution import _loop_boundary, validate_prompt


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def validate(prompt, *outputs):
    starts = {node_id for node_id, value in prompt.items() if value["class_type"] == "StartLoop"}
    ends = {node_id for node_id, value in prompt.items() if value["class_type"] == "EndLoop"}
    return validate_loops(prompt, set(outputs) or {"output"}, prompt, starts, ends)


def test_pairs_simple_loop():
    prompt = {
        "start": node("StartLoop"),
        "body": node("Body", value=["start", 0]),
        "end": node("EndLoop", value=["body", 0]),
        "output": node("Output", value=["end", 0]),
    }

    assert validate(prompt) == {"start": "end"}


def test_accepts_branched_body_when_every_branch_reaches_end():
    prompt = {
        "start": node("StartLoop"),
        "left": node("Body", value=["start", 0]),
        "right": node("Body", value=["start", 0]),
        "end": node("EndLoop", left=["left", 0], right=["right", 0]),
        "output": node("Output", value=["end", 0]),
    }

    assert validate(prompt) == {"start": "end"}


def test_accepts_dependencies_entering_loop_body():
    prompt = {
        "source": node("Source"),
        "start": node("StartLoop"),
        "body": node("Body", iteration=["start", 0], value=["source", 0]),
        "end": node("EndLoop", value=["body", 0]),
        "output": node("Output", value=["end", 0]),
    }

    assert validate(prompt) == {"start": "end"}


def test_accepts_accumulate_control_from_outside_loop():
    prompt = {
        "control": node("Source"),
        "start": node("StartLoop"),
        "body": node("Body", value=["start", 0]),
        "end": node("EndLoop", value=["body", 0], accumulate=["control", 0]),
        "output": node("Output", value=["end", 0]),
    }

    assert validate(prompt) == {"start": "end"}


def test_pairs_nested_loops_with_outer_value_entering_inner_body():
    prompt = {
        "outer": node("StartLoop"),
        "inner": node("StartLoop", parent=["outer", 0]),
        "inner_body": node("Body", outer=["outer", 0], inner=["inner", 0]),
        "inner_end": node("EndLoop", value=["inner_body", 0]),
        "outer_body": node("Body", outer=["outer", 0], inner=["inner_end", 0]),
        "outer_end": node("EndLoop", value=["outer_body", 0]),
        "output": node("Output", value=["outer_end", 0]),
    }

    assert validate(prompt) == {"inner": "inner_end", "outer": "outer_end"}


def test_pairs_sequential_inner_loops():
    prompt = {
        "outer": node("StartLoop"),
        "first": node("StartLoop", parent=["outer", 0]),
        "first_body": node("Body", value=["first", 0]),
        "first_end": node("EndLoop", value=["first_body", 0]),
        "second": node("StartLoop", value=["first_end", 0]),
        "second_body": node("Body", outer=["outer", 0], value=["second", 0]),
        "second_end": node("EndLoop", value=["second_body", 0]),
        "outer_end": node("EndLoop", value=["second_end", 0]),
        "output": node("Output", value=["outer_end", 0]),
    }

    assert validate(prompt) == {
        "first": "first_end",
        "second": "second_end",
        "outer": "outer_end",
    }


def test_pairs_independent_loops():
    prompt = {
        "left": node("StartLoop"),
        "left_end": node("EndLoop", value=["left", 0]),
        "left_output": node("Output", value=["left_end", 0]),
        "right": node("StartLoop"),
        "right_end": node("EndLoop", value=["right", 0]),
        "right_output": node("Output", value=["right_end", 0]),
    }

    assert validate(prompt, "left_output", "right_output") == {
        "left": "left_end",
        "right": "right_end",
    }


def test_ignores_boundaries_outside_selected_outputs():
    prompt = {
        "start": node("StartLoop"),
        "end": node("EndLoop", value=["start", 0]),
        "output": node("Output", value=["end", 0]),
        "unused_start": node("StartLoop"),
    }

    assert validate_loops(
        prompt,
        {"output"},
        {"start", "end", "output"},
        {"start", "unused_start"},
        {"end"},
    ) == {"start": "end"}


def test_rejects_end_without_start():
    prompt = {
        "source": node("Source"),
        "end": node("EndLoop", value=["source", 0]),
        "output": node("Output", value=["end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["type"] == "custom_validation_failed"
    assert exc.value.error["extra_info"]["input_name"] == "loop boundary"
    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_end_without_start"
    assert exc.value.error["extra_info"]["node_ids"] == ["end"]


def test_rejects_start_without_end():
    prompt = {
        "start": node("StartLoop"),
        "output": node("Output", value=["start", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["type"] == "custom_validation_failed"
    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_start_without_end"
    assert exc.value.error["extra_info"]["node_ids"] == ["start"]


def test_rejects_all_unpaired_starts_together():
    prompt = {
        "first": node("StartLoop"),
        "second": node("StartLoop"),
        "output": node("Output", first=["first", 0], second=["second", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_start_without_end"
    assert exc.value.error["extra_info"]["node_ids"] == ["first", "second"]


def test_rejects_ambiguous_unrelated_starts():
    prompt = {
        "left": node("StartLoop"),
        "right": node("StartLoop"),
        "body": node("Body", left=["left", 0], right=["right", 0]),
        "end": node("EndLoop", value=["body", 0]),
        "output": node("Output", value=["end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "ambiguous_loop_nesting"
    assert exc.value.error["extra_info"]["node_ids"] == ["end", "left", "right"]
    assert exc.value.error["details"] == "End Loop end can close multiple unrelated Start Loops: left, right"


def test_rejects_second_end_reached_before_pair():
    prompt = {
        "start": node("StartLoop"),
        "body": node("Body", value=["start", 0]),
        "end_a": node("EndLoop", value=["body", 0]),
        "end_b": node("EndLoop", value=["body", 0]),
        "output_a": node("Output", value=["end_a", 0]),
        "output_b": node("Output", value=["end_b", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt, "output_a", "output_b")

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_escape"
    assert exc.value.error["extra_info"]["node_ids"] == ["end_a", "end_b", "start"]


def test_rejects_output_route_around_end():
    prompt = {
        "start": node("StartLoop"),
        "body": node("Body", value=["start", 0]),
        "end": node("EndLoop", value=["body", 0]),
        "output": node("Output", closed=["end", 0], bypass=["body", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_escape"
    assert exc.value.error["extra_info"]["node_ids"] == ["end", "output", "start"]


def test_rejects_inner_loop_route_to_unpaired_outer_end():
    prompt = {
        "outer": node("StartLoop"),
        "inner": node("StartLoop", parent=["outer", 0]),
        "body": node("Body", value=["inner", 0]),
        "inner_end": node("EndLoop", value=["body", 0]),
        "outer_end": node("EndLoop", closed=["inner_end", 0], bypass=["body", 0]),
        "output": node("Output", value=["outer_end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_escape"
    assert exc.value.error["extra_info"]["node_ids"] == ["inner", "inner_end", "outer_end"]


def test_rejects_end_after_start_was_already_paired():
    prompt = {
        "start": node("StartLoop"),
        "first_end": node("EndLoop", value=["start", 0]),
        "second_end": node("EndLoop", value=["first_end", 0]),
        "output": node("Output", value=["second_end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_end_without_start"
    assert exc.value.error["extra_info"]["node_ids"] == ["second_end"]


def test_rejects_accumulate_control_from_loop_body():
    prompt = {
        "start": node("StartLoop"),
        "body": node("Body", value=["start", 0]),
        "control": node("Body", value=["body", 0]),
        "end": node("EndLoop", value=["body", 0], accumulate=["control", 0]),
        "output": node("Output", value=["end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_accumulate_from_body"
    assert exc.value.error["extra_info"]["node_ids"] == ["control", "end", "start"]
    assert exc.value.error["details"] == (
        "End Loop end accumulate is driven by loop node control under Start Loop start"
    )


def test_rejects_accumulate_control_directly_from_start():
    prompt = {
        "start": node("StartLoop"),
        "end": node("EndLoop", value=["start", 0], accumulate=["start", 1]),
        "output": node("Output", value=["end", 0]),
    }

    with pytest.raises(LoopValidationError) as exc:
        validate(prompt)

    assert exc.value.error["extra_info"]["loop_error_type"] == "loop_accumulate_from_body"
    assert exc.value.error["extra_info"]["node_ids"] == ["end", "start"]


class Body:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left": ("*",), "right": ("*",)}}

    RETURN_TYPES = ("*",)


class Output:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True


class InvalidOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "label": ("STRING",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True


class Source:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("*",)


def test_prompt_validation_includes_end_after_terminated_output(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Output", Output)
    prompt = {
        "start": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["start", 0], right=["start", 0]),
        "carry": node("Body", left=["body", 0], right=["start", 0]),
        "output": node("Output", value=["body", 0]),
        "end": node(
            "EndLoop",
            output_value=["body", 0],
            next_iteration_value=["carry", 0],
            termination0=["output", 0],
            accumulate=False,
        ),
    }

    valid, error, good_outputs, node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert valid
    assert error is None
    assert good_outputs == ["output"]
    assert node_errors == {}
    assert prompt["start"]["_loop_end"] == "end"
    assert "carry" in prompt["start"]["_loop_body"]


def test_prompt_validation_reports_every_ambiguous_boundary(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Output", Output)
    prompt = {
        "left": node("StartLoop", cache_iterations=False),
        "right": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["left", 0], right=["right", 0]),
        "end": node("EndLoop", output_value=["body", 0], accumulate=False),
        "output": node("Output", value=["end", 0]),
    }

    valid, error, good_outputs, node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert not valid
    assert error["details"] == (
        "End Loop has ambiguous Start Loops: "
        "End Loop end can close multiple unrelated Start Loops: left, right"
    )
    assert good_outputs == []
    assert set(node_errors) == {"left", "right", "end"}
    assert all(
        value["errors"][0]["extra_info"]["node_ids"] == ["end", "left", "right"]
        for value in node_errors.values()
    )


def test_prompt_validation_stacks_loop_and_input_errors(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "InvalidOutput", InvalidOutput)
    prompt = {
        "left": node("StartLoop", cache_iterations=False),
        "right": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["left", 0], right=["right", 0]),
        "end": node("EndLoop", output_value=["body", 0], accumulate=False),
        "output": node("InvalidOutput", value=["end", 0]),
    }

    valid, error, good_outputs, node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert not valid
    assert good_outputs == []
    assert {reason["type"] for reason in node_errors["output"]["errors"]} == {"required_input_missing"}
    assert node_errors["end"]["errors"][0]["type"] == "custom_validation_failed"
    assert node_errors["end"]["errors"][0]["extra_info"]["loop_error_type"] == "ambiguous_loop_nesting"
    assert "Required input is missing" in error["details"]
    assert "End Loop has ambiguous Start Loops" in error["details"]


def test_prompt_validation_reports_loop_escape_as_recognized_node_error(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Output", Output)
    prompt = {
        "start": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["start", 0], right=["start", 0]),
        "end": node("EndLoop", output_value=["body", 0], accumulate=False),
        "output": node("Output", value=["end", 0], bypass=["body", 0]),
    }

    valid, error, good_outputs, node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert not valid
    assert good_outputs == []
    assert error["details"].count("Loop body is not closed") == 1
    for node_id in ("start", "end", "output"):
        node_error = node_errors[node_id]["errors"][0]
        assert node_error["type"] == "custom_validation_failed"
        assert node_error["extra_info"]["input_name"] == "loop boundary"
        assert node_error["extra_info"]["loop_error_type"] == "loop_escape"


def test_loop_error_does_not_reject_independent_output(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Output", Output)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Source", Source)
    prompt = {
        "left": node("StartLoop", cache_iterations=False),
        "right": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["left", 0], right=["right", 0]),
        "end": node("EndLoop", output_value=["body", 0], accumulate=False),
        "loop_output": node("Output", value=["end", 0]),
        "source": node("Source"),
        "independent_output": node("Output", value=["source", 0]),
    }

    valid, error, good_outputs, node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert valid
    assert error is None
    assert good_outputs == ["independent_output"]
    assert set(node_errors) == {"left", "right", "end"}
    assert all(value["dependent_outputs"] == ["loop_output"] for value in node_errors.values())


def test_prompt_validation_builds_each_node_schema_once(monkeypatch):
    """Validation asks every node for its loop boundary, and that must not cost a
    second schema build.

    ``GET_SCHEMA`` re-runs ``define_schema`` on every call. ``validate_inputs``
    needs that once per node, because a combo whose options come from
    ``folder_paths`` has to see files added since startup before it can judge the
    supplied value. A loop boundary is a literal, so asking for it the same way
    doubled the work: measured at two builds per node here, and 3.4ms against
    0.02ms for the boundary scan alone on a 200-node graph.
    """
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StartLoop", StartLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "EndLoop", EndLoop)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Body", Body)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Output", Output)

    StartLoop.GET_SCHEMA()
    EndLoop.GET_SCHEMA()

    builds = Counter()
    for cls in (StartLoop, EndLoop):
        original = cls.define_schema.__func__

        def counting(inner_cls, _original=original, _name=cls.__name__):
            builds[_name] += 1
            return _original(inner_cls)

        monkeypatch.setattr(cls, "define_schema", classmethod(counting))

    prompt = {
        "start": node("StartLoop", cache_iterations=False),
        "body": node("Body", left=["start", 0], right=["start", 0]),
        "end": node("EndLoop", output_value=["body", 0], accumulate=False),
        "output": node("Output", value=["end", 0]),
    }

    valid, _error, _good_outputs, _node_errors = asyncio.run(validate_prompt("prompt", prompt, None))

    assert valid
    assert builds == Counter({"StartLoop": 1, "EndLoop": 1})


def test_loop_boundary_does_not_rebuild_a_schema_the_class_already_has():
    StartLoop.GET_SCHEMA()
    builds = []
    original = StartLoop.define_schema.__func__

    def counting(inner_cls):
        builds.append(1)
        return original(inner_cls)

    StartLoop.define_schema = classmethod(counting)
    try:
        assert _loop_boundary(StartLoop) == "start"
    finally:
        StartLoop.define_schema = classmethod(original)

    assert builds == []


def test_loop_boundary_reads_a_node_that_was_registered_without_a_schema():
    """The fallback is load-bearing: the loop nodes themselves reach validation with
    SCHEMA unset, and it has to fill it in rather than answer None."""
    StartLoop.SCHEMA = None
    try:
        assert _loop_boundary(StartLoop) == "start"
        assert StartLoop.SCHEMA is not None
    finally:
        StartLoop.GET_SCHEMA()

    assert _loop_boundary(Body) is None


def test_loop_boundary_does_not_read_a_subclass_off_its_parent():
    """A subclass that has never been asked must build its own schema.

    GET_SCHEMA caches with `cls.SCHEMA = schema`, so a subclass inherits whatever
    its parent was asked for. Reading that would hand validate_loops the parent's
    boundary for a node that declares a different one, or none at all.
    """
    StartLoop.GET_SCHEMA()

    class NotABoundary(StartLoop):
        @classmethod
        def define_schema(cls):
            schema = StartLoop.define_schema()
            schema.node_id = "NotABoundary"
            schema.loop_boundary = None
            return schema

    assert StartLoop.SCHEMA.loop_boundary == "start"
    assert _loop_boundary(NotABoundary) is None
