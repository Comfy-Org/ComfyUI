import asyncio

import pytest

from comfy_execution.graph import DynamicPrompt, ExecutionList


def test_dynamic_prompt_can_override_ephemeral_node_without_mutating_it():
    original = {"node": {"class_type": "Original", "inputs": {"value": 1}}}
    prompt = DynamicPrompt(original)
    prompt.add_ephemeral_node("child", {"class_type": "Child", "inputs": {"value": 2}}, "node", "child")

    prompt.override_node("child", {"class_type": "Child", "inputs": {"value": 3}})

    assert prompt.get_node("child")["inputs"] == {"value": 3}
    assert prompt.ephemeral_prompt["child"]["inputs"] == {"value": 2}


def execution_list():
    result = ExecutionList.__new__(ExecutionList)
    result.staged_node_id = "start"
    result.pendingNodes = {node_id: True for node_id in ("start", "first", "second", "end")}
    result.blockCount = {"start": 0, "first": 1, "second": 1, "end": 2}
    result.blocking = {
        "start": {"first": {0: True}},
        "first": {"second": {0: True}, "end": {0: True}},
        "second": {"end": {0: True}},
        "end": {},
    }
    result.execution_cache = {node_id: {} for node_id in result.pendingNodes}
    result.execution_cache_listeners = {node_id: set() for node_id in result.pendingNodes}
    result.execution_cache_listeners["start"] = {("first", 0), ("end", 0)}
    result.externalBlocks = 0
    result.externalBlockResults = {}
    result.unblockedEvent = asyncio.Event()
    return result


def test_inhibit_nodes_removes_body_and_releases_downstream_node():
    graph = execution_list()

    graph.inhibit_nodes({"first", "second"})

    assert graph.pendingNodes == {"start": True, "end": True}
    assert graph.blockCount == {"start": 0, "end": 0}
    assert graph.blocking == {"start": {}, "end": {}}
    assert set(graph.execution_cache) == {"start", "end"}
    assert set(graph.execution_cache_listeners) == {"start", "end"}
    assert graph.execution_cache_listeners["start"] == {("end", 0)}


def test_inhibit_nodes_ignores_nodes_which_are_not_pending():
    graph = execution_list()

    graph.inhibit_nodes({"missing"})

    assert set(graph.pendingNodes) == {"start", "first", "second", "end"}
    assert graph.blockCount["end"] == 2


def test_inhibit_nodes_requires_staged_control_node():
    graph = execution_list()
    graph.staged_node_id = None

    with pytest.raises(AssertionError, match="while a control node is staged"):
        graph.inhibit_nodes({"first"})


def test_control_node_cannot_inhibit_itself():
    graph = execution_list()

    with pytest.raises(AssertionError, match="cannot inhibit itself"):
        graph.inhibit_nodes({"start"})


def test_staged_node_reports_dependencies_added_during_lazy_status():
    graph = execution_list()

    assert not graph.is_staged_node_blocked()
    graph.blockCount["start"] = 1

    assert graph.is_staged_node_blocked()


def test_external_block_carries_a_result_to_its_node():
    graph = execution_list()

    block = graph.add_external_block("end")
    block([1, 2])

    assert graph.get_external_block_result("end") == [1, 2]
    assert graph.externalBlocks == 0
    assert graph.blockCount["end"] == 2


def test_multiple_external_blocks_can_be_released():
    graph = execution_list()

    first = graph.add_external_block("end")
    second = graph.add_external_block("end")

    first()
    second()
    assert graph.get_external_block_result("end") is None
    assert graph.externalBlocks == 0
    assert graph.blockCount["end"] == 2
