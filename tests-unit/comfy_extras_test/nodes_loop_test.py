import math
from types import SimpleNamespace

from comfy_api.latest import io
from comfy_extras.nodes_loop import EndLoop, LoopIteration, LoopProgress, LoopResult, StartLoop


def test_loop_schema_exposes_cache_policy_and_integrated_carry():
    inputs = StartLoop.INPUT_TYPES()

    assert inputs["required"]["cache_iterations"][1]["default"] is False
    assert inputs["required"]["cache_iterations"][1]["advanced"] is True
    assert list(inputs["optional"]) == ["parent_iteration", "initial_iteration_value"]
    assert StartLoop.RETURN_NAMES[:5] == [
        "iteration_index",
        "is_first",
        "is_last",
        "list_item",
        "current_iteration_value",
    ]
    assert len(StartLoop.RETURN_NAMES) == 5
    assert "_loop_result" not in EndLoop.INPUT_TYPES().get("optional", {})


def test_iteration_cache_policy_is_stable_or_unique():
    assert math.isnan(StartLoop.fingerprint_inputs(False))
    assert math.isnan(StartLoop.fingerprint_inputs([False]))
    assert math.isnan(StartLoop.fingerprint_inputs(True))
    assert math.isnan(StartLoop.fingerprint_inputs([True]))
    assert math.isnan(LoopIteration.fingerprint_inputs(False))
    assert math.isnan(LoopIteration.fingerprint_inputs([False]))
    assert LoopIteration.fingerprint_inputs(True) is None
    assert LoopIteration.fingerprint_inputs([True]) is None
    assert math.isnan(LoopProgress.fingerprint_inputs())


def test_loop_progress_reports_on_start_node(monkeypatch):
    updates = []
    monkeypatch.setattr(
        "comfy_extras.nodes_loop.PromptServer",
        SimpleNamespace(instance=SimpleNamespace(send_progress_text=lambda *args: updates.append(args))),
    )

    assert LoopProgress.execute(["start"], [2], [4]).result == (2,)
    assert updates == [("Iteration 2 / 4", "start")]


def test_loop_result_passes_iteration_outputs_to_end_loop():
    unblocked = []
    result = {}
    def release(node_id, outputs):
        assert node_id == "close"
        result["close"] = outputs
        unblocked.append(True)
    execution_list = SimpleNamespace(
        release_external_block=release,
        get_external_block_result=lambda node_id: result.get(node_id),
    )
    result_node = LoopResult.PREPARE_CLASS_CLONE({"hidden_inputs": {io.Hidden.execution_list: execution_list}})
    end_node = EndLoop.PREPARE_CLASS_CLONE(
        {"hidden_inputs": {io.Hidden.execution_list: execution_list, io.Hidden.unique_id: "close"}}
    )

    result_node.execute(["close"], output0=[1, "a"], output1=[2, "b"])

    assert unblocked == [True]
    assert end_node.execute([True]).result == ([1, "a", 2, "b"],)
