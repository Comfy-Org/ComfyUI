import pytest

from comfy.model_management import interrupt_current_processing
from comfy_api import v4
from comfy_execution.utils import CurrentNodeContext
from node_test import kept


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_check_interrupted_clears_the_cancel_that_is_interrupted_only_reports(
    run_node, synchronous
):
    executor, server = await run_node(
        "comfy.model_management.interrupt_current_processing()\n"
        "seen = [{call}.execution.is_interrupted(), {call}.execution.is_interrupted()]\n"
        "try:\n"
        "    {call}.execution.check_interrupted()\n"
        "except InterruptProcessingException:\n"
        "    seen.append('raised')\n"
        "seen.append({call}.execution.is_interrupted())\n"
        "{call}.execution.check_interrupted()\n"
        "{call}.ui.send('seen', {{'seen': seen}})",
        synchronous,
    )

    assert executor.success, executor.status_messages
    assert [data for event, data, _ in server.sent if event == "seen"] == [
        {"seen": [True, True, "raised", False]}
    ]


@pytest.mark.asyncio
async def test_check_interrupted_in_a_finished_nodes_context_leaves_the_cancel():
    interrupt_current_processing()

    with CurrentNodeContext("an earlier prompt", "node"):
        with pytest.raises(v4.InterruptProcessingException):
            await v4.eng.execution.check_interrupted()
    assert await v4.eng.execution.is_interrupted()


@pytest.mark.asyncio
async def test_check_interrupted_outside_a_node_raises_and_leaves_the_cancel_for_the_prompt():
    interrupt_current_processing()

    with pytest.raises(v4.InterruptProcessingException):
        await v4.eng.execution.check_interrupted()
    assert await v4.eng.execution.is_interrupted()


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize(
    "stop",
    [
        "{call}.execution.check_interrupted()",
        "{call}.progress.update(1, 2)",
        "raise InterruptProcessingException()",
    ],
)
async def test_a_cancelled_node_ends_the_prompt_as_interrupted(
    run_node, stop, synchronous
):
    cancel = "comfy.model_management.interrupt_current_processing()\n"

    executor, server = await run_node(cancel + stop, synchronous)

    assert not executor.success
    assert executor.status_messages[-1][0] == "execution_interrupted"


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_a_node_can_see_a_cancel_and_finish_its_work(run_node, synchronous):
    cancel = "comfy.model_management.interrupt_current_processing()\n"

    executor, server = await run_node(
        cancel + "assert {call}.execution.is_interrupted()", synchronous
    )

    assert executor.success, executor.status_messages


LEFTOVER_TASK_PACK = """
import asyncio

import comfy.model_management
from comfy_api.v4 import ComfyExtension, InterruptProcessingException, eng, io
from server import PromptServer


def record(what):
    PromptServer.instance.sent.append(("kept", what, None))


TASKS = []


async def watch():
    await asyncio.sleep(0.05)
    try:
        await eng.execution.check_interrupted()
    except InterruptProcessingException:
        record("the finished node's task saw the cancel")


class Starts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="LeftoverStarts", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        TASKS.append(asyncio.create_task(watch()))
        return io.NodeOutput(1)


class Cancelled(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="LeftoverCancelled", inputs=[io.Int.Input("x")], outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls, x):
        comfy.model_management.interrupt_current_processing()  # the user presses cancel
        await asyncio.sleep(0.2)
        return io.NodeOutput(x)


class After(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="LeftoverAfter", inputs=[io.Int.Input("x")], is_output_node=True)

    @classmethod
    def execute(cls, x):
        record("a node after the cancel ran")
        return io.NodeOutput()


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Starts, Cancelled, After]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
async def test_a_task_left_over_from_a_finished_node_does_not_consume_the_users_cancel(
    load_pack, run_prompt
):
    await load_pack(LEFTOVER_TASK_PACK, "v4_leftover")

    executor, server = await run_prompt(
        {
            "1": {"class_type": "LeftoverStarts", "inputs": {}},
            "2": {"class_type": "LeftoverCancelled", "inputs": {"x": ["1", 0]}},
            "3": {"class_type": "LeftoverAfter", "inputs": {"x": ["2", 0]}},
        },
        ["3"],
    )

    events = [event for event, _ in executor.status_messages]
    assert "execution_interrupted" in events, (events, kept(server))
    assert kept(server) == ["the finished node's task saw the cancel"]


@pytest.mark.asyncio
async def test_a_cancel_while_a_later_node_waits_stops_the_prompt(
    load_pack, run_prompt
):
    await load_pack(
        LEFTOVER_TASK_PACK.replace("check_interrupted()", "is_interrupted()"),
        "v4_leftover_control",
    )

    executor, server = await run_prompt(
        {
            "1": {"class_type": "LeftoverStarts", "inputs": {}},
            "2": {"class_type": "LeftoverCancelled", "inputs": {"x": ["1", 0]}},
            "3": {"class_type": "LeftoverAfter", "inputs": {"x": ["2", 0]}},
        },
        ["3"],
    )

    assert "execution_interrupted" in [event for event, _ in executor.status_messages]
    assert "a node after the cancel ran" not in kept(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_a_cancel_passes_through_a_nodes_except_exception(run_node, synchronous):
    executor, server = await run_node(
        "comfy.model_management.interrupt_current_processing()\n"
        "try:\n"
        "    {call}.execution.check_interrupted()\n"
        "except Exception:\n"
        "    pass",
        synchronous,
    )

    assert not executor.success
    assert executor.status_messages[-1][0] == "execution_interrupted"
