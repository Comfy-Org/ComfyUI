# execution.py pulls in comfy.model_management, whose module-level device
# probe needs a working accelerator unless --cpu is set. Unit-test runners
# (and CI) run on CPU-only torch, so force the CPU state before importing.
from comfy.cli_args import args

args.cpu = True

import execution


class StubServer:
    def __init__(self):
        self.updates = 0

    def queue_updated(self):
        self.updates += 1


def make_item(number, prompt_id, prompt):
    return (number, prompt_id, prompt, {}, [], {})


def make_queue():
    return execution.PromptQueue(StubServer())


def test_put_rejects_duplicate_pending_prompt_id_without_mutating_the_heap():
    queue = make_queue()
    prompt_id = "00000000-0000-4000-8000-000000000001"

    assert queue.put(make_item(1.0, prompt_id, {"node": "first"})) is True
    assert queue.put(make_item(1.0, prompt_id, {"node": "second"})) is False

    # The heap was not mutated: the first submission stays intact and no
    # partially-inserted entry is left behind.
    assert len(queue.queue) == 1
    assert queue.queue[0][2] == {"node": "first"}


def test_put_rejects_duplicate_pending_prompt_id_regardless_of_number():
    queue = make_queue()
    prompt_id = "00000000-0000-4000-8000-000000000002"

    assert queue.put(make_item(1.0, prompt_id, {"node": "first"})) is True
    # A different queue number must not slip a duplicate ID through: the
    # equal (number, prompt_id) prefix is what makes heapq compare the
    # prompt dicts and raise, but any duplicate ID corrupts the
    # one-ID-one-job contract the queue state relies on.
    assert queue.put(make_item(7.5, prompt_id, {"node": "second"})) is False
    assert len(queue.queue) == 1


def test_put_rejects_prompt_id_that_is_currently_running():
    queue = make_queue()
    prompt_id = "00000000-0000-4000-8000-000000000003"

    assert queue.put(make_item(1.0, prompt_id, {"node": "only"})) is True
    item, task_id = queue.get(timeout=0.1)
    assert item[1] == prompt_id

    assert queue.put(make_item(5.0, prompt_id, {"node": "again"})) is False
    assert len(queue.queue) == 0


def test_put_accepts_distinct_ids_with_equal_priorities():
    queue = make_queue()
    a = "00000000-0000-4000-8000-000000000004"
    b = "00000000-0000-4000-8000-000000000005"

    assert queue.put(make_item(1.0, a, {"node": "a"})) is True
    assert queue.put(make_item(1.0, b, {"node": "b"})) is True
    assert len(queue.queue) == 2


def test_put_accepts_an_id_again_once_it_is_no_longer_in_flight():
    queue = make_queue()
    prompt_id = "00000000-0000-4000-8000-000000000006"

    assert queue.put(make_item(1.0, prompt_id, {"node": "first"})) is True
    _, task_id = queue.get(timeout=0.1)
    queue.task_done(
        task_id,
        {},
        execution.PromptQueue.ExecutionStatus("success", True, []),
    )

    # Completed-history ID reuse is unchanged: the ID may be submitted again
    # once nothing is pending or running under it.
    assert queue.put(make_item(2.0, prompt_id, {"node": "second"})) is True
    assert len(queue.queue) == 1
