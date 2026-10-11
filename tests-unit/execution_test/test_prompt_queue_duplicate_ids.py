import threading

import pytest

from execution import DuplicatePromptIdError, PromptQueue


class Server:
    def queue_updated(self):
        pass


def item(number, prompt_id):
    return (number, prompt_id, {}, {}, [], {})


def test_duplicate_prompt_id_is_rejected_while_pending_or_running():
    queue = PromptQueue(Server())
    original = item(7, "duplicate")

    queue.put(original)
    with pytest.raises(DuplicatePromptIdError):
        queue.put(item(7, "duplicate"))
    assert queue.queue == [original]

    _, item_id = queue.get()
    with pytest.raises(DuplicatePromptIdError):
        queue.put(item(8, "duplicate"))
    assert queue.queue == []
    assert queue.currently_running == {item_id: original}

    queue.task_done(item_id, {}, None)
    queue.put(item(9, "duplicate"))


def test_distinct_prompt_ids_with_equal_priority_are_accepted():
    queue = PromptQueue(Server())

    queue.put(item(7, "first"))
    queue.put(item(7, "second"))

    assert sorted(queued[1] for queued in queue.queue) == ["first", "second"]


def test_concurrent_duplicate_prompt_id_has_one_winner():
    queue = PromptQueue(Server())
    barrier = threading.Barrier(8)
    outcomes = []

    def put(index):
        barrier.wait()
        try:
            queue.put(item(index, "duplicate"))
            outcomes.append("accepted")
        except DuplicatePromptIdError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=put, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("accepted") == 1
    assert outcomes.count("rejected") == 7
    assert len(queue.queue) == 1
