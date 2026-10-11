import ast
import asyncio
import copy
import heapq
import logging
import threading
import types
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional

ROOT = Path(__file__).resolve().parents[2]


def _definitions(path, *names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {
        "copy": copy,
        "heapq": heapq,
        "threading": threading,
        "List": List,
        "Literal": Literal,
        "NamedTuple": NamedTuple,
        "Optional": Optional,
        "MAXIMUM_HISTORY_SIZE": 10000,
        "nodes": types.SimpleNamespace(interrupt_processing=lambda: None),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _queue_types():
    return _definitions(ROOT / "execution.py", "DuplicatePromptIdError", "PromptQueue")


class Server:
    def __init__(self):
        self.updates = 0

    def queue_updated(self):
        self.updates += 1


def _item(number, prompt_id, prompt):
    return (number, prompt_id, prompt, {}, [], {})


def test_duplicate_pending_and_running_ids_are_rejected_without_mutating_the_heap():
    definitions = _queue_types()
    duplicate_error = definitions["DuplicatePromptIdError"]
    queue = definitions["PromptQueue"](Server())
    prompt_id = "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7"
    original = _item(7, prompt_id, {"one": {"class_type": "First"}})

    queue.put(original)
    try:
        queue.put(_item(7, prompt_id, {"two": {"class_type": "Second"}}))
    except duplicate_error:
        pass
    else:
        raise AssertionError("duplicate pending prompt_id was accepted")
    assert queue.queue == [original]

    queued, item_id = queue.get()
    assert queued == original
    try:
        queue.put(_item(8, prompt_id, {"three": {"class_type": "Third"}}))
    except duplicate_error:
        pass
    else:
        raise AssertionError("duplicate running prompt_id was accepted")
    assert queue.queue == []
    assert queue.currently_running == {item_id: original}

    queue.task_done(item_id, {}, None)
    queue.put(_item(9, prompt_id, {"four": {"class_type": "Fourth"}}))
    assert queue.queue[0][1] == prompt_id


def _post_prompt_function():
    tree = ast.parse((ROOT / "server.py").read_text())
    post_prompt = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "post_prompt"
    )
    post_prompt.decorator_list = []
    factory = ast.FunctionDef(
        name="bind_post_prompt",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[post_prompt, ast.Return(value=ast.Name(id="post_prompt", ctx=ast.Load()))],
        decorator_list=[],
    )
    ast.fix_missing_locations(factory)
    return factory


def test_duplicate_route_returns_conflict_without_consuming_an_automatic_number():
    definitions = _queue_types()
    duplicate_error = definitions["DuplicatePromptIdError"]

    class Response:
        def __init__(self, payload, status):
            self.payload = payload
            self.status = status

    class Request:
        headers = {}

        async def json(self):
            return {
                "prompt_id": "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7",
                "prompt": {"one": {"class_type": "Test"}},
            }

    class Queue:
        def put(self, item):
            raise duplicate_error(item[1])

    async def validate_prompt(*_args):
        return True, None, [], {}

    namespace = {
        "execution": types.SimpleNamespace(
            DuplicatePromptIdError=duplicate_error,
            SENSITIVE_EXTRA_DATA_KEYS=[],
            validate_prompt=validate_prompt,
        ),
        "logging": logging,
        "time": types.SimpleNamespace(time=lambda: 0),
        "uuid": types.SimpleNamespace(uuid4=lambda: "generated-id"),
        "validate_job_id": lambda value: value,
        "valid_workflow_metadata": lambda _data: None,
        "workflow_metadata_from_prompt": lambda _data: None,
        "web": types.SimpleNamespace(json_response=lambda payload, status=200: Response(payload, status)),
    }
    exec(compile(ast.Module(body=[_post_prompt_function()], type_ignores=[]), str(ROOT / "server.py"), "exec"), namespace)
    server = types.SimpleNamespace(
        number=12,
        prompt_queue=Queue(),
        node_replace_manager=types.SimpleNamespace(apply_replacements=lambda _prompt: None),
        trigger_on_prompt=lambda data: data,
    )

    response = asyncio.run(namespace["bind_post_prompt"](server)(Request()))

    assert response.status == 409
    assert response.payload["error"]["type"] == "duplicate_prompt_id"
    assert server.number == 12
