import pytest
import torch
from aiohttp import web

from comfy.cli_args import args

# Must precede the import: comfy.model_management picks its device at import time, and a CUDA
# build with no driver raises there.
_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import main
finally:
    args.cpu = _original_cpu

class LoopEscape(Exception):
    pass


class Queue:
    def __init__(
        self,
        completion_error: RuntimeError | None = None,
        queued_after_first: int = 1,
        on_second_get=None,
    ) -> None:
        self.completion_error = completion_error
        # Prompts still waiting once the first is taken; tests opt in to an idle queue.
        self.queued = queued_after_first
        self.running = 0
        self.on_second_get = on_second_get
        self.get_calls = 0

    def get(self, timeout=None):
        self.get_calls += 1
        if self.get_calls > 1:
            if self.on_second_get is not None:
                self.on_second_get()
            raise LoopEscape("prompt worker requested a second item")
        self.running = 1
        return (0, "prompt-id", {}, {}, [], {}), 1

    def get_tasks_remaining(self) -> int:
        # Like PromptQueue: queued plus running, and the running prompt only leaves in task_done.
        return self.queued + self.running

    def task_done(self, *args, **kwargs) -> None:
        if self.completion_error is not None:
            raise self.completion_error
        self.running = 0

    def get_flags(self):
        return {}


class Server:
    last_prompt_id = None
    client_id = None


class AssetManager:
    def __init__(self, resume_error: RuntimeError | None = None) -> None:
        self.paused = False
        self.resume_error = resume_error

    def pause_background_scan(self) -> None:
        self.paused = True

    def resume_background_scan(self) -> None:
        self.paused = False
        if self.resume_error is not None:
            raise self.resume_error


class Executor:
    def __init__(self, *args, **kwargs) -> None:
        self.history_result = {}
        self.success = True
        self.status_messages = []

    def execute(self, *args, **kwargs) -> None:
        return None


class ExecuteFailureExecutor(Executor):
    def execute(self, *args, **kwargs) -> None:
        raise RuntimeError("forced execute failure")


def test_prompt_worker_resumes_background_scan_when_execute_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", ExecuteFailureExecutor)
    asset_manager = AssetManager()

    with pytest.raises(RuntimeError, match="^forced execute failure$"):
        main.prompt_worker(Queue(), Server(), asset_manager)

    assert asset_manager.paused is False


def test_prompt_worker_resumes_background_scan_when_completion_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    asset_manager = AssetManager()

    with pytest.raises(RuntimeError, match="^forced completion failure$"):
        main.prompt_worker(
            Queue(completion_error=RuntimeError("forced completion failure")),
            Server(),
            asset_manager,
        )

    assert asset_manager.paused is False


def test_prompt_worker_preserves_execute_error_when_resume_raises(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", ExecuteFailureExecutor)
    asset_manager = AssetManager(resume_error=RuntimeError("forced resume failure"))

    with pytest.raises(RuntimeError, match="^forced execute failure$"):
        main.prompt_worker(Queue(), Server(), asset_manager)

    assert asset_manager.paused is False


def test_free_endpoint_stores_explicit_false_unload_models() -> None:
    """Regression test for issue #16620: an explicit "unload_models": false
    in a POST /free body must not be dropped by server.py's flag storage, or
    main.py's flags.get("unload_models", free_memory) fallback would still
    trigger an unload whenever free_memory is set.
    """
    flags = main.server._free_endpoint_flags({"free_memory": True, "unload_models": False})

    free_memory = flags.get("free_memory", False)
    assert flags.get("unload_models", free_memory) is False


@pytest.mark.asyncio
async def test_post_free_stores_explicit_false_unload_models(aiohttp_client) -> None:
    """Drives POST /free over HTTP against a route that mirrors server.py's
    real handler wiring, so a regression in the handler itself (not just in
    _free_endpoint_flags) would also be caught."""

    class FakeQueue:
        def __init__(self) -> None:
            self.flags = {}

        def set_flag(self, name, value) -> None:
            self.flags[name] = value

    queue = FakeQueue()
    routes = web.RouteTableDef()

    @routes.post("/free")
    async def post_free(request):
        json_data = await request.json()
        for name, value in main.server._free_endpoint_flags(json_data).items():
            queue.set_flag(name, value)
        return web.Response(status=200)

    app = web.Application()
    app.add_routes(routes)
    client = await aiohttp_client(app)

    resp = await client.post("/free", json={"free_memory": True, "unload_models": False})

    assert resp.status == 200
    assert queue.flags["unload_models"] is False


def test_prompt_worker_resumes_scan_when_later_iteration_raises_before_gc(monkeypatch) -> None:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    clock = iter((1.0, 2.0, 2.0))
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(clock))
    asset_manager = AssetManager()
    queue = Queue()

    with pytest.raises(LoopEscape, match="^prompt worker requested a second item$"):
        main.prompt_worker(queue, Server(), asset_manager)

    assert queue.get_calls == 2
    assert asset_manager.paused is False


def _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first: int) -> list[bool]:
    monkeypatch.setattr(main.execution, "PromptExecutor", Executor)
    # The clock never reaches the 10 s GC interval, so only the idle check can resume the scan.
    monkeypatch.setattr(main.time, "perf_counter", lambda: 1.0)
    asset_manager = AssetManager()
    seen: list[bool] = []
    queue = Queue(queued_after_first=queued_after_first, on_second_get=lambda: seen.append(asset_manager.paused))

    with pytest.raises(LoopEscape, match="^prompt worker requested a second item$"):
        main.prompt_worker(queue, Server(), asset_manager)

    return seen


def test_prompt_worker_resumes_scan_once_the_queue_is_empty(monkeypatch) -> None:
    assert _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first=0) == [False]


def test_prompt_worker_keeps_scan_paused_while_prompts_are_queued(monkeypatch) -> None:
    assert _paused_when_the_worker_asks_for_the_next_prompt(monkeypatch, queued_after_first=1) == [True]
