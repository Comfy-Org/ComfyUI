from __future__ import annotations

import asyncio
import threading

import pytest

import execution
from comfy_api.latest import _sdk


class _Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *_args):
        pass


class _Backend:
    maintenance_interval_seconds = 0.25

    def __init__(self) -> None:
        self.events = []
        self.loops = []
        self.maintenance_loops = []

    async def on_prompt_start(self, prompt_id, extra_data):
        self.events.append(("start", prompt_id, extra_data))
        self.loops.append(asyncio.get_running_loop())

    async def on_prompt_end(self, prompt_id, extra_data):
        self.events.append(("end", prompt_id, extra_data))

    async def on_prompt_abort(self, prompt_id, extra_data):
        self.events.append(("abort", prompt_id, extra_data))

    async def maintenance(self):
        self.maintenance_loops.append(asyncio.get_running_loop())

    async def dispatch(self, _plan, local_call, _runtime=None):
        return await local_call()


def _executor():
    return execution.PromptExecutor(
        _Server(),
        cache_args={"ram": 0, "ram_inactive": 0},
    )


def test_prompt_awaits_execution_backend_start_and_end_hooks():
    backend = _Backend()
    original = _sdk.providers.execution_backend
    _sdk.providers.execution_backend = backend
    extra_data = {"comfy_secure_tenant_id": "tenant-alice"}
    try:
        asyncio.run(_executor().execute_async({}, "job-1", extra_data, []))
    finally:
        _sdk.providers.execution_backend = original

    assert backend.events == [
        ("start", "job-1", extra_data),
        ("end", "job-1", extra_data),
    ]


def test_prompt_ends_backend_lifecycle_when_execution_setup_raises(monkeypatch):
    backend = _Backend()
    original = _sdk.providers.execution_backend
    _sdk.providers.execution_backend = backend
    monkeypatch.setattr(
        execution,
        "DynamicPrompt",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )
    try:
        with pytest.raises(RuntimeError, match="setup failed"):
            asyncio.run(_executor().execute_async({}, "job-failed", {}, []))
    finally:
        _sdk.providers.execution_backend = original

    assert [event[0] for event in backend.events] == ["start", "abort"]


def test_synchronous_prompt_worker_keeps_one_async_loop_for_warm_realms():
    backend = _Backend()
    original = _sdk.providers.execution_backend
    _sdk.providers.execution_backend = backend
    executor = _executor()
    try:
        executor.execute({}, "job-1", {}, [])
        executor.execute({}, "job-2", {}, [])
        assert executor.execution_backend_maintenance_interval() == 0.25
        executor.maintain_execution_backend()
    finally:
        executor.close()
        _sdk.providers.execution_backend = original

    assert backend.loops[0] is backend.loops[1]
    assert backend.maintenance_loops == [backend.loops[0]]


def test_cleanup_failure_does_not_mask_prompt_failure_or_stop_maintenance(
    monkeypatch,
):
    class FailingCleanupBackend(_Backend):
        async def on_prompt_abort(self, prompt_id, extra_data):
            raise RuntimeError("cleanup failed")

        async def maintenance(self):
            raise RuntimeError("maintenance failed")

    backend = FailingCleanupBackend()
    original = _sdk.providers.execution_backend
    _sdk.providers.execution_backend = backend
    monkeypatch.setattr(
        execution,
        "DynamicPrompt",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )
    executor = _executor()
    try:
        with pytest.raises(RuntimeError, match="setup failed"):
            executor.execute({}, "job-failed", {}, [])
        executor.maintain_execution_backend()
    finally:
        executor.close()
        _sdk.providers.execution_backend = original


def test_prompt_executor_shutdown_cancels_active_prompt_and_scrubs_backend():
    class BlockingBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.shutdown_called = False

        async def on_prompt_start(self, prompt_id, extra_data):
            await super().on_prompt_start(prompt_id, extra_data)
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.001)

        async def shutdown(self):
            self.shutdown_called = True

    backend = BlockingBackend()
    original = _sdk.providers.execution_backend
    _sdk.providers.execution_backend = backend
    executor = _executor()
    outcome = []

    def execute_prompt():
        try:
            executor.execute({}, "job-active", {}, [])
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=execute_prompt)
    worker.start()
    assert backend.started.wait(timeout=1)
    try:
        executor.request_shutdown()
    finally:
        backend.release.set()
        worker.join(timeout=2)
        executor.close()
        _sdk.providers.execution_backend = original

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], asyncio.CancelledError)
    assert [event[0] for event in backend.events] == ["start", "abort"]
    assert backend.shutdown_called
