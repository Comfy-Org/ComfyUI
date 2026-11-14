import asyncio
import contextvars
import threading

import pytest

from comfy_api.v4._sync import SyncGroup


@pytest.mark.asyncio
async def test_sync_call_runs_on_the_calling_thread_in_its_context_and_loop():
    current = contextvars.ContextVar("current")
    current.set("node")

    class Calls:
        async def where(self):
            return threading.get_ident(), current.get(), asyncio.get_running_loop()

    assert SyncGroup(Calls, Calls()).where() == (
        threading.get_ident(),
        "node",
        asyncio.get_running_loop(),
    )


@pytest.mark.parametrize("exception_type", [ValueError, BaseException])
def test_sync_call_raises_the_exception_the_call_raised(exception_type):
    error = exception_type("failure")

    class Calls:
        async def fail(self):
            raise error

    with pytest.raises(exception_type) as raised:
        SyncGroup(Calls, Calls()).fail()

    assert raised.value is error


def test_sync_call_that_needs_the_event_loop_is_stopped_and_not_run_again():
    events = []

    class Calls:
        async def wait(self):
            events.append("started")
            try:
                await asyncio.sleep(0)
                events.append("finished")
            finally:
                events.append("closed")

    with pytest.raises(RuntimeError, match="needs an event loop; await it"):
        try:
            SyncGroup(Calls, Calls()).wait()
        finally:
            assert events == ["started", "closed"]


def test_sync_group_has_the_declared_calls_and_runs_the_given_implementation():
    class Declared:
        async def name(self):
            return "declared"

    class Provider(Declared):
        async def name(self):
            return "provider"

        async def extra(self):
            return "not part of the API"

    group = SyncGroup(Declared, Provider())

    assert group.name() == "provider"
    assert not hasattr(group, "extra")
