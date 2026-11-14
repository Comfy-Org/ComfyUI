import pytest

from comfy.cli_args import args
from comfy_api import v4
from server import PromptServer


class Tab:
    def __init__(self):
        self.received = []

    async def send_json(self, message):
        self.received.append(message)


class Sockets:
    send_json = PromptServer.send_json

    def __init__(self, *names):
        self.sockets = {name: Tab() for name in names}


async def deliver(server, event, *client_ids):
    """Deliver the recorded `event` messages the way ComfyUI's socket layer does."""
    sockets = Sockets(*client_ids)
    for sent_event, data, sid in server.sent:
        if sent_event == event:
            await sockets.send_json(sent_event, data, sid)
    return {name: tab.received for name, tab in sockets.sockets.items()}


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_send_reaches_only_the_client_that_queued_the_prompt(
    run_node, synchronous
):
    executor, server = await run_node(
        '{call}.ui.send("my_pack.done", {{"node": "v4", "count": 2}})', synchronous
    )

    assert executor.success, executor.status_messages
    assert await deliver(server, "my_pack.done", "tab", "other") == {
        "tab": [{"type": "my_pack.done", "data": {"node": "v4", "count": 2}}],
        "other": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("client_id", [None, ""])
async def test_send_without_a_client_sends_nothing(run_node, synchronous, client_id):
    executor, server = await run_node(
        '{call}.ui.send("my_pack.done", {{"count": 2}})',
        synchronous,
        client_id=client_id,
    )

    assert executor.success, executor.status_messages
    assert await deliver(server, "my_pack.done", "tab", "other") == {
        "tab": [],
        "other": [],
    }


@pytest.mark.asyncio
async def test_send_from_a_worker_thread_reaches_the_client(run_node):
    executor, server = await run_node(
        "loop = asyncio.get_running_loop()\n"
        "def v1_method():\n"
        '    send = eng.ui.send("RememberingUtils:change", {{"node": "v4", "next": 7}})\n'
        "    asyncio.run_coroutine_threadsafe(send, loop).result()\n"
        "await asyncio.to_thread(v1_method)"
    )

    assert executor.success, executor.status_messages
    assert ("RememberingUtils:change", {"node": "v4", "next": 7}, "tab") in server.sent


@pytest.mark.asyncio
async def test_send_copies_data_so_later_changes_are_not_sent(run_node):
    executor, server = await run_node(
        'data = {{"images": ["a.png"]}}\n'
        '{call}.ui.send("my_pack.saved", data)\n'
        'data["images"].append("b.png")'
    )

    assert executor.success, executor.status_messages
    assert ("my_pack.saved", {"images": ["a.png"]}, "tab") in server.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("client_id", ["tab", None])
@pytest.mark.parametrize(
    "value, exception_type",
    [("torch.zeros(1)", "TypeError"), ('float("nan")', "ValueError")],
)
async def test_send_of_data_that_is_not_json_fails_the_node_and_sends_nothing(
    run_node, synchronous, client_id, value, exception_type
):
    executor, server = await run_node(
        f'{{call}}.ui.send("my_pack.done", {{{{"value": {value}}}}})',
        synchronous,
        client_id=client_id,
    )

    assert not executor.success
    [error] = [
        data for event, data in executor.status_messages if event == "execution_error"
    ]
    assert error["exception_type"] == exception_type
    assert all(event != "my_pack.done" for event, _, _ in server.sent)


@pytest.mark.asyncio
async def test_send_outside_a_running_node_raises():
    with pytest.raises(RuntimeError):
        await v4.eng.ui.send("my_pack.done", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["disable_metadata", "multi_user"])
async def test_settings_get_reads_comfyui_value_at_each_call(monkeypatch, name):
    sync_settings = v4.ComfyAPISync().settings

    for value in (True, False):
        monkeypatch.setattr(args, name, value)
        assert await v4.eng.settings.get(name) is value
        assert sync_settings.get(name) is value


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_settings_get_from_a_worker_thread_reads_the_live_value(
    run_node, monkeypatch, value
):
    monkeypatch.setattr(args, "disable_metadata", value)

    executor, _ = await run_node(
        "loop = asyncio.get_running_loop()\n"
        "def v1_method():\n"
        '    get = eng.settings.get("disable_metadata")\n'
        "    return asyncio.run_coroutine_threadsafe(get, loop).result()\n"
        f"if await asyncio.to_thread(v1_method) is not {value}:\n"
        '    raise AssertionError("wrong value")'
    )

    assert executor.success, executor.status_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["port", "listen", "default_hashing_function"])
async def test_settings_get_of_an_undeclared_name_raises_value_error(name):
    with pytest.raises(ValueError):
        await v4.eng.settings.get(name)
    with pytest.raises(ValueError):
        v4.ComfyAPISync().settings.get(name)
