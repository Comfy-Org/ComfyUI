import json

from comfy_api.internal.singleton import ProxiedSingleton

from ._execution import _running_node

# The message types ComfyUI's frontend acts on itself, before any listener a pack registered.
_FRONTEND_MESSAGES = frozenset(
    {
        "agent_status",
        "autoQueueGraphChanged",
        "b_preview",
        "executed",
        "executing",
        "execution_cached",
        "execution_error",
        "execution_interrupted",
        "execution_start",
        "execution_success",
        "feature_flags",
        "graphChanged",
        "logs",
        "notification",
        "progress",
        "progress_state",
        "prompt_outputs_skipped",
        "promptQueued",
        "status",
    }
)


class Ui(ProxiedSingleton):
    async def progress_text(self, text: str) -> None:
        """Show `text` on the running node to the client that queued the prompt.

        Nothing is sent if the prompt was queued without a client id. Raises `RuntimeError` unless
        made while the node runs, from its `execute` or a function it calls.
        """
        from server import PromptServer

        node_id = _running_node().node_id
        server = PromptServer.instance
        client_id = server.client_id
        if client_id:
            server.send_progress_text(text, node_id, client_id)

    async def send(self, event: str, data: dict) -> None:
        """Send `{"type": event, "data": data}` to the client that queued the prompt.

        Name the event after the pack, as `"<pack>.<event>"`. The pack's frontend code must have
        subscribed to it before it is sent: the frontend drops a message no listener has asked
        for. Raises `ValueError` for a message type the frontend acts on itself, such as
        `"progress"` or `"executed"`.

        Nothing is sent if the prompt was queued without a client id. `data` is copied as JSON now,
        so later changes to it are not sent; if it has a `"prompt_id"` key, ComfyUI adds the
        workflow's metadata, such as `workflow_id`, to what it sends. Raises `TypeError` if it is
        not JSON-serialisable, `ValueError` for NaN, infinity or a circular reference, and
        `RuntimeError` unless made while the node runs, from its `execute` or a function it
        calls.
        """
        from server import PromptServer

        if event in _FRONTEND_MESSAGES:
            raise ValueError(
                f"{event!r} is a message type the frontend acts on itself; "
                "name a pack's event '<pack>.<event>'"
            )
        _running_node()
        message = json.loads(json.dumps(data, allow_nan=False))
        server = PromptServer.instance
        client_id = server.client_id
        if client_id:
            server.send_sync(event, message, client_id)
