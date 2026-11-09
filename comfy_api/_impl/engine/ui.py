"""Core engine extension: what the running node shows in the UI."""
from comfy_api._impl import _runtime


class EngineOps:
    def progress_text(self, text):
        import server

        instance = getattr(server.PromptServer, "instance", None)
        plan = _runtime.current_runtime().plan
        node_id = getattr(plan, "node_id", None)
        if instance is not None and node_id is not None:
            instance.send_progress_text(str(text), node_id, getattr(instance, "client_id", None))
        return None

    def send(self, event, data):
        """A custom event to the client that queued the running prompt (``PromptServer.send_sync``)."""
        import server

        if not isinstance(event, str) or not event or len(event) > 128:
            raise ValueError("event is a short name")
        instance = getattr(server.PromptServer, "instance", None)
        if instance is not None:
            instance.send_sync(event, data, getattr(instance, "client_id", None))
        return None
