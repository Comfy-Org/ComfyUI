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

    async def interact(self, kind, payload, reuse_last=False, remember=False, timeout=540.0):
        """Ask the user something and wait: the secure overlay's interaction broker, unchanged."""
        import server
        from comfy_api._impl import _runtime
        from comfy_api._impl.interactions import BROKER

        if not isinstance(reuse_last, bool) or not isinstance(remember, bool):
            raise TypeError("interaction cache switches must be booleans")
        rt = _runtime.current_runtime()
        instance = getattr(server.PromptServer, "instance", None)
        return await BROKER.request(
            kind=kind, payload=payload, tenant="", pack=(rt.pack or "pack").split(".")[0] or "pack",
            node_id=str(getattr(rt.plan, "node_id", "") or ""), sid=getattr(instance, "client_id", None),
            timeout=float(timeout), reuse_last=reuse_last, remember=remember)
