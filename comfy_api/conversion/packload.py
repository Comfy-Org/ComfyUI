"""A pack imported in this process, as pack isolation imports it: its nodes, routes and web directory."""
import asyncio
import importlib.util
import os
import sys
import types
from typing import Any, Optional


def module_name_for(entry: str) -> str:
    """The name ComfyUI gives a custom-node directory it imports (``nodes.load_custom_node``)."""
    return entry.replace(".", "_x_")


class _Inert:
    """A part of ComfyUI's server a pack reaches for while it loads (its queue, its user manager). There is
    none while a pack is described: every call does nothing and answers None, so the pack's nodes can be
    listed. What the pack does with it when a node runs is the converter's to report."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str) -> "_Inert":
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        return _Inert(f"{self._name}.{attribute}")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:
        return f"<{self._name}: not there while a pack is described>"


class _PromptServerStandIn:
    """``server.PromptServer.instance`` for pack code while it's described: collects routes, drops UI events."""

    def __init__(self) -> None:
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.supports: set = set()
        self.loop = None
        self.number = 0
        self.on_prompt_handlers: list = []
        self.prompt_queue = _Inert("PromptServer.instance.prompt_queue")
        self.node_replace_manager = _Inert("PromptServer.instance.node_replace_manager")
        self.user_manager = _Inert("PromptServer.instance.user_manager")
        self.internal_routes = _Inert("PromptServer.instance.internal_routes")

    def send_sync(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    async def send(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    def send_progress_text(self, text: str, node_id: Any, sid: Any = None) -> None:
        pass

    async def send_json(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    def queue_updated(self) -> None:
        pass

    def add_on_prompt_handler(self, handler: Any) -> None:
        self.on_prompt_handlers.append(handler)


def install_server_stand_in() -> _PromptServerStandIn:
    instance = _PromptServerStandIn()
    module = types.ModuleType("server", "Stand-in for ComfyUI's server module while a pack is described.")
    from protocol import BinaryEventTypes

    from aiohttp import web

    module.BinaryEventTypes = BinaryEventTypes
    module.web = web  # ComfyUI's server module has aiohttp's ``web`` (packs use ``server.web.json_response``)
    module.PromptServer = type("PromptServer", (), {"instance": instance})
    sys.modules["server"] = module
    return instance


class LoadedPack:
    """A pack imported in this process: its nodes, routes and web directory."""

    def __init__(self, entry: str) -> None:
        self.entry = os.path.realpath(entry)
        self.module_name = module_name_for(self.entry)
        self.prompt_server = install_server_stand_in()
        init = os.path.join(self.entry, "__init__.py")
        spec = importlib.util.spec_from_file_location(self.module_name, init, submodule_search_locations=[self.entry])
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        self.module = module
        self.registrations: dict = {}  # samplers, schedulers, model folders, node replacements
        self.nodes: dict[str, Any] = dict(getattr(module, "NODE_CLASS_MAPPINGS", None) or {})
        self.display_names: dict[str, str] = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {})
        if hasattr(module, "comfy_entrypoint"):
            asyncio.run(self._load_extension(module.comfy_entrypoint))
        web = getattr(module, "WEB_DIRECTORY", None)
        self.web_directory: Optional[str] = os.path.abspath(os.path.join(self.entry, web)) if web else None

    async def _load_extension(self, entrypoint: Any) -> None:
        extension = entrypoint()
        if asyncio.iscoroutine(extension):
            extension = await extension
        await extension.on_load()
        from comfy_api.latest import registration

        self.registrations = await registration.collect(extension)
        for cls in await extension.get_node_list():
            schema = cls.GET_SCHEMA()
            self.nodes[schema.node_id] = cls
            if schema.display_name:
                self.display_names[schema.node_id] = schema.display_name

    def routes(self) -> list[tuple[str, str]]:
        found = [(route.method, route.path) for route in self.prompt_server.routes
                 if hasattr(route, "method") and hasattr(route, "path")]
        for resource in self.prompt_server.app.router.resources():
            info = resource.get_info()
            path = info.get("path") or info.get("formatter")
            if "directory" in info:
                continue  # static files
            for route in resource:
                if path and route.method != "HEAD":
                    found.append((route.method, path))
        return sorted(set(found))

    def declared_routes(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``[[tool.comfy.routes]]`` in the pack's pyproject.toml, as ComfyUI serves them."""
        from comfy_api.latest import registration

        declared = registration.declarations(self.entry)
        return {(r["method"].lower(), r["path"]): (r["module"], r["handler"]) for r in declared["comfy"].get("routes") or []}
