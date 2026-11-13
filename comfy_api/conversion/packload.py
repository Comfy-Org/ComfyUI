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
        for value in (*args, *kwargs.values()):  # a coroutine handed to the server's loop: never run
            if asyncio.iscoroutine(value):
                value.close()
        return None

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:
        return f"<{self._name}: not there while a pack is described>"


class _PromptServerStandIn:
    """``server.PromptServer.instance`` for pack code while it's described: collects routes, drops UI events.
    It has what ComfyUI's server has and nothing else (packs test ``hasattr(server, ...)`` before adding their own)."""

    def __init__(self) -> None:
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.supports: set = set()
        self.loop = _Inert("PromptServer.instance.loop")  # no loop runs: what's scheduled on it never happens
        self.number = 0
        self.on_prompt_handlers: list = []
        self.sockets: dict = {}
        self.sockets_metadata: dict = {}
        self.address, self.port = "127.0.0.1", 8188
        self.web_root = _web_root()
        for name in ("prompt_queue", "node_replace_manager", "user_manager", "internal_routes", "messages",
                     "client_session", "model_file_manager", "custom_node_manager", "subgraph_manager",
                     "asset_manager", "asset_export_manager", "deploy_to_cloud_routes", "workflow_metadata"):
            setattr(self, name, _Inert(f"PromptServer.instance.{name}"))

    def send_sync(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    async def send(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    def send_progress_text(self, text: str, node_id: Any, sid: Any = None) -> None:
        pass

    async def send_json(self, event: str, data: Any, sid: Any = None) -> None:
        pass

    async def send_image(self, image_data: Any, sid: Any = None) -> None:
        pass

    async def send_image_with_metadata(self, image_data: Any, sid: Any = None) -> None:
        pass

    async def send_bytes(self, event: Any, data: Any, sid: Any = None) -> None:
        pass

    def encode_bytes(self, event: int, data: bytes) -> bytes:
        import struct

        return struct.pack(">I", event) + bytes(data)

    def queue_updated(self) -> None:
        pass

    def get_queue_info(self) -> dict:
        return {"exec_info": {"queue_remaining": 0}}

    def add_on_prompt_handler(self, handler: Any) -> None:
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data: Any) -> Any:
        return json_data

    def add_routes(self) -> None:
        pass


_SERVER_METHODS = ("send_sync", "send", "send_json", "send_progress_text", "send_image", "send_image_with_metadata",
                   "send_bytes", "encode_bytes", "queue_updated", "get_queue_info", "add_on_prompt_handler",
                   "trigger_on_prompt", "add_routes")


def _web_root() -> str:
    """Where ComfyUI's frontend files are (packs read it to find or add web files)."""
    try:
        import importlib.resources

        import comfyui_frontend_package
        return str(importlib.resources.files(comfyui_frontend_package) / "static")
    except Exception:
        return os.path.join(os.environ.get("COMFY_ROOT") or os.getcwd(), "web")


def install_server_stand_in() -> _PromptServerStandIn:
    instance = _PromptServerStandIn()
    module = types.ModuleType("server", "Stand-in for ComfyUI's server module while a pack is described.")
    from protocol import BinaryEventTypes

    from aiohttp import web

    module.BinaryEventTypes = BinaryEventTypes
    module.web = web  # ComfyUI's server module has aiohttp's ``web`` (packs use ``server.web.json_response``)
    # packs also reach the server's methods through the class (``PromptServer.send_sync`` patched or called)
    module.PromptServer = type("PromptServer", (), {
        "instance": instance, **{name: getattr(_PromptServerStandIn, name) for name in _SERVER_METHODS}})
    sys.modules["server"] = module
    return instance


class LoadedPack:
    """A pack imported in this process: its nodes, routes and web directory."""

    def __init__(self, entry: str, plan: Any = None) -> None:
        import standin

        self.entry = os.path.realpath(entry)
        self.module_name = module_name_for(self.entry)
        self.prompt_server = install_server_stand_in()
        init = os.path.join(self.entry, "__init__.py")
        # What the plan says this pack needs and isn't here is stood in for while it's imported, and armed
        # after: using a stand-in then raises (standin.py). No plan: the pack loads as it is.
        self.stand_ins = standin.Session(plan, self.entry, self.module_name)
        with self.stand_ins:
            spec = importlib.util.spec_from_file_location(self.module_name, init, submodule_search_locations=[self.entry])
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = module
            self.stand_ins.prepare(module)
            spec.loader.exec_module(module)
            self.module = module
            self.registrations: dict = {}  # samplers, schedulers, model folders, node replacements
            self.unlisted: dict[str, str] = {}  # V3 node classes whose schema raised: class name -> the error
            self.nodes: dict[str, Any] = dict(getattr(module, "NODE_CLASS_MAPPINGS", None) or {})
            self.display_names: dict[str, str] = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {})
            if hasattr(module, "comfy_entrypoint"):
                asyncio.run(self._load_extension(module.comfy_entrypoint))
        web = getattr(module, "WEB_DIRECTORY", None)
        web = web if isinstance(web, (str, os.PathLike)) else None  # not what a stand-in gave it
        self.web_directory: Optional[str] = os.path.abspath(os.path.join(self.entry, web)) if web else None

    async def _load_extension(self, entrypoint: Any) -> None:
        extension = entrypoint()
        if asyncio.iscoroutine(extension):
            extension = await extension
        if not hasattr(extension, "get_node_list"):
            return  # not an extension (a pack that returns its V1 tables): ComfyUI skips it too
        await extension.on_load()
        from comfy_api.latest import registration

        self.registrations = await registration.collect(extension)
        for cls in await extension.get_node_list():
            try:
                schema = cls.GET_SCHEMA()
            except Exception as error:  # one node that can't say what it is (no GPU to list, say): the rest still load
                self.unlisted[getattr(cls, "__name__", str(cls))] = f"{type(error).__name__}: {error}"[:500]
                continue
            self.nodes[schema.node_id] = cls
            if schema.display_name:
                self.display_names[schema.node_id] = schema.display_name

    def routes(self) -> list[tuple[str, str]]:
        found = [(route.method, route.path) for route in self.prompt_server.routes
                 if isinstance(getattr(route, "method", None), str) and isinstance(getattr(route, "path", None), str)]
        for resource in self.prompt_server.app.router.resources():
            info = resource.get_info()
            path = info.get("path") or info.get("formatter")
            if "directory" in info:
                continue  # static files
            for route in resource:
                if path and isinstance(path, str) and route.method != "HEAD":
                    found.append((route.method, path))
        return sorted(set(found))

    def declared_routes(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``[[tool.comfy.routes]]`` in the pack's pyproject.toml, as ComfyUI serves them."""
        from comfy_api.latest import registration

        declared = registration.declarations(self.entry)
        return {(r["method"].lower(), r["path"]): (r["module"], r["handler"]) for r in declared["comfy"].get("routes") or []}
