import asyncio
import glob
import hmac
import importlib.util
import json
import os
import uuid
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from gateway.remote import GatewayError, RemoteTarget, normalize_url
from gateway.transfers import FileAccessError, UploadCache, file_inputs, remote_file_exists, transfer_input

SESSION_COOKIE = "comfy_remote_gateway"
# Routes the editor sends to the remote in remote mode: execution, node definitions, models, files,
# system information and logs. Workflows, settings, extensions and the rest stay local.
PROXY_ROUTES = (
    "/prompt", "/queue", "/history", "/interrupt", "/jobs", "/free",
    "/view", "/view_metadata", "/upload",
    "/object_info", "/models", "/experiment/models", "/embeddings", "/system_stats",
    "/internal/logs", "/internal/folder_paths", "/internal/files",
    # Model downloads by remote_companion, installed on the remote.
    "/remote_execution/models",
    # ComfyUI-Manager on the remote: installing node packs, its task queue and restart.
    "/v2/manager", "/v2/customnode", "/v2/externalmodel", "/v2/snapshot",
)
# Local credentials the stock frontend adds to extra_data; they must not leave this machine.
LOCAL_SECRETS = ("auth_token_comfy_org", "api_key_comfy_org")


# Download sources the editor and remote_companion.py accept.
MODEL_LINK_PREFIXES = ("https://huggingface.co/", "https://civitai.com/", "https://civitai.red/")
# Which saved access key a download host gets.
KEY_FOR_HOST = {"huggingface.co": "huggingface", "civitai.com": "civitai", "civitai.red": "civitai"}


def build_model_links():
    # Download links that ComfyUI's bundled workflow templates declare for model files, by file name.
    links = {}
    spec = importlib.util.find_spec("comfyui_workflow_templates_json")
    for root in (spec.submodule_search_locations if spec else []):
        for path in sorted(glob.glob(os.path.join(root, "templates", "*.json"))):
            with open(path, encoding="utf-8") as f:
                stack = [json.load(f)]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    name, url, directory = item.get("name"), item.get("url"), item.get("directory")
                    if isinstance(name, str) and isinstance(url, str) and isinstance(directory, str) and url.startswith(MODEL_LINK_PREFIXES):
                        links.setdefault(name, {"name": name, "url": url, "directory": directory})
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    return list(links.values())


def cors_headers(resp, origin):
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Vary"] = "Origin"


class Hub:
    """Browser sockets that receive connection status updates."""

    def __init__(self):
        # One ordered queue per socket so a newer status never arrives before an older one.
        self.queues = {}

    def send_json(self, message):
        data = json.dumps(message)
        for queue in self.queues.values():
            queue.put_nowait(data)

    async def serve(self, ws, first_message):
        queue = asyncio.Queue()
        queue.put_nowait(json.dumps(first_message))
        self.queues[ws] = queue
        writer = asyncio.create_task(self._write(ws, queue))
        try:
            async for _ in ws:
                pass
        finally:
            del self.queues[ws]
            writer.cancel()

    async def _write(self, ws, queue):
        while True:
            data = await queue.get()
            try:
                await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                return


class Gateway:
    def __init__(self, local_url, token, allowed_origins, data_dir):
        self.local_url = local_url.rstrip("/")
        self.token = token
        self.allowed_origins = set(allowed_origins)
        self.tmp_dir = os.path.join(data_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        # The connected remote and whether remote mode uses it, restored after a gateway restart.
        self.state_path = os.path.join(data_dir, "state.json")
        self.uploads = UploadCache(os.path.join(data_dir, "uploads.json"))
        # Hugging Face and Civitai keys for model downloads; never returned to the browser.
        self.keys_path = os.path.join(data_dir, "keys.json")
        self.keys = {}
        if os.path.exists(self.keys_path):
            with open(self.keys_path, encoding="utf-8") as f:
                self.keys = json.load(f)
        self.hub = Hub()
        self.targets = {}
        self.connected_url = None
        self.proxy_target_id = None
        self.model_links = None
        self.session = None

    def target_for(self, url):
        url = normalize_url(url)
        for target in self.targets.values():
            if target.url == url:
                return target
        target = RemoteTarget(url, self.session, self.hub)
        self.targets[target.id] = target
        return target

    def save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"url": self.connected_url, "proxy": self.proxy_target_id is not None}, f)

    def proxy_target(self):
        target = self.targets.get(self.proxy_target_id)
        if target is None:
            raise GatewayError("Remote mode has no connected server; nothing was sent.", status=409)
        if target.status != "ready":
            raise GatewayError(f"Remote {target.url} is {target.status}; nothing was sent.", status=503)
        return target

    def snapshot(self):
        return {"type": "state", "targets": [t.summary() for t in self.targets.values()], "proxy_target_id": self.proxy_target_id}

    @web.middleware
    async def guard(self, request, handler):
        origin = request.headers.get("Origin")
        if origin is not None and origin not in self.allowed_origins:
            return web.json_response({"error": "Origin not allowed."}, status=403)
        if request.method == "OPTIONS":
            resp = web.Response(status=204)
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE"
            resp.headers["Access-Control-Allow-Headers"] = request.headers.get("Access-Control-Request-Headers", "Content-Type, X-Gateway-Token")
            resp.headers["Access-Control-Max-Age"] = "600"
            if request.headers.get("Access-Control-Request-Private-Network") == "true":
                resp.headers["Access-Control-Allow-Private-Network"] = "true"
        else:
            # The cookie covers requests the native frontend makes without our header: its socket and <img> loads.
            # Its socket URL may also carry an unrelated comfy.org "token", so any matching credential is accepted.
            presented = (request.headers.get("X-Gateway-Token"), request.query.get("token"), request.cookies.get(SESSION_COOKIE))
            if not any(hmac.compare_digest(t.encode(), self.token.encode()) for t in presented if t):
                resp = web.json_response({"error": "Missing or invalid gateway token."}, status=401)
            else:
                try:
                    resp = await handler(request)
                except GatewayError as e:
                    resp = web.json_response({"error": str(e)}, status=e.status)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    resp = web.json_response({"error": f"Remote request failed: {e}"}, status=502)
        if origin is not None and not resp.prepared:
            cors_headers(resp, origin)
        return resp

    async def session_cookie(self, request):
        resp = web.json_response({"ok": True})
        resp.set_cookie(SESSION_COOKIE, self.token, httponly=True, samesite="Strict", path="/")
        return resp

    async def state(self, request):
        return web.json_response(self.snapshot())

    async def connect(self, request):
        target = self.target_for((await request.json()).get("url", ""))
        target.connect()
        self.connected_url = target.url
        self.save_state()
        return web.json_response(target.summary())

    async def disconnect(self, request):
        target = self.targets.get(request.match_info["target_id"])
        if target is None:
            raise GatewayError("Unknown remote target.", status=404)
        await target.disconnect()
        if self.connected_url == target.url:
            self.connected_url = None
            self.save_state()
        return web.json_response(target.summary())

    async def set_proxy(self, request):
        target_id = (await request.json()).get("target_id")
        if target_id is not None and target_id not in self.targets:
            raise GatewayError("Unknown remote target; connect first.", status=404)
        self.proxy_target_id = target_id
        self.save_state()
        return web.json_response({"proxy_target_id": target_id})

    async def key_status(self, request):
        return web.json_response({name: name in self.keys for name in ("huggingface", "civitai")})

    async def set_keys(self, request):
        for name, value in (await request.json()).items():
            if name not in ("huggingface", "civitai"):
                raise GatewayError(f"Unknown key {name!r}.")
            if value:
                self.keys[name] = value.strip()
            else:
                self.keys.pop(name, None)
        tmp = self.keys_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.keys, f)
        os.replace(tmp, self.keys_path)
        return await self.key_status(request)

    async def links(self, request):
        if self.model_links is None:
            self.model_links = await asyncio.get_running_loop().run_in_executor(None, build_model_links)
        return web.json_response(self.model_links)

    async def events(self, request):
        if request.headers.get("Origin") not in self.allowed_origins:
            return web.json_response({"error": "Origin not allowed."}, status=403)
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        await self.hub.serve(ws, self.snapshot())
        return ws

    async def prepare_prompt(self, target, body):
        data = json.loads(body)
        for key in LOCAL_SECRETS:
            data.get("extra_data", {}).pop(key, None)
        # A caller-supplied id lets the queue/history show whether a prompt with a lost response was accepted.
        data.setdefault("prompt_id", str(uuid.uuid4()))
        async with self.session.get(f"{self.local_url}/object_info") as resp:
            local_defs = await resp.json()
        for node in data.get("prompt", {}).values():
            names = file_inputs(local_defs.get(node.get("class_type"), {}))
            for name, value in node.get("inputs", {}).items():
                # File names refer to remote files in remote mode; a local copy is sent only when the remote lacks it.
                if name in names and isinstance(value, str) and not await remote_file_exists(self.session, target, value):
                    try:
                        node["inputs"][name] = await transfer_input(self.session, self.local_url, target, self.uploads, self.tmp_dir, value)
                    except FileNotFoundError:
                        pass  # on neither side; remote validation reports it
        return data

    async def proxy_http(self, request):
        path = "/" + request.match_info["tail"]
        route = path[len("/api"):] if path.startswith("/api/") else path
        if not any(route == r or route.startswith(r + "/") for r in PROXY_ROUTES):
            raise GatewayError("Not a remote route.", status=404)
        target = self.proxy_target()
        params = [(k, v) for k, v in request.query.items() if k != "token"]
        if request.method == "POST" and route == "/prompt":
            return await self.proxy_prompt(request, target, path, params)
        kwargs = {}
        if request.method == "POST" and route == "/remote_execution/models/download":
            # The matching saved key goes only with downloads from its own host.
            body = await request.json()
            key = self.keys.get(KEY_FOR_HOST.get(urlsplit(body.get("url", "")).hostname))
            kwargs["json"] = {**body, "token": key} if key else body
        elif request.can_read_body:
            kwargs["data"] = await request.read()
            kwargs["headers"] = {"Content-Type": request.headers.get("Content-Type", "application/json")}
        async with self.session.request(request.method, target.url + path, params=params, timeout=aiohttp.ClientTimeout(total=120), **kwargs) as remote:
            resp = web.StreamResponse(status=remote.status, headers={k: v for k, v in remote.headers.items() if k in ("Content-Type", "Cache-Control", "Content-Disposition")})
            if request.headers.get("Origin"):
                cors_headers(resp, request.headers["Origin"])
            await resp.prepare(request)
            async for chunk in remote.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def proxy_prompt(self, request, target, path, params):
        try:
            data = await self.prepare_prompt(target, await request.read())
        except FileAccessError as e:
            return web.json_response({"error": {"type": "invalid_input_file", "message": str(e), "details": "", "extra_info": {}}, "node_errors": {}}, status=400)
        try:
            async with self.session.post(target.url + path, params=params, json=data, timeout=aiohttp.ClientTimeout(total=120)) as remote:
                body = await remote.read()
                # Only ComfyUI's own answers are passed on; a tunnel error page can follow an accepted prompt.
                if remote.status in (200, 400) and remote.content_type == "application/json":
                    return web.Response(status=remote.status, body=body, content_type="application/json")
                problem = f"HTTP {remote.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            problem = str(e) or type(e).__name__
        # Acceptance is unknown and the prompt is never resent; the user checks the queue first.
        message = f"No usable response from the remote ({problem}). Prompt {data['prompt_id']} may still have been queued; check the queue before running again."
        return web.json_response({"error": {"type": "remote_unreachable", "message": message, "details": "", "extra_info": {}}, "node_errors": {}}, status=502)

    async def proxy_ws(self, request):
        if request.headers.get("Origin") not in self.allowed_origins:
            return web.json_response({"error": "Origin not allowed."}, status=403)
        target = self.proxy_target()
        # Only the client id is forwarded; the frontend may append a local comfy.org token.
        params = {"clientId": request.query["clientId"]} if request.query.get("clientId") else {}
        async with self.session.ws_connect(f"{target.ws_base()}/ws", params=params, heartbeat=30, max_msg_size=0) as remote:
            ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
            await ws.prepare(request)

            async def pump(source, sink):
                async for msg in source:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await sink.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await sink.send_bytes(msg.data)

            tasks = [asyncio.create_task(pump(ws, remote)), asyncio.create_task(pump(remote, ws))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                task.cancel()
            await ws.close()
            return ws

    async def on_startup(self, app):
        self.session = aiohttp.ClientSession()
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved["url"]:
                target = self.target_for(saved["url"])
                target.connect()
                self.connected_url = target.url
                if saved["proxy"]:
                    self.proxy_target_id = target.id

    async def on_cleanup(self, app):
        for target in self.targets.values():
            if target.task is not None:
                target.task.cancel()
        for ws in list(self.hub.queues):
            await ws.close()
        await self.session.close()

    def make_app(self):
        app = web.Application(middlewares=[self.guard], client_max_size=64 * 1024 * 1024)
        app.router.add_get("/state", self.state)
        app.router.add_post("/targets/connect", self.connect)
        app.router.add_post("/targets/{target_id}/disconnect", self.disconnect)
        app.router.add_post("/session", self.session_cookie)
        app.router.add_get("/events", self.events)
        app.router.add_post("/proxy", self.set_proxy)
        app.router.add_get("/model_links", self.links)
        app.router.add_get("/keys", self.key_status)
        app.router.add_post("/keys", self.set_keys)
        # The native frontend opens {api_host}/ws, so the remote socket proxy lives at /ws.
        app.router.add_get("/ws", self.proxy_ws)
        app.router.add_route("*", "/remote/{tail:.*}", self.proxy_http)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app
