import asyncio
import json
import os
import socket
import struct
import sys
import tempfile
import unittest
import uuid

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.app import Gateway  # noqa: E402

TOKEN = "test-token"
ORIGIN = "http://127.0.0.1:8188"
HEADERS = {"X-Gateway-Token": TOKEN, "Origin": ORIGIN}

DEFS = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["model.safetensors"]]}}, "output": ["MODEL", "CLIP", "VAE"]},
    "LoadImage": {"input": {"required": {"image": [["a.png"], {"image_upload": True}]}}, "output": ["IMAGE", "MASK"]},
    "LoadImageMask": {"input": {"required": {"image": [["a.png"], {"image_upload": True}], "channel": [["alpha", "red"]]}}, "output": ["MASK"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}, "output": []},
}

PNG = b"\x89PNG\r\n\x1a\nfake"


def t2i(prefix="out"):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
    }


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeRemote:
    """Just enough of the ComfyUI server API, driven step by step by the tests."""

    def __init__(self):
        self.pending = []
        self.prompts = {}
        self.files = {("input", "", "a.png"): b"remote file with the same name"}
        self.sockets = {}
        self.prompt_posts = []
        self.uploads = 0
        self.lose_response = False
        self.download_requests = []

    def app(self):
        routes = [
            ("GET", "/system_stats", lambda r: web.json_response({"system": {"comfyui_version": "0.42.0", "os": "linux"}, "devices": [{"name": "cuda:0 Fake GPU : cudaMallocAsync", "vram_total": 24 * 1024 ** 3}]})),
            ("GET", "/object_info", lambda r: web.json_response(DEFS)),
            ("POST", "/prompt", self.post_prompt),
            ("GET", "/queue", lambda r: web.json_response({"queue_running": [], "queue_pending": [[0, p, {}, {}, []] for p in self.pending]})),
            ("POST", "/upload/image", self.upload),
            ("GET", "/view", self.view),
            ("POST", "/remote_execution/models/download", self.download),
        ]
        app = web.Application()
        # Like ComfyUI, every route is also served under /api.
        for method, path, handler in routes:
            add = app.router.add_get if method == "GET" else app.router.add_post
            add(path, handler)
            add("/api" + path, handler)
        app.router.add_get("/internal/logs/raw", lambda r: web.json_response({"entries": [{"m": "remote log line"}]}))
        app.router.add_get("/ws", self.ws)
        return app

    async def post_prompt(self, request):
        body = await request.json()
        self.prompt_posts.append(body)
        if any(v["class_type"] not in DEFS for v in body["prompt"].values()):
            return web.json_response({"error": {"type": "invalid_prompt", "message": "Cannot execute"}, "node_errors": {}}, status=400)
        self.prompts[body["prompt_id"]] = body
        self.pending.append(body["prompt_id"])
        if self.lose_response:
            # A tunnel error page after the prompt was accepted.
            return web.Response(status=502, text="<html>Bad gateway</html>", content_type="text/html")
        return web.json_response({"prompt_id": body["prompt_id"], "number": len(self.prompts), "node_errors": {}})

    async def download(self, request):
        self.download_requests.append(await request.json())
        return web.json_response({"id": "d1", "status": "downloading"})

    async def upload(self, request):
        form = await request.post()
        field = form["image"]
        self.files[("input", form.get("subfolder", ""), field.filename)] = field.file.read()
        self.uploads += 1
        return web.json_response({"name": field.filename, "subfolder": form.get("subfolder", ""), "type": "input"})

    async def view(self, request):
        key = (request.query.get("type"), request.query.get("subfolder", ""), request.query.get("filename"))
        if key not in self.files:
            return web.Response(status=404)
        return web.Response(body=self.files[key], content_type="image/png")

    async def ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get("clientId") or uuid.uuid4().hex
        self.sockets[sid] = ws
        await ws.send_json({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": len(self.pending)}}, "sid": sid}})
        async for msg in ws:
            if json.loads(msg.data).get("type") == "feature_flags":
                await ws.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": True}})
        return ws

    async def run(self, prompt_id):
        ws = self.sockets[self.prompts[prompt_id]["client_id"]]
        self.pending.remove(prompt_id)
        await ws.send_json({"type": "execution_start", "data": {"prompt_id": prompt_id}})
        await ws.send_json({"type": "progress", "data": {"value": 1, "max": 2, "node": "9", "prompt_id": prompt_id}})
        meta = json.dumps({"prompt_id": prompt_id, "node_id": "9", "display_node_id": "9", "image_type": "image/jpeg"}).encode()
        await ws.send_bytes(struct.pack(">I", 4) + struct.pack(">I", len(meta)) + meta + b"jpegdata")
        image = {"filename": "out_00001_.png", "subfolder": "", "type": "output"}
        self.files[("output", "", image["filename"])] = PNG + prompt_id.encode()
        await ws.send_json({"type": "executed", "data": {"node": "9", "output": {"images": [image]}, "prompt_id": prompt_id}})


class FakeLocal:
    def __init__(self):
        self.files = {("input", "", "a.png"): PNG + b"local A", ("input", "", "only_local.png"): PNG + b"only local", ("input", "clipspace", "mask.png"): PNG + b"local mask"}
        self.views = 0

    def app(self):
        app = web.Application()
        app.router.add_get("/object_info", lambda r: web.json_response(DEFS))
        app.router.add_get("/view", self.view)
        return app

    async def view(self, request):
        self.views += 1
        key = (request.query.get("type"), request.query.get("subfolder", ""), request.query.get("filename"))
        if key not in self.files:
            return web.Response(status=404)
        return web.Response(body=self.files[key])


class GatewayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.remote = FakeRemote()
        self.local = FakeLocal()
        self.remote_port = free_port()
        self.remote_url = f"http://127.0.0.1:{self.remote_port}"
        self.remote_server = TestServer(self.remote.app(), port=self.remote_port)
        self.local_server = TestServer(self.local.app())
        await self.remote_server.start_server()
        await self.local_server.start_server()
        self.data_dir = tempfile.mkdtemp()
        self.client = await self.start_gateway()

    async def start_gateway(self):
        self.gateway = Gateway(str(self.local_server.make_url("")), TOKEN, {ORIGIN}, self.data_dir)
        client = TestClient(TestServer(self.gateway.make_app()))
        await client.start_server()
        return client

    async def asyncTearDown(self):
        await self.client.close()
        await self.remote_server.close()
        await self.local_server.close()

    async def post(self, path, body):
        resp = await self.client.post(path, json=body, headers=HEADERS)
        return resp.status, await resp.json()

    async def until(self, cond, timeout=10):
        for _ in range(int(timeout / 0.02)):
            if cond():
                return
            await asyncio.sleep(0.02)
        self.fail("condition not reached")

    def target(self):
        return next(iter(self.gateway.targets.values()))

    async def connect(self):
        status, target = await self.post("/targets/connect", {"url": self.remote_url + "/#fragment"})
        self.assertEqual(status, 200, target)
        await self.until(lambda: self.gateway.targets[target["id"]].status == "ready")
        return target["id"]

    async def remote_mode(self):
        target_id = await self.connect()
        status, body = await self.post("/proxy", {"target_id": target_id})
        self.assertEqual(status, 200, body)
        return target_id

    async def test_requires_token_and_allowed_origin(self):
        resp = await self.client.get("/state", headers={"Origin": ORIGIN})
        self.assertEqual(resp.status, 401)
        resp = await self.client.get("/state", headers={"X-Gateway-Token": TOKEN, "Origin": "https://evil.example"})
        self.assertEqual(resp.status, 403)
        resp = await self.client.get("/state", headers=HEADERS)
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers["Access-Control-Allow-Origin"], ORIGIN)
        upgrade = {"Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}
        for path in ("/events", "/ws"):
            resp = await self.client.get(f"{path}?token={TOKEN}", headers={"Origin": "https://evil.example", **upgrade})
            self.assertEqual(resp.status, 403, path)

    async def test_connect_reports_server_and_queue(self):
        await self.connect()
        summary = self.target().summary()
        self.assertEqual((summary["url"], summary["comfyui_version"], summary["queue_remaining"]), (self.remote_url, "0.42.0", 0))
        self.assertEqual(summary["devices"][0]["vram_total"], 24 * 1024 ** 3)
        status, url = await self.post("/targets/connect", {"url": "file:///etc/passwd"})
        self.assertEqual(status, 400)

    async def test_retries_until_the_remote_appears(self):
        await self.remote_server.close()
        status, _ = await self.post("/targets/connect", {"url": self.remote_url})
        await self.until(lambda: self.target().status == "disconnected" and "retrying" in (self.target().detail or ""))
        self.remote_server = TestServer(self.remote.app(), port=self.remote_port)
        await self.remote_server.start_server()
        await self.until(lambda: self.target().status == "ready")

    async def test_proxy_requires_remote_mode_and_a_remote_route(self):
        resp = await self.client.get("/remote/api/queue", headers=HEADERS)
        self.assertEqual(resp.status, 409)
        await self.remote_mode()
        for path in ("/remote/api/userdata/workflows/x.json", "/remote/api/settings", "/remote/api/viewx", "/remote/internal/other"):
            resp = await self.client.get(path, headers=HEADERS)
            self.assertEqual(resp.status, 404, path)
        resp = await self.client.get("/remote/api/queue", headers=HEADERS)
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers["Access-Control-Allow-Credentials"], "true")
        resp = await self.client.get("/remote/api/object_info", headers=HEADERS)
        self.assertEqual(await resp.json(), DEFS)
        resp = await self.client.get("/remote/api/system_stats", headers=HEADERS)
        self.assertEqual((await resp.json())["system"]["os"], "linux")
        resp = await self.client.get("/remote/internal/logs/raw", headers=HEADERS)
        self.assertEqual((await resp.json())["entries"][0]["m"], "remote log line")
        form = aiohttp.FormData()
        form.add_field("image", b"uploaded in remote mode", filename="new.png", content_type="image/png")
        resp = await self.client.post("/remote/api/upload/image", data=form, headers=HEADERS)
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.remote.files[("input", "", "new.png")], b"uploaded in remote mode")

    async def test_remote_mode_refuses_instead_of_running_locally_while_remote_is_down(self):
        await self.remote_mode()
        await self.remote_server.close()
        await self.until(lambda: self.target().status == "disconnected")
        status, body = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertEqual(status, 503)
        self.assertIn("nothing was sent", body["error"])
        self.assertEqual(self.local.views, 0)
        self.remote_server = TestServer(self.remote.app(), port=self.remote_port)
        await self.remote_server.start_server()
        await self.until(lambda: self.target().status == "ready")
        status, body = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertEqual(status, 200, body)

    async def test_prompt_strips_local_secrets_and_uploads_only_missing_inputs(self):
        await self.remote_mode()
        prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "2": {"class_type": "LoadImageMask", "inputs": {"image": "clipspace/mask.png [input]", "channel": "alpha"}},
            "3": {"class_type": "LoadImage", "inputs": {"image": "nowhere.png"}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "p"}},
        }
        body = {"client_id": "abc", "prompt": prompt, "extra_data": {"auth_token_comfy_org": "secret", "api_key_comfy_org": "key", "extra_pnginfo": {"workflow": {}}, "preview_method": "auto"}}
        status, result = await self.post("/remote/api/prompt", body)
        self.assertEqual(status, 200, result)
        sent = self.remote.prompt_posts[0]
        self.assertEqual(sent["extra_data"], {"extra_pnginfo": {"workflow": {}}, "preview_method": "auto"})
        self.assertEqual((sent["client_id"], result["prompt_id"]), ("abc", sent["prompt_id"]))
        # The remote's own a.png is what the remote-mode editor shows, so it is used as is.
        self.assertEqual(sent["prompt"]["1"]["inputs"]["image"], "a.png")
        sub, _, name = sent["prompt"]["2"]["inputs"]["image"].removesuffix(" [input]").rpartition("/")
        self.assertEqual((sub, self.remote.files[("input", sub, name)]), ("remote_execution", PNG + b"local mask"))
        self.assertEqual(sent["prompt"]["3"]["inputs"]["image"], "nowhere.png")

    async def test_local_only_input_is_uploaded_once_and_again_after_a_remote_reset(self):
        await self.remote_mode()
        prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "only_local.png"}}, "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "p"}}}
        for _ in range(2):
            await self.post("/remote/api/prompt", {"client_id": "x", "prompt": prompt})
        self.assertEqual(self.remote.uploads, 1)
        self.assertEqual(self.remote.prompt_posts[0]["prompt"]["1"]["inputs"]["image"], self.remote.prompt_posts[1]["prompt"]["1"]["inputs"]["image"])
        self.remote.files = {k: v for k, v in self.remote.files.items() if k[1] != "remote_execution"}
        await self.post("/remote/api/prompt", {"client_id": "x", "prompt": prompt})
        self.assertEqual(self.remote.uploads, 2)

    async def test_invalid_file_paths_are_rejected(self):
        await self.remote_mode()
        for value in ("../secret.png", "sub/../../x.png", "C:\\x.png", "/etc/passwd"):
            prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": value}}}
            status, body = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": prompt})
            self.assertEqual(status, 400, value)
            self.assertIn("Rejected file reference", body["error"]["message"])
        self.assertEqual((self.local.views, self.remote.prompt_posts), (0, []))

    async def test_lost_prompt_response_is_reported_not_retried(self):
        await self.remote_mode()
        self.remote.lose_response = True
        status, body = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertEqual(status, 502)
        prompt_id = self.remote.prompt_posts[0]["prompt_id"]
        self.assertIn(f"Prompt {prompt_id} may still have been queued", body["error"]["message"])
        self.assertEqual((len(self.remote.prompt_posts), self.remote.pending), (1, [prompt_id]))
        # ComfyUI's own validation errors pass through unchanged.
        self.remote.lose_response = False
        status, body = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": {"1": {"class_type": "Unknown", "inputs": {}}}})
        self.assertEqual((status, body["error"]["type"]), (400, "invalid_prompt"))

    async def test_native_socket_and_images_use_the_session_cookie(self):
        await self.remote_mode()
        resp = await self.client.get("/ws?clientId=browser1", headers={"Origin": ORIGIN})
        self.assertEqual(resp.status, 401)
        await self.client.post("/session", headers=HEADERS)
        # The frontend may append its comfy.org token; the cookie still authenticates and the token is not forwarded.
        socket = await self.client.ws_connect("/ws?clientId=browser1&token=comfy-org-token", headers={"Origin": ORIGIN})
        self.assertEqual((await socket.receive_json())["type"], "status")
        await socket.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": True}})
        self.assertEqual((await socket.receive_json())["type"], "feature_flags")

        _, result = await self.post("/remote/api/prompt", {"client_id": "browser1", "prompt": t2i()})
        await self.remote.run(result["prompt_id"])
        self.assertEqual([(await socket.receive_json())["type"] for _ in range(2)], ["execution_start", "progress"])
        frame = await socket.receive_bytes()
        self.assertEqual(struct.unpack(">I", frame[:4])[0], 4)
        image = (await socket.receive_json())["data"]["output"]["images"][0]
        resp = await self.client.get("/remote/api/view", params={"filename": image["filename"], "type": "output", "subfolder": ""})
        self.assertEqual(await resp.read(), PNG + result["prompt_id"].encode())
        await socket.close()

    async def test_gateway_restart_restores_connection_and_remote_mode(self):
        target_id = await self.remote_mode()
        await self.client.close()
        self.client = await self.start_gateway()
        self.assertEqual(self.gateway.proxy_target_id, target_id)
        await self.until(lambda: self.gateway.targets[target_id].status == "ready")
        status, _ = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertEqual(status, 200)

    async def test_disconnect_is_remembered(self):
        target_id = await self.remote_mode()
        status, summary = await self.post(f"/targets/{target_id}/disconnect", {})
        self.assertEqual(summary["status"], "disconnected")
        status, _ = await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertEqual(status, 503)
        await self.client.close()
        self.client = await self.start_gateway()
        self.assertEqual(self.gateway.targets, {})
        self.assertEqual(await self.connect(), target_id)


    async def test_access_keys_go_only_to_their_own_host(self):
        await self.remote_mode()
        status, keys = await self.post("/keys", {"huggingface": " hf_secret ", "civitai": "civ_secret"})
        self.assertEqual(keys, {"huggingface": True, "civitai": True})
        resp = await self.client.get("/keys", headers=HEADERS)
        self.assertNotIn("secret", await resp.text())
        for url in ("https://huggingface.co/a/resolve/main/m.safetensors", "https://civitai.com/api/download/models/1", "https://example.com/m.safetensors"):
            await self.post("/remote/remote_execution/models/download", {"url": url, "directory": "vae", "name": "m.safetensors"})
        self.assertEqual([r.get("token") for r in self.remote.download_requests], ["hf_secret", "civ_secret", None])
        # Other remote routes never carry a key.
        await self.post("/remote/api/prompt", {"client_id": "x", "prompt": t2i()})
        self.assertNotIn("secret", json.dumps(self.remote.prompt_posts))
        status, keys = await self.post("/keys", {"huggingface": ""})
        self.assertEqual(keys, {"huggingface": False, "civitai": True})
        status, _ = await self.post("/keys", {"other": "x"})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
