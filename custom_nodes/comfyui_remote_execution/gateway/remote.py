import asyncio
import hashlib
import json
from urllib.parse import urlsplit

import aiohttp


class GatewayError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def normalize_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise GatewayError("Remote URL must be an http:// or https:// address.")
    if parts.username or parts.password:
        raise GatewayError("Remote URL must not contain credentials.")
    if parts.query:
        raise GatewayError("Remote URL must not contain a query string.")
    # The fragment never reaches the server, so it is dropped.
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


class RemoteTarget:
    def __init__(self, url, session, hub):
        self.url = url
        self.id = hashlib.sha256(url.encode()).hexdigest()[:16]
        self.session = session
        self.hub = hub
        self.status = "disconnected"
        self.detail = None
        self.system_stats = None
        self.queue_remaining = None
        self.task = None

    def summary(self):
        system = (self.system_stats or {}).get("system", {})
        devices = (self.system_stats or {}).get("devices", [])
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "detail": self.detail,
            "comfyui_version": system.get("comfyui_version"),
            "os": system.get("os"),
            "devices": [{"name": d.get("name"), "vram_total": d.get("vram_total")} for d in devices],
            "queue_remaining": self.queue_remaining,
        }

    def set_status(self, status, detail=None):
        self.status = status
        self.detail = detail
        self.hub.send_json({"type": "target", "target": self.summary()})

    def ws_base(self):
        scheme = "wss" if self.url.startswith("https") else "ws"
        return f"{scheme}{self.url[self.url.index(':'):]}"

    def connect(self):
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    async def disconnect(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None
        self.set_status("disconnected", "Disconnected by user.")

    async def run(self):
        # Stays connected until disconnect(): /system_stats identifies the server, its event
        # socket reports liveness and queue size, and failures are retried with backoff.
        self.set_status("connecting")
        delay = 1
        while True:
            try:
                async with self.session.get(f"{self.url}/system_stats", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    self.system_stats = await resp.json(content_type=None)
                if "comfyui_version" not in self.system_stats.get("system", {}):
                    self.task = None
                    self.set_status("incompatible", "The remote did not identify itself as ComfyUI.")
                    return
                async with self.session.ws_connect(f"{self.ws_base()}/ws", heartbeat=30, max_msg_size=0) as ws:
                    self.set_status("ready")
                    delay = 1
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            message = json.loads(msg.data)
                            if message.get("type") == "status":
                                self.queue_remaining = message["data"]["status"]["exec_info"]["queue_remaining"]
                                self.hub.send_json({"type": "target", "target": self.summary()})
                detail = "The remote closed the connection; reconnecting."
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
                detail = f"Cannot reach the remote ({e}); retrying."
            self.set_status("disconnected", detail)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
