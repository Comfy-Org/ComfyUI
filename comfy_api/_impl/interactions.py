# Lifted unchanged from the secure overlay (comfy_secure_nodes/interactions.py), its tested interaction
# broker: one broker for ComfyUI, its own pack processes and the overlay's sandboxed packs.
"""Brokered, one-use UI interactions for sandboxed node executions.

Guests choose one reviewed interaction *kind* and a bounded JSON payload.  The
host owns the websocket event name, request token, node identity, timeout, and
HTTP response route, so a pack cannot impersonate core events or install an
ambient callback endpoint.
"""
from __future__ import annotations

import asyncio
import collections
import copy
import json
import math
import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any


EVENT = "secure-node-interaction"
RESPONSE_ROUTE = "/secure-nodes/interactions/respond"
MAX_JSON_BYTES = 64 * 1024
MAX_PENDING = 128
MAX_CACHE = 256
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 540.0
KINDS = frozenset({"image-choice", "prompt-await", "mask-edit"})


def _bounded_string(value: Any, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > maximum
        or "\0" in value
    ):
        raise TypeError(f"{label} must be a bounded string")
    return value


def _asset_identity(
    value: Any, label: str, *, result: bool = False,
) -> dict[str, str]:
    name_key = "name" if result else "filename"
    if not isinstance(value, dict) or set(value) != {
        name_key, "type", "subfolder",
    }:
        raise TypeError(f"{label} must be an exact managed asset identity")
    name = _bounded_string(value[name_key], f"{label}.{name_key}", 1024)
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"{label}.{name_key} must be a safe basename")
    asset_type = value["type"]
    allowed_types = {"temp"} if result else {"input", "output", "temp"}
    if asset_type not in allowed_types:
        raise ValueError(f"{label}.type is not permitted")
    subfolder = _bounded_string(
        value["subfolder"], f"{label}.subfolder", 2048)
    if (
        subfolder.startswith(("/", "\\"))
        or "\\" in subfolder
        or any(part in {"", ".", ".."} for part in subfolder.split("/")
               if subfolder)
        or re.match(r"^[A-Za-z]:", subfolder)
    ):
        raise ValueError(f"{label}.subfolder must be a confined relative path")
    return {name_key: name, "type": asset_type, "subfolder": subfolder}


def _extras(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 3:
        raise TypeError(f"{label} must contain exactly three strings")
    return [
        _bounded_string(item, f"{label}[{index}]", 4096)
        for index, item in enumerate(value)
    ]


def _mask_edit_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("mask-edit payload must be an object")
    required = {"variant", "image", "extras", "tip"}
    allowed = required | {"initial_mask", "sound"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise TypeError("mask-edit payload has unsupported fields")
    if value["variant"] != "cg-image-filter.mask-edit-v1":
        raise ValueError("unsupported mask-edit payload variant")
    result: dict[str, Any] = {
        "variant": value["variant"],
        "image": _asset_identity(value["image"], "mask-edit image"),
        "extras": _extras(value["extras"], "mask-edit extras"),
        "tip": _bounded_string(value["tip"], "mask-edit tip", 16384),
    }
    if "initial_mask" in value:
        result["initial_mask"] = _asset_identity(
            value["initial_mask"], "mask-edit initial_mask")
    if "sound" in value:
        if value["sound"] not in {"ding.mp3", "beep.mp3", "honk.mp3"}:
            raise ValueError("unsupported mask-edit sound")
        result["sound"] = value["sound"]
    return result


def _mask_edit_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and set(value) == {"reset"}:
        if value["reset"] is not True:
            raise TypeError("mask-edit reset must be true")
        return {"reset": True}
    if not isinstance(value, dict) or not isinstance(
        value.get("cancelled"), bool
    ):
        raise TypeError("mask-edit response must contain cancelled")
    if value["cancelled"]:
        if set(value) != {"cancelled"}:
            raise TypeError("cancelled mask-edit response has unsupported fields")
        return {"cancelled": True}
    if set(value) != {"cancelled", "mask", "extras"}:
        raise TypeError("successful mask-edit response has unsupported fields")
    return {
        "cancelled": False,
        "mask": _asset_identity(value["mask"], "mask-edit mask", result=True),
        "extras": _extras(value["extras"], "mask-edit extras"),
    }


def _validate_payload(kind: str, value: Any) -> Any:
    if kind == "mask-edit":
        value = _mask_edit_payload(value)
    return _bounded_json(value, "interaction payload")


def _validate_response(kind: str, value: Any) -> Any:
    if kind == "mask-edit":
        value = _mask_edit_response(value)
    return _bounded_json(value, "interaction response")


def _bounded_json(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be finite JSON data") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds {MAX_JSON_BYTES} encoded bytes")
    return copy.deepcopy(value)


@dataclass
class _Pending:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    cache_key: tuple[str, str, str, str]
    remember: bool


class InteractionBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._cache: collections.OrderedDict[
            tuple[str, str, str, str], Any
        ] = collections.OrderedDict()

    async def request(
        self, *, kind: str, payload: Any, tenant: str, pack: str,
        node_id: str, sid: str | None, timeout: float,
        reuse_last: bool = False, remember: bool = False,
    ) -> Any:
        if kind not in KINDS:
            raise ValueError(f"unknown secure interaction kind {kind!r}")
        timeout = float(timeout)
        if (
            not math.isfinite(timeout)
            or not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("interaction timeout is outside the allowed range")
        payload = _validate_payload(kind, payload)
        cache_key = (str(tenant), str(pack), str(node_id), kind)
        with self._lock:
            if reuse_last and cache_key in self._cache:
                value = self._cache.pop(cache_key)
                self._cache[cache_key] = value
                return copy.deepcopy(value)
            if len(self._pending) >= MAX_PENDING:
                raise RuntimeError("too many secure UI interactions are pending")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        token = secrets.token_urlsafe(32)
        pending = _Pending(loop, future, cache_key, bool(remember))
        with self._lock:
            self._pending[token] = pending

        from server import PromptServer

        PromptServer.instance.send_sync(EVENT, {
            "request_id": token,
            "kind": kind,
            "pack": str(pack),
            "node_id": str(node_id),
            "timeout_seconds": timeout,
            "payload": payload,
        }, sid)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            with self._lock:
                self._pending.pop(token, None)

    def respond(self, token: str, response: Any) -> bool:
        if not isinstance(token, str) or len(token) > 128:
            return False
        with self._lock:
            pending = self._pending.get(token)
        if pending is None:
            return False
        try:
            response = _validate_response(pending.cache_key[3], response)
        except (TypeError, ValueError):
            return False
        with self._lock:
            current = self._pending.get(token)
            if current is not pending:
                return False
            self._pending.pop(token, None)
            if pending.remember:
                self._cache[pending.cache_key] = copy.deepcopy(response)
                self._cache.move_to_end(pending.cache_key)
                while len(self._cache) > MAX_CACHE:
                    self._cache.popitem(last=False)

        def resolve() -> None:
            if not pending.future.done():
                pending.future.set_result(response)

        pending.loop.call_soon_threadsafe(resolve)
        return True


BROKER = InteractionBroker()


def register_routes(prompt_server) -> None:
    if getattr(prompt_server, "_secure_interaction_routes", False):
        return
    from aiohttp import web

    @prompt_server.routes.post(RESPONSE_ROUTE)
    async def secure_interaction_response(request):
        try:
            if request.content_length is not None and request.content_length > MAX_JSON_BYTES:
                raise ValueError("response body is too large")
            body = await request.json()
            token = body.get("request_id") if isinstance(body, dict) else None
            response = body.get("response") if isinstance(body, dict) else None
            if not BROKER.respond(token, response):
                return web.json_response(
                    {"ok": False, "error": "unknown, expired, or invalid request"},
                    status=404,
                )
            return web.json_response({"ok": True})
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid interaction response"},
                status=400,
            )

    prompt_server._secure_interaction_routes = True


__all__ = [
    "BROKER", "EVENT", "KINDS", "RESPONSE_ROUTE", "register_routes",
]
