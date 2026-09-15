"""Closed Ollama vendor integration for Secure Nodes V2.

This is intentionally not a general HTTP client. Direct node-supplied origins
are the default Ollama service on loopback only; any other deployment must be
named in host-admin configuration and nodes receive only that profile name.
Requests and responses are projected onto the small Ollama fields used by the
public node pack.
"""
from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class InProcessOllama:
    _PROFILE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
    _LOOPBACK = {"localhost", "127.0.0.1", "::1"}
    _MAX_REQUEST_BYTES = 64 * 1024 * 1024
    _MAX_RESPONSE_BYTES = 32 * 1024 * 1024
    _MAX_TEXT_BYTES = 4 * 1024 * 1024
    _MAX_CONTEXT_TOKENS = 262_144
    _MAX_IMAGES = 16
    _MAX_IMAGE_PIXELS = 67_108_864
    _MAX_IMAGE_BYTES = 48 * 1024 * 1024
    _OPTIONS = {
        "mirostat": (int, 0, 2),
        "mirostat_eta": (float, 0.0, 1000.0),
        "mirostat_tau": (float, 0.0, 1000.0),
        "num_ctx": (int, 0, 2**31),
        "repeat_last_n": (int, -1, 64),
        "repeat_penalty": (float, 0.0, 2.0),
        "temperature": (float, -10.0, 10.0),
        "seed": (int, 0, 2**31),
        "tfs_z": (float, 1.0, 1000.0),
        "num_predict": (int, -2, 32_768),
        "top_k": (int, 0, 100),
        "top_p": (float, 0.0, 1.0),
        "min_p": (float, 0.0, 1.0),
        "main_gpu": (int, 0, 0),
    }
    _BOOLEAN_OPTIONS = {"low_vram"}
    _TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

    @staticmethod
    def _text(
        value: Any, field: str, *, maximum: int,
        allow_empty: bool = True, strip: bool = False,
    ) -> str:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"Ollama {field} must be a string")
        result = value.strip() if strip else value
        size = len(result.encode("utf-8"))
        if size > maximum or (not allow_empty and not result):
            raise ValueError(f"Ollama {field} is invalid or too large")
        return result

    @classmethod
    def _validate_origin(cls, value: str, *, direct: bool) -> str:
        parts = urllib.parse.urlsplit(value)
        if (parts.scheme not in ({"http"} if direct else {"http", "https"})
                or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError("Ollama endpoint must be an origin only")
        try:
            port = parts.port
        except ValueError as error:
            raise ValueError("Ollama endpoint has an invalid port") from error
        if direct:
            if parts.hostname.lower() not in cls._LOOPBACK or port != 11434:
                raise ValueError(
                    "direct Ollama endpoint must be loopback port 11434")
            if parts.hostname.lower() == "localhost":
                try:
                    addresses = socket.getaddrinfo(
                        "localhost", 11434, type=socket.SOCK_STREAM)
                except OSError as error:
                    raise ValueError("localhost did not resolve") from error
                if not addresses or any(
                    not ipaddress.ip_address(item[4][0]).is_loopback
                    for item in addresses
                ):
                    raise ValueError("localhost must resolve only to loopback")
        default_port = 80 if parts.scheme == "http" else 443
        resolved_port = port or default_port
        host = parts.hostname.lower()
        rendered_host = f"[{host}]" if ":" in host else host
        rendered_port = "" if resolved_port == default_port else f":{resolved_port}"
        return f"{parts.scheme}://{rendered_host}{rendered_port}"

    @classmethod
    def _profiles(cls) -> dict[str, str]:
        raw = os.environ.get("COMFY_SECURE_OLLAMA_PROFILES", "{}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "COMFY_SECURE_OLLAMA_PROFILES is invalid JSON") from error
        if not isinstance(value, dict) or len(value) > 64:
            raise RuntimeError("Ollama profile configuration must be an object")
        result = {}
        for name, origin in value.items():
            if (not isinstance(name, str) or not cls._PROFILE.fullmatch(name)
                    or not isinstance(origin, str)):
                raise RuntimeError("Ollama profile configuration is invalid")
            result[name] = cls._validate_origin(origin, direct=False)
        return result

    @classmethod
    def _origin(cls, endpoint: Any) -> str:
        endpoint = cls._text(
            endpoint, "endpoint", maximum=2048,
            allow_empty=False, strip=True)
        if endpoint.startswith("ollama://"):
            name = endpoint.removeprefix("ollama://")
            if not cls._PROFILE.fullmatch(name):
                raise ValueError("Ollama profile name is invalid")
            origin = cls._profiles().get(name)
            if origin is None:
                raise ValueError(f"Ollama profile {name!r} is not configured")
            return origin
        return cls._validate_origin(endpoint, direct=True)

    @staticmethod
    def _json_body(value: dict[str, Any]) -> bytes:
        try:
            body = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("Ollama request is not JSON-safe") from error
        if len(body) > InProcessOllama._MAX_REQUEST_BYTES:
            raise ValueError("Ollama request exceeds the size limit")
        return body

    @classmethod
    def _format(cls, value: Any) -> str | dict[str, Any]:
        if isinstance(value, str):
            if value not in {"", "json"}:
                raise ValueError("Ollama format must be text, json, or a schema")
            return value
        if not isinstance(value, dict):
            raise TypeError("Ollama format must be text, json, or a schema")
        entries = 0

        def validate(item: Any, depth: int) -> None:
            nonlocal entries
            if depth > 16 or entries > 4096:
                raise ValueError("Ollama response schema exceeds its bounds")
            entries += 1
            if item is None or isinstance(item, (str, bool, int)):
                return
            if isinstance(item, float):
                if not math.isfinite(item):
                    raise ValueError("Ollama response schema must be finite")
                return
            if isinstance(item, list):
                if len(item) > 1024:
                    raise ValueError("Ollama response schema exceeds its bounds")
                for child in item:
                    validate(child, depth + 1)
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    if (
                        not isinstance(key, str) or "\x00" in key
                        or len(key.encode("utf-8")) > 512
                    ):
                        raise ValueError("Ollama response schema has an invalid key")
                    validate(child, depth + 1)
                return
            raise TypeError("Ollama response schema must be JSON data")

        validate(value, 0)
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("Ollama response schema is not JSON-safe") from error
        if len(encoded) > 64 * 1024:
            raise ValueError("Ollama response schema exceeds its size limit")
        return json.loads(encoded.decode("utf-8"))

    @classmethod
    def _json_object(
        cls, value: Any, field: str, *, maximum: int = 64 * 1024,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"Ollama {field} must be a JSON object")
        entries = 0

        def validate(item: Any, depth: int) -> None:
            nonlocal entries
            entries += 1
            if depth > 16 or entries > 4096:
                raise ValueError(f"Ollama {field} exceeds its bounds")
            if item is None or isinstance(item, (str, bool, int)):
                return
            if isinstance(item, float):
                if not math.isfinite(item):
                    raise ValueError(f"Ollama {field} must be finite")
                return
            if isinstance(item, list):
                if len(item) > 1024:
                    raise ValueError(f"Ollama {field} exceeds its bounds")
                for child in item:
                    validate(child, depth + 1)
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    if (
                        not isinstance(key, str) or "\x00" in key
                        or len(key.encode("utf-8")) > 512
                    ):
                        raise ValueError(f"Ollama {field} has an invalid key")
                    validate(child, depth + 1)
                return
            raise TypeError(f"Ollama {field} must contain JSON data")

        validate(value, 0)
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Ollama {field} is not JSON-safe") from error
        if len(encoded) > maximum:
            raise ValueError(f"Ollama {field} exceeds its size limit")
        return json.loads(encoded.decode("utf-8"))

    @classmethod
    def _tool_name(cls, value: Any) -> str:
        if not isinstance(value, str) or not cls._TOOL_NAME.fullmatch(value):
            raise ValueError("Ollama tool name is invalid")
        return value

    @classmethod
    def _tool_calls(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > 32:
            raise ValueError("Ollama tool calls must be a bounded list")
        result = []
        for call in value:
            if not isinstance(call, dict):
                raise ValueError("Ollama tool call has an invalid shape")
            function = call.get("function")
            if (set(call) != {"function"} or not isinstance(function, dict)
                    or set(function) != {"name", "arguments"}):
                raise ValueError("Ollama tool call has an invalid shape")
            result.append({
                "name": cls._tool_name(function["name"]),
                "arguments": cls._json_object(
                    function["arguments"], "tool arguments"),
            })
        return result

    @classmethod
    def _tools(cls, value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not 1 <= len(value) <= 32:
            raise ValueError("Ollama tools must contain 1 to 32 entries")
        result = []
        for tool in value:
            if not isinstance(tool, dict) or set(tool) != {
                "name", "description", "parameters",
            }:
                raise ValueError("Ollama tool has an invalid shape")
            parameters = cls._json_object(
                tool["parameters"], "tool parameters")
            if parameters.get("type") != "object":
                raise ValueError("Ollama tool parameters must describe an object")
            result.append({
                "type": "function",
                "function": {
                    "name": cls._tool_name(tool["name"]),
                    "description": cls._text(
                        tool["description"], "tool description",
                        maximum=4096),
                    "parameters": parameters,
                },
            })
        return result

    @staticmethod
    def _timeout(value: Any) -> float:
        if isinstance(value, bool) or type(value) not in {int, float}:
            raise TypeError("Ollama timeout_seconds must be numeric")
        result = float(value)
        if not math.isfinite(result) or not 1.0 <= result <= 600.0:
            raise ValueError("Ollama timeout_seconds must be in [1, 600]")
        return result

    @classmethod
    def _request_json(
        cls, origin: str, path: str, payload: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        if path not in {"/api/tags", "/api/generate", "/api/chat"}:
            raise ValueError("Ollama request path is not permitted")
        data = None if payload is None else cls._json_body(payload)
        request = urllib.request.Request(
            origin + path,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ComfyUI-Secure-Nodes/2",
            },
            method="GET" if data is None else "POST",
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                final = urllib.parse.urlsplit(response.geturl())
                expected = urllib.parse.urlsplit(origin + path)
                if (final.scheme, final.hostname, final.port, final.path) != (
                    expected.scheme, expected.hostname, expected.port,
                    expected.path,
                ) or final.query or final.fragment:
                    raise RuntimeError("Ollama redirected outside its fixed origin")
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"application/json", "text/json"}:
                    raise RuntimeError("Ollama returned a non-JSON response")
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as error:
                        raise RuntimeError(
                            "Ollama returned an invalid response size") from error
                    if not 0 <= declared_size <= cls._MAX_RESPONSE_BYTES:
                        raise RuntimeError("Ollama response exceeds the size limit")
                body = response.read(cls._MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Ollama request failed with HTTP {error.code}") from error
        if len(body) > cls._MAX_RESPONSE_BYTES:
            raise RuntimeError("Ollama response exceeds the size limit")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Ollama returned invalid JSON") from error
        if not isinstance(value, dict):
            raise RuntimeError("Ollama returned an invalid response object")
        return value

    @classmethod
    def _options(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict) or not set(value) <= (
            set(cls._OPTIONS) | cls._BOOLEAN_OPTIONS | {"stop"}
        ):
            raise ValueError("Ollama options contain an unsupported field")
        result: dict[str, Any] = {}
        for name, item in value.items():
            if name == "stop":
                result[name] = cls._text(
                    item, "stop option", maximum=4096, strip=False)
                continue
            if name in cls._BOOLEAN_OPTIONS:
                if type(item) is not bool:
                    raise TypeError(f"Ollama option {name} must be a boolean")
                result[name] = item
                continue
            kind, minimum, maximum = cls._OPTIONS[name]
            if kind is int:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise TypeError(f"Ollama option {name} must be an integer")
                normalized: int | float = item
            else:
                if isinstance(item, bool) or type(item) not in (int, float):
                    raise TypeError(f"Ollama option {name} must be numeric")
                normalized = float(item)
                if not math.isfinite(normalized):
                    raise ValueError(f"Ollama option {name} must be finite")
            if not minimum <= normalized <= maximum:
                raise ValueError(f"Ollama option {name} is out of range")
            result[name] = normalized
        return result

    @classmethod
    def _context(cls, value: Any, *, required: bool = False) -> list[int] | None:
        if value is None and not required:
            return None
        if (not isinstance(value, list)
                or len(value) > cls._MAX_CONTEXT_TOKENS):
            raise ValueError("Ollama context must be a bounded integer list")
        result = []
        for token in value:
            if (isinstance(token, bool) or not isinstance(token, int)
                    or not 0 <= token <= 2**31 - 1):
                raise ValueError("Ollama context contains an invalid token")
            result.append(token)
        return result

    @classmethod
    def _keep_alive(cls, value: Any, unit: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value <= 120:
            raise ValueError("Ollama keep_alive must be in [-1, 120]")
        if unit not in {"minutes", "hours"}:
            raise ValueError("Ollama keep_alive_unit must be minutes or hours")
        return f"{value}{'m' if unit == 'minutes' else 'h'}"

    @classmethod
    async def _images(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        from ._sdk import ImageRef, current_runtime
        import torch
        from PIL import Image

        if not isinstance(value, ImageRef):
            raise TypeError("Ollama images must be an IMAGE ref")
        pixels = await current_runtime().refs.resolve(value)
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                or not 1 <= int(pixels.shape[0]) <= cls._MAX_IMAGES
                or int(pixels.shape[-1]) < 3):
            raise ValueError("Ollama images require a bounded BHWC RGB batch")
        batch, height, width = map(int, pixels.shape[:3])
        if (height < 1 or width < 1
                or batch * height * width > cls._MAX_IMAGE_PIXELS):
            raise ValueError("Ollama image dimensions exceed the limit")
        if not torch.isfinite(pixels).all():
            raise ValueError("Ollama images must contain finite pixels")
        rgb = (pixels[..., :3].detach().to("cpu").clamp(0.0, 1.0) * 255.0)
        rgb = rgb.to(torch.uint8).numpy()
        result = []
        total = 0
        for frame in rgb:
            buffer = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            total += len(encoded)
            if total > cls._MAX_IMAGE_BYTES:
                raise ValueError("Ollama encoded images exceed the size limit")
            result.append(encoded)
        return result

    @classmethod
    def _response_text(cls, value: Any, field: str) -> str:
        return cls._text(
            value, f"response {field}", maximum=cls._MAX_TEXT_BYTES)

    async def list_models(self, endpoint: str) -> list[str]:
        origin = self._origin(endpoint)
        value = await asyncio.to_thread(
            self._request_json, origin, "/api/tags", None, 10.0)
        raw_models = value.get("models")
        if not isinstance(raw_models, list):
            raise RuntimeError("Ollama model list is missing")
        models = []
        for item in raw_models[:512]:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("model"))
            try:
                models.append(self._text(
                    name, "model name", maximum=512,
                    allow_empty=False, strip=True))
            except ValueError:
                continue
        return list(dict.fromkeys(models))

    async def generate(
        self, endpoint: str, model: str, system: str, prompt: str,
        images=None, context: list[int] | None = None, think: bool = False,
        options: dict[str, Any] | None = None, keep_alive: int = 5,
        keep_alive_unit: str = "minutes", format: str | dict[str, Any] = "",
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        if not isinstance(think, bool):
            raise TypeError("Ollama think must be a bool")
        format_value = self._format(format)
        timeout_value = self._timeout(timeout_seconds)
        payload: dict[str, Any] = {
            "model": self._text(
                model, "model", maximum=512, allow_empty=False, strip=True),
            "system": self._text(system, "system", maximum=1_048_576),
            "prompt": self._text(prompt, "prompt", maximum=4_194_304),
            "stream": False,
            "think": think,
            "keep_alive": self._keep_alive(keep_alive, keep_alive_unit),
            "format": format_value,
        }
        image_data = await self._images(images)
        context_data = self._context(context)
        option_data = self._options(options)
        if image_data is not None:
            payload["images"] = image_data
        if context_data is not None:
            payload["context"] = context_data
        if option_data is not None:
            payload["options"] = option_data
        origin = self._origin(endpoint)
        response = await asyncio.to_thread(
            self._request_json, origin, "/api/generate", payload, timeout_value)
        result: dict[str, Any] = {
            "response": self._response_text(response.get("response"), "text"),
            "context": self._context(response.get("context"), required=True),
        }
        thinking = response.get("thinking")
        if think and thinking is not None:
            result["thinking"] = self._response_text(thinking, "thinking")
        return result

    async def chat(
        self, endpoint: str, model: str,
        messages: list[dict[str, Any]], images=None, think: bool = False,
        options: dict[str, Any] | None = None, keep_alive: int = 5,
        keep_alive_unit: str = "minutes", format: str | dict[str, Any] = "",
        timeout_seconds: float = 600.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(think, bool):
            raise TypeError("Ollama think must be a bool")
        format_value = self._format(format)
        timeout_value = self._timeout(timeout_seconds)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
            raise ValueError("Ollama messages must contain 1 to 256 entries")
        projected = []
        total = 0
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Ollama message has an invalid shape")
            role = message.get("role")
            allowed = {"role", "content"}
            if role == "assistant":
                allowed |= {"thinking", "tool_calls"}
            elif role == "tool":
                allowed |= {"tool_name"}
            elif role not in {"system", "user"}:
                raise ValueError("Ollama message has an invalid shape")
            if not {"role", "content"}.issubset(message) or not set(message) <= allowed:
                raise ValueError("Ollama message has an invalid shape")
            if role == "tool" and "tool_name" not in message:
                raise ValueError("Ollama tool message requires tool_name")
            content = self._text(
                message["content"], "message content", maximum=1_048_576)
            total += len(content.encode("utf-8"))
            if total > 4_194_304:
                raise ValueError("Ollama message history exceeds the size limit")
            item: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                if "thinking" in message:
                    item["thinking"] = self._text(
                        message["thinking"], "message thinking",
                        maximum=1_048_576)
                if "tool_calls" in message:
                    calls = message["tool_calls"]
                    if not isinstance(calls, list) or len(calls) > 32:
                        raise ValueError(
                            "Ollama message tool calls must be a bounded list")
                    native_calls = []
                    for call in calls:
                        if not isinstance(call, dict) or set(call) != {
                            "name", "arguments",
                        }:
                            raise ValueError(
                                "Ollama message tool call has an invalid shape")
                        native_calls.append({"function": {
                            "name": self._tool_name(call["name"]),
                            "arguments": self._json_object(
                                call["arguments"], "tool arguments"),
                        }})
                    item["tool_calls"] = native_calls
            elif role == "tool":
                item["tool_name"] = self._tool_name(message["tool_name"])
            projected.append(item)
        image_data = await self._images(images)
        if image_data is not None:
            user = next((item for item in reversed(projected)
                         if item["role"] == "user"), None)
            if user is None:
                raise ValueError("Ollama images require a user message")
            user["images"] = image_data
        payload: dict[str, Any] = {
            "model": self._text(
                model, "model", maximum=512, allow_empty=False, strip=True),
            "messages": projected,
            "stream": False,
            "think": think,
            "keep_alive": self._keep_alive(keep_alive, keep_alive_unit),
            "format": format_value,
        }
        option_data = self._options(options)
        if option_data is not None:
            payload["options"] = option_data
        tool_data = self._tools(tools)
        if tool_data is not None:
            payload["tools"] = tool_data
        origin = self._origin(endpoint)
        response = await asyncio.to_thread(
            self._request_json, origin, "/api/chat", payload, timeout_value)
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response has no message")
        result = {
            "response": self._response_text(message.get("content"), "text"),
        }
        thinking = message.get("thinking")
        if think and thinking is not None:
            result["thinking"] = self._response_text(thinking, "thinking")
        if message.get("tool_calls") is not None:
            result["tool_calls"] = self._tool_calls(message["tool_calls"])
        return result
