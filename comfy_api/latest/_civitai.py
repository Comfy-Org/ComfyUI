"""Closed, read-only Civitai API projection for the Secure Nodes SDK.

This is deliberately not a general HTTP client.  It can reach one fixed
vendor endpoint, accepts no credentials or caller-supplied URL, and returns a
small field projection rather than the vendor's unbounded response objects.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any


class InProcessCivitai:
    _ORIGIN = "https://civitai.com"
    _API_PREFIX = "/api/v1/"
    _MAX_RESPONSE_BYTES = 4 * 1024 * 1024
    _CACHE_TTL_SECONDS = 300.0
    _CACHE_MAX_ENTRIES = 16
    _CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
    _CACHE_LOCK = threading.Lock()
    _HASH = re.compile(r"[0-9A-Fa-f]{8,128}")
    _HASH_NAME = re.compile(r"[A-Za-z0-9_-]{1,32}")

    @staticmethod
    def _bounded_text(value: Any, field: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Civitai {field} must be a string")
        value = value.strip()
        if not value or len(value) > maximum or "\x00" in value:
            raise ValueError(f"Civitai {field} is invalid")
        return value

    @staticmethod
    def _bounded_id(value: Any, field: str) -> int:
        if type(value) is not int or not 1 <= value <= 2**63 - 1:
            raise ValueError(f"Civitai {field} must be a positive integer")
        return value

    @classmethod
    def _fetch_json(cls, path: str, query: dict[str, Any] | None = None) -> dict:
        if not isinstance(path, str) or not path.startswith(cls._API_PREFIX):
            raise ValueError("Civitai request path is outside the fixed API")
        encoded = urllib.parse.urlencode(query or {})
        url = f"{cls._ORIGIN}{path}" + (f"?{encoded}" if encoded else "")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ComfyUI-Secure-Nodes/2",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=15.0) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if (final.scheme != "https" or final.hostname != "civitai.com"
                    or final.port not in (None, 443)
                    or not final.path.startswith(cls._API_PREFIX)):
                raise RuntimeError("Civitai redirected outside its fixed API")
            content_type = response.headers.get_content_type().lower()
            if content_type not in {"application/json", "text/json"}:
                raise RuntimeError("Civitai returned a non-JSON response")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise RuntimeError(
                        "Civitai returned an invalid response size") from exc
                if not 0 <= declared_size <= cls._MAX_RESPONSE_BYTES:
                    raise RuntimeError("Civitai response exceeds the size limit")
            payload = response.read(cls._MAX_RESPONSE_BYTES + 1)
        if len(payload) > cls._MAX_RESPONSE_BYTES:
            raise RuntimeError("Civitai response exceeds the size limit")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Civitai returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Civitai returned an invalid response object")
        return value

    @classmethod
    def _cached_fetch(
        cls, path: str, query: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> dict:
        if type(refresh) is not bool:
            raise TypeError("Civitai refresh must be a bool")
        key = path + "?" + urllib.parse.urlencode(query or {})
        now = time.monotonic()
        if not refresh:
            with cls._CACHE_LOCK:
                cached = cls._CACHE.pop(key, None)
                if cached is not None and now - cached[0] <= cls._CACHE_TTL_SECONDS:
                    cls._CACHE[key] = cached
                    return copy.deepcopy(cached[1])
        value = cls._fetch_json(path, query)
        with cls._CACHE_LOCK:
            cls._CACHE[key] = (time.monotonic(), copy.deepcopy(value))
            while len(cls._CACHE) > cls._CACHE_MAX_ENTRIES:
                cls._CACHE.popitem(last=False)
        return value

    @classmethod
    def _project_hashes(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for name, digest in list(value.items())[:16]:
            if (isinstance(name, str) and cls._HASH_NAME.fullmatch(name)
                    and isinstance(digest, str)
                    and 1 <= len(digest) <= 256 and "\x00" not in digest):
                result[name] = digest
        return result

    @classmethod
    def _project_files(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value[:100]:
            if not isinstance(item, dict):
                continue
            try:
                name = cls._bounded_text(item.get("name"), "file name", 512)
            except ValueError:
                continue
            result.append({
                "name": name,
                "hashes": cls._project_hashes(item.get("hashes")),
            })
        return result

    @staticmethod
    def _project_trained_words(value: Any) -> list[str]:
        """Return only the bounded public trigger-word projection.

        Civitai responses are vendor-controlled, so the projection is capped
        before it crosses the broker even when the upstream array is very
        large or contains malformed entries.
        """
        if not isinstance(value, list):
            return []
        result: list[str] = []
        total_bytes = 0
        for word in value[:2048]:
            if not isinstance(word, str) or not word or "\x00" in word:
                continue
            encoded_size = len(word.encode("utf-8"))
            if encoded_size > 512:
                continue
            if total_bytes + encoded_size > 1024 * 1024:
                break
            result.append(word)
            total_bytes += encoded_size
        return result

    @classmethod
    def _project_meta_value(
        cls,
        value: Any,
        state: dict[str, int],
        depth: int = 0,
    ) -> Any:
        """Project bounded image metadata without admitting vendor objects."""
        if depth > 4 or state["items"] >= 256:
            raise ValueError("Civitai image metadata exceeds its bounds")
        state["items"] += 1
        if value is None or isinstance(value, bool):
            return value
        if type(value) is int:
            if not -(2**63) <= value <= 2**63 - 1:
                raise ValueError("Civitai image metadata integer is invalid")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Civitai image metadata number is invalid")
            return value
        if isinstance(value, str):
            if "\x00" in value:
                raise ValueError("Civitai image metadata text is invalid")
            encoded = value.encode("utf-8")
            if len(encoded) > 64 * 1024:
                raise ValueError("Civitai image metadata text is too large")
            state["bytes"] += len(encoded)
            if state["bytes"] > 512 * 1024:
                raise ValueError("Civitai image metadata exceeds its size limit")
            return value
        if isinstance(value, list):
            if len(value) > 256:
                raise ValueError("Civitai image metadata list is too large")
            return [
                cls._project_meta_value(item, state, depth + 1)
                for item in value
            ]
        if isinstance(value, dict):
            if len(value) > 64:
                raise ValueError("Civitai image metadata object is too large")
            result = {}
            for key, item in value.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key.encode("utf-8")) > 128
                    or "\x00" in key
                    or key in {"__proto__", "constructor", "prototype"}
                ):
                    raise ValueError("Civitai image metadata key is invalid")
                state["bytes"] += len(key.encode("utf-8"))
                if state["bytes"] > 512 * 1024:
                    raise ValueError(
                        "Civitai image metadata exceeds its size limit")
                result[key] = cls._project_meta_value(
                    item, state, depth + 1)
            return result
        raise ValueError("Civitai image metadata value is invalid")

    @classmethod
    def _project_images(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result = []
        state = {"items": 0, "bytes": 0}
        for item in value[:32]:
            if not isinstance(item, dict):
                continue
            try:
                url = cls._bounded_text(item.get("url"), "image URL", 2048)
                parsed = urllib.parse.urlsplit(url)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.port not in (None, 443)
                ):
                    raise ValueError("Civitai image URL must be bounded HTTPS")
            except (ValueError, TypeError):
                continue
            projected = {"url": url}
            meta = item.get("meta")
            if isinstance(meta, dict):
                try:
                    projected["meta"] = cls._project_meta_value(
                        meta, state)
                except ValueError:
                    # The image identity remains useful even when a malformed
                    # vendor metadata object is omitted.
                    pass
            result.append(projected)
        return result

    @classmethod
    def _project_version_summary(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        try:
            return {
                "id": cls._bounded_id(value.get("id"), "model version id"),
                "name": cls._bounded_text(
                    value.get("name"), "model version name", 512),
            }
        except ValueError:
            return None

    @classmethod
    def _project_search(cls, value: dict) -> dict[str, Any]:
        items = value.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Civitai model search has no items list")
        result = []
        for item in items[:100]:
            if not isinstance(item, dict):
                continue
            try:
                model_id = cls._bounded_id(item.get("id"), "model id")
                name = cls._bounded_text(item.get("name"), "model name", 512)
            except ValueError:
                continue
            versions = []
            raw_versions = item.get("modelVersions")
            if isinstance(raw_versions, list):
                for version in raw_versions[:100]:
                    projected = cls._project_version_summary(version)
                    if projected is not None:
                        versions.append(projected)
            result.append({
                "id": model_id,
                "name": name,
                "modelVersions": versions,
            })
        return {"items": result}

    @classmethod
    def _project_version(cls, value: dict) -> dict[str, Any]:
        return {
            "id": cls._bounded_id(value.get("id"), "model version id"),
            "name": cls._bounded_text(
                value.get("name"), "model version name", 512),
            "files": cls._project_files(value.get("files")),
        }

    @classmethod
    def _project_version_by_hash(cls, value: dict) -> dict[str, Any]:
        model = value.get("model")
        if not isinstance(model, dict):
            raise RuntimeError("Civitai model-version response has no model")
        projected_model = {
            "name": cls._bounded_text(model.get("name"), "model name", 512),
        }
        model_type = model.get("type")
        if isinstance(model_type, str) and 1 <= len(model_type) <= 128:
            projected_model["type"] = model_type
        result: dict[str, Any] = {
            "id": cls._bounded_id(value.get("id"), "model version id"),
            "name": cls._bounded_text(
                value.get("name"), "model version name", 512),
            "modelId": cls._bounded_id(value.get("modelId"), "model id"),
            "model": projected_model,
            "files": cls._project_files(value.get("files")),
            "trainedWords": cls._project_trained_words(
                value.get("trainedWords")),
            "images": cls._project_images(value.get("images")),
        }
        base_model = value.get("baseModel")
        if isinstance(base_model, str):
            try:
                result["baseModel"] = cls._bounded_text(
                    base_model, "base model", 512)
            except ValueError:
                pass
        air = value.get("air")
        if isinstance(air, str) and 1 <= len(air) <= 512 and "\x00" not in air:
            result["air"] = air
        return result

    async def search_models(
        self, username: str, query: str | None = None,
        limit: int = 20, nsfw: bool = False,
    ) -> dict[str, Any]:
        username = self._bounded_text(username, "username", 128)
        if query is not None:
            query = self._bounded_text(query, "query", 512)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Civitai search limit must be in [1, 100]")
        if type(nsfw) is not bool:
            raise TypeError("Civitai nsfw must be a bool")
        params: dict[str, Any] = {
            "username": username,
            "limit": limit,
            "nsfw": "true" if nsfw else "false",
        }
        if query is not None:
            params["query"] = query
        value = await asyncio.to_thread(
            self._cached_fetch, "/api/v1/models", params)
        return self._project_search(value)

    async def model_version(self, model_version_id: int) -> dict[str, Any]:
        model_version_id = self._bounded_id(
            model_version_id, "model version id")
        value = await asyncio.to_thread(
            self._cached_fetch, f"/api/v1/model-versions/{model_version_id}")
        return self._project_version(value)

    async def model_version_by_hash(
        self, hash_value: str, refresh: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(hash_value, str) or not self._HASH.fullmatch(hash_value):
            raise ValueError("Civitai model hash must be 8-128 hex digits")
        if type(refresh) is not bool:
            raise TypeError("Civitai refresh must be a bool")
        normalized = hash_value.upper()
        value = await asyncio.to_thread(
            self._cached_fetch,
            f"/api/v1/model-versions/by-hash/{normalized}",
            None,
            refresh,
        )
        return self._project_version_by_hash(value)
