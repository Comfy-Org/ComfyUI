"""Bounded fixed-provider media integrations used by Secure Nodes V2.

This module deliberately is not a general HTTP client.  Every request target,
request shape, response projection, poll loop, and downloaded-media bound is
owned here.  Pack code supplies only the provider credential and the small set
of fields admitted by the public integration protocols.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import ipaddress
import json
import math
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


_LUMA_ORIGIN = "https://api.lumalabs.ai"
_LUMA_BASE = _LUMA_ORIGIN + "/dream-machine/v1/generations"
_IMGBB_UPLOAD = "https://api.imgbb.com/1/upload"
_SENSENOVA_BASE = "https://token.sensenova.cn/v1"
_JSON_MAX = 4 * 1024 * 1024
_SENSENOVA_IMAGE_JSON_MAX = 72 * 1024 * 1024
_SENSENOVA_IMAGE_BYTES_MAX = 50 * 1024 * 1024
_IMAGE_BYTES_MAX = 32 * 1024 * 1024
_VIDEO_BYTES_MAX = 1024 * 1024 * 1024
_IMAGE_PIXELS_MAX = 64 * 1024 * 1024
_POLL_TIMEOUT_SECONDS = 900.0
_POLL_INTERVAL_SECONDS = 1.0
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _secret(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"{name} must be a non-empty bounded string")
    if "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} contains invalid characters")
    return value


def _text(value: Any, name: str, *, maximum: int = 5000,
          allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds its bounded text limit")
    if not allow_empty and not value:
        raise ValueError(f"{name} is required")
    return value


def _identifier(value: Any, name: str) -> str:
    value = _text(value, name, maximum=256, allow_empty=False)
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    value = str(value)
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(choices))}")
    return value


def _https_url(value: Any, name: str) -> str:
    value = _text(value, name, maximum=2048, allow_empty=False)
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https" or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a credential-free HTTPS URL")
    return value


def _public_https_url(value: Any, name: str) -> str:
    value = _https_url(value, name)
    parsed = urllib.parse.urlsplit(value)
    assert parsed.hostname is not None
    try:
        infos = socket.getaddrinfo(
            parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ValueError(f"{name} host could not be resolved") from error
    addresses = {info[4][0].split("%", 1)[0] for info in infos}
    if not addresses:
        raise ValueError(f"{name} host did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"{name} must resolve only to public addresses")
    return value


class _PublicHttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_https_url(newurl, "provider media redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_provider_error(error: BaseException) -> RuntimeError:
    if isinstance(error, urllib.error.HTTPError):
        return RuntimeError(f"provider request failed with HTTP {error.code}")
    if isinstance(error, urllib.error.URLError):
        return RuntimeError("provider request failed")
    return RuntimeError("provider request failed")


def _request_json(
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    payload: Optional[dict[str, Any]] = None,
    form: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    maximum: int = _JSON_MAX,
    attempts: int = 1,
    request_maximum: int = 1024 * 1024,
) -> dict[str, Any]:
    data: Optional[bytes] = None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        data = json.dumps(
            payload, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > request_maximum:
            raise ValueError("provider request JSON exceeds its byte limit")
        request_headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("ascii")
        if len(data) > 48 * 1024 * 1024:
            raise ValueError("provider upload form exceeds 48 MiB")
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method)
    if not 1 <= attempts <= 3:
        raise ValueError("provider request attempts must be in [1, 3]")
    if not 1 <= maximum <= _SENSENOVA_IMAGE_JSON_MAX:
        raise ValueError("provider response limit is invalid")
    if not 1 <= request_maximum <= 49 * 1024 * 1024:
        raise ValueError("provider request limit is invalid")
    raw = b""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > maximum:
                    raise ValueError("provider response exceeds its byte limit")
                raw = response.read(maximum + 1)
            break
        except urllib.error.HTTPError as error:
            if error.code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(2**attempt)
                continue
            raise _safe_provider_error(error) from None
        except urllib.error.URLError as error:
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
                continue
            raise _safe_provider_error(error) from None
    if len(raw) > maximum:
        raise ValueError("provider response exceeds its byte limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provider returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def _sensenova_key() -> str:
    return _secret(os.environ.get("SN_API_KEY", ""), "SenseNova API key")


def _bounded_json(value: Any, secret: str, *, maximum: int) -> Any:
    """Validate provider JSON and remove an echoed credential.

    Byte bounds are enforced before parsing. These structural bounds stop a
    small, deeply nested response from consuming unbounded recursion or object
    overhead while preserving the pinned nodes' raw JSON output.
    """
    remaining = [200_000]

    def visit(item: Any, depth: int) -> Any:
        remaining[0] -= 1
        if remaining[0] < 0:
            raise ValueError("SenseNova response contains too many JSON values")
        if depth > 16:
            raise ValueError("SenseNova response exceeds the JSON depth limit")
        if item is None or type(item) is bool or type(item) is int:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("SenseNova response contains a non-finite number")
            return item
        if isinstance(item, str):
            if len(item.encode("utf-8")) > maximum:
                raise ValueError("SenseNova response contains oversized text")
            return item.replace(secret, "[REDACTED]") if secret else item
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str) or "\x00" in key or len(
                        key.encode("utf-8")) > 512:
                    raise ValueError("SenseNova response has an invalid JSON key")
                result[key] = visit(child, depth + 1)
            return result
        raise ValueError("SenseNova response contains an unsupported JSON value")

    return visit(value, 0)


def _sensenova_post(
    path: str, payload: dict[str, Any], *, timeout: int, maximum: int = _JSON_MAX,
    request_maximum: int = 1024 * 1024,
) -> dict[str, Any]:
    key = _sensenova_key()
    raw = _request_json(
        "POST",
        _SENSENOVA_BASE + path,
        headers={"Authorization": f"Bearer {key}"},
        payload=payload,
        timeout=float(timeout),
        maximum=maximum,
        attempts=3,
        request_maximum=request_maximum,
    )
    projected = _bounded_json(raw, key, maximum=maximum)
    if not isinstance(projected, dict):
        raise ValueError("SenseNova response must be a JSON object")
    return projected


def _luma_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_secret(api_key, 'Luma API key')}"}


def _luma_post(api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request_json(
        "POST", _LUMA_BASE + path,
        headers=_luma_headers(api_key), payload=payload)


def _luma_get(api_key: str, generation_id: str) -> dict[str, Any]:
    return _request_json(
        "GET", _LUMA_BASE + "/" + urllib.parse.quote(
            _identifier(generation_id, "generation id"), safe=""),
        headers=_luma_headers(api_key))


def _generation_id(response: dict[str, Any]) -> str:
    return _identifier(response.get("id"), "provider generation id")


def _poll_generation(api_key: str, generation_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while True:
        generation = _luma_get(api_key, generation_id)
        state = generation.get("state")
        if state == "completed":
            return generation
        if state == "failed":
            reason = generation.get("failure_reason")
            if isinstance(reason, str):
                reason = reason[:512].replace("\x00", "")
            else:
                reason = "unspecified provider failure"
            raise RuntimeError(f"Luma generation failed: {reason}")
        if state not in {"queued", "dreaming", "processing", "pending"}:
            raise ValueError("Luma returned an unknown generation state")
        if time.monotonic() >= deadline:
            raise TimeoutError("Luma generation exceeded the 900 second deadline")
        time.sleep(_POLL_INTERVAL_SECONDS)


def _asset_url(generation: dict[str, Any], kind: str) -> str:
    assets = generation.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("completed Luma generation has no assets object")
    return _public_https_url(assets.get(kind), f"Luma {kind} asset")


def _download_bytes(url: str, *, maximum: int) -> bytes:
    url = _public_https_url(url, "provider media URL")
    opener = urllib.request.build_opener(_PublicHttpsRedirect())
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with opener.open(request, timeout=60.0) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > maximum:
                raise ValueError("provider media exceeds its byte limit")
            data = response.read(maximum + 1)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise _safe_provider_error(error) from None
    if len(data) > maximum:
        raise ValueError("provider media exceeds its byte limit")
    return data


def _output_video_target(filename: str, generation_id: str) -> tuple[str, str]:
    import folder_paths

    logical = _text(filename, "output filename", maximum=1024).replace("\\", "/")
    if logical:
        root, extension = os.path.splitext(logical)
        logical = root if extension else logical
        if logical.endswith("/"):
            logical += generation_id
    else:
        logical = generation_id
    logical += ".mp4"
    parts = logical.split("/")
    if (
        logical.startswith("/") or len(parts) > 32
        or any(part in {"", ".", ".."} for part in parts)
        or any(len(part.encode("utf-8")) > 255 for part in parts)
    ):
        raise ValueError("output filename must be a confined relative name")
    output_root = os.path.realpath(folder_paths.get_output_directory())
    parent = os.path.realpath(os.path.join(output_root, *parts[:-1]))
    if os.path.commonpath((output_root, parent)) != output_root:
        raise ValueError("output filename escapes the output directory")
    os.makedirs(parent, exist_ok=True)
    parent = os.path.realpath(os.path.join(output_root, *parts[:-1]))
    target = os.path.realpath(os.path.join(parent, parts[-1]))
    if os.path.commonpath((output_root, target)) != output_root:
        raise ValueError("output filename escapes the output directory")
    return target, logical


def _save_video(url: str, filename: str, generation_id: str) -> str:
    target, logical = _output_video_target(filename, generation_id)
    if os.path.lexists(target):
        raise FileExistsError(f"output video already exists: {logical!r}")
    parent = os.path.dirname(target)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".comfy-luma-", suffix=".tmp", dir=parent)
    written = 0
    try:
        url = _public_https_url(url, "Luma video asset")
        opener = urllib.request.build_opener(_PublicHttpsRedirect())
        request = urllib.request.Request(url, headers={"Accept": "video/mp4,*/*"})
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                with opener.open(request, timeout=60.0) as response:
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) > _VIDEO_BYTES_MAX:
                        raise ValueError("Luma video exceeds the 1 GiB limit")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > _VIDEO_BYTES_MAX:
                            raise ValueError("Luma video exceeds the 1 GiB limit")
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise _safe_provider_error(error) from None
        if written == 0:
            raise ValueError("Luma returned an empty video")
        os.link(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return logical


def _references(value: Any, name: str, *, maximum: int = 4) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{name} must contain between 1 and {maximum} references")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"url", "weight"}:
            raise ValueError(f"{name} entries must contain only url and weight")
        weight = item["weight"]
        if isinstance(weight, bool) or type(weight) not in {int, float}:
            raise TypeError(f"{name} weight must be numeric")
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"{name} weight must be in [0, 1]")
        result.append({"url": _https_url(item["url"], f"{name} URL"), "weight": weight})
    return result


def _keyframes(value: Any) -> Optional[dict[str, dict[str, str]]]:
    if value is None:
        return None
    if not isinstance(value, dict) or not 1 <= len(value) <= 2:
        raise ValueError("Luma keyframes must contain one or two frames")
    result: dict[str, dict[str, str]] = {}
    for frame, item in value.items():
        if frame not in {"frame0", "frame1"} or not isinstance(item, dict):
            raise ValueError("Luma keyframes must use frame0 or frame1")
        if item.get("type") == "image" and set(item) == {"type", "url"}:
            result[frame] = {"type": "image", "url": _https_url(
                item["url"], "keyframe image URL")}
        elif item.get("type") == "generation" and set(item) == {"type", "id"}:
            result[frame] = {"type": "generation", "id": _identifier(
                item["id"], "keyframe generation id")}
        else:
            raise ValueError("Luma keyframe has an unsupported shape")
    return result


@dataclass(frozen=True)
class InProcessLuma:
    async def create_video(
        self, api_key: str, prompt: str, model: str, *,
        loop: bool = False, aspect_ratio: Optional[str] = None,
        duration: Optional[str] = None, resolution: str = "720p",
        keyframes: Optional[dict[str, Any]] = None,
        save: bool = True, filename: str = "",
    ) -> dict[str, Any]:
        if type(loop) is not bool or type(save) is not bool:
            raise TypeError("loop and save must be booleans")
        payload: dict[str, Any] = {
            "generation_type": "video",
            "prompt": _text(prompt, "prompt", allow_empty=True),
            "model": _choice(model, "model", {"ray-flash-2", "ray-2", "ray-1.6"}),
            "loop": loop,
            "resolution": _choice(resolution, "resolution", {"540p", "720p"}),
        }
        if aspect_ratio is not None:
            payload["aspect_ratio"] = _choice(
                aspect_ratio, "aspect ratio",
                {"9:16", "3:4", "1:1", "4:3", "16:9", "21:9"})
        if duration is not None:
            payload["duration"] = _choice(duration, "duration", {"5s", "9s"})
        frames = _keyframes(keyframes)
        if frames is not None:
            payload["keyframes"] = frames
        created = await asyncio.to_thread(_luma_post, api_key, "/video", payload)
        generation_id = _generation_id(created)
        completed = await asyncio.to_thread(_poll_generation, api_key, generation_id)
        url = _asset_url(completed, "video")
        saved = await asyncio.to_thread(
            _save_video, url, filename, generation_id) if save else None
        return {"generation_id": generation_id, "url": url, "saved": saved}

    async def upscale_video(
        self, api_key: str, generation_id: str, resolution: str, *,
        save: bool = True, filename: str = "",
    ) -> dict[str, Any]:
        if type(save) is not bool:
            raise TypeError("save must be a boolean")
        source_id = _identifier(generation_id, "generation id")
        resolution = _choice(
            resolution, "resolution", {"540p", "720p", "1080p", "4k"})
        created = await asyncio.to_thread(
            _luma_post, api_key,
            "/" + urllib.parse.quote(source_id, safe="") + "/upscale",
            {"generation_type": "upscale_video", "resolution": resolution})
        result_id = _generation_id(created)
        completed = await asyncio.to_thread(_poll_generation, api_key, result_id)
        url = _asset_url(completed, "video")
        saved = await asyncio.to_thread(
            _save_video, url, filename, result_id) if save else None
        return {"generation_id": result_id, "url": url, "saved": saved}

    async def add_audio(
        self, api_key: str, generation_id: str, prompt: str,
        negative_prompt: str, *, save: bool = True, filename: str = "",
    ) -> dict[str, Any]:
        if type(save) is not bool:
            raise TypeError("save must be a boolean")
        source_id = _identifier(generation_id, "generation id")
        created = await asyncio.to_thread(
            _luma_post, api_key,
            "/" + urllib.parse.quote(source_id, safe="") + "/audio",
            {
                "generation_type": "add_audio",
                "prompt": _text(prompt, "audio prompt"),
                "negative_prompt": _text(
                    negative_prompt, "negative audio prompt"),
            })
        result_id = _generation_id(created)
        completed = await asyncio.to_thread(_poll_generation, api_key, result_id)
        url = _asset_url(completed, "video")
        saved = await asyncio.to_thread(
            _save_video, url, filename, result_id) if save else None
        return {"generation_id": result_id, "url": url, "saved": saved}

    async def create_image(
        self, api_key: str, prompt: str, model: str, *,
        aspect_ratio: str = "1:1",
        image_ref: Optional[list[dict[str, Any]]] = None,
        style_ref: Optional[list[dict[str, Any]]] = None,
        character_ref: Optional[dict[str, Any]] = None,
        modify_image_ref: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generation_type": "image",
            "prompt": _text(prompt, "prompt", allow_empty=False),
            "model": _choice(model, "model", {"photon-1", "photon-flash-1"}),
            "aspect_ratio": _choice(
                aspect_ratio, "aspect ratio",
                {"9:16", "3:4", "1:1", "4:3", "16:9", "21:9"}),
        }
        if image_ref is not None:
            payload["image_ref"] = _references(image_ref, "image reference")
        if style_ref is not None:
            payload["style_ref"] = _references(style_ref, "style reference", maximum=1)
        if modify_image_ref is not None:
            payload["modify_image_ref"] = _references(
                [modify_image_ref], "modify image reference", maximum=1)[0]
        if character_ref is not None:
            if (
                not isinstance(character_ref, dict)
                or set(character_ref) != {"identity0"}
                or not isinstance(character_ref["identity0"], dict)
                or set(character_ref["identity0"]) != {"images"}
            ):
                raise ValueError("character reference has an unsupported shape")
            images = character_ref["identity0"]["images"]
            if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 4:
                raise ValueError("character reference must contain 1 to 4 images")
            payload["character_ref"] = {"identity0": {"images": [
                _https_url(url, "character image URL") for url in images
            ]}}
        created = await asyncio.to_thread(_luma_post, api_key, "/image", payload)
        generation_id = _generation_id(created)
        completed = await asyncio.to_thread(_poll_generation, api_key, generation_id)
        url = _asset_url(completed, "image")
        encoded = await asyncio.to_thread(
            _download_bytes, url, maximum=_IMAGE_BYTES_MAX)
        from PIL import Image
        import numpy as np
        import torch
        from ._sdk import ImageRef

        try:
            with Image.open(io.BytesIO(encoded)) as image:
                image.load()
                width, height = image.size
                if width < 1 or height < 1 or width * height > _IMAGE_PIXELS_MAX:
                    raise ValueError("Luma image exceeds the decoded pixel limit")
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        except (ValueError, Image.DecompressionBombError):
            raise
        except Exception as error:
            raise ValueError("Luma returned an invalid image") from error
        tensor = torch.from_numpy(array).unsqueeze(0)
        image_ref_value = await ImageRef._from_raw(tensor)
        return {
            "generation_id": generation_id,
            "url": url,
            "image": image_ref_value,
        }


@dataclass(frozen=True)
class InProcessImgBB:
    async def upload(
        self, api_key: str, image: Any, *,
        expiration_seconds: Optional[int] = None,
    ) -> str:
        from PIL import Image
        import numpy as np
        from ._sdk import current_runtime

        api_key = _secret(api_key, "ImgBB API key")
        value = await current_runtime().refs.resolve(image)
        if not hasattr(value, "ndim") or int(value.ndim) != 4 or len(value) < 1:
            raise TypeError("ImgBB upload requires a non-empty IMAGE batch")
        first = value[0]
        if int(first.shape[-1]) not in {1, 3, 4}:
            raise ValueError("ImgBB image must have 1, 3, or 4 channels")
        array = np.clip(
            first.detach().cpu().float().numpy() * 255.0, 0, 255
        ).astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[..., 0]
        rendered = Image.fromarray(array)
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG", optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) > _IMAGE_BYTES_MAX:
            raise ValueError("ImgBB PNG exceeds the 32 MiB upload limit")
        query = {"key": api_key}
        if expiration_seconds is not None:
            if isinstance(expiration_seconds, bool) or not isinstance(
                    expiration_seconds, int):
                raise TypeError("ImgBB expiration must be an integer")
            if not 60 <= expiration_seconds <= 15_552_000:
                raise ValueError("ImgBB expiration must be in [60, 15552000]")
            query["expiration"] = str(expiration_seconds)
        url = _IMGBB_UPLOAD + "?" + urllib.parse.urlencode(query)
        response = await asyncio.to_thread(
            _request_json, "POST", url,
            form={"image": base64.b64encode(encoded).decode("ascii")})
        if response.get("success") is not True:
            raise RuntimeError("ImgBB rejected the image upload")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("ImgBB response has no data object")
        return _public_https_url(data.get("url"), "ImgBB image URL")


_SENSENOVA_CHAT_MODELS = {
    "sensenova-6.7-flash-lite",
    "deepseek-v4",
}
_SENSENOVA_VISION_MODELS = {"sensenova-6.7-flash-lite"}
_SENSENOVA_IMAGE_MODELS = {"sensenova-u1-fast"}
_SENSENOVA_IMAGE_SIZES = {
    "2752x1536", "1536x2752", "2048x2048", "2496x1664",
    "1664x2496", "2368x1760", "1760x2368", "2272x1824",
    "1824x2272", "3072x1376", "1344x3136",
}


def _sensenova_timeout(value: Any, *, image: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("SenseNova timeout must be an integer")
    lower = 30 if image else 10
    upper = 900 if image else 600
    if not lower <= value <= upper:
        raise ValueError(f"SenseNova timeout must be in [{lower}, {upper}]")
    return value


def _sensenova_sampling(
    temperature: Any, top_p: Any, max_tokens: Any,
) -> tuple[float, float, int]:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("SenseNova temperature must be a number")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("SenseNova top_p must be a number")
    temperature = float(temperature)
    top_p = float(top_p)
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise ValueError("SenseNova temperature must be in [0, 2]")
    if not math.isfinite(top_p) or not 0.0 <= top_p <= 1.0:
        raise ValueError("SenseNova top_p must be in [0, 1]")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("SenseNova max_tokens must be an integer")
    if not 1 <= max_tokens <= 65_536:
        raise ValueError("SenseNova max_tokens must be in [1, 65536]")
    return temperature, top_p, max_tokens


def _sensenova_url(value: Any) -> str:
    value = _text(
        value, "SenseNova vision URL", maximum=8192, allow_empty=False)
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "SenseNova vision URL must be credential-free HTTP(S); "
            "use Vision Image for inline image data")
    return value


async def _sensenova_text_refs(values: dict[str, str]) -> dict[str, Any]:
    from ._sdk import ValueRef

    result: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(value, str):
            raise TypeError(f"SenseNova {key} output must be text")
        result[key] = await ValueRef.from_value(value)
    return result


def _sensenova_chat_payload(
    *, text: str, system_prompt: str, model: str, temperature: float,
    top_p: float, max_tokens: int,
) -> dict[str, Any]:
    temperature, top_p, max_tokens = _sensenova_sampling(
        temperature, top_p, max_tokens)
    return {
        "model": _choice(model, "SenseNova chat model", _SENSENOVA_CHAT_MODELS),
        "messages": [
            {"role": "system", "content": _text(
                system_prompt, "SenseNova system prompt", maximum=256 * 1024)},
            {"role": "user", "content": _text(
                text, "SenseNova chat text", maximum=256 * 1024,
                allow_empty=False)},
        ],
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def _sensenova_chat_strings(raw: dict[str, Any]) -> dict[str, str]:
    try:
        text = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "SenseNova chat response has no choices[0].message.content") from error
    if not isinstance(text, str):
        raise ValueError("SenseNova chat content must be text")
    usage = raw.get("usage", {})
    return {
        "text": text,
        "usage_json": json.dumps(usage, ensure_ascii=False),
        "raw_json": json.dumps(raw, ensure_ascii=False),
    }


def _sensenova_png_data_url(value: Any) -> str:
    from PIL import Image
    import numpy as np

    if not hasattr(value, "ndim") or int(value.ndim) != 4 or len(value) < 1:
        raise TypeError("SenseNova vision image requires a non-empty IMAGE batch")
    first = value[0]
    if int(first.shape[-1]) not in {3, 4}:
        raise ValueError("SenseNova vision image must have 3 or 4 channels")
    height, width = int(first.shape[0]), int(first.shape[1])
    if height < 1 or width < 1 or height * width > _IMAGE_PIXELS_MAX:
        raise ValueError("SenseNova vision image exceeds the pixel limit")
    array = np.asarray(first.detach().cpu().float().numpy())
    array = np.clip(array, 0.0, 1.0)
    array = np.rint(array * 255.0).astype(np.uint8)
    rendered = Image.fromarray(array).convert("RGB")
    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    if len(encoded) > _IMAGE_BYTES_MAX:
        raise ValueError("SenseNova vision PNG exceeds the 32 MiB limit")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _sensenova_decode_image(encoded: bytes):
    from PIL import Image
    import numpy as np
    import torch

    if not encoded or len(encoded) > _SENSENOVA_IMAGE_BYTES_MAX:
        raise ValueError("SenseNova image exceeds the 50 MiB limit")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            width, height = image.size
            if width < 1 or height < 1 or width * height > _IMAGE_PIXELS_MAX:
                raise ValueError("SenseNova image exceeds the decoded pixel limit")
            array = np.array(
                image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    except (ValueError, Image.DecompressionBombError):
        raise
    except Exception as error:
        raise ValueError("SenseNova returned an invalid image") from error
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).contiguous().float()


def _sensenova_image_info(image: Any) -> str:
    shape = tuple(image.shape)
    value_range = f"{float(image.min()):.6f}..{float(image.max()):.6f}"
    return (
        f"shape={shape}; dtype={image.dtype}; device={image.device}; "
        f"contiguous={image.is_contiguous()}; range={value_range}"
    )


@dataclass(frozen=True)
class InProcessSenseNova:
    """Fixed SenseNova U1 cloud operations documented by D33."""

    async def chat(
        self, text: str, system_prompt: str, model: str, *,
        temperature: float = 0.7, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        payload = _sensenova_chat_payload(
            text=text, system_prompt=system_prompt, model=model,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        timeout = _sensenova_timeout(timeout_seconds)
        raw = await asyncio.to_thread(
            _sensenova_post, "/chat/completions", payload, timeout=timeout)
        return await _sensenova_text_refs(_sensenova_chat_strings(raw))

    async def vision_url(
        self, image_url: str, prompt: str, system_prompt: str, model: str, *,
        temperature: float = 0.2, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
        _allow_data_url: bool = False,
    ) -> dict[str, Any]:
        temperature, top_p, max_tokens = _sensenova_sampling(
            temperature, top_p, max_tokens)
        if _allow_data_url and not image_url.startswith(
                "data:image/png;base64,"):
            raise ValueError("internal SenseNova image must be a PNG data URL")
        payload = {
            "model": _choice(
                model, "SenseNova vision model", _SENSENOVA_VISION_MODELS),
            "messages": [
                {"role": "system", "content": _text(
                    system_prompt, "SenseNova system prompt", maximum=256 * 1024)},
                {"role": "user", "content": [
                    {"type": "text", "text": _text(
                        prompt, "SenseNova vision prompt", maximum=256 * 1024,
                        allow_empty=False)},
                    {"type": "image_url", "image_url": {"url": (
                        _text(
                            image_url, "SenseNova encoded image",
                            maximum=48 * 1024 * 1024, allow_empty=False)
                        if _allow_data_url else _sensenova_url(image_url)
                    )}},
                ]},
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        timeout = _sensenova_timeout(timeout_seconds)
        raw = await asyncio.to_thread(
            _sensenova_post, "/chat/completions", payload, timeout=timeout,
            request_maximum=(49 * 1024 * 1024 if _allow_data_url else 1024 * 1024))
        return await _sensenova_text_refs(_sensenova_chat_strings(raw))

    async def vision_image(
        self, image: Any, prompt: str, system_prompt: str, model: str, *,
        temperature: float = 0.2, top_p: float = 1.0,
        max_tokens: int = 2048, timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        from ._sdk import current_runtime

        value = await current_runtime().refs.resolve(image)
        return await self.vision_url(
            _sensenova_png_data_url(value), prompt, system_prompt, model,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            _allow_data_url=True,
        )

    async def generate_image(
        self, prompt: str, model: str, size: str, *,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        prompt = _text(
            prompt, "SenseNova image prompt", maximum=256 * 1024,
            allow_empty=False)
        model = _choice(
            model, "SenseNova image model", _SENSENOVA_IMAGE_MODELS)
        size = str(size).split("|", 1)[0].strip()
        size = _choice(size, "SenseNova image size", _SENSENOVA_IMAGE_SIZES)
        timeout = _sensenova_timeout(timeout_seconds, image=True)
        raw = await asyncio.to_thread(
            _sensenova_post, "/images/generations",
            {"model": model, "prompt": prompt, "size": size, "n": 1},
            timeout=timeout, maximum=_SENSENOVA_IMAGE_JSON_MAX)
        try:
            first = raw["data"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("SenseNova image response has no data[0]") from error
        if not isinstance(first, dict):
            raise ValueError("SenseNova image data[0] must be an object")
        image_base64 = first.get("b64_json") or first.get("base64") or first.get(
            "image_base64") or ""
        image_url = first.get("url") or ""
        if not isinstance(image_base64, str) or not isinstance(image_url, str):
            raise ValueError("SenseNova image payload must contain text fields")
        if image_base64:
            encoded_text = image_base64.split(",", 1)[1] if (
                image_base64.startswith("data:") and "," in image_base64
            ) else image_base64
            try:
                encoded = base64.b64decode(encoded_text, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError("SenseNova image contains invalid base64") from error
            if image_url:
                image_url = _public_https_url(image_url, "SenseNova image URL")
        elif image_url:
            image_url = _public_https_url(image_url, "SenseNova image URL")
            encoded = await asyncio.to_thread(
                _download_bytes, image_url, maximum=_SENSENOVA_IMAGE_BYTES_MAX)
        else:
            raise ValueError("SenseNova image response has no image payload")
        tensor = _sensenova_decode_image(encoded)
        from ._sdk import ImageRef

        image_ref = await ImageRef._from_raw(tensor)
        strings = await _sensenova_text_refs({
            "image_base64": image_base64,
            "image_url": image_url,
            "raw_json": json.dumps(raw, ensure_ascii=False),
            "image_info": _sensenova_image_info(tensor),
        })
        return {"image": image_ref, **strings}


__all__ = ["InProcessImgBB", "InProcessLuma", "InProcessSenseNova"]
