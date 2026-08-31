"""Closed llama.cpp vendor integration for Secure Nodes V2.

The guest sees managed GGUF names and an opaque model ref only.  Host paths,
chat-handler objects, model sessions, and encoded image bytes never cross the
boundary.
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass, field
import inspect
import io
import math
import os
import threading
from typing import Any


@dataclass
class _Entry:
    llm: Any
    handler: Any
    family: str
    lock: threading.Lock = field(default_factory=threading.Lock)


def _classes():
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format
    except ImportError as error:
        raise RuntimeError(
            "llama.cpp inference requires a host-managed llama-cpp-python "
            "build with Qwen vision support") from error
    return Llama, llama_chat_format


def _supported_kwargs(callable_value, values: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_value).parameters
    except (TypeError, ValueError):
        return values
    if any(item.kind == inspect.Parameter.VAR_KEYWORD
           for item in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _validate_gguf(path: str) -> None:
    # Reuse the host's closed GGUF header/count validation.  A catalogue name
    # is confinement, not proof that arbitrary local bytes are model weights.
    from ._sdk import _InProcessModels

    _InProcessModels._verify_weight_file(path, ".gguf")


def _load(
    model_path: str,
    mmproj_path: str | None,
    family: str,
    *,
    device: str,
    context_length: int,
    batch_size: int,
    gpu_layers: int,
    image_max_tokens: int,
    top_k: int,
    pool_size: int,
) -> _Entry:
    Llama, formats = _classes()
    handler = None
    if mmproj_path is not None:
        handler_name = (
            "Qwen3VLChatHandler"
            if family == "qwen3_vl" else "Qwen25VLChatHandler"
        )
        handler_class = getattr(formats, handler_name, None)
        if handler_class is None:
            raise RuntimeError(
                f"the host llama.cpp build lacks {handler_name}")
        handler_options = _supported_kwargs(handler_class.__init__, {
            "clip_model_path": mmproj_path,
            "image_max_tokens": image_max_tokens,
            "force_reasoning": False,
            "verbose": False,
        })
        handler = handler_class(**handler_options)

    import torch

    # Preserve the pack's placement intent without trusting a requested
    # accelerator that the host does not actually own.  Its legacy backend
    # offloaded layers only on CUDA; MPS and unavailable CUDA fell back to CPU.
    wants_cuda = device == "auto" or device.startswith("cuda")
    selected_gpu_layers = (
        gpu_layers if wants_cuda and torch.cuda.is_available() else 0)
    options = {
        "model_path": model_path,
        "n_ctx": context_length,
        "n_batch": batch_size,
        "n_gpu_layers": selected_gpu_layers,
        "swa_full": True,
        "verbose": False,
        "pool_size": pool_size,
        "top_k": top_k,
    }
    if handler is not None:
        options.update({
            "chat_handler": handler,
            "image_min_tokens": 1024,
            "image_max_tokens": image_max_tokens,
        })
    elif family == "qwen3":
        options["chat_format"] = "qwen"
    llm = Llama(**_supported_kwargs(Llama.__init__, options))
    return _Entry(llm=llm, handler=handler, family=family)


class _Cache:
    def __init__(self, maximum: int = 1):
        self.maximum = maximum
        self.entries: OrderedDict[tuple[Any, ...], _Entry] = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def _file(path: str | None):
        if path is None:
            return None
        status = os.stat(path)
        return (
            os.path.realpath(path), status.st_dev, status.st_ino,
            status.st_size, status.st_mtime_ns, status.st_ctime_ns,
        )

    def get(self, model_path, mmproj_path, family, options, cache):
        if not cache:
            return _load(
                model_path, mmproj_path, family, **options)
        key = (
            self._file(model_path), self._file(mmproj_path), family,
            tuple(sorted(options.items())),
        )
        with self.lock:
            entry = self.entries.pop(key, None)
            if entry is not None:
                self.entries[key] = entry
                return entry
            entry = _load(model_path, mmproj_path, family, **options)
            while len(self.entries) >= self.maximum:
                self.entries.popitem(last=False)
            self.entries[key] = entry
            return entry

    def clear(self):
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
        return count


_CACHE = _Cache()


class InProcessLlamaCpp:
    _MAX_TEXT = 4 * 1024 * 1024
    _MAX_PIXELS = 268_435_456
    _MAX_IMAGE_BYTES = 64 * 1024 * 1024

    @staticmethod
    def _text(value, field, maximum):
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"llama.cpp {field} must be a string")
        if len(value.encode("utf-8")) > maximum:
            raise ValueError(f"llama.cpp {field} exceeds its size limit")
        return value

    async def load_chat_model(
        self, model_weight: str, mmproj_weight: str | None = None, *,
        family: str = "qwen3_vl", device: str = "auto",
        context_length: int = 8192, batch_size: int = 512,
        gpu_layers: int = -1, image_max_tokens: int = 4096,
        top_k: int = 0, pool_size: int = 4_194_304,
        cache: bool = True,
    ):
        from ._sdk import LlamaCppModelRef, current_runtime
        import folder_paths

        if family not in {"qwen3_vl", "qwen2_5_vl", "qwen3"}:
            raise ValueError("unknown llama.cpp Qwen family")
        if device not in {"auto", "cpu", "mps", "cuda"} and not (
            isinstance(device, str) and device.startswith("cuda:")
            and device[5:].isdigit()
        ):
            raise ValueError("invalid llama.cpp device")
        bounds = {
            "context_length": (context_length, 1024, 262144),
            "batch_size": (batch_size, 64, 32768),
            "gpu_layers": (gpu_layers, -1, 200),
            "image_max_tokens": (image_max_tokens, 256, 1_024_000),
            "top_k": (top_k, 0, 32768),
            "pool_size": (pool_size, 1_048_576, 10_485_760),
        }
        checked = {}
        for name, (value, minimum, maximum) in bounds.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"llama.cpp {name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"llama.cpp {name} is outside its bounds")
            checked[name] = value
        if type(cache) is not bool:
            raise TypeError("llama.cpp cache must be a boolean")
        if (not isinstance(model_weight, str)
                or not model_weight.lower().endswith(".gguf")):
            raise ValueError("llama.cpp model weight must be managed GGUF")
        model_path = folder_paths.get_full_path_or_raise(
            "text_encoders", model_weight)
        _validate_gguf(model_path)
        mmproj_path = None
        if mmproj_weight is not None:
            if (not isinstance(mmproj_weight, str)
                    or not mmproj_weight.lower().endswith(".gguf")):
                raise ValueError("llama.cpp projector weight must be managed GGUF")
            mmproj_path = folder_paths.get_full_path_or_raise(
                "text_encoders", mmproj_weight)
            _validate_gguf(mmproj_path)
        if family == "qwen3" and mmproj_path is not None:
            raise ValueError("text-only Qwen must not receive an mmproj")
        if family != "qwen3" and mmproj_path is None:
            raise ValueError("Qwen vision models require an mmproj")

        options = {
            "device": device,
            **checked,
        }
        entry = await asyncio.to_thread(
            _CACHE.get,
            model_path,
            mmproj_path,
            family,
            options,
            cache,
        )
        return LlamaCppModelRef._wrap(
            await current_runtime().refs.create("LLAMA_CPP_MODEL", entry))

    async def _images(self, image, video):
        from ._sdk import ImageRef, current_runtime
        import torch
        from PIL import Image

        batches = []
        total_pixels = 0
        for name, value, maximum in (
            ("image", image, 1), ("video", video, 64),
        ):
            if value is None:
                continue
            if not isinstance(value, ImageRef):
                raise TypeError(f"llama.cpp {name} must be an IMAGE ref")
            pixels = await current_runtime().refs.resolve(value)
            if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                    or not 1 <= int(pixels.shape[0]) <= maximum
                    or int(pixels.shape[-1]) < 3):
                raise ValueError(f"llama.cpp {name} has an invalid shape")
            if not torch.isfinite(pixels).all():
                raise ValueError(f"llama.cpp {name} contains non-finite pixels")
            batch, height, width = map(int, pixels.shape[:3])
            if height <= 0 or width <= 0:
                raise ValueError(f"llama.cpp {name} has an invalid shape")
            total_pixels += batch * height * width
            if total_pixels > self._MAX_PIXELS:
                raise ValueError("llama.cpp media exceeds the pixel limit")
            batches.append(pixels[..., :3])
        if not batches:
            return []
        result = []
        total = 0
        for pixels in batches:
            arrays = (pixels.detach().to("cpu").clamp(0, 1) * 255).to(
                torch.uint8).numpy()
            for array in arrays:
                output = io.BytesIO()
                Image.fromarray(array, mode="RGB").save(output, format="PNG")
                encoded = base64.b64encode(output.getvalue()).decode("ascii")
                total += len(encoded)
                if total > self._MAX_IMAGE_BYTES:
                    raise ValueError(
                        "llama.cpp encoded media exceeds the size limit")
                result.append(encoded)
        return result

    async def generate(
        self, model, system: str, prompt: str,
        image=None, video=None, max_tokens: int = 512,
        temperature: float = 0.7, top_p: float = 0.9,
        repetition_penalty: float = 1.0, seed: int = 1,
    ) -> str:
        from ._sdk import LlamaCppModelRef, current_runtime

        if not isinstance(model, LlamaCppModelRef):
            raise TypeError("llama.cpp model must be an opaque model ref")
        entry = await current_runtime().refs.resolve(model)
        if not isinstance(entry, _Entry):
            raise TypeError("invalid llama.cpp model ref")
        system = self._text(system, "system prompt", 1_048_576)
        prompt = self._text(prompt, "prompt", self._MAX_TEXT)
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or not 1 <= max_tokens <= 4096):
            raise ValueError("llama.cpp max_tokens must be in [1, 4096]")
        numeric = {
            "temperature": (temperature, 0.0, 2.0),
            "top_p": (top_p, 0.0, 1.0),
            "repetition_penalty": (repetition_penalty, 0.5, 2.0),
        }
        checked = {}
        for name, (value, minimum, maximum) in numeric.items():
            if isinstance(value, bool) or type(value) not in {int, float}:
                raise TypeError(f"llama.cpp {name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"llama.cpp {name} is outside its bounds")
            checked[name] = value
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed <= 0xFFFFFFFF):
            raise ValueError("llama.cpp seed must be a uint32")
        images = await self._images(image, video)
        if images and entry.handler is None:
            raise ValueError("text-only llama.cpp model cannot receive media")

        if images:
            content = [{"type": "text", "text": prompt}]
            content.extend({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            } for encoded in images)
        else:
            content = prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        def invoke():
            with entry.lock:
                return entry.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=checked["temperature"],
                    top_p=checked["top_p"],
                    repeat_penalty=checked["repetition_penalty"],
                    seed=seed,
                    stop=["<|im_end|>", "<|im_start|>"],
                )

        response = await asyncio.to_thread(invoke)
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("llama.cpp returned an invalid response") from error
        return self._text(text, "response", self._MAX_TEXT).strip()

    def clear(self):
        return _CACHE.clear()
