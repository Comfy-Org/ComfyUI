import asyncio
import base64
import io

import pytest
import torch
from PIL import Image

from comfy_api.latest._ollama import InProcessOllama
from comfy_api.latest._sdk import (
    ExecutionPlan,
    ImageRef,
    InProcessCtxProvider,
    InProcessOps,
    InProcessRefResolver,
    bind_runtime,
)


def _plan():
    return ExecutionPlan(
        prompt_id="ollama",
        node_id="1",
        node_type="ollama-test",
        prompt={"1": {"class_type": "ollama-test"}},
        extra_pnginfo={},
    )


def test_ollama_vendor_projection_is_closed_bounded_and_encodes_images(
    monkeypatch,
):
    calls = []

    def request(cls, origin, path, payload, timeout):
        calls.append((origin, path, payload, timeout))
        if path == "/api/tags":
            return {"models": [{"name": "vision:latest"}, {"model": "qwen"}]}
        if path == "/api/generate":
            return {
                "response": "generated",
                "thinking": "considered",
                "context": [2, 3, 5],
            }
        if path == "/api/chat":
            return {"message": {"content": "chatted", "thinking": "reasoned"}}
        raise AssertionError(path)

    monkeypatch.setattr(InProcessOllama, "_request_json", classmethod(request))
    monkeypatch.setenv(
        "COMFY_SECURE_OLLAMA_PROFILES",
        '{"studio":"https://ollama.example.test"}',
    )

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        integration = context.integrations.ollama
        with bind_runtime(refs, context, InProcessOps()):
            image = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((2, 3, 4, 3))))
            assert await integration.list_models(
                "http://127.0.0.1:11434") == ["vision:latest", "qwen"]
            generated = await integration.generate(
                endpoint="http://127.0.0.1:11434",
                model="vision:latest",
                system="be precise",
                prompt="describe",
                images=image,
                context=[1, 2],
                think=True,
                options={
                    "temperature": 0.4,
                    "top_k": 20,
                    "stop": "END",
                    "low_vram": True,
                    "main_gpu": 0,
                },
                keep_alive=7,
                keep_alive_unit="minutes",
                format={
                    "type": "object",
                    "properties": {"caption": {"type": "string"}},
                },
                timeout_seconds=42,
            )
            assert generated == {
                "response": "generated",
                "thinking": "considered",
                "context": [2, 3, 5],
            }
            chatted = await integration.chat(
                endpoint="ollama://studio",
                model="qwen",
                messages=[{"role": "user", "content": "hello"}],
                images=image,
                think=True,
                keep_alive=1,
                keep_alive_unit="hours",
            )
            assert chatted == {"response": "chatted", "thinking": "reasoned"}

            with pytest.raises(ValueError, match="loopback"):
                await integration.list_models("http://example.com:11434")
            with pytest.raises(ValueError, match="unsupported"):
                await integration.generate(
                    "http://127.0.0.1:11434", "qwen", "", "x",
                    options={"arbitrary": True})

    asyncio.run(run())

    assert calls[0][:2] == (
        "http://127.0.0.1:11434", "/api/tags")
    generate_payload = calls[1][2]
    assert generate_payload["stream"] is False
    assert generate_payload["think"] is True
    assert generate_payload["keep_alive"] == "7m"
    assert generate_payload["format"] == {
        "type": "object",
        "properties": {"caption": {"type": "string"}},
    }
    assert generate_payload["context"] == [1, 2]
    assert generate_payload["options"] == {
        "temperature": 0.4,
        "top_k": 20,
        "stop": "END",
        "low_vram": True,
        "main_gpu": 0,
    }
    assert calls[1][3] == 42.0
    assert len(generate_payload["images"]) == 2
    for encoded in generate_payload["images"]:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.size == (4, 3)
            assert decoded.mode == "RGB"

    assert calls[2][0] == "https://ollama.example.test"
    chat_payload = calls[2][2]
    assert chat_payload["keep_alive"] == "1h"
    assert chat_payload["think"] is True
    assert chat_payload["messages"][0]["role"] == "user"
    assert len(chat_payload["messages"][0]["images"]) == 2


def test_generic_llm_tool_chat_normalizes_messages_and_calls(monkeypatch):
    calls = []

    def request(cls, origin, path, payload, timeout):
        calls.append((origin, path, payload, timeout))
        return {
            "message": {
                "content": "",
                "thinking": "I should search",
                "tool_calls": [{
                    "function": {
                        "name": "search_internet",
                        "arguments": {"query": "current time"},
                    },
                }],
            },
        }

    monkeypatch.setattr(InProcessOllama, "_request_json", classmethod(request))

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            result = await context.integrations.llm.chat(
                provider="ollama",
                profile="http://127.0.0.1:11434",
                model="qwen",
                messages=[
                    {"role": "system", "content": "Use tools."},
                    {"role": "user", "content": "What time is it?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "name": "search_internet",
                            "arguments": {"query": "time"},
                        }],
                    },
                    {
                        "role": "tool",
                        "name": "search_internet",
                        "content": "12:34",
                    },
                ],
                tools=[{
                    "name": "search_internet",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }],
                temperature=0.2,
                max_tokens=2048,
                thinking=True,
                response_format="json",
                timeout_seconds=120,
                vendor_options={
                    "ollama": {
                        "keep_alive": 2,
                        "keep_alive_unit": "hours",
                    },
                },
            )
            assert result == {
                "content": "",
                "thinking": "I should search",
                "tool_calls": [{
                    "name": "search_internet",
                    "arguments": {"query": "current time"},
                }],
            }

            with pytest.raises(ValueError, match="provider"):
                await context.integrations.llm.chat(
                    "unknown", "profile", "model",
                    [{"role": "user", "content": "x"}],
                )

    asyncio.run(run())

    _, path, payload, timeout = calls[0]
    assert path == "/api/chat"
    assert timeout == 120.0
    assert payload["options"] == {
        "temperature": 0.2,
        "num_predict": 2048,
    }
    assert payload["keep_alive"] == "2h"
    assert payload["format"] == "json"
    assert payload["messages"][2]["tool_calls"] == [{
        "function": {
            "name": "search_internet",
            "arguments": {"query": "time"},
        },
    }]
    assert payload["messages"][3] == {
        "role": "tool",
        "tool_name": "search_internet",
        "content": "12:34",
    }
    assert payload["tools"] == [{
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
