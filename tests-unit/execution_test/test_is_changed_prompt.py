import asyncio

from comfy.cli_args import args
args.cpu = True

import nodes

from comfy_execution.caching import BasicCache, CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt
from execution import IsChangedCache


class ChangingSource:
    version = "v1"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls):
        return cls.version


def test_resubmitted_prompt_rechecks_source_and_misses_stale_cache(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "ChangingSource", ChangingSource)
    monkeypatch.setattr(ChangingSource, "version", "v1")
    cache = BasicCache(CacheKeySetInputSignature)
    prompt = {"1": {"class_type": "ChangingSource", "inputs": {}}}

    async def set_prompt(submitted):
        dynamic = DynamicPrompt(submitted)
        changed = IsChangedCache("prompt", dynamic, cache)
        await cache.set_prompt(dynamic, submitted.keys(), changed)

    asyncio.run(set_prompt(prompt))
    cache.set_local("1", "v1 result")

    monkeypatch.setattr(ChangingSource, "version", "v2")
    resubmitted = {"1": {"class_type": "ChangingSource", "inputs": {}, "is_changed": ["v1"]}}
    asyncio.run(set_prompt(resubmitted))
    assert cache.get_local("1") is None
    assert "is_changed" not in prompt["1"]
