import inspect
from types import SimpleNamespace

import pytest

from app.node_replace_manager import NodeReplaceManager
from comfy_api import v4
from comfy_api.latest import ComfyAPI_latest
from server import PromptServer


async def finish(result):
    return await result if inspect.isawaitable(result) else result


def test_new_v3_service_members_do_not_enter_the_v4_contract(monkeypatch):
    monkeypatch.setattr(ComfyAPI_latest, "new_service", object(), raising=False)
    monkeypatch.setattr(
        ComfyAPI_latest.Execution, "new_call", lambda self: "new", raising=False
    )

    assert not hasattr(v4.eng, "new_service")
    assert not hasattr(v4.eng.execution, "new_call")
    assert not hasattr(v4.ComfyAPISync().execution, "new_call")


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_a_registered_node_replacement_is_applied_to_prompts(
    monkeypatch, synchronous
):
    manager = NodeReplaceManager()
    monkeypatch.setattr(
        PromptServer,
        "instance",
        SimpleNamespace(node_replace_manager=manager),
        raising=False,
    )
    replacement = v4.io.NodeReplace("NewNode", "OldNode")

    if synchronous:
        v4.ComfyAPISync().node_replacement.register(replacement)
    else:
        await v4.eng.node_replacement.register(replacement)

    assert manager.get_replacement("OldNode") == [replacement]


def test_a_change_to_a_v3_service_call_does_not_change_the_v4_call(monkeypatch):
    async def changed(self, node_replace):
        raise AssertionError("V4 must not run V3's code")

    manager = NodeReplaceManager()
    monkeypatch.setattr(
        PromptServer,
        "instance",
        SimpleNamespace(node_replace_manager=manager),
        raising=False,
    )
    monkeypatch.setattr(ComfyAPI_latest.NodeReplacement, "register", changed)

    v4.ComfyAPISync().node_replacement.register(v4.io.NodeReplace("NewNode", "OldNode"))

    assert manager.has_replacement("OldNode")


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_v4_cache_registration_is_visible_to_the_existing_host(synchronous):
    from comfy_execution.cache_provider import (
        _get_cache_providers,
        unregister_cache_provider,
    )

    caching = v4.ComfyAPISync().caching if synchronous else v4.eng.caching

    class Provider(v4.CacheProvider):
        async def on_lookup(self, context):
            return None

        async def on_store(self, context, value):
            pass

    provider = Provider()
    before = _get_cache_providers()
    try:
        await finish(caching.register_provider(provider))
        assert _get_cache_providers() == (*before, provider)
        await finish(caching.unregister_provider(provider))
        assert _get_cache_providers() == before
    finally:
        if provider in _get_cache_providers():
            unregister_cache_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_the_first_replacement_applies_only_to_ids_no_node_has_and_repeats_are_ignored(
    monkeypatch, synchronous
):
    manager = NodeReplaceManager()
    monkeypatch.setattr(
        PromptServer,
        "instance",
        SimpleNamespace(node_replace_manager=manager),
        raising=False,
    )
    replacements = (
        v4.ComfyAPISync().node_replacement if synchronous else v4.eng.node_replacement
    )
    first = v4.io.NodeReplace("EmptyLatentImage", "OldNode")

    for replacement in (
        first,
        v4.io.NodeReplace("VAEDecode", "OldNode"),
        v4.io.NodeReplace("EmptyLatentImage", "OldNode"),
        v4.io.NodeReplace("EmptyLatentImage", "VAEDecode"),
    ):
        await finish(replacements.register(replacement))
    prompt = {
        "1": {"class_type": "OldNode", "inputs": {}},
        "2": {"class_type": "VAEDecode", "inputs": {}},
    }
    manager.apply_replacements(prompt)

    assert manager.get_replacement("OldNode")[0] is first
    assert len(manager.get_replacement("OldNode")) == 2
    assert [prompt[n]["class_type"] for n in "12"] == ["EmptyLatentImage", "VAEDecode"]


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_registering_a_provider_again_or_removing_an_unknown_one_does_nothing(
    synchronous,
):
    from comfy_execution.cache_provider import (
        _get_cache_providers,
        unregister_cache_provider,
    )

    caching = v4.ComfyAPISync().caching if synchronous else v4.eng.caching

    class Provider(v4.CacheProvider):
        async def on_lookup(self, context):
            return None

        async def on_store(self, context, value):
            pass

    provider = Provider()
    before = _get_cache_providers()
    try:
        await finish(caching.unregister_provider(provider))
        await finish(caching.register_provider(provider))
        await finish(caching.register_provider(provider))
        assert _get_cache_providers() == (*before, provider)
    finally:
        if provider in _get_cache_providers():
            unregister_cache_provider(provider)
