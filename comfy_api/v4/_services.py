import threading
from dataclasses import dataclass
from functools import partial

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import ComfyAPI_latest, io
from comfy_execution.cache_provider import (
    _get_cache_providers,
    register_cache_provider,
    unregister_cache_provider,
)

CacheProvider = ComfyAPI_latest.Caching.CacheProvider
CacheContext = ComfyAPI_latest.Caching.CacheContext
CacheValue = ComfyAPI_latest.Caching.CacheValue


class NodeReplacement(ProxiedSingleton):
    async def register(self, node_replace: io.NodeReplace) -> None:
        """Register a mapping from a removed node type to its replacement.

        ComfyUI applies it to queued prompts only while no node is registered under the old id
        and one is under the new id. The first mapping registered for an old id is
        the one applied; registering the same pair again does nothing. A mapping registered
        as its pack loads, in its import or `on_load` or in a task or `asyncio.to_thread` call
        they start, is removed if ComfyUI disables the pack or takes it away. Any other stays
        until ComfyUI restarts: one registered from a node or a route, or on a thread started
        another way, which ComfyUI cannot tell is the pack's.
        """
        from server import PromptServer

        from . import _packs

        manager = PromptServer.instance.node_replace_manager
        if manager.register(node_replace):
            _packs._registered(partial(manager.unregister, node_replace))


@dataclass(eq=False)
class _Use:
    """One registration of a provider made here."""

    provider: CacheProvider


# The registrations made here that have not been taken back or unregistered: ComfyUI keeps one
# provider for all who registered it, or one equal to it, until the last has gone.
_provider_uses: list[_Use] = []
_provider_lock = threading.Lock()


def _used(provider: CacheProvider) -> bool:
    return any(use.provider == provider for use in _provider_uses)


def _stop_using(use: _Use) -> None:
    with _provider_lock:
        if use not in _provider_uses:
            return
        _provider_uses.remove(use)
        if not _used(use.provider):
            unregister_cache_provider(use.provider)


class Caching(ProxiedSingleton):
    async def register_provider(self, provider: CacheProvider) -> None:
        """Register an external cache provider; does nothing if `provider` is already registered.

        On a local cache miss providers are asked in registration order and the first value one
        returns is used. Every provider is offered each result ComfyUI stores under a key it can
        compare and serialise: not the result of a node whose `fingerprint_inputs` returns NaN, nor
        of the nodes after it, nor of a node another node expanded into. With `--cache-none`
        no provider is asked or offered anything. A provider registered as its pack loads, in
        its import or `on_load` or in a task or `asyncio.to_thread` call they start, is
        unregistered if ComfyUI disables the pack or takes it away, unless another pack
        registered it too. Any other stays until unregistered or until ComfyUI restarts: one
        registered from a node or a route, or on a thread started another way, which ComfyUI
        cannot tell is the pack's.
        """
        from . import _packs

        use = _Use(provider)
        with _provider_lock:
            # ComfyUI registers a provider once, and takes one equal to it for the same.
            if provider not in _get_cache_providers():
                register_cache_provider(provider)
            elif not _used(provider):
                # Registered with ComfyUI directly, so it is not this API's to take back.
                return
            _provider_uses.append(use)
        _packs._registered(partial(_stop_using, use))

    async def unregister_provider(self, provider: CacheProvider) -> None:
        """Stop asking `provider` from now on; does nothing if it is not registered.

        Stores already handed to it still finish.
        """
        with _provider_lock:
            _provider_uses[:] = [use for use in _provider_uses if use.provider != provider]
            unregister_cache_provider(provider)
