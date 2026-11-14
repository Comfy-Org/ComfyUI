from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import ComfyAPI_latest, io

CacheProvider = ComfyAPI_latest.Caching.CacheProvider
CacheContext = ComfyAPI_latest.Caching.CacheContext
CacheValue = ComfyAPI_latest.Caching.CacheValue


class NodeReplacement(ProxiedSingleton):
    async def register(self, node_replace: io.NodeReplace) -> None:
        """Register a mapping from a removed node type to its replacement.

        ComfyUI applies it to queued prompts and to `eng.node.run` only while no node is registered
        under the old id and one is under the new id. The first mapping registered for an old id is
        the one applied; registering the same pair again does nothing. It stays until ComfyUI
        restarts, also if the pack that registered it is disabled.
        """
        from server import PromptServer

        PromptServer.instance.node_replace_manager.register(node_replace)


class Caching(ProxiedSingleton):
    async def register_provider(self, provider: CacheProvider) -> None:
        """Register an external cache provider; does nothing if `provider` is already registered.

        On a local cache miss providers are asked in registration order and the first value one
        returns is used. Every provider is offered each result ComfyUI stores under a key it can
        compare and serialise: not the result of a node whose `fingerprint_inputs` returns NaN, nor
        of the nodes after it. It stays until unregistered or until ComfyUI restarts, also if the
        pack that registered it is disabled.
        """
        from comfy_execution.cache_provider import register_cache_provider

        register_cache_provider(provider)

    async def unregister_provider(self, provider: CacheProvider) -> None:
        """Stop asking `provider` from now on; does nothing if it is not registered.

        Stores already handed to it still finish.
        """
        from comfy_execution.cache_provider import unregister_cache_provider

        unregister_cache_provider(provider)
