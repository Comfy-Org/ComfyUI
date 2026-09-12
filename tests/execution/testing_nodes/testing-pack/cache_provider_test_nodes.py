from comfy.comfy_types.node_typing import ComfyNodeABC
from comfy_api.latest._caching import CacheProvider
from comfy_execution.cache_provider import register_cache_provider


class _RecordingCacheProvider(CacheProvider):
    """Records the class types of every externally stored cache entry so tests
    can assert that failed or failure-blocked outputs never leave the process."""

    def __init__(self):
        self.stored_class_types = []

    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        self.stored_class_types.append(context.class_type)


RECORDING_CACHE_PROVIDER = _RecordingCacheProvider()
register_cache_provider(RECORDING_CACHE_PROVIDER)


class TestCacheProviderRecord(ComfyNodeABC):
    """Reports which node class types have been stored through the external
    cache provider interface since the server started."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "report"
    CATEGORY = "Testing/Nodes"

    def report(self):
        return {"ui": {"stored_class_types": list(RECORDING_CACHE_PROVIDER.stored_class_types)}}


CACHE_PROVIDER_TEST_NODE_CLASS_MAPPINGS = {
    "TestCacheProviderRecord": TestCacheProviderRecord,
}

CACHE_PROVIDER_TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestCacheProviderRecord": "Test Cache Provider Record",
}
