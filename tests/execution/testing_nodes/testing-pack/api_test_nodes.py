import asyncio
import time
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy_api.v0_0_2 import ComfyAPI, ComfyAPISync

api = ComfyAPI()
api_sync = ComfyAPISync()


class TestAsyncProgressUpdate(ComfyNodeABC):
    """Test node with async VALIDATE_INPUTS."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "value": (IO.ANY, {}),
                "sleep_seconds": (IO.FLOAT, {"default": 1.0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute"
    CATEGORY = "experimental/async"

    async def execute(self, value, sleep_seconds):
        start = time.time()
        expiration = start + sleep_seconds
        now = start
        while now < expiration:
            now = time.time()
            await api.execution.set_progress(
                value=(now - start) / sleep_seconds,
                max_value=1.0,
            )
            await asyncio.sleep(0.01)
        return (value,)


class TestSyncProgressUpdate(ComfyNodeABC):
    """Test node with async VALIDATE_INPUTS."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "value": (IO.ANY, {}),
                "sleep_seconds": (IO.FLOAT, {"default": 1.0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute"
    CATEGORY = "experimental/async"

    def execute(self, value, sleep_seconds):
        start = time.time()
        expiration = start + sleep_seconds
        now = start
        while now < expiration:
            now = time.time()
            api_sync.execution.set_progress(
                value=(now - start) / sleep_seconds,
                max_value=1.0,
            )
            time.sleep(0.01)
        return (value,)


API_TEST_NODE_CLASS_MAPPINGS = {
    "TestAsyncProgressUpdate": TestAsyncProgressUpdate,
    "TestSyncProgressUpdate": TestSyncProgressUpdate,
}

API_TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestAsyncProgressUpdate": "Async Progress Update Test Node",
    "TestSyncProgressUpdate": "Sync Progress Update Test Node",
}
