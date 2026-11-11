import asyncio
import os
import tempfile

import pytest

from test_inmemory_assets import InMemoryAssets

_BASE = os.path.join(tempfile.gettempdir(), "execute-reentry-test-base")


def _find_ids(value) -> list:
    found: list = []
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "id":
                found.append(sub)
            found.extend(_find_ids(sub))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_ids(item))
    return found


class _Server:
    def __init__(self, client_id=None) -> None:
        self.client_id = client_id
        self.last_node_id = None
        self.sent: list = []

    def send_sync(self, event, payload, client_id) -> None:
        self.sent.append((event, payload))


class _AsyncDictCache:

    def __init__(self) -> None:
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value) -> None:
        self.store[key] = value

    async def ensure_subcache_for(self, unique_id, node_ids):
        return self


class _Caches:
    def __init__(self) -> None:
        self.outputs = _AsyncDictCache()
        self.objects = _AsyncDictCache()
        self.all = [self.outputs, self.objects]


class _ExecutionList:
    def cache_update(self, unique_id, entry) -> None:
        pass

    def add_external_block(self, unique_id):
        return lambda: None

    def get_cache(self, a, b):
        return None


class _NoProgress:
    def start_progress(self, *a, **k) -> None:
        pass

    def finish_progress(self, *a, **k) -> None:
        pass


class _AsyncUINode:

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    async def run(self):
        await asyncio.sleep(0.02)
        return {"ui": {"images": [{"filename": "async.png", "subfolder": "", "type": "output"}]}}


@pytest.fixture
def execution_env(monkeypatch):
    try:
        from comfy.cli_args import args
        monkeypatch.setattr(args, "cpu", True, raising=False)
        import execution
        import folder_paths
        import nodes
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"execution module could not be imported in CPU mode: {exc!r}")

    os.makedirs(_BASE, exist_ok=True)
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda t: _BASE)
    monkeypatch.setattr(execution, "get_progress_state", lambda: _NoProgress())
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "AsyncUINode", _AsyncUINode)

    return execution, InMemoryAssets()


async def _drive_async_reentry(execution, asset_manager):
    from comfy_execution.graph import DynamicPrompt

    unique_id = "1"
    with open(os.path.join(_BASE, "async.png"), "wb") as f:
        f.write(b"x")

    dynprompt = DynamicPrompt({unique_id: {"class_type": "AsyncUINode", "inputs": {}}})
    caches = _Caches()
    server = _Server(client_id=None)
    exec_list = _ExecutionList()
    pending_subgraph_results: dict = {}
    pending_async_nodes: dict = {}
    ui_outputs: dict = {}
    executed: set = set()

    common = (server, dynprompt, caches, unique_id, {}, executed, "job-1", exec_list)

    r1, _, _ = await execution.execute(
        *common, pending_subgraph_results, pending_async_nodes, ui_outputs, asset_manager
    )
    ui_outputs_had_uid_after_entry1 = unique_id in ui_outputs

    tasks = [t for t in pending_async_nodes.get(unique_id, []) if isinstance(t, asyncio.Task)]
    if tasks:
        await asyncio.gather(*tasks)
    for _ in range(3):
        await asyncio.sleep(0)

    r2, _, _ = await execution.execute(
        *common, pending_subgraph_results, pending_async_nodes, ui_outputs, asset_manager
    )

    cached = caches.outputs.store.get(unique_id)
    return {
        "r1": r1,
        "r2": r2,
        "ui_outputs_had_uid_after_entry1": ui_outputs_had_uid_after_entry1,
        "ui_ids": _find_ids(ui_outputs.get(unique_id)),
        "cache_entry": cached,
        "cache_ids": _find_ids(cached.ui) if cached is not None else None,
    }


def test_async_reentry_keeps_cache_id_free(execution_env):
    execution, asset_manager = execution_env
    from execution import ExecutionResult

    obs = asyncio.run(_drive_async_reentry(execution, asset_manager))

    assert obs["r1"] == ExecutionResult.PENDING
    assert obs["r2"] == ExecutionResult.SUCCESS
    # WHY it is safe: the async branch returns PENDING *before* writing ui_outputs,
    # so on re-entry ui_outputs.get(unique_id) is None and cannot seed cache_ui_value.
    assert obs["ui_outputs_had_uid_after_entry1"] is False

    assert obs["ui_ids"] == ["asset-1"]

    assert obs["cache_entry"] is not None
    assert obs["cache_ids"] == []
    assert [call.method for call in asset_manager.calls] == ["register_executed_output"]
