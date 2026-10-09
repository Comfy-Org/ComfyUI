"""A save node's asset_preview goes to registration only: never to clients, history or the cache."""

import asyncio
import os

import pytest

from test_execute_reentry import _BASE, _Caches, _ExecutionList, _NoProgress, _Server
from test_inmemory_assets import InMemoryAssets

_REF = {"filename": f"{'a' * 64}.jpg", "width": 4, "height": 3}


class _SaveNode:
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        entry = {"filename": "frame.exr", "subfolder": "", "type": "output", "asset_preview": dict(_REF)}
        return {"ui": {"images": [entry]}}


class _Off(InMemoryAssets):
    @property
    def enabled(self) -> bool:
        return False


@pytest.fixture
def execution(monkeypatch):
    try:
        from comfy.cli_args import args
        monkeypatch.setattr(args, "cpu", True, raising=False)
        import execution
        import folder_paths
        import nodes
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"execution module could not be imported in CPU mode: {exc!r}")
    os.makedirs(_BASE, exist_ok=True)
    with open(os.path.join(_BASE, "frame.exr"), "wb") as f:
        f.write(b"x")
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda t: _BASE)
    monkeypatch.setattr(execution, "get_progress_state", lambda: _NoProgress())
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SaveNode", _SaveNode)
    return execution


def _execute(execution, asset_manager, client_id="client"):
    from comfy_execution.graph import DynamicPrompt

    server, caches, ui_outputs = _Server(client_id=client_id), _Caches(), {}
    dynprompt = DynamicPrompt({"1": {"class_type": "SaveNode", "inputs": {}}})
    asyncio.run(execution.execute(server, dynprompt, caches, "1", {}, set(), "job", _ExecutionList(), {}, {}, ui_outputs, asset_manager))
    return server.sent, ui_outputs, caches.outputs.store["1"].ui


@pytest.mark.parametrize("manager", [InMemoryAssets, _Off])
def test_the_key_never_escapes(execution, manager):
    sent, ui_outputs, cached = _execute(execution, manager())

    for value in ([payload for _, payload in sent], ui_outputs, cached):
        assert "asset_preview" not in repr(value)
    assert "executed" in [event for event, _ in sent]


def test_registration_gets_the_ref_for_its_path(execution):
    asset_manager = InMemoryAssets()

    _execute(execution, asset_manager, client_id=None)

    assert asset_manager.preview_refs == {os.path.join(_BASE, "frame.exr"): _REF}
