import ast
import asyncio
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from comfy_api import index
from comfy_api._impl import _runtime


ROOT = Path(__file__).resolve().parents[2]


def test_native_api_does_not_load_optional_runtime(tmp_path):
    code = '''
import asyncio
import importlib.abc
import sys

blocked = ("comfy_api._impl.isolation", "comfy_api._impl.pack_storage",
           "comfy_api._impl.interactions", "comfy_api._impl.manifest", "comfy_secure_nodes",
           "multiprocessing.shared_memory")
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("native path imported optional runtime: " + fullname)
sys.meta_path.insert(0, BlockOptional())
def audit(event, args):
    if event in ("subprocess.Popen", "socket.connect", "socket.getaddrinfo"):
        raise AssertionError("native path attempted external operation: " + event)
sys.addaudithook(audit)
from comfy_api.index import eng
from comfy_api._impl import _runtime
assert not _runtime.load_execution_provider()
assert type(_runtime.providers.execution_backend) is _runtime.InProcessExecutionBackend
class Clip:
    def tokenize(self, text, **kwargs):
        return {"text": text, "options": kwargs}
async def run():
    with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.providers.ops_provider):
        assert await eng.clip.tokenize(Clip(), "hello", return_word_ids=True) == {
            "text": "hello", "options": {"return_word_ids": True}}
asyncio.run(run())
assert not any(name == item or name.startswith(item + ".") for name in sys.modules for item in blocked)
'''
    env = os.environ.copy()
    env.pop("COMFY_EXECUTION_PROVIDER", None)
    env.update(COMFY_PACK_ISOLATION="true", OOP_CUSTOM_NODES="true", PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_provider_registers_once_per_registry(tmp_path, monkeypatch):
    name = "test_explicit_execution_provider"
    (tmp_path / f"{name}.py").write_text(
        'registrations = 0\n'
        'def register(providers):\n'
        '    global registrations\n'
        '    registrations += 1\n'
        '    providers.test_registration = registrations\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("COMFY_EXECUTION_PROVIDER", name)
    monkeypatch.delitem(sys.modules, name, raising=False)
    original_types = (index.io.Int, index.io.Image, index.io.NodeOutput)
    first = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", first)
    assert _runtime.load_execution_provider()
    _runtime.load_execution_provider()
    module = sys.modules[name]
    assert module.registrations == 1
    assert first.test_registration == 1
    second = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", second)
    assert _runtime.load_execution_provider()
    assert module.registrations == 2
    assert second.test_registration == 2
    assert original_types == (index.io.Int, index.io.Image, index.io.NodeOutput)
    monkeypatch.delitem(sys.modules, name)


def test_failed_registration_can_be_retried(tmp_path, monkeypatch):
    name = "test_retry_execution_provider"
    (tmp_path / f"{name}.py").write_text(
        'calls = 0\n'
        'def register(providers):\n'
        '    global calls\n'
        '    calls += 1\n'
        '    if calls == 1:\n'
        '        raise RuntimeError("registration failed")\n'
        '    providers.test_ready = True\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("COMFY_EXECUTION_PROVIDER", name)
    monkeypatch.delitem(sys.modules, name, raising=False)
    registry = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", registry)
    with pytest.raises(RuntimeError, match="registration failed"):
        _runtime.load_execution_provider()
    assert _runtime.load_execution_provider()
    assert sys.modules[name].calls == 2
    assert registry.test_ready
    monkeypatch.delitem(sys.modules, name)


def test_provider_services_accept_sync_and_async_handlers(monkeypatch):
    registry = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", registry)
    calls = []
    values = {}

    async def get(key, scope="pack", default=None):
        calls.append(("get", key, scope, default))
        return values.get((scope, key), default)

    def set_value(key, value, scope="pack"):
        calls.append(("set", key, value, scope))
        values[scope, key] = value

    def delete(key, scope="pack"):
        calls.append(("delete", key, scope))
        return values.pop((scope, key), None) is not None

    async def keys(scope="pack"):
        calls.append(("keys", scope))
        return [key for item_scope, key in values if item_scope == scope]

    async def interact(kind, payload, reuse_last=False, remember=False, timeout=540.0):
        calls.append(("interact", kind, payload, reuse_last, remember, timeout))
        return {"answer": True}

    registry.service_calls.update({"store.get": get, "store.set": set_value, "store.delete": delete,
                                   "store.keys": keys, "ui.interact": interact})

    async def run():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, registry.ops_provider, pack="fixture"):
            assert await index.eng.store.get("missing", default="fallback") == "fallback"
            assert await index.eng.store.set("key", {"n": [1]}, scope="workflow") is None
            assert await index.eng.store.get("key", scope="workflow") == {"n": [1]}
            assert await index.eng.store.keys("workflow") == ["key"]
            assert await index.eng.store.delete("key", scope="workflow") is True
            assert await index.eng.ui.interact("confirm", {"text": "ok"}, reuse_last=True, remember=True, timeout=2.5) == {"answer": True}
            assert await index.eng.ui.interact("choice", {}) == {"answer": True}
    asyncio.run(run())
    assert calls == [
        ("get", "missing", "pack", "fallback"), ("set", "key", {"n": [1]}, "workflow"),
        ("get", "key", "workflow", None), ("keys", "workflow"), ("delete", "key", "workflow"),
        ("interact", "confirm", {"text": "ok"}, True, True, 2.5),
        ("interact", "choice", {}, False, False, 540.0),
    ]


def test_missing_services_and_provider_denials_are_explicit(monkeypatch):
    registry = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", registry)
    async def run():
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, registry.ops_provider):
            for name, call in (
                ("store.get", lambda: index.eng.store.get("key")),
                ("store.set", lambda: index.eng.store.set("key", "value")),
                ("store.delete", lambda: index.eng.store.delete("key")),
                ("store.keys", lambda: index.eng.store.keys()),
                ("ui.interact", lambda: index.eng.ui.interact("confirm", {})),
            ):
                with pytest.raises(_runtime.CallNotSupported, match=re.escape(name)):
                    await call()
            def deny(*args, **kwargs):
                raise PermissionError("grant denied")
            registry.service_calls["store.get"] = deny
            with pytest.raises(PermissionError, match="grant denied"):
                await index.eng.store.get("key")
    asyncio.run(run())


def test_public_service_stub_signatures():
    tree = ast.parse((ROOT / "comfy_api/index.pyi").read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    expected = {
        "_Store": [
            'async def get(self, key: str, scope: str = "pack", default: Any = None) -> Any: ...',
            'async def set(self, key: str, value: Any, scope: str = "pack") -> None: ...',
            'async def delete(self, key: str, scope: str = "pack") -> bool: ...',
            'async def keys(self, scope: str = "pack") -> list[str]: ...',
        ],
        "_Ui": ['async def interact(self, kind: str, payload: dict, reuse_last: bool = False, remember: bool = False, timeout: float = 540.0) -> Any: ...'],
    }
    for class_name, signatures in expected.items():
        methods = {node.name: node for node in classes[class_name].body if isinstance(node, ast.AsyncFunctionDef)}
        for signature in signatures:
            wanted = ast.parse(signature).body[0]
            actual = methods[wanted.name]
            assert ast.dump(actual.args) == ast.dump(wanted.args)
            assert ast.dump(actual.returns) == ast.dump(wanted.returns)


def test_custom_node_admission_uses_outer_name_and_checks_policies(monkeypatch):
    import types
    import nodes

    args = types.SimpleNamespace(disable_all_custom_nodes=True, whitelist_custom_nodes=["allowed"], enable_manager=True)
    monkeypatch.setattr(nodes, "args", args)
    checked = []
    manager = types.SimpleNamespace(should_be_disabled=lambda path: checked.append(("manager", path)) or False)
    monkeypatch.setattr(nodes, "comfyui_manager", manager, raising=False)
    monkeypatch.setattr(nodes.governance, "pack_refusal", lambda path: checked.append(("governance", path)) or None)
    assert not nodes.custom_node_allowed("/custom/other")
    assert checked == []
    assert nodes.custom_node_allowed("/custom/allowed/")
    assert checked == [("manager", "/custom/allowed/"), ("governance", "/custom/allowed/")]
    checked.clear()
    manager.should_be_disabled = lambda path: checked.append(("manager", path)) or True
    assert not nodes.custom_node_allowed("/custom/allowed")
    assert checked == [("manager", "/custom/allowed")]
    manager.should_be_disabled = lambda path: False
    monkeypatch.setattr(nodes.governance, "pack_refusal", lambda path: "blocked by governance")
    assert not nodes.custom_node_allowed("/custom/allowed")
    args.disable_all_custom_nodes = False
    args.enable_manager = False
    monkeypatch.setattr(nodes.governance, "pack_refusal", lambda path: None)
    assert nodes.custom_node_allowed("/custom/other")


def test_provider_routes_wait_for_a_server_in_headless_execution(monkeypatch):
    from types import SimpleNamespace
    import nodes
    import server

    calls = []
    registry = _runtime.Providers()
    registry.execution_backend = SimpleNamespace(setup_routes=calls.append)
    monkeypatch.setattr(_runtime, "providers", registry)
    async def builtins():
        return []
    monkeypatch.setattr(nodes, "init_builtin_extra_nodes", builtins)
    monkeypatch.setattr(server.PromptServer, "instance", None, raising=False)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert calls == []
    instance = object()
    monkeypatch.setattr(server.PromptServer, "instance", instance)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert calls == [instance]
