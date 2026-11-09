"""Core engine extension: the ``eng.store`` rows, small values a pack keeps between runs.

Kept by the secure overlay's pack storage (``comfy_api/_impl/pack_storage.py``: atomic, bounded files
under ``user/secure-node-storage``), so a pack's values are the same in ComfyUI's process, a pack
process and the sandbox. Scopes: ``"pack"`` (every workflow), ``"workflow"`` (this workflow, by its id)
and ``"node"`` (this node in this workflow). Values are plain data, stored as JSON: keys up to 128
bytes, values up to 1 MB, and 256 entries and 8 MB per scope.
"""
import json
import re

from comfy_api._impl import _runtime

SCOPES = ("pack", "workflow", "node")
TENANT = "default"  # the overlay's storage tenant when none is configured


def _pack_name():
    rt = _runtime.current_runtime()
    parts = [p for p in (rt.pack or "").replace("\\", "/").replace("/", ".").split(".") if p not in ("", "custom_nodes")]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", parts[0] if parts else "pack")


def _scope_id(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope is one of {SCOPES}, got {scope!r}")
    pack = _pack_name()
    if scope == "pack":
        return pack
    plan = _runtime.current_runtime().plan
    workflow = ((getattr(plan, "extra_pnginfo", None) or {}).get("workflow") or {}).get("id")
    if not workflow:
        raise ValueError(f"scope {scope!r} needs the workflow's id, and this prompt has none")
    return f"{pack}/workflow/{workflow}" if scope == "workflow" else f"{pack}/node/{workflow}/{getattr(plan, 'node_id', '')}"


def _store():
    from comfy_api._impl.pack_storage import pack_store
    return pack_store()


class EngineOps:
    def get(self, key, scope="pack", default=None):
        text = _store().get(TENANT, _scope_id(scope), str(key))
        return default if text is None else json.loads(text)

    def set(self, key, value, scope="pack"):
        _store().set(TENANT, _scope_id(scope), str(key), json.dumps(value))  # plain data only

    def delete(self, key, scope="pack"):
        return _store().delete(TENANT, _scope_id(scope), str(key))

    def keys(self, scope="pack"):
        return _store().keys(TENANT, _scope_id(scope))
