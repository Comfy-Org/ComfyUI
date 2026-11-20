"""Tests for NodeReplaceManager registration behavior."""
import importlib
import sys
import types

import pytest


@pytest.fixture
def NodeReplaceManager(monkeypatch):
    """Provide NodeReplaceManager with `nodes` stubbed.

    `app.node_replace_manager` does `import nodes` at module level, which pulls in
    torch + the full ComfyUI graph. register() doesn't actually need it, so we
    stub `nodes` per-test (via monkeypatch so it's torn down) and reload the
    module so it picks up the stub instead of any cached real import.
    """
    fake_nodes = types.ModuleType("nodes")
    fake_nodes.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.delitem(sys.modules, "app.node_replace_manager", raising=False)
    module = importlib.import_module("app.node_replace_manager")
    yield module.NodeReplaceManager
    # Drop the freshly-imported module so the next test (or a later real import
    # of `nodes`) starts from a clean slate.
    sys.modules.pop("app.node_replace_manager", None)


class FakeNodeReplace:
    """Lightweight stand-in for comfy_api.latest._io.NodeReplace."""
    def __init__(self, new_node_id, old_node_id, old_widget_ids=None,
                 input_mapping=None, output_mapping=None, old_pack=None, kind="deprecated"):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.old_pack = old_pack
        self.kind = kind


def test_register_adds_replacement(NodeReplaceManager):
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="NewNode", old_node_id="OldNode"))
    assert manager.has_replacement("OldNode")
    assert len(manager.get_replacement("OldNode")) == 1


def test_register_allows_multiple_alternatives_for_same_old_node(NodeReplaceManager):
    """Different new_node_ids for the same old_node_id should all be kept."""
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="AltA", old_node_id="OldNode"))
    manager.register(FakeNodeReplace(new_node_id="AltB", old_node_id="OldNode"))
    replacements = manager.get_replacement("OldNode")
    assert len(replacements) == 2
    assert {r.new_node_id for r in replacements} == {"AltA", "AltB"}


def test_register_is_idempotent_for_duplicate_pair(NodeReplaceManager):
    """Re-registering the same (old_node_id, new_node_id) should be a no-op."""
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="NewNode", old_node_id="OldNode"))
    manager.register(FakeNodeReplace(new_node_id="NewNode", old_node_id="OldNode"))
    manager.register(FakeNodeReplace(new_node_id="NewNode", old_node_id="OldNode"))
    assert len(manager.get_replacement("OldNode")) == 1


def test_register_idempotent_preserves_first_registration(NodeReplaceManager):
    """First registration wins; later duplicates with different mappings are ignored."""
    manager = NodeReplaceManager()
    first = FakeNodeReplace(
        new_node_id="NewNode", old_node_id="OldNode",
        input_mapping=[{"new_id": "a", "old_id": "x"}],
    )
    second = FakeNodeReplace(
        new_node_id="NewNode", old_node_id="OldNode",
        input_mapping=[{"new_id": "b", "old_id": "y"}],
    )
    manager.register(first)
    manager.register(second)
    replacements = manager.get_replacement("OldNode")
    assert len(replacements) == 1
    assert replacements[0] is first


def test_register_dedupe_does_not_affect_other_old_nodes(NodeReplaceManager):
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="NewA", old_node_id="OldA"))
    manager.register(FakeNodeReplace(new_node_id="NewA", old_node_id="OldA"))
    manager.register(FakeNodeReplace(new_node_id="NewB", old_node_id="OldB"))
    assert len(manager.get_replacement("OldA")) == 1
    assert len(manager.get_replacement("OldB")) == 1


def test_register_keeps_same_type_from_different_packs(NodeReplaceManager):
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="PrimitiveString", old_node_id="StringConstant", old_pack="pack-a"))
    manager.register(FakeNodeReplace(new_node_id="PrimitiveString", old_node_id="StringConstant", old_pack="pack-b"))
    assert {r.old_pack for r in manager.get_replacement("StringConstant")} == {"pack-a", "pack-b"}


@pytest.mark.parametrize("entry_packs, node_pack, expected", [
    (["pack-a", "pack-b"], "pack-b", "New-pack-b"),
    (["pack-a", "pack-b"], "pack-c", None),
    (["pack-a", "pack-b"], None, None),
    (["pack-a", None], "pack-z", "New-None"),
    (["pack-a", None], "pack-a", "New-pack-a"),
    (["pack-a"], None, "New-pack-a"),
    (["pack-a"], "pack-b", None),
])
def test_select_replacement_matches_node_pack(NodeReplaceManager, entry_packs, node_pack, expected):
    manager = NodeReplaceManager()
    for pack in entry_packs:
        manager.register(FakeNodeReplace(new_node_id=f"New-{pack}", old_node_id="Old", old_pack=pack))
    selected = manager.select_replacement("Old", node_pack)
    assert (selected.new_node_id if selected else None) == expected


def test_apply_replacements_selects_by_prompt_pack(NodeReplaceManager):
    nodes = sys.modules["nodes"]
    nodes.NODE_CLASS_MAPPINGS = {"NewA": object, "NewB": object}
    manager = NodeReplaceManager()
    manager.register(FakeNodeReplace(new_node_id="NewA", old_node_id="Old", old_pack="pack-a",
                                     input_mapping=[{"new_id": "value", "old_id": "string"}]))
    manager.register(FakeNodeReplace(new_node_id="NewB", old_node_id="Old", old_pack="pack-b",
                                     input_mapping=[{"new_id": "value", "old_id": "string"}]))
    prompt = {
        "1": {"class_type": "Old", "inputs": {"string": "hi"}, "_meta": {"cnr_id": "pack-b"}},
        "2": {"class_type": "Old", "inputs": {"string": "yo"}, "_meta": {"aux_id": "pack-a"}},
    }
    manager.apply_replacements(prompt)
    assert prompt["1"]["class_type"] == "NewB"
    assert prompt["2"]["class_type"] == "Old"
    assert prompt["1"]["inputs"] == {"value": "hi"}
