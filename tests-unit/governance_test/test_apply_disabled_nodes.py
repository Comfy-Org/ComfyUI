import copy
import logging
from pathlib import Path

import pytest
import torch

from app import governance
from comfy.cli_args import args


if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from app.node_replace_manager import NodeReplaceManager
from comfy_api.latest import io


class TestNode:
    pass


@pytest.fixture(autouse=True)
def isolated_governance_state(monkeypatch: pytest.MonkeyPatch):
    class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    load_custom_node = nodes.load_custom_node
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset(), raising=False)
    monkeypatch.setattr(governance, "_original_load_custom_node", None, raising=False)

    try:
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
        nodes.load_custom_node = load_custom_node


def _prompt(class_type: str, with_meta: bool) -> dict:
    node = {"class_type": class_type, "inputs": {}}
    if with_meta:
        node["_meta"] = {"title": class_type}
    return {"1": node}


def _replacement(old_node_id: str = "LegacyNode", new_node_id: str = "CurrentNode") -> io.NodeReplace:
    return io.NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id)


def test_apply_disabled_nodes_removes_class_and_display_name(caplog: pytest.LogCaptureFixture) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS["DisabledNode"] = TestNode
    nodes.NODE_DISPLAY_NAME_MAPPINGS["DisabledNode"] = "Disabled Node"

    # When
    with caplog.at_level(logging.INFO):
        governance.apply_disabled_nodes({"DisabledNode"})

    # Then
    assert "DisabledNode" not in nodes.NODE_CLASS_MAPPINGS
    assert "DisabledNode" not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert "Pruned 1 disabled node" in caplog.text


def test_apply_disabled_nodes_warns_once_for_all_missing_ids(caplog: pytest.LogCaptureFixture) -> None:
    # Given
    missing = {"MissingNodeA", "MissingNodeB"}

    # When
    with caplog.at_level(logging.WARNING):
        governance.apply_disabled_nodes(missing)

    # Then
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert all(node_id in warnings[0].getMessage() for node_id in missing)


@pytest.mark.parametrize("with_meta", [False, True])
def test_disabled_old_id_forwards_to_allowed_target(with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)

    # When
    governance.apply_disabled_nodes({"LegacyNode"})
    manager.apply_replacements(prompt)

    # Then
    assert manager.has_replacement("LegacyNode") is True
    assert prompt["1"]["class_type"] == "CurrentNode"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_meta", [False, True])
async def test_disabled_target_is_not_applied_and_prompt_is_refused(with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS["CurrentNode"] = TestNode
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)

    # When
    governance.apply_disabled_nodes({"CurrentNode"})
    manager.apply_replacements(prompt)
    valid = await execution.validate_prompt("prompt-id", prompt, None)

    # Then
    assert prompt["1"]["class_type"] == "LegacyNode"
    assert valid[0] is False
    assert valid[1]["type"] == "missing_node_type"
    assert valid[1]["extra_info"]["class_type"] == "LegacyNode"


@pytest.mark.asyncio
async def test_disabling_both_replacement_ends_refuses_prompt() -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", False)

    # When
    governance.apply_disabled_nodes({"LegacyNode", "CurrentNode"})
    manager.apply_replacements(prompt)
    valid = await execution.validate_prompt("prompt-id", prompt, None)

    # Then
    assert prompt["1"]["class_type"] == "LegacyNode"
    assert valid[0] is False
    assert valid[1]["type"] == "missing_node_type"


@pytest.mark.parametrize("with_meta", [False, True])
def test_replacement_behavior_is_unchanged_when_neither_end_is_disabled(with_meta: bool) -> None:
    # Given
    nodes.NODE_CLASS_MAPPINGS.update({"LegacyNode": TestNode, "CurrentNode": TestNode})
    manager = NodeReplaceManager()
    manager.register(_replacement())
    prompt = _prompt("LegacyNode", with_meta)
    original_prompt = copy.deepcopy(prompt)

    # When
    governance.apply_disabled_nodes(set())
    manager.apply_replacements(prompt)

    # Then
    assert prompt == original_prompt


@pytest.mark.asyncio
async def test_post_prune_v1_load_cannot_register_disabled_id(tmp_path: Path) -> None:
    # Given
    module_path = tmp_path / "sealed_v1_node.py"
    module_path.write_text(
        "class SealedV1Node:\n"
        "    pass\n\n"
        "NODE_CLASS_MAPPINGS = {'DisabledV1': SealedV1Node}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {'DisabledV1': 'Disabled V1'}\n",
        encoding="utf-8",
    )
    governance.apply_disabled_nodes({"DisabledV1"})

    # When
    loaded = await nodes.load_custom_node(str(module_path), ignore={"OtherNode"})

    # Then
    assert loaded is True
    assert "DisabledV1" not in nodes.NODE_CLASS_MAPPINGS
    assert "DisabledV1" not in nodes.NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.asyncio
async def test_post_prune_v3_load_cannot_register_disabled_schema_id(tmp_path: Path) -> None:
    # Given
    module_path = tmp_path / "sealed_v3_node.py"
    module_path.write_text(
        "from comfy_api.latest import ComfyExtension\n\n"
        "class SealedV3Node:\n"
        "    @classmethod\n"
        "    def GET_SCHEMA(cls):\n"
        "        class Schema:\n"
        "            node_id = 'DisabledV3'\n"
        "            display_name = 'Disabled V3'\n\n"
        "        return Schema()\n\n\n"
        "class SealedExtension(ComfyExtension):\n"
        "    async def get_node_list(self):\n"
        "        return [SealedV3Node]\n\n\n"
        "async def comfy_entrypoint():\n"
        "    return SealedExtension()\n",
        encoding="utf-8",
    )
    governance.apply_disabled_nodes({"DisabledV3"})

    # When
    loaded = await nodes.load_custom_node(str(module_path))

    # Then
    assert loaded is True
    assert "DisabledV3" not in nodes.NODE_CLASS_MAPPINGS
    assert "DisabledV3" not in nodes.NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.asyncio
async def test_seal_keeps_replacement_target_absent_after_v1_load(tmp_path: Path) -> None:
    # Given
    module_path = tmp_path / "sealed_target.py"
    module_path.write_text(
        "class SealedTargetNode:\n"
        "    pass\n\n"
        "NODE_CLASS_MAPPINGS = {'DisabledTarget': SealedTargetNode}\n",
        encoding="utf-8",
    )
    nodes.NODE_CLASS_MAPPINGS["DisabledTarget"] = TestNode
    manager = NodeReplaceManager()
    manager.register(_replacement(new_node_id="DisabledTarget"))
    governance.apply_disabled_nodes({"DisabledTarget"})
    loaded = await nodes.load_custom_node(str(module_path))
    prompt = _prompt("LegacyNode", False)

    # When
    manager.apply_replacements(prompt)

    # Then
    assert loaded is True
    assert "DisabledTarget" not in nodes.NODE_CLASS_MAPPINGS
    assert prompt["1"]["class_type"] == "LegacyNode"
