import logging

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


@pytest.mark.asyncio
async def test_custom_node_collision_keeps_existing_registration(
    tmp_path, monkeypatch, caplog
):
    node_id = "AssetIssueStableIdCollision"
    first_pack = tmp_path / "first_pack"
    second_pack = tmp_path / "second_pack"
    first_pack.mkdir()
    second_pack.mkdir()
    (first_pack / "__init__.py").write_text(
        f"class FirstNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: FirstNode}}\n"
        f"NODE_DISPLAY_NAME_MAPPINGS = {{{node_id!r}: 'First Node'}}\n",
        encoding="utf-8",
    )
    (second_pack / "__init__.py").write_text(
        f"class SecondNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: SecondNode}}\n"
        f"NODE_DISPLAY_NAME_MAPPINGS = {{{node_id!r}: 'Second Node'}}\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, node_id, raising=False)
    caplog.set_level(logging.WARNING)

    assert await nodes.load_custom_node(str(first_pack), module_parent="custom_nodes")
    first_node = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert await nodes.load_custom_node(str(second_pack), module_parent="custom_nodes")

    assert nodes.NODE_CLASS_MAPPINGS[node_id] is first_node
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == "First Node"
    assert "AssetIssueStableIdCollision" in caplog.text
    assert "custom_nodes.first_pack" in caplog.text
    assert "custom_nodes.second_pack" in caplog.text


@pytest.mark.asyncio
async def test_same_custom_node_source_can_reload_registration(
    tmp_path, monkeypatch, caplog
):
    node_id = "AssetIssueSameSourceReload"
    pack = tmp_path / "reloadable_pack"
    pack.mkdir()
    (pack / "__init__.py").write_text(
        f"class ReloadedNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: ReloadedNode}}\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, node_id, raising=False)
    caplog.set_level(logging.WARNING)

    assert await nodes.load_custom_node(str(pack), module_parent="custom_nodes")
    first_node = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert await nodes.load_custom_node(str(pack), module_parent="custom_nodes")

    assert nodes.NODE_CLASS_MAPPINGS[node_id] is not first_node
    assert "conflicts with" not in caplog.text


@pytest.mark.asyncio
async def test_custom_node_roots_with_same_pack_name_are_distinct(
    tmp_path, monkeypatch, caplog
):
    node_id = "AssetIssueSamePackNameCollision"
    first_pack = tmp_path / "root_a" / "shared_pack"
    second_pack = tmp_path / "root_b" / "shared_pack"
    first_pack.mkdir(parents=True)
    second_pack.mkdir(parents=True)
    (first_pack / "__init__.py").write_text(
        f"class FirstNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: FirstNode}}\n",
        encoding="utf-8",
    )
    (second_pack / "__init__.py").write_text(
        f"class SecondNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: SecondNode}}\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, node_id, raising=False)
    caplog.set_level(logging.WARNING)

    assert await nodes.load_custom_node(str(first_pack), module_parent="custom_nodes")
    first_node = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert await nodes.load_custom_node(str(second_pack), module_parent="custom_nodes")

    assert nodes.NODE_CLASS_MAPPINGS[node_id] is first_node
    assert str(first_pack) in caplog.text
    assert str(second_pack) in caplog.text


@pytest.mark.asyncio
async def test_custom_node_does_not_replace_registration_with_unknown_source(
    tmp_path, monkeypatch, caplog
):
    node_id = "AssetIssueUnknownSourceCollision"
    existing_node = type("ExistingNode", (), {})
    pack = tmp_path / "custom_pack"
    pack.mkdir()
    (pack / "__init__.py").write_text(
        f"class NewNode: pass\nNODE_CLASS_MAPPINGS = {{{node_id!r}: NewNode}}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, existing_node)
    caplog.set_level(logging.WARNING)

    assert await nodes.load_custom_node(str(pack), module_parent="custom_nodes")

    assert nodes.NODE_CLASS_MAPPINGS[node_id] is existing_node
    assert "unknown source" in caplog.text
    assert "unknown path" in caplog.text


@pytest.mark.asyncio
async def test_v3_custom_node_collision_keeps_existing_schema(
    tmp_path, monkeypatch, caplog
):
    node_id = "AssetIssueV3StableIdCollision"
    first_pack = tmp_path / "first_v3_pack"
    second_pack = tmp_path / "second_v3_pack"
    first_pack.mkdir()
    second_pack.mkdir()

    for pack, class_name, display_name in (
        (first_pack, "FirstNode", "First Node"),
        (second_pack, "SecondNode", "Second Node"),
    ):
        (pack / "__init__.py").write_text(
            "from comfy_api.latest import ComfyExtension\n\n"
            f"class {class_name}:\n"
            "    @classmethod\n"
            "    def GET_SCHEMA(cls):\n"
            "        class Schema:\n"
            f"            node_id = {node_id!r}\n"
            f"            display_name = {display_name!r}\n"
            "        return Schema()\n\n"
            "class TestExtension(ComfyExtension):\n"
            f"    async def get_node_list(self): return [{class_name}]\n\n"
            "async def comfy_entrypoint(): return TestExtension()\n",
            encoding="utf-8",
        )
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, node_id, raising=False)
    monkeypatch.delitem(nodes.NODE_DISPLAY_NAME_MAPPINGS, node_id, raising=False)
    caplog.set_level(logging.WARNING)

    assert await nodes.load_custom_node(str(first_pack), module_parent="custom_nodes")
    first_node = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert await nodes.load_custom_node(str(second_pack), module_parent="custom_nodes")

    assert nodes.NODE_CLASS_MAPPINGS[node_id] is first_node
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == "First Node"
    assert "custom_nodes.first_v3_pack" in caplog.text
    assert "custom_nodes.second_v3_pack" in caplog.text
