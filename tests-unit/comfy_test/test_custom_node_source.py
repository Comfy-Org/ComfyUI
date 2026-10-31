import logging

import nodes


def test_read_pack_version(tmp_path):
    pack = tmp_path / "my_pack"
    pack.mkdir()
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "my_pack"\nversion = "1.2.3" # release\n',
        encoding="utf-8",
    )
    assert nodes._read_pack_version(str(pack)) == "1.2.3"


def test_read_pack_version_missing_or_invalid(tmp_path):
    pack = tmp_path / "my_pack"
    pack.mkdir()
    assert nodes._read_pack_version(str(pack)) is None

    (pack / "pyproject.toml").write_text('[project]\nname = "my_pack"\n', encoding="utf-8")
    assert nodes._read_pack_version(str(pack)) is None

    (pack / "pyproject.toml").write_text("[\n", encoding="utf-8")
    assert nodes._read_pack_version(str(pack)) is None


def test_read_pack_version_ignores_parent_of_single_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "9.9.9"\n', encoding="utf-8")
    node_file = tmp_path / "node.py"
    node_file.write_text("# node\n", encoding="utf-8")
    assert nodes._read_pack_version(str(node_file)) is None


def test_warn_if_node_replaced(caplog, tmp_path, monkeypatch):
    name = "TestReplacedNodeSource"
    old = type("OldNode", (), {"RELATIVE_PYTHON_MODULE": "custom_nodes.old_pack@1.0.0"})
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, old)
    with caplog.at_level(logging.WARNING):
        source = nodes._warn_if_node_replaced(name, str(tmp_path / "new_pack"), "custom_nodes", "2.0.0")
    assert source == "custom_nodes.new_pack@2.0.0"
    assert "Node 'TestReplacedNodeSource' was already registered by custom_nodes.old_pack@1.0.0, replaced by custom_nodes.new_pack@2.0.0" in caplog.text


def test_warn_if_node_replaced_same_source_is_quiet(caplog, tmp_path, monkeypatch):
    name = "TestSameNodeSource"
    old = type("OldNode", (), {"RELATIVE_PYTHON_MODULE": "custom_nodes.new_pack@2.0.0"})
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, old)
    with caplog.at_level(logging.WARNING):
        source = nodes._warn_if_node_replaced(name, str(tmp_path / "new_pack"), "custom_nodes", "2.0.0")
    assert source == "custom_nodes.new_pack@2.0.0"
    assert caplog.text == ""
