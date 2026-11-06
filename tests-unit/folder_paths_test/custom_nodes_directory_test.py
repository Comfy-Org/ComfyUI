import importlib
import os

import pytest

import folder_paths
from comfy.cli_args import args, parser


@pytest.fixture
def reload_folder_paths(monkeypatch):
    def reload(**overrides):
        for name, value in overrides.items():
            monkeypatch.setattr(args, name, value)
        return importlib.reload(folder_paths)

    yield reload
    monkeypatch.undo()
    importlib.reload(folder_paths)


def test_flag_parsed(tmp_path):
    parsed = parser.parse_args(["--custom-nodes-directory", str(tmp_path)])
    assert parsed.custom_nodes_directory == str(tmp_path)


def test_flag_rejects_missing_directory(tmp_path):
    with pytest.raises(SystemExit):
        parser.parse_args(["--custom-nodes-directory", str(tmp_path / "missing")])


def test_default_custom_nodes_path(reload_folder_paths):
    fp = reload_folder_paths(base_directory=None, custom_nodes_directory=None)
    assert fp.get_folder_paths("custom_nodes") == [os.path.join(fp.base_path, "custom_nodes")]


def test_custom_nodes_directory_flag(reload_folder_paths, tmp_path):
    fp = reload_folder_paths(base_directory=None, custom_nodes_directory=str(tmp_path))
    assert fp.get_folder_paths("custom_nodes") == [str(tmp_path)]


def test_custom_nodes_directory_overrides_base_directory(reload_folder_paths, tmp_path):
    base = tmp_path / "base"
    nodes = tmp_path / "nodes"
    base.mkdir()
    nodes.mkdir()
    fp = reload_folder_paths(base_directory=str(base), custom_nodes_directory=str(nodes))
    assert fp.get_folder_paths("custom_nodes") == [str(nodes)]
    assert fp.get_folder_paths("checkpoints") == [os.path.join(str(base), "models", "checkpoints")]
