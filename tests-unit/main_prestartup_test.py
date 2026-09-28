from __future__ import annotations

import ast
import importlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time

import folder_paths


def _load_execute_prestartup_script():
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "execute_prestartup_script")
    compiled = compile(ast.Module(body=[function], type_ignores=[]), filename=str(main_path), mode="exec")
    namespace = {
        "args": SimpleNamespace(disable_all_custom_nodes=False, whitelist_custom_nodes=[], enable_manager=False),
        "folder_paths": folder_paths,
        "importlib": importlib,
        "logging": logging,
        "os": os,
        "time": time,
    }
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["execute_prestartup_script"]


def _load_prestartup_script_for_paths(monkeypatch, custom_nodes_paths: list[str]):
    monkeypatch.setattr(
        folder_paths,
        "get_folder_paths",
        lambda name: list(custom_nodes_paths) if name == "custom_nodes" else [],
    )
    return _load_execute_prestartup_script()


def _make_pack(root: Path, name: str) -> Path:
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "prestartup_script.py").write_text("VALUE = 1\n")
    return pack


def test_execute_prestartup_script_handles_empty_custom_nodes_paths(monkeypatch):
    execute_prestartup_script = _load_prestartup_script_for_paths(monkeypatch, [])

    execute_prestartup_script()


def test_execute_prestartup_script_keeps_all_timing_entries(monkeypatch, tmp_path):
    first_custom_nodes = tmp_path / "custom_nodes_1"
    second_custom_nodes = tmp_path / "custom_nodes_2"
    pack_one = _make_pack(first_custom_nodes, "pack_one")
    pack_two = _make_pack(second_custom_nodes, "pack_two")

    execute_prestartup_script = _load_prestartup_script_for_paths(monkeypatch, [str(first_custom_nodes), str(second_custom_nodes)])

    messages: list[str] = []
    monkeypatch.setattr(logging, "info", lambda message, *args, **kwargs: messages.append(message))

    execute_prestartup_script()

    joined = "\n".join(messages)
    assert str(pack_one) in joined
    assert str(pack_two) in joined
