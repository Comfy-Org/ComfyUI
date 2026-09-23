from __future__ import annotations

import ast
import importlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import folder_paths
from comfy.cli_args import parser


def _is_agent_check(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "args"
        and node.test.attr == "enable_agent"
    )


def _run_agent_check(monkeypatch, caplog, find_spec):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    handler = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "handle_comfy_agent_unavailable")
    check = next(node for node in module.body if _is_agent_check(node))
    compiled = compile(ast.Module(body=[handler, check], type_ignores=[]), filename=str(main_path), mode="exec")
    args = SimpleNamespace(enable_agent=True)
    namespace = {
        "args": args,
        "folder_paths": folder_paths,
        "importlib": importlib,
        "logging": logging,
        "os": os,
    }
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    with caplog.at_level(logging.INFO):
        exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return args


def test_enable_agent_flag_parses():
    assert parser.parse_args(["--enable-agent"]).enable_agent is True
    assert parser.parse_args([]).enable_agent is False


def test_missing_agent_package_disables_flag_without_install_command(monkeypatch, caplog):
    args = _run_agent_check(monkeypatch, caplog, lambda name: None)

    assert args.enable_agent is False
    warning = next(record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
    assert "agent_requirements.txt" in warning
    assert "pip install" not in warning


def test_namespace_agent_package_counts_as_missing(monkeypatch, caplog):
    args = _run_agent_check(monkeypatch, caplog, lambda name: SimpleNamespace(origin=None))

    assert args.enable_agent is False


@pytest.mark.parametrize(
    ("find_spec", "expected"),
    [
        (lambda name: SimpleNamespace(origin="comfy_agent/__init__.py"), ["[agent-event] flag_enabled"]),
        (lambda name: None, ["[agent-event] flag_enabled", "[agent-event] package_missing"]),
    ],
)
def test_agent_event_lines(monkeypatch, caplog, find_spec, expected):
    _run_agent_check(monkeypatch, caplog, find_spec)

    events = [record.getMessage() for record in caplog.records if record.getMessage().startswith("[agent-event]")]
    assert events == expected
    assert all(record.levelno == logging.INFO for record in caplog.records if record.getMessage().startswith("[agent-event]"))
