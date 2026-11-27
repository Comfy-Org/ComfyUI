"""``--auto-launch`` must hand the browser a reachable URL, not a bind wildcard.

``main.py`` cannot be imported from a unit test (it starts the application), so
the ``startup_server`` callback is pulled out of its source with the same AST
extraction used by ``main_prestartup_test.py``.
"""

from __future__ import annotations

import ast
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_startup_server(os_name: str):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    start_comfyui = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "start_comfyui"
    )
    startup_server = next(
        node
        for node in ast.walk(start_comfyui)
        if isinstance(node, ast.FunctionDef) and node.name == "startup_server"
    )
    compiled = compile(
        ast.Module(body=[startup_server], type_ignores=[]),
        filename=str(main_path),
        mode="exec",
    )
    namespace = {"os": SimpleNamespace(name=os_name)}
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself
    return namespace["startup_server"]


@pytest.mark.parametrize("os_name", ["posix", "nt"])
@pytest.mark.parametrize(
    "scheme, address, port, expected_url",
    [
        ("http", "0.0.0.0", 8188, "http://127.0.0.1:8188"),
        ("https", "0.0.0.0", 8443, "https://127.0.0.1:8443"),
        ("http", "127.0.0.1", 8188, "http://127.0.0.1:8188"),
        ("http", "192.168.1.10", 8188, "http://192.168.1.10:8188"),
        ("http", "localhost", 8188, "http://localhost:8188"),
        ("http", "::1", 8188, "http://[::1]:8188"),
        ("http", "2001:db8::1", 8188, "http://[2001:db8::1]:8188"),
    ],
)
def test_auto_launch_opens_a_reachable_url(
    monkeypatch, os_name, scheme, address, port, expected_url
):
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    _load_startup_server(os_name)(scheme, address, port)

    assert opened == [expected_url]
