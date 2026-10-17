import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import webbrowser

import pytest


def _load_startup_server(os_name):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    start_comfyui = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "start_comfyui")
    function = next(node for node in ast.walk(start_comfyui) if isinstance(node, ast.FunctionDef) and node.name == "startup_server")
    compiled = compile(ast.Module(body=[function], type_ignores=[]), filename=str(main_path), mode="exec")
    namespace = {"os": SimpleNamespace(name=os_name)}
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["startup_server"]


@pytest.mark.parametrize("os_name", ["posix", "nt"])
@pytest.mark.parametrize(
    "scheme,address,port,expected_url",
    [
        ("http", "0.0.0.0", 8188, "http://127.0.0.1:8188"),
        ("https", "0.0.0.0", 8443, "https://127.0.0.1:8443"),
        ("http", "127.0.0.1", 8188, "http://127.0.0.1:8188"),
        ("http", "192.168.1.10", 8188, "http://192.168.1.10:8188"),
        ("http", "localhost", 8188, "http://localhost:8188"),
        ("http", "::1", 8188, "http://[::1]:8188"),
        ("http", "2001:db8::1", 8188, "http://[2001:db8::1]:8188"),
        ("http", "::", 8188, "http://[::]:8188"),
    ],
)
def test_startup_server_browser_url(monkeypatch, os_name, scheme, address, port, expected_url):
    startup_server = _load_startup_server(os_name)
    open_browser = Mock()
    monkeypatch.setattr(webbrowser, "open", open_browser)

    startup_server(scheme, address, port)

    open_browser.assert_called_once_with(expected_url)
