from __future__ import annotations

import ast
import ipaddress
from pathlib import Path
from types import SimpleNamespace


def _load_warn_if_manager_exposed(args):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "warn_if_manager_exposed")
    compiled = compile(ast.Module(body=[function], type_ignores=[]), filename=str(main_path), mode="exec")
    warnings: list[str] = []
    namespace = {
        "args": args,
        "app": SimpleNamespace(logger=SimpleNamespace(log_startup_warning=warnings.append)),
        "ipaddress": ipaddress,
    }
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["warn_if_manager_exposed"], warnings


def _args(**overrides):
    values = {"enable_manager": True, "disable_manager_ui": False, "listen": "127.0.0.1"}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_no_warning_on_loopback():
    warn, warnings = _load_warn_if_manager_exposed(_args())

    warn()

    assert warnings == []


def test_no_warning_on_ipv6_loopback_and_localhost():
    warn, warnings = _load_warn_if_manager_exposed(_args(listen="::1,localhost"))

    warn()

    assert warnings == []


def test_no_warning_when_manager_disabled():
    warn, warnings = _load_warn_if_manager_exposed(_args(enable_manager=False, listen="0.0.0.0"))

    warn()

    assert warnings == []


def test_no_warning_when_manager_ui_disabled():
    warn, warnings = _load_warn_if_manager_exposed(_args(disable_manager_ui=True, listen="0.0.0.0"))

    warn()

    assert warnings == []


def test_warning_on_wildcard_listen():
    warn, warnings = _load_warn_if_manager_exposed(_args(listen="0.0.0.0,::"))

    warn()

    assert len(warnings) == 1
    assert "ComfyUI-Manager" in warnings[0]
    assert "0.0.0.0,::" in warnings[0]


def test_warning_when_any_listen_address_is_not_loopback():
    warn, warnings = _load_warn_if_manager_exposed(_args(listen="127.0.0.1,192.168.1.10"))

    warn()

    assert len(warnings) == 1
