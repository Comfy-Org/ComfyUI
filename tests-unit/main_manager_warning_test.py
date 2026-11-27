from __future__ import annotations

import ast
import asyncio
import ipaddress
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _load_warn_if_manager_exposed(args, custom_nodes_paths=()):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in ("legacy_manager_pack_will_load", "warn_if_manager_exposed")]
    compiled = compile(ast.Module(body=functions, type_ignores=[]), filename=str(main_path), mode="exec")
    warnings: list[str] = []
    namespace = {
        "args": args,
        "app": SimpleNamespace(logger=SimpleNamespace(log_startup_warning=warnings.append)),
        "ipaddress": ipaddress,
        "os": os,
        "folder_paths": SimpleNamespace(get_folder_paths=lambda name: list(custom_nodes_paths)),
    }
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["warn_if_manager_exposed"], warnings


def _args(**overrides):
    values = {
        "enable_manager": True,
        "disable_manager_ui": False,
        "listen": "127.0.0.1",
        "disable_all_custom_nodes": False,
        "whitelist_custom_nodes": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_custom_nodes(tmp_path, pack_name):
    pack = Path(tmp_path) / "custom_nodes" / pack_name
    (pack / "glob").mkdir(parents=True)
    (pack / "__init__.py").write_text("")
    (pack / "glob" / "manager_server.py").write_text("")
    return [str(Path(tmp_path) / "custom_nodes")]


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


def test_no_warning_on_uppercase_localhost():
    warn, warnings = _load_warn_if_manager_exposed(_args(listen="LOCALHOST,LocalHost"))

    warn()

    assert warnings == []


def test_warning_when_localhost_mixed_with_non_loopback():
    warn, warnings = _load_warn_if_manager_exposed(_args(listen="LOCALHOST,0.0.0.0"))

    warn()

    assert len(warnings) == 1


def test_warning_when_ui_disabled_but_legacy_pack_loads():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "manager")
        warn, warnings = _load_warn_if_manager_exposed(
            _args(disable_manager_ui=True, listen="0.0.0.0"), custom_nodes
        )

        warn()

    assert len(warnings) == 1
    assert "ComfyUI-Manager" in warnings[0]


def test_no_warning_when_ui_disabled_and_legacy_pack_blocked_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "ComfyUI-Manager")
        warn, warnings = _load_warn_if_manager_exposed(
            _args(disable_manager_ui=True, listen="0.0.0.0"), custom_nodes
        )

        warn()

    assert warnings == []


def test_no_warning_when_ui_disabled_and_legacy_pack_is_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "manager.disabled")
        warn, warnings = _load_warn_if_manager_exposed(
            _args(disable_manager_ui=True, listen="0.0.0.0"), custom_nodes
        )

        warn()

    assert warnings == []


def test_no_warning_when_ui_disabled_and_custom_nodes_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "manager")
        warn, warnings = _load_warn_if_manager_exposed(
            _args(disable_manager_ui=True, listen="0.0.0.0", disable_all_custom_nodes=True),
            custom_nodes,
        )

        warn()

    assert warnings == []


def test_no_warning_when_legacy_pack_loads_on_loopback():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "manager")
        warn, warnings = _load_warn_if_manager_exposed(
            _args(disable_manager_ui=True, listen="127.0.0.1"), custom_nodes
        )

        warn()

    assert warnings == []


def _load_start_comfyui(args, custom_nodes_paths=()):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in ("legacy_manager_pack_will_load", "warn_if_manager_exposed", "start_comfyui")]
    compiled = compile(ast.Module(body=functions, type_ignores=[]), filename=str(main_path), mode="exec")
    warnings: list[str] = []
    prompt_server = SimpleNamespace(prompt_queue=None, add_routes=lambda: None)
    namespace = {
        "args": args,
        "app": SimpleNamespace(logger=SimpleNamespace(log_startup_warning=warnings.append)),
        "ipaddress": ipaddress,
        "os": os,
        "logging": logging,
        "asyncio": asyncio,
        "folder_paths": SimpleNamespace(
            set_temp_directory=lambda path: None,
            get_temp_directory=lambda: tempfile.gettempdir(),
            get_folder_paths=lambda name: list(custom_nodes_paths),
        ),
        "dependencies_available": lambda: True,
        "missing_dependencies": lambda: [],
        "stop_startup": lambda *a, **k: None,
        "get_missing_requirements_message": lambda: "",
        "WITHOUT_ASSETS": "",
        "default_asset_manager": lambda: SimpleNamespace(enabled=True),
        "feature_flags": SimpleNamespace(set_server_feature=lambda *a, **k: None),
        "cleanup_temp_filesystem": lambda: None,
        "server": SimpleNamespace(PromptServer=lambda loop, asset_manager: prompt_server),
        "comfyui_manager": SimpleNamespace(start=lambda: None),
        "hook_breaker_ac10a0": SimpleNamespace(save_functions=lambda: None, restore_functions=lambda: None),
        "nodes": SimpleNamespace(init_extra_nodes=lambda **k: None),
        "governance": SimpleNamespace(apply_disabled_nodes=lambda disabled: None),
        "comfy": SimpleNamespace(model_management=SimpleNamespace(set_cudnn_benchmark=lambda: None)),
        "cuda_malloc_warning": lambda: None,
        "setup_database": lambda asset_manager: None,
        "hijack_progress": lambda server: None,
        "prompt_worker": lambda *a, **k: None,
        "threading": SimpleNamespace(Thread=lambda *a, **k: SimpleNamespace(start=lambda: None)),
    }
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["start_comfyui"], warnings


def _startup_args(**overrides):
    values = {
        "enable_manager": True,
        "disable_manager_ui": False,
        "listen": "0.0.0.0",
        "temp_directory": None,
        "disable_assets": True,
        "disable_all_custom_nodes": False,
        "whitelist_custom_nodes": [],
        "disable_partner_nodes": False,
        "disabled_nodes_config": None,
        "quick_test_for_ci": False,
        "auto_launch": False,
        "enable_agent": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_start_comfyui_warns_when_manager_exposed():
    start, warnings = _load_start_comfyui(_startup_args())

    start(SimpleNamespace(run_until_complete=lambda value: None))

    assert len(warnings) == 1
    assert "ComfyUI-Manager" in warnings[0]


def test_start_comfyui_no_warning_on_loopback():
    start, warnings = _load_start_comfyui(_startup_args(listen="127.0.0.1"))

    start(SimpleNamespace(run_until_complete=lambda value: None))

    assert warnings == []


def test_start_comfyui_warns_when_ui_disabled_but_legacy_pack_loads():
    with tempfile.TemporaryDirectory() as tmp:
        custom_nodes = _make_custom_nodes(tmp, "manager")
        start, warnings = _load_start_comfyui(_startup_args(disable_manager_ui=True), custom_nodes)

        start(SimpleNamespace(run_until_complete=lambda value: None))

    assert len(warnings) == 1
    assert "ComfyUI-Manager" in warnings[0]
