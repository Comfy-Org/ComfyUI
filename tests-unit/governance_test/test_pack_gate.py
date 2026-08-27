from __future__ import annotations

import ast
from collections.abc import Callable
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from app import governance
from comfy.cli_args import args


if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
import nodes


COMFYUI_ROOT = Path(__file__).parents[2]
MAIN_PATH = COMFYUI_ROOT / "main.py"


def _load_execute_prestartup_script() -> Callable[[], None]:
    module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "execute_prestartup_script")
    compiled = compile(ast.Module(body=[function], type_ignores=[]), filename=str(MAIN_PATH), mode="exec")
    namespace = {
        "args": args,
        "folder_paths": folder_paths,
        "governance": governance,
        "importlib": importlib,
        "logging": logging,
        "os": os,
        "time": time,
    }
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself
    return namespace["execute_prestartup_script"]


def _make_directory_pack(root: Path, name: str = "TestPack") -> tuple[Path, Path, Path]:
    pack_path = root / name
    pack_path.mkdir(parents=True)
    prestartup_sentinel = root.parent / f"{name}-prestartup"
    import_sentinel = root.parent / f"{name}-import"
    (pack_path / "prestartup_script.py").write_text(
        f"from pathlib import Path\nPath({str(prestartup_sentinel)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (pack_path / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(import_sentinel)!r}).write_text('ran', encoding='utf-8')\nNODE_CLASS_MAPPINGS = {{}}\n",
        encoding="utf-8",
    )
    return pack_path, prestartup_sentinel, import_sentinel


async def _run_both_gates(
    monkeypatch: pytest.MonkeyPatch,
    custom_nodes_path: Path,
    prestartup_sentinel: Path,
    import_sentinel: Path,
) -> tuple[bool, bool]:
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(custom_nodes_path)] if name == "custom_nodes" else [])

    _load_execute_prestartup_script()()
    await nodes.init_external_custom_nodes()

    return prestartup_sentinel.exists(), import_sentinel.exists()


@pytest.fixture(autouse=True)
def isolated_pack_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(args, "enable_manager", False)
    monkeypatch.setattr(args, "disable_all_custom_nodes", False)
    monkeypatch.setattr(args, "whitelist_custom_nodes", [])
    governance.set_custom_node_policy(None, frozenset(), {})
    yield
    governance.set_custom_node_policy(None, frozenset(), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "has_entry", "tampered", "denied", "expected_to_run"),
    [
        pytest.param("allowlist", True, False, False, True, id="allowlist-match"),
        pytest.param("allowlist", True, True, False, False, id="allowlist-tamper"),
        pytest.param("allowlist", False, False, False, False, id="allowlist-no-entry"),
        pytest.param("blocklist", True, False, False, True, id="blocklist-match"),
        pytest.param("blocklist", True, True, False, False, id="blocklist-tamper"),
        pytest.param("blocklist", False, False, False, True, id="blocklist-unknown"),
        pytest.param("blocklist", False, False, True, False, id="blocklist-denied"),
        pytest.param("blocklist", True, False, True, False, id="blocklist-shipped-and-denied"),
    ],
)
async def test_posture_matrix_applies_at_both_gates_with_manager_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    has_entry: bool,
    tampered: bool,
    denied: bool,
    expected_to_run: bool,
) -> None:
    # Given a pack policy while Manager is absent
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    digest = governance.pack_digest(str(pack_path))
    entry_name = "manifest-name" if mode == "allowlist" else pack_path.name
    allowed_packs = {entry_name: digest} if has_entry else {}
    denied_packs = frozenset({pack_path.name.lower()}) if denied else frozenset()
    governance.set_custom_node_policy(mode, denied_packs, allowed_packs)
    if tampered:
        with (pack_path / "__init__.py").open("a", encoding="utf-8") as stream:
            stream.write("# tampered\n")

    # When both code-execution gates enumerate the pack
    prestartup_ran, import_ran = await _run_both_gates(
        monkeypatch,
        custom_nodes_path,
        prestartup_sentinel,
        import_sentinel,
    )

    # Then Manager absence never bypasses either gate
    assert args.enable_manager is False
    assert (prestartup_ran, import_ran) == (expected_to_run, expected_to_run)


@pytest.mark.asyncio
async def test_blocklist_denied_basename_matching_is_case_insensitive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given a mixed-case pack denied by its lowercase basename
    custom_nodes_path = tmp_path / "custom_nodes"
    _, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, "MiXeDcAsE")
    governance.set_custom_node_policy("blocklist", frozenset({"mixedcase"}), {})

    # When both gates enumerate it
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then neither entry point executes
    assert result == (False, False)


@pytest.mark.asyncio
async def test_blocklist_renamed_denied_pack_documents_known_limitation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given denied shipped bytes moved under an unknown basename
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, "renamed-pack")
    governance.set_custom_node_policy(
        "blocklist",
        frozenset({"blocked-pack"}),
        {"blocked-pack": governance.pack_digest(str(pack_path))},
    )

    # When both gates enumerate the renamed pack
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then blocklist posture treats it as an unknown permitted pack
    assert result == (True, True)


@pytest.mark.asyncio
async def test_absent_policy_preserves_stock_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given policy data with custom-node governance unset
    custom_nodes_path = tmp_path / "custom_nodes"
    _, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    governance.set_custom_node_policy(None, frozenset({"testpack"}), {"TestPack": "blake3:" + "0" * 64})

    # When both stock loading paths run
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then governance does not alter either path
    assert result == (True, True)


@pytest.mark.asyncio
async def test_existing_disable_all_flag_still_narrows_allowed_pack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given a governance-allowed pack disabled by the existing CLI flag
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    governance.set_custom_node_policy("allowlist", frozenset(), {"manifest-name": governance.pack_digest(str(pack_path))})
    monkeypatch.setattr(args, "disable_all_custom_nodes", True)
    monkeypatch.setattr(args, "whitelist_custom_nodes", [])

    # When both loading paths run
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then governance cannot broaden the existing restriction
    assert result == (False, False)


@pytest.mark.asyncio
async def test_denied_single_file_module_is_never_imported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given an unknown single-file pack under strict allowlist posture
    custom_nodes_path = tmp_path / "custom_nodes"
    custom_nodes_path.mkdir()
    sentinel = tmp_path / "single-file-import"
    module_path = custom_nodes_path / "unknown.py"
    module_path.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran', encoding='utf-8')\nNODE_CLASS_MAPPINGS = {{}}\n",
        encoding="utf-8",
    )
    governance.set_custom_node_policy("allowlist", frozenset(), {})
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(custom_nodes_path)] if name == "custom_nodes" else [])

    # When the import loop enumerates it
    await nodes.init_external_custom_nodes()

    # Then the module body never executes
    assert not sentinel.exists()


def test_real_main_rejects_unknown_pack_without_manager(tmp_path: Path) -> None:
    # Given a real startup with Manager absent and an unknown allowlist pack
    custom_nodes_path = tmp_path / "custom_nodes"
    _, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, "unknown-pack")
    setup = (
        "import runpy, sys\n"
        "from app import governance\n"
        "governance.initialize = lambda: governance.set_custom_node_policy('allowlist', frozenset(), {})\n"
        f"sys.argv = [{str(MAIN_PATH)!r}, '--base-directory', {str(tmp_path)!r}, '--cpu', '--disable-api-nodes', '--quick-test-for-ci']\n"
        f"runpy.run_path({str(MAIN_PATH)!r}, run_name='__main__')\n"
    )

    # When main.py runs through custom-node initialization
    result = subprocess.run(
        [sys.executable, "-c", setup],
        cwd=COMFYUI_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    # Then startup succeeds without executing either unknown-pack entry point
    assert result.returncode == 0, result.stdout + result.stderr
    assert not prestartup_sentinel.exists()
    assert not import_sentinel.exists()
