from __future__ import annotations

import ast
from collections.abc import Callable
import importlib
import importlib.util
import logging
import os
from pathlib import Path
import py_compile
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
GENERIC_REFUSAL = "Custom node pack '{name}' is not permitted by your organization's policy."
MANAGER_REFUSAL = (
    "Custom node pack '{name}' is not loaded: ComfyUI-Manager cannot run under a custom-node policy, "
    "because its startup script installs packs before they are checked."
)
BYTECODE_REFUSAL = (
    "Custom node pack '{name}' is not loaded: it carries compiled Python files the policy cannot check ({paths}). "
    "Delete them and restart ComfyUI."
)


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
async def test_loose_bytecode_beside_source_denies_the_pack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given an approved pack shipping bytecode outside __pycache__
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    digest = governance.pack_digest(str(pack_path))
    (pack_path / "shadow.pyc").write_bytes(b"sourceless\n")
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: digest})

    # When both gates enumerate the pack
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then neither entry point executes
    assert result == (False, False)


def _plant_bytecode(source_path: Path, sentinel: Path) -> Path:
    # Compile other code into the cache file Python would read for source_path; an unchecked-hash .pyc runs without the source.
    impostor = source_path.parent.parent / f"impostor-{source_path.stem}.py"
    impostor.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bytecode', encoding='utf-8')\nNODE_CLASS_MAPPINGS = {{}}\n", encoding="utf-8")
    cache_path = Path(importlib.util.cache_from_source(str(source_path)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(impostor), cfile=str(cache_path), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH, doraise=True)
    impostor.unlink()
    return cache_path


@pytest.mark.asyncio
async def test_allowed_single_file_pack_never_runs_bytecode_beside_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given an approved single-file pack, and bytecode for it in custom_nodes/__pycache__, outside what its digest measures
    monkeypatch.setattr(sys, "pycache_prefix", None)
    custom_nodes_path = tmp_path / "custom_nodes"
    custom_nodes_path.mkdir()
    sentinel = tmp_path / "single-file-import"
    module_path = custom_nodes_path / "goodpack.py"
    module_path.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('source', encoding='utf-8')\nNODE_CLASS_MAPPINGS = {{}}\n",
        encoding="utf-8",
    )
    governance.set_custom_node_policy("allowlist", frozenset(), {module_path.name: governance.pack_digest(str(module_path))})
    _plant_bytecode(module_path, sentinel)
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(custom_nodes_path)] if name == "custom_nodes" else [])

    # When the import loop loads it
    await nodes.init_external_custom_nodes()

    # Then the measured source runs, not the bytecode
    assert sentinel.read_text(encoding="utf-8") == "source"


@pytest.mark.asyncio
async def test_allowed_pack_never_runs_bytecode_from_a_pycache_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given PYTHONPYCACHEPREFIX in effect, and bytecode in the prefix tree for an approved pack's prestartup script and submodule
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "pycache-prefix"))
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    (pack_path / "__init__.py").write_text("from . import helper\nNODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (pack_path / "helper.py").write_text(f"from pathlib import Path\nPath({str(import_sentinel)!r}).write_text('source', encoding='utf-8')\n", encoding="utf-8")
    (pack_path / "prestartup_script.py").write_text(
        f"from pathlib import Path\nPath({str(prestartup_sentinel)!r}).write_text('source', encoding='utf-8')\n", encoding="utf-8"
    )
    _plant_bytecode(pack_path / "helper.py", import_sentinel)
    _plant_bytecode(pack_path / "prestartup_script.py", prestartup_sentinel)
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: governance.pack_digest(str(pack_path))})

    # When both gates load it
    await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then both run the measured source, not the bytecode
    assert prestartup_sentinel.read_text(encoding="utf-8") == "source"
    assert import_sentinel.read_text(encoding="utf-8") == "source"


def test_child_interpreter_never_runs_bytecode_from_a_pycache_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given PYTHONPYCACHEPREFIX in effect, and bytecode in the prefix tree for an approved pack's module
    prefix = tmp_path / "pycache-prefix"
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(prefix))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "")
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    sentinel = tmp_path / "child-import"
    (pack_path / "helper.py").write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('source', encoding='utf-8')\n", encoding="utf-8")
    (pack_path / "install.py").write_text("import helper\n", encoding="utf-8")
    _plant_bytecode(pack_path / "helper.py", sentinel)
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: governance.pack_digest(str(pack_path))})

    # When the pack starts a child interpreter that imports the module
    subprocess.run([sys.executable, str(pack_path / "install.py")], check=True)

    # Then the child runs the measured source, not the bytecode
    assert sentinel.read_text(encoding="utf-8") == "source"


@pytest.mark.asyncio
async def test_pack_carrying_bytecode_logs_the_cause_not_the_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given an approved pack that an earlier ungoverned run left __pycache__ folders in
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path)
    (pack_path / "sub").mkdir()
    (pack_path / "sub" / "__init__.py").write_text("", encoding="utf-8")
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: governance.pack_digest(str(pack_path))})
    for folder in (pack_path / "__pycache__", pack_path / "sub" / "__pycache__"):
        folder.mkdir()
        (folder / "__init__.cpython-313.pyc").write_bytes(b"compiled\n")
        (folder / "other.cpython-313.pyc").write_bytes(b"compiled\n")

    # When both gates enumerate it
    with caplog.at_level(logging.WARNING):
        result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then neither entry point runs, and each gate names the folders to delete instead of blaming the policy
    assert result == (False, False)
    messages = [record.getMessage() for record in caplog.records]
    paths = f"{pack_path / '__pycache__'}, {pack_path / 'sub' / '__pycache__'}"
    assert messages.count(BYTECODE_REFUSAL.format(name=pack_path.name, paths=paths)) == 2
    assert GENERIC_REFUSAL.format(name=pack_path.name) not in messages


def test_pack_gate_denies_pack_when_digest_cannot_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given a governed pack that cannot be measured
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: "blake3:" + "0" * 64})
    monkeypatch.setattr(governance, "pack_digest", lambda _path: (_ for _ in ()).throw(OSError("unreadable")))

    # When the pack gate measures it, then the pack is denied closed
    assert governance.pack_allowed(str(pack_path)) is False


def test_pack_install_in_child_interpreter_keeps_pack_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given an allowed pack whose install.py imports a pack module
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "")
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    (pack_path / "helper.py").write_text("", encoding="utf-8")
    (pack_path / "install.py").write_text("import helper\n", encoding="utf-8")
    governance.set_custom_node_policy("allowlist", frozenset(), {pack_path.name: governance.pack_digest(str(pack_path))})

    # When the install runs in a child interpreter
    subprocess.run([sys.executable, str(pack_path / "install.py")], check=True)

    # Then it leaves no bytecode behind and the pack is still allowed
    assert governance.pack_allowed(str(pack_path)) is True


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


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ComfyUI-Manager", "comfyui-manager", "COMFYUI-MANAGER"])
@pytest.mark.parametrize("mode", ["allowlist", "blocklist"])
async def test_legacy_manager_pack_is_refused_under_any_custom_node_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    name: str,
) -> None:
    # Given a legacy Manager pack the policy would otherwise admit: listed by digest, and not denied
    custom_nodes_path = tmp_path / "custom_nodes"
    pack_path, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, name)
    governance.set_custom_node_policy(mode, frozenset(), {name: governance.pack_digest(str(pack_path))})

    # When both gates enumerate it
    with caplog.at_level(logging.WARNING):
        result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then neither its prestartup script (which runs scheduled installs) nor its import runs, and each gate gives the Manager reason, not a list mistake
    assert result == (False, False)
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count(MANAGER_REFUSAL.format(name=name)) == 2
    assert GENERIC_REFUSAL.format(name=name) not in messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "denied"),
    [
        pytest.param("allowlist", False, id="allowlist-no-entry"),
        pytest.param("blocklist", True, id="blocklist-denied"),
    ],
)
async def test_other_refused_pack_logs_the_generic_policy_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    denied: bool,
) -> None:
    # Given an ordinary pack the policy refuses
    custom_nodes_path = tmp_path / "custom_nodes"
    _, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, "OtherPack")
    governance.set_custom_node_policy(mode, frozenset({"otherpack"}) if denied else frozenset(), {})

    # When both gates enumerate it
    with caplog.at_level(logging.WARNING):
        result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then each gate logs the generic policy message, never the Manager reason
    assert result == (False, False)
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count(GENERIC_REFUSAL.format(name="OtherPack")) == 2
    assert MANAGER_REFUSAL.format(name="OtherPack") not in messages


@pytest.mark.asyncio
async def test_legacy_manager_pack_loads_without_a_custom_node_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given a legacy Manager pack and no custom-node policy
    custom_nodes_path = tmp_path / "custom_nodes"
    _, prestartup_sentinel, import_sentinel = _make_directory_pack(custom_nodes_path, "ComfyUI-Manager")
    governance.set_custom_node_policy(None, frozenset(), {})

    # When both stock loading paths run
    result = await _run_both_gates(monkeypatch, custom_nodes_path, prestartup_sentinel, import_sentinel)

    # Then governance leaves it alone
    assert result == (True, True)
