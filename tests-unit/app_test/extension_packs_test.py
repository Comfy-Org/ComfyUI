import ast
import importlib
import logging
import os
from pathlib import Path
import time

import pytest
import torch

from app import extension_packs, governance
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
import nodes


MAIN_PATH = Path(__file__).parents[2] / "main.py"
REGISTER = "import { comfy } from '/comfy/api/v4.js'\nexport const pack = comfy.registerPack(import.meta)\n"


def _pack(root: Path, folder: str, pyproject: str | None = '[project]\nname = "test-pack"\nversion = "1.2.0"\n', js: str | None = REGISTER) -> Path:
    path = root / folder
    (path / "web").mkdir(parents=True)
    if pyproject is not None:
        (path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    if js is not None:
        (path / "web" / "main.js").write_text(js, encoding="utf-8")
    return path


def _refusal(root: Path, folder: str, pyproject: str | None) -> str:
    path = _pack(root, folder, pyproject)
    identities, refusals = extension_packs.check_packs([str(path)], {})
    assert identities == {}
    return refusals[str(path)]


@pytest.mark.parametrize("file_name", ["main.js", "main.mjs", "deep/nested/main.js"])
def test_register_pack_call_marks_a_v4_pack(tmp_path, file_name):
    target = tmp_path / "pack" / file_name
    target.parent.mkdir(parents=True)
    target.write_text("const pack = comfy.registerPack(import.meta)\n", encoding="utf-8")
    assert extension_packs.calls_register_pack(str(tmp_path / "pack"))


@pytest.mark.parametrize("file_name", ["node_modules/x/main.js", ".git/main.js", "__pycache__/main.js", "web/main.ts", "nodes.py", "web/main.js.map"])
def test_register_pack_outside_scanned_files_is_not_a_v4_pack(tmp_path, file_name):
    target = tmp_path / "pack" / file_name
    target.parent.mkdir(parents=True)
    target.write_text("comfy.registerPack(import.meta)\n", encoding="utf-8")
    assert not extension_packs.calls_register_pack(str(tmp_path / "pack"))


@pytest.mark.parametrize("source", [
    "comfy.registerPack( import.meta )",
    "comfy.registerPack(\n  import.meta,\n)",
    "registerPack(import . meta)",
    "comfy.registerPack?.(import.meta)",
    "comfy.registerPack ?. ( import.meta )",
    "comfy.registerPack?.(\n  import.meta\n)",
])
def test_register_pack_call_may_span_whitespace(tmp_path, source):
    _pack(tmp_path, "pack", js=f"export const pack = {source}\n")
    assert extension_packs.calls_register_pack(str(tmp_path / "pack"))


@pytest.mark.parametrize("source", ["function registerPack(name) {}", "// call registerPack() first", "// registerPack?.() when available", "lib.registerPack(meta)", "lib.registerPack?.(meta)", "registerPack(import.metadata)", "registerPack(importMeta)", "registerPack?(import.meta)"])
def test_other_register_pack_is_not_a_v4_pack(tmp_path, source):
    _pack(tmp_path, "pack", js=source + "\n")
    assert not extension_packs.calls_register_pack(str(tmp_path / "pack"))


def test_legacy_pack_without_pyproject_is_not_checked(tmp_path):
    path = _pack(tmp_path, "legacy", pyproject=None, js="app.registerExtension({ name: 'legacy' })\n")
    assert extension_packs.check_packs([str(path), str(tmp_path / "single_file.py")], {}) == ({}, {})


@pytest.mark.parametrize(
    ("pyproject", "expected"),
    [
        ('[project]\nname = "test-pack"\nversion = "1.2.0"\n', ("test-pack", "1.2.0")),
        ('[project]\nname = "  Test.Pack_2  "\nversion = "0.1"\n', ("test.pack_2", "0.1")),
        ('[project]\nname = "a"\n', ("a", None)),
        ('[project]\nname = "test-pack"\ndynamic = ["version"]\n', ("test-pack", None)),
        ('[project]\nname = "test-pack"\nversion = 2\n', ("test-pack", None)),
        ('[project]\nname = "' + "a" * 100 + '"\n', ("a" * 100, None)),
    ],
)
def test_pyproject_gives_the_identity(tmp_path, pyproject, expected):
    path = _pack(tmp_path, "pack", pyproject)
    assert extension_packs.check_packs([str(path)], {}) == ({str(path): expected}, {})


def test_missing_pyproject_is_refused(tmp_path):
    assert _refusal(tmp_path, "pack", None).endswith('is not loaded: it has no pyproject.toml; add one beside its __init__.py with [project] name = "<registry id>".')


def test_unparseable_pyproject_is_refused(tmp_path):
    reason = _refusal(tmp_path, "pack", '[project\nname = "test-pack"\n')
    assert "is not loaded: its pyproject.toml does not parse (Expected ']'" in reason
    assert reason.endswith("); fix the TOML so it gives [project] name.")


@pytest.mark.parametrize("pyproject", ['[tool.comfy]\nweb = "web"\n', '[project]\nversion = "1.0"\n', '[project]\nname = "  "\n', '[project]\nname = 5\n'])
def test_missing_name_is_refused(tmp_path, pyproject):
    assert _refusal(tmp_path, "pack", pyproject).endswith('is not loaded: its pyproject.toml has no [project] name; add name = "<registry id>" under [project].')


@pytest.mark.parametrize("name", ["-pack", "pack-", "pack.", "my pack", "pack/../x", "päck", "a" * 101])
def test_name_that_is_not_a_registry_id_is_refused(tmp_path, name):
    assert _refusal(tmp_path, "pack", f'[project]\nname = "{name}"\n').endswith(f"[project] name '{name}' is not a registry id; change it to 1-100 letters, digits, '.', '_' or '-', starting and ending with a letter or digit.")


@pytest.mark.parametrize("name", ["comfy-core", "My-Pack", " my-pack "])
def test_reserved_name_is_refused(tmp_path, name):
    assert _refusal(tmp_path, "pack", f'[project]\nname = "{name}"\n').endswith(f"[project] name '{name}' is reserved; change it to the pack's own registry id.")


@pytest.mark.parametrize("name", ["title", "id", "pos", "showAdvanced", "ShowAdvanced", "widgets_values_named", " Extensions "])
def test_node_field_name_is_refused(tmp_path, name):
    assert _refusal(tmp_path, "pack", f'[project]\nname = "{name}"\n').endswith(f"[project] name '{name}' is a field every node saves; change it to the pack's own registry id.")


def test_duplicate_id_refuses_every_claimant(tmp_path):
    first = _pack(tmp_path, "first", '[project]\nname = "test-pack"\n')
    second = _pack(tmp_path, "second", '[project]\nname = "Test-Pack"\n')
    other = _pack(tmp_path, "other", '[project]\nname = "other-pack"\n')
    identities, refusals = extension_packs.check_packs([str(first), str(second), str(other)], {})
    assert identities == {str(other): ("other-pack", None)}
    assert set(refusals) == {str(first), str(second)}
    for reason in refusals.values():
        assert reason.endswith(f"its id 'test-pack' is claimed by more than one folder ({first}, {second}); remove or disable all but one of them.")


def test_duplicate_id_of_a_legacy_pack_is_not_a_conflict(tmp_path):
    v4 = _pack(tmp_path, "v4", '[project]\nname = "test-pack"\n')
    legacy = _pack(tmp_path, "legacy", '[project]\nname = "test-pack"\n', js="app.registerExtension({})\n")
    assert extension_packs.check_packs([str(v4), str(legacy)], {}) == ({str(v4): ("test-pack", None)}, {})


def test_records_cover_each_served_folder_of_a_verified_pack(tmp_path):
    pack = str(tmp_path / "pack")
    web_dirs = {
        "pack": str(tmp_path / "pack" / "web"),
        "test-pack": str(tmp_path / "pack" / "js"),
        "legacy": str(tmp_path / "legacy" / "web"),
        "outside": str(tmp_path / "shared" / "web"),
    }
    assert extension_packs.pack_records(web_dirs, {pack: ("test-pack", "1.2.0")}) == [
        {"segment": "pack", "id": "test-pack", "version": "1.2.0"},
        {"segment": "test-pack", "id": "test-pack", "version": "1.2.0"},
    ]


def test_check_custom_nodes_reuses_the_result_for_the_same_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(extension_packs, "_checked", None)
    first = _pack(tmp_path, "first", '[project]\nname = "comfy-core"\n')
    (tmp_path / "single_file.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    result = extension_packs.check_custom_nodes([str(first)])
    assert extension_packs.check_custom_nodes([str(tmp_path / "__pycache__"), str(first), str(tmp_path / "single_file.py")]) is result
    second = _pack(tmp_path, "second", '[project]\nname = "second-pack"\n')
    assert extension_packs.check_custom_nodes([str(first), str(second)]) == ({str(second): ("second-pack", None)}, result[1])


def test_refusal_stays_when_the_folders_change(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(extension_packs, "_checked", None)
    first = _pack(tmp_path, "first", '[project]\nname = "test-pack"\n')
    second = _pack(tmp_path, "second", '[project]\nname = "test-pack"\n')
    earlier = extension_packs.check_custom_nodes([str(first), str(second)])[1]
    assert set(earlier) == {str(first), str(second)}

    new_bad = _pack(tmp_path, "new_bad", '[project]\nname = "comfy-core"\n')
    new_good = _pack(tmp_path, "new_good", '[project]\nname = "new-pack"\n')
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        identities, refusals = extension_packs.check_custom_nodes([str(first), str(new_bad), str(new_good)])
    assert identities == {str(new_good): ("new-pack", None)}
    assert refusals[str(first)] == earlier[str(first)]
    assert set(refusals) == {str(first), str(new_bad)}
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR] == [refusals[str(new_bad)]]


def test_new_folder_claiming_a_verified_id_is_refused_alone(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(extension_packs, "_checked", None)
    holder = _pack(tmp_path, "holder", '[project]\nname = "test-pack"\nversion = "1.0"\n')
    assert extension_packs.check_custom_nodes([str(holder)]) == ({str(holder): ("test-pack", "1.0")}, {})

    first = _pack(tmp_path, "newcomer_1", '[project]\nname = "Test-Pack"\n')
    second = _pack(tmp_path, "newcomer_2", '[project]\nname = "test-pack"\n')
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        identities, refusals = extension_packs.check_custom_nodes([str(holder), str(first), str(second)])
    assert identities == {str(holder): ("test-pack", "1.0")}
    assert refusals == {
        str(first): f"V4 pack {first} is not loaded: its id 'test-pack' is already held by {holder}; remove or disable one of them.",
        str(second): f"V4 pack {second} is not loaded: its id 'test-pack' is already held by {holder}; remove or disable one of them.",
    }
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR] == [refusals[str(first)], refusals[str(second)]]


def _execute_prestartup_script():
    module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "execute_prestartup_script")
    namespace = {"args": args, "extension_packs": extension_packs, "folder_paths": folder_paths, "governance": governance, "importlib": importlib, "logging": logging, "os": os, "time": time}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename=str(MAIN_PATH), mode="exec"), namespace)  # noqa: S102 - trusted AST extracted from main.py itself
    namespace["execute_prestartup_script"]()


@pytest.mark.asyncio
async def test_refused_pack_runs_no_code(tmp_path, monkeypatch, caplog):
    custom_nodes = tmp_path / "custom_nodes"
    stub = "from pathlib import Path\nPath(__file__).with_name('imported').write_text('ran')\nWEB_DIRECTORY = './web'\nNODE_CLASS_MAPPINGS = {}\n"
    prestartup = "from pathlib import Path\nPath(__file__).with_name('prestarted').write_text('ran')\n"
    good = _pack(custom_nodes, "good_pack", '[project]\nname = "Good-Pack"\nversion = "1.2.0"\n')
    bad = _pack(custom_nodes, "bad_pack", '[project]\nname = "comfy-core"\n')
    legacy = _pack(custom_nodes, "legacy_pack", pyproject=None, js="app.registerExtension({})\n")
    for path in (good, bad, legacy):
        (path / "__init__.py").write_text(stub, encoding="utf-8")
        (path / "prestartup_script.py").write_text(prestartup, encoding="utf-8")

    monkeypatch.setattr(args, "enable_manager", False)
    monkeypatch.setattr(args, "disable_all_custom_nodes", False)
    governance.set_custom_node_policy(None, frozenset(), {})
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(custom_nodes)] if name == "custom_nodes" else [])
    monkeypatch.setattr(nodes, "EXTENSION_WEB_DIRS", {})
    monkeypatch.setattr(nodes, "EXTENSION_PACKS", [])
    monkeypatch.setattr(extension_packs, "_checked", None)

    with caplog.at_level(logging.INFO):
        _execute_prestartup_script()
        await nodes.init_external_custom_nodes()

    for path in (good, legacy):
        assert (path / "prestarted").exists()
        assert (path / "imported").exists()
    assert not (bad / "prestarted").exists()
    assert not (bad / "imported").exists()
    assert "bad_pack" not in nodes.EXTENSION_WEB_DIRS
    assert nodes.EXTENSION_PACKS == [{"segment": "good_pack", "id": "good-pack", "version": "1.2.0"}]
    assert caplog.text.count("Checked V4 packs in ") == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [f"V4 pack {bad} is not loaded: [project] name 'comfy-core' is reserved; change it to the pack's own registry id."]
    assert f"(IMPORT FAILED): {bad}" in caplog.text
