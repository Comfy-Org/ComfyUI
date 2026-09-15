import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest

from app import governance
from comfy.cli_args import args


COMFYUI_ROOT = Path(__file__).parents[2]
MAIN_PATH = COMFYUI_ROOT / "main.py"
POLICY_MESSAGE = "organization's policy"


@pytest.fixture
def governed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    policy_path = tmp_path / "policy.signed.json"
    monkeypatch.setattr(governance, "GOVERNANCE_REQUIRED", True, raising=False)
    monkeypatch.setattr(governance, "_POLICY_PATH", policy_path, raising=False)
    monkeypatch.setattr(governance, "_EXTRA_MODEL_PATHS_CONFIG_PATH", tmp_path / "extra_model_paths.yaml", raising=False)
    monkeypatch.setattr(governance, "_policy", None, raising=False)
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset(), raising=False)
    monkeypatch.setattr(args, "disabled_nodes_config", None)
    monkeypatch.setattr(args, "extra_model_paths_config", None)
    return policy_path


def _assert_policy_exit(exc_info: pytest.ExceptionInfo[SystemExit], log_text: str) -> None:
    assert exc_info.value.code != 0
    assert POLICY_MESSAGE in log_text


def _run_governed(
    main_path: Path,
    *arguments: str,
    policy_path: Path | None = None,
    extra_model_paths_path: Path | None = None,
    import_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    setup = [
        "from pathlib import Path",
        "import logging",
        "import runpy",
        "import sys",
        "import types",
        "from app import governance",
        "governance.GOVERNANCE_REQUIRED = True",
        "def stub(name, **values):\n    module = types.ModuleType(name)\n    module.__dict__.update(values)\n    sys.modules[name] = module\n    return module",
        "noop = lambda *args, **kwargs: None",
        "folder_paths = stub('folder_paths', __file__='folder_paths.py', base_path='', models_dir='', get_output_directory=lambda: '', add_model_folder_path=noop, set_output_directory=noop, set_input_directory=noop, set_user_directory=noop, get_folder_paths=lambda name: [])",
        "stub('app.logger', setup_logger=noop)",
        "stub('app.assets.seeder', asset_seeder=types.SimpleNamespace(shutdown=noop))",
        "stub('app.assets.services', register_output_files=noop)",
        "import utils",
        "utils.extra_config = stub('utils.extra_config', load_extra_path_config=lambda path: logging.warning('Adding extra search path checkpoints %s', path))",
        "stub('utils.mime_types', init_mime_types=noop)",
        "stub('comfy_execution.progress', get_progress_state=noop)",
        "stub('comfy_execution.utils', get_executing_context=noop)",
        "stub('comfy_api', feature_flags=types.SimpleNamespace())",
        "stub('app.database.db', init_db=noop, dependencies_available=lambda: False)",
        "control = stub('comfy_aimdo.control', init=noop)",
        "stub('comfy_aimdo', control=control)",
        "stub('cuda_malloc', get_torch_version_noimport=lambda: '')",
    ]
    if policy_path is not None:
        setup.extend(
            (
                f"governance._POLICY_PATH = Path({str(policy_path)!r})",
                "governance.verify_and_load = lambda envelope_bytes: {}",
            )
        )
    if extra_model_paths_path is not None:
        setup.append(f"governance._EXTRA_MODEL_PATHS_CONFIG_PATH = Path({str(extra_model_paths_path)!r})")
    if import_path is not None:
        setup.append(f"sys.path.insert(0, {str(import_path)!r})")
    setup.append(f"runpy.run_path({str(main_path)!r}, run_name='__main__')")

    return subprocess.run(
        [sys.executable, "-c", "\n".join(setup), *arguments],
        cwd=COMFYUI_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_build_constants_have_upstream_safe_defaults() -> None:
    assert governance.GOVERNANCE_REQUIRED is False
    assert governance.GOVERNANCE_BUILD_IDENTITY == ""
    assert governance.GOVERNANCE_PUBLIC_KEY == ""
    assert governance.GOVERNANCE_MIN_POLICY_GENERATION == 0
    assert governance.GOVERNANCE_CAPABILITY_VERSION == 1


def test_governance_import_does_not_require_policy_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

original_import = builtins.__import__

def import_without_policy_dependencies(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'blake3', 'cryptography'}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_policy_dependencies
from app import governance
assert governance.GOVERNANCE_REQUIRED is False
""",
        ],
        cwd=COMFYUI_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_initialize_is_noop_without_filesystem_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_filesystem_access(*_args, **_kwargs):
        pytest.fail("initialize accessed the filesystem")

    monkeypatch.setattr(governance, "GOVERNANCE_REQUIRED", False, raising=False)
    monkeypatch.setattr(builtins, "open", unexpected_filesystem_access)
    monkeypatch.setattr(os.path, "isfile", unexpected_filesystem_access)
    monkeypatch.setattr(Path, "is_file", unexpected_filesystem_access)
    monkeypatch.setattr(Path, "read_bytes", unexpected_filesystem_access)

    governance.initialize()


def test_initialize_exits_when_manifest_is_missing(governed: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)


def test_initialize_exits_on_internal_exception(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    governed.write_bytes(b"signed policy")

    def fail_verification(_envelope_bytes: bytes):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(governance, "verify_and_load", fail_verification, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)


def test_initialize_rejects_disabled_nodes_config(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(args, "disabled_nodes_config", str(governed.parent / "disabled.yaml"))

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)


def test_initialize_does_not_gate_on_capability_version(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "GOVERNANCE_CAPABILITY_VERSION", -1, raising=False)
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: {}, raising=False)

    governance.initialize()


@pytest.mark.parametrize("active_forms", [["model"], ["customNode", "model"], ["unknownForm"]])
def test_initialize_exits_on_form_this_build_cannot_enforce(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    active_forms: list[str],
) -> None:
    governed.write_bytes(b"signed policy")
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: {"activeForms": active_forms}, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        governance.initialize()

    _assert_policy_exit(exc_info, caplog.text)
    assert governance._policy is None


def test_initialize_applies_policy_limited_to_enforced_forms(
    governed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governed.write_bytes(b"signed policy")
    policy = {"activeForms": ["nodeId"], "disabledNodes": ["SomeNode"]}
    monkeypatch.setattr(governance, "verify_and_load", lambda envelope_bytes: policy, raising=False)

    governance.initialize()

    assert governance._disabled_nodes == frozenset({"SomeNode"})


def test_manager_import_waits_until_after_governance(tmp_path: Path) -> None:
    sentinel_path = tmp_path / "manager-imported"
    manager_package = tmp_path / "comfyui_manager"
    manager_package.mkdir()
    (manager_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_path)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = _run_governed(
        MAIN_PATH,
        "--enable-manager",
        "--quick-test-for-ci",
        import_path=tmp_path,
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert not sentinel_path.exists()


def test_governed_main_rejects_cli_extra_model_paths(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.signed.json"
    policy_path.write_bytes(b"signed policy")
    extra_paths = tmp_path / "extra-paths.yaml"
    extra_paths.write_text("test:\n  base_path: .\n  checkpoints: models\n", encoding="utf-8")

    result = _run_governed(
        MAIN_PATH,
        "--extra-model-paths-config",
        str(extra_paths),
        "--quick-test-for-ci",
        policy_path=policy_path,
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert "Adding extra search path checkpoints" not in result.stdout + result.stderr


def test_governed_main_rejects_applied_default_extra_model_paths(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.signed.json"
    policy_path.write_bytes(b"signed policy")
    main_path = tmp_path / "main.py"
    main_path.write_bytes(MAIN_PATH.read_bytes())
    extra_paths = tmp_path / "extra_model_paths.yaml"
    extra_paths.write_text("test:\n  base_path: .\n  checkpoints: models\n", encoding="utf-8")

    result = _run_governed(
        main_path,
        "--quick-test-for-ci",
        policy_path=policy_path,
        extra_model_paths_path=extra_paths,
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert "Adding extra search path checkpoints" not in result.stdout + result.stderr


def test_governance_failure_precedes_prestartup_scripts(tmp_path: Path) -> None:
    sentinel_path = tmp_path / "prestartup-ran"
    pack_path = tmp_path / "custom_nodes" / "sentinel_pack"
    pack_path.mkdir(parents=True)
    (pack_path / "prestartup_script.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_path)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = _run_governed(
        MAIN_PATH,
        "--base-directory",
        str(tmp_path),
        "--quick-test-for-ci",
    )

    assert result.returncode != 0
    assert POLICY_MESSAGE in result.stdout + result.stderr
    assert not sentinel_path.exists()
