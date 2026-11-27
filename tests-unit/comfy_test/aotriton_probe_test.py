import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture
def probe(monkeypatch):
    # Load the wrapper without importing torch or initializing a GPU.
    path = Path(__file__).resolve().parents[2] / "comfy" / "model_management.py"
    module = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "aotriton_supported_isolated")
    namespace = {
        "__file__": str(path),
        "os": os,
        "sys": sys,
        "subprocess": subprocess,
        "logging": logging,
        "AOTRITON_PROBE_CHILD_ENV": "COMFYUI_AOTRITON_PROBE",
        "aotriton_supported": Mock(return_value=True),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    monkeypatch.delenv("COMFYUI_AOTRITON_PROBE", raising=False)
    return namespace


@pytest.mark.parametrize("supported", [True, False])
def test_child_reuses_import_time_probe(probe, tmp_path, supported):
    package = tmp_path / "comfy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "model_management.py"
    module.write_text(
        "import os\n"
        "assert os.environ['COMFYUI_AOTRITON_PROBE'] == '1'\n"
        "calls = 0\n"
        "def aotriton_supported():\n"
        "    global calls\n"
        "    calls += 1\n"
        "    assert calls == 1, 'probe ran twice'\n"
        f"    return {supported!r}\n"
        "AOTRITON_SUPPORTED = aotriton_supported()\n"
    )
    probe["__file__"] = str(module)

    assert probe["aotriton_supported_isolated"]() is supported
    probe["aotriton_supported"].assert_not_called()


def test_in_process_override(probe, monkeypatch):
    monkeypatch.setenv("COMFYUI_AOTRITON_PROBE", "1")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    assert probe["aotriton_supported_isolated"]() is True
    probe["aotriton_supported"].assert_called_once_with()
    run.assert_not_called()


@pytest.mark.parametrize(
    "returncode, stdout, expected",
    [
        (0, "AOTRITON 1\n", True),
        (0, "startup output\nAOTRITON 1\n", True),
        (-11, "AOTRITON 1\n", False),
        (1, "AOTRITON 1\n", False),
        (0, "AOTRITON 0\n", False),
        (0, "not AOTRITON 1\n", False),
    ],
)
def test_probe_result(probe, monkeypatch, returncode, stdout, expected):
    result = subprocess.CompletedProcess([], returncode, stdout, "")
    monkeypatch.setattr(subprocess, "run", Mock(return_value=result))

    assert probe["aotriton_supported_isolated"]() is expected


@pytest.mark.parametrize("returncode, stdout", [(0, "AOTRITON 0\n"), (1, "")])
def test_failure_diagnostics(probe, monkeypatch, caplog, returncode, stdout):
    result = subprocess.CompletedProcess([], returncode, stdout, "Could not run flash attention: missing kernel image\n")
    monkeypatch.setattr(subprocess, "run", Mock(return_value=result))

    assert probe["aotriton_supported_isolated"]() is False
    assert "missing kernel image" in caplog.text
    if returncode:
        assert "exited with 1" in caplog.text


@pytest.mark.parametrize("error", [OSError("cannot launch"), subprocess.TimeoutExpired("probe", 120)])
def test_probe_process_errors(probe, monkeypatch, caplog, error):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))

    assert probe["aotriton_supported_isolated"]() is False
    assert "Could not run the flash attention probe process" in caplog.text
