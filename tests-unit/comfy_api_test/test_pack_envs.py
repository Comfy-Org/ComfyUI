"""Pack isolation's per-pack environments: no dependency conflicts between packs, or with ComfyUI."""
import importlib.util
import os
import subprocess
import zipfile

import pytest

from comfy_api._impl.isolation import envs, server


def _wheel(folder, name, version):
    """A minimal pure-Python wheel: ``import <name>; <name>.VERSION``."""
    dist = f"{name}-{version}.dist-info"
    path = folder / f"{name}-{version}-py3-none-any.whl"
    files = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for file, text in files.items():
            archive.writestr(file, text)
        archive.writestr(f"{dist}/RECORD", "".join(f"{file},,\n" for file in [*files, f"{dist}/RECORD"]))
    return path


def _pack(root, name, requirements, check):
    pack = root / name
    pack.mkdir()
    (pack / "requirements.txt").write_text(requirements)
    (pack / "__init__.py").write_text(check + "\nNODE_CLASS_MAPPINGS = {}\n")
    return str(pack)


@pytest.fixture
def index(tmp_path, monkeypatch):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "tiny_dep", "1.0")
    _wheel(wheels, "tiny_dep", "2.0")
    monkeypatch.setenv(envs.ENV_DIR_ENV, str(tmp_path / "envs"))
    monkeypatch.setenv(envs.PIP_ARGS_ENV, f"--no-index --find-links {wheels}")
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    monkeypatch.delenv(envs.ENVS_ENV, raising=False)
    return tmp_path


def _run(python, code):
    return subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.split()


def test_two_packs_use_conflicting_versions_side_by_side(index):
    one = _pack(index, "pack_one", "tiny_dep==1.0\n", "import tiny_dep\nassert tiny_dep.VERSION == '1.0'")
    two = _pack(index, "pack_two", "tiny_dep==2.0\n", "import tiny_dep\nassert tiny_dep.VERSION == '2.0'")
    python_one, python_two = envs.python_for(one), envs.python_for(two)
    assert python_one != python_two
    import torch
    code = "import tiny_dep, torch; print(tiny_dep.VERSION, torch.__version__)"
    assert _run(python_one, code) == ["1.0", torch.__version__]  # its own package; ComfyUI's torch
    assert _run(python_two, code) == ["2.0", torch.__version__]
    assert importlib.util.find_spec("tiny_dep") is None  # ComfyUI's environment is untouched
    for entry, python in ((one, python_one), (two, python_two)):  # each pack imports in its own
        assert server.PackServer.describe(entry, python=python)["nodes"] == {}


def test_an_environment_is_made_once_per_version_of_the_requirements(index):
    pack = _pack(index, "pack", "tiny_dep==1.0\n", "")
    first = envs.python_for(pack)
    assert envs.python_for(pack) == first
    (index / "pack" / "requirements.txt").write_text("tiny_dep==2.0\n")
    assert envs.python_for(pack) != first


def test_a_pack_that_needs_another_torch_fails_alone(index):
    pack = _pack(index, "wants_old_torch", "torch==0.0.1\n", "")
    with pytest.raises(envs.PackEnvError, match="next to ComfyUI's packages"):
        envs.python_for(pack)


def test_a_pack_without_requirements_runs_on_comfyuis_interpreter(index):
    import sys
    pack = index / "plain"
    pack.mkdir()
    (pack / "__init__.py").write_text("")
    assert envs.python_for(str(pack)) == sys.executable


def test_environments_can_be_turned_off(index, monkeypatch):
    import sys
    pack = _pack(index, "pack", "tiny_dep==1.0\n", "")
    monkeypatch.setenv(envs.ENVS_ENV, "off")
    assert envs.python_for(pack) == sys.executable
    assert not os.listdir(index / "envs") if (index / "envs").exists() else True
