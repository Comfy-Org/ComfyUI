"""Per-pack Python environments: pack isolation's answer to dependency conflicts.

A pack that ships a ``requirements.txt`` gets its own environment, made once per version of its
requirements and kept in ``COMFY_PACK_ENV_DIR`` (default ``user/pack_envs``):

- the pack's own packages come first, so one pack's ``transformers`` or ``opencv`` can't break
  another pack or ComfyUI;
- ComfyUI's packages are visible underneath (a ``.pth`` file adds ComfyUI's site directories), so
  torch and the rest aren't installed again for every pack;
- ``torch``, ``torchvision`` and ``torchaudio`` are held at ComfyUI's versions, because tensors
  cross between the processes. A pack that needs another torch fails to install, alone.

A pack without requirements runs on ComfyUI's interpreter: it has nothing to conflict with.
``COMFY_PACK_ENVS=off`` runs every pack on ComfyUI's interpreter, sharing its packages.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shlex
import site
import subprocess
import sys
import threading
from typing import Optional

ENVS_ENV = "COMFY_PACK_ENVS"
ENV_DIR_ENV = "COMFY_PACK_ENV_DIR"
PIP_ARGS_ENV = "COMFY_PACK_PIP_ARGS"  # extra pip arguments, e.g. an index or --no-index
#: Held at ComfyUI's versions in every pack environment: tensors cross between the processes.
PINNED = ("torch", "torchvision", "torchaudio")
READY = ".comfy-pack-env-ready"
logger = logging.getLogger(__name__)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class PackEnvError(RuntimeError):
    """A pack's requirements can't be installed next to ComfyUI's."""


def enabled() -> bool:
    return os.environ.get(ENVS_ENV, "on").strip().lower() not in ("0", "off", "false", "no")


def requirements_file(entry: str) -> Optional[str]:
    """The pack's ``requirements.txt``: in the directory it's loaded from, or, for a ``v2/``
    conversion, in the pack's own directory."""
    candidates = [os.path.join(entry, "requirements.txt")]
    if os.path.basename(os.path.normpath(entry)) == "v2":
        candidates.append(os.path.join(os.path.dirname(os.path.normpath(entry)), "requirements.txt"))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                if any(line.split("#", 1)[0].strip() for line in file):
                    return path
    return None


def env_root() -> str:
    path = os.environ.get(ENV_DIR_ENV)
    if not path:
        import folder_paths
        path = os.path.join(folder_paths.get_user_directory(), "pack_envs")
    os.makedirs(path, exist_ok=True)
    return path


def comfy_site_packages() -> list[str]:
    """Where ComfyUI's own packages are, in its order."""
    paths = list(site.getsitepackages())
    if site.ENABLE_USER_SITE and os.path.isdir(site.getusersitepackages()):
        paths.append(site.getusersitepackages())
    return [p for p in dict.fromkeys(paths) if os.path.isdir(p)]


def pins() -> list[str]:
    import importlib.metadata

    found = []
    for name in PINNED:
        try:
            found.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def _env_python(path: str) -> str:
    return os.path.join(path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(path, "bin", "python")


def _key(requirements: str) -> str:
    with open(requirements, "rb") as file:
        text = file.read()
    digest = hashlib.sha256(text)
    digest.update(sys.version.encode())
    digest.update("\n".join(pins() + comfy_site_packages()).encode())
    return digest.hexdigest()[:16]


def _pack_name(entry: str) -> str:
    name = os.path.basename(os.path.normpath(entry))
    if name == "v2":
        name = os.path.basename(os.path.dirname(os.path.normpath(entry)))
    return name


def python_for(entry: str, requirements: Optional[list[str]] = None) -> str:
    """The interpreter a pack's process runs on, making the pack's environment if it needs one.

    ``requirements``: the pack manifest's list; without one, the pack's ``requirements.txt``."""
    if not enabled():
        return sys.executable
    name = _pack_name(entry)
    if requirements is not None:
        lines = [line.strip() for line in requirements if line.strip()]
        if not lines:
            return sys.executable
        text = "\n".join(lines) + "\n"
        requirements_path = os.path.join(env_root(), f"{name}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.requirements.txt")
        if not os.path.isfile(requirements_path):
            with open(requirements_path + ".tmp", "w", encoding="utf-8") as file:
                file.write(text)
            os.replace(requirements_path + ".tmp", requirements_path)
    else:
        requirements_path = requirements_file(entry)
        if requirements_path is None:
            return sys.executable
    path = os.path.join(env_root(), f"{name}-{_key(requirements_path)}")
    with _locks_guard:
        lock = _locks.setdefault(path, threading.Lock())
    with lock:
        if not os.path.isfile(os.path.join(path, READY)):
            _create(path, requirements_path, cwd=entry)
    return _env_python(path)


def _site_packages(python: str) -> str:
    return subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _create(path: str, requirements: str, cwd: Optional[str] = None) -> None:
    import shutil

    logger.info("preparing the environment for %s (once per version of its requirements)", requirements)
    shutil.rmtree(path, ignore_errors=True)  # a half-made environment from an earlier attempt
    log_path = path + ".log"
    with open(log_path, "w") as log:
        def run(command: list[str]) -> None:
            log.write("$ " + " ".join(shlex.quote(c) for c in command) + "\n")
            log.flush()
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=cwd or os.path.dirname(requirements))
            if done.returncode != 0:
                log.flush()
                with open(log_path) as text:
                    tail = text.read()[-3000:]
                raise PackEnvError(f"can't install {requirements} next to ComfyUI's packages "
                                   f"(log: {log_path}):\n{tail}")

        run([sys.executable, "-m", "venv", path])
        python = _env_python(path)
        # The pack's own packages first, then ComfyUI's, added as site directories so that
        # ComfyUI's own .pth files (packages installed elsewhere) apply here too.
        with open(os.path.join(_site_packages(python), "comfy_parent_site.pth"), "w") as pth:
            pth.write("".join(f"import site; site.addsitedir({path!r})\n" for path in comfy_site_packages()))
        constraints = os.path.join(path, "comfy-constraints.txt")
        with open(constraints, "w") as file:
            file.write("\n".join(pins()) + "\n")
        extra = shlex.split(os.environ.get(PIP_ARGS_ENV, ""))
        run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
             "-r", requirements, "-c", constraints, *extra])
    with open(os.path.join(path, READY), "w") as file:
        file.write(requirements + "\n")
