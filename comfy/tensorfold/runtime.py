import hashlib
import json
import platform
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


SOURCE_REVISION = "1ddac962008367ec2c50e10202c910ecf430b5eb"
SOURCE_PATCH_SHA256 = "e6156ed11263a2d32586245b1702a6bc47ac0049e65bf1724afad69242eb617f"
MAX_MANIFEST_BYTES = 65536


@dataclass(frozen=True)
class Runtime:
    artifact: Path
    engine_version: str
    source_revision: str
    digest: str
    source_patch_sha256: str


def _relative_file(root, value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("TensorFold runtime contains an invalid artifact path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("TensorFold runtime artifacts must stay inside their installation.")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("TensorFold runtime artifacts must not use symbolic links.")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("TensorFold runtime artifact is not a regular file.")
    return path


def verify_file(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("TensorFold runtime has an invalid SHA-256 digest.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("TensorFold runtime checksum failed. Reinstall it before running this workflow.")


def read_manifest(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
    except FileNotFoundError as error:
        raise RuntimeError("TensorFold media runtime is missing. Run the separate TensorFold setup CLI first.") from error
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("TensorFold runtime manifest is too large.")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("TensorFold runtime manifest is invalid JSON.") from error
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError("Unsupported TensorFold runtime manifest schema.")
    if data.get("source_revision") != SOURCE_REVISION:
        raise ValueError("This TensorFold media revision needs a matching ComfyUI integration update.")
    if data.get("backend") != "cuda":
        raise ValueError("This TensorFold integration requires the CUDA media runtime.")
    if not isinstance(data.get("engine"), dict) or not isinstance(data.get("artifacts"), dict):
        raise ValueError("TensorFold runtime manifest is incomplete.")
    version = data["engine"].get("version")
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError("TensorFold runtime has an invalid engine version.")
    return data


def load_runtime(path):
    data = read_manifest(path)
    if data.get("source_patch_sha256") != SOURCE_PATCH_SHA256:
        raise ValueError("TensorFold MOSS needs the ComfyUI numerical compatibility build. Reinstall it with the current setup CLI.")
    machine = platform.machine().lower()
    architectures = {"x86_64": "linux-x86_64", "amd64": "linux-x86_64", "aarch64": "linux-aarch64", "arm64": "linux-aarch64"}
    if platform.system() != "Linux" or architectures.get(machine) != data.get("platform"):
        raise RuntimeError("The installed TensorFold media runtime does not match this Linux host.")
    entry = data["artifacts"].get("moss")
    if not isinstance(entry, dict):
        raise RuntimeError("TensorFold's MOSS component is not installed. Run the setup CLI first.")
    target = _relative_file(path.parent, entry.get("path"))
    verify_file(target, entry.get("sha256"))
    if not target.stat().st_mode & stat.S_IXUSR:
        raise ValueError("TensorFold's MOSS executable is not executable. Reinstall the runtime.")
    return Runtime(target, data["engine"]["version"], data["source_revision"], entry["sha256"], data["source_patch_sha256"])


def manifest_fingerprint(path):
    try:
        with path.open("rb") as stream:
            return hashlib.sha256(stream.read(MAX_MANIFEST_BYTES + 1)).hexdigest()
    except FileNotFoundError:
        return "missing"
