import hashlib
import json

import pytest

from comfy.tensorfold import runtime


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    binary = tmp_path / "installs" / "verified" / "tf-moss-image"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"verified native artifact")
    binary.chmod(0o755)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    data = {
        "schema": 1, "source_revision": runtime.SOURCE_REVISION, "source_patch_sha256": runtime.SOURCE_PATCH_SHA256,
        "backend": "cuda", "platform": "linux-x86_64",
        "engine": {"version": "1.2.0"},
        "artifacts": {"moss": {"path": "installs/verified/tf-moss-image", "sha256": digest}},
    }
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps(data))
    return manifest, binary, data


def test_verified_runtime_is_local_only(installed, monkeypatch):
    manifest, binary, _ = installed
    import socket
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("unexpected network access"))
    result = runtime.load_runtime(manifest)
    assert result.artifact == binary
    assert result.engine_version == "1.2.0"


def test_tampered_artifact_is_rejected(installed):
    manifest, binary, _ = installed
    binary.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        runtime.load_runtime(manifest)


@pytest.mark.parametrize("path", ["../outside.so", "/tmp/native.so", "a/../b.so", "a//b.so", "a\\b.so", "./x.so", ""])
def test_manifest_cannot_escape_installation(installed, path):
    manifest, _, data = installed
    data["artifacts"]["moss"]["path"] = path
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runtime.load_runtime(manifest)


def test_symlink_is_rejected(installed, tmp_path):
    manifest, binary, _ = installed
    outside = tmp_path / "outside.so"
    binary.rename(outside)
    binary.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        runtime.load_runtime(manifest)


@pytest.mark.parametrize("field,value", [("schema", True), ("schema", 2), ("source_revision", "f" * 40), ("backend", "metal"), ("platform", "linux-aarch64")])
def test_incompatible_runtime_is_rejected(installed, field, value):
    manifest, _, data = installed
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        runtime.load_runtime(manifest)


def test_manifest_size_is_bounded(tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_bytes(b" " * (runtime.MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        runtime.read_manifest(manifest)


def test_install_update_invalidates_node_fingerprint(installed):
    manifest, _, data = installed
    before = runtime.manifest_fingerprint(manifest)
    data["engine"]["version"] = "1.2.1"
    manifest.write_text(json.dumps(data))
    assert runtime.manifest_fingerprint(manifest) != before


def test_missing_runtime_has_setup_instruction(tmp_path):
    with pytest.raises(RuntimeError, match="setup CLI"):
        runtime.load_runtime(tmp_path / "runtime.json")


def test_moss_requires_executable_file(installed):
    manifest, binary, data = installed
    binary.chmod(0o644)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not executable"):
        runtime.load_runtime(manifest)


@pytest.mark.parametrize("patch_digest", [None, "0" * 64])
def test_moss_rejects_legacy_or_other_numerical_build(installed, patch_digest):
    manifest, binary, data = installed
    binary.chmod(0o755)
    if patch_digest is None:
        data.pop("source_patch_sha256")
    else:
        data["source_patch_sha256"] = patch_digest
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="numerical compatibility"):
        runtime.load_runtime(manifest)
