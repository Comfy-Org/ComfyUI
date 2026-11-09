# Lifted unchanged from the secure overlay (comfy_secure_nodes/storage.py), its tested pack storage;
# ``PackStore`` below adds delete and keys on the same records.
"""Bounded, tenant-and-pack scoped persistent storage for secure guests.

The public SDK intentionally exposes only string ``get``/``set`` operations.
This host implementation keeps paths, filenames, locking, and atomic writes on
the trusted side; a guest cannot enumerate another pack's keys or recover a
filesystem location.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path


MAX_KEY_BYTES = 128
MAX_VALUE_BYTES = 1024 * 1024
MAX_ENTRIES = 256
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def _default_root() -> Path:
    configured = os.environ.get("COMFY_SECURE_STORAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    import folder_paths

    return (Path(folder_paths.get_user_directory()).resolve()
            / "secure-node-storage")


def _bounded_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"invalid storage {label}")
    size = len(value.encode("utf-8"))
    if not 1 <= size <= 1024:
        raise ValueError(f"invalid storage {label}")
    return value


def _bounded_key(key: str) -> str:
    if not isinstance(key, str) or "\x00" in key:
        raise ValueError("storage key must be a string without NUL")
    size = len(key.encode("utf-8"))
    if not 1 <= size <= MAX_KEY_BYTES:
        raise ValueError(
            f"storage key must be 1-{MAX_KEY_BYTES} UTF-8 bytes")
    return key


def _bounded_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("storage value must be a string")
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        raise ValueError(
            f"storage value exceeds {MAX_VALUE_BYTES} UTF-8 bytes")
    return value


class PackStorage:
    """Small atomic KV store whose namespace is chosen only by the host."""

    _lock = threading.RLock()

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None else _default_root()
        )

    @staticmethod
    def _digest(*parts: str) -> str:
        payload = "\x00".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _scope(self, tenant: str, pack_id: str) -> Path:
        tenant = _bounded_identity(tenant, "tenant")
        pack_id = _bounded_identity(pack_id, "pack identity")
        return self.root / self._digest(tenant, pack_id)

    @staticmethod
    def _record_path(scope: Path, key: str) -> Path:
        return scope / f"{PackStorage._digest(key)}.json"

    @staticmethod
    def _read_record(path: Path) -> tuple[str, str] | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_VALUE_BYTES + 4096:
            raise RuntimeError("secure storage record is invalid")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            payload = os.read(descriptor, MAX_VALUE_BYTES + 4097)
        finally:
            os.close(descriptor)
        if len(payload) > MAX_VALUE_BYTES + 4096:
            raise RuntimeError("secure storage record exceeds its bound")
        try:
            record = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("secure storage record is corrupt") from error
        if not isinstance(record, dict) or set(record) != {"key", "value"}:
            raise RuntimeError("secure storage record has an invalid shape")
        key = _bounded_key(record.get("key"))
        value = _bounded_value(record.get("value"))
        return key, value

    def get(self, tenant: str, pack_id: str, key: str) -> str | None:
        key = _bounded_key(key)
        scope = self._scope(tenant, pack_id)
        path = self._record_path(scope, key)
        with self._lock:
            record = self._read_record(path)
            if record is None:
                return None
            stored_key, value = record
            if stored_key != key:
                raise RuntimeError("secure storage key digest collision")
            return value

    def set(self, tenant: str, pack_id: str, key: str, value: str) -> None:
        key = _bounded_key(key)
        value = _bounded_value(value)
        scope = self._scope(tenant, pack_id)
        path = self._record_path(scope, key)
        encoded = json.dumps(
            {"key": key, "value": value},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_VALUE_BYTES + 4096:
            raise ValueError("storage record exceeds its bound")

        with self._lock:
            scope.mkdir(parents=True, exist_ok=True, mode=0o700)
            records = []
            for candidate in scope.iterdir():
                if candidate.suffix != ".json":
                    continue
                info = candidate.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError("secure storage contains an invalid entry")
                records.append((candidate, info.st_size))
            existing_size = next(
                (size for candidate, size in records if candidate == path), 0)
            existing_count = len(records)
            if existing_size == 0 and existing_count >= MAX_ENTRIES:
                raise ValueError("secure storage entry quota exceeded")
            projected = sum(size for _candidate, size in records)
            projected = projected - existing_size + len(encoded)
            if projected > MAX_TOTAL_BYTES:
                raise ValueError("secure storage total quota exceeded")

            temporary_name = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".storage-", suffix=".tmp", dir=scope)
                try:
                    os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    descriptor = -1
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                os.replace(temporary_name, path)
                temporary_name = None
            finally:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass


_STORAGE: PackStorage | None = None
_STORAGE_LOCK = threading.Lock()


def host_storage() -> PackStorage:
    global _STORAGE
    with _STORAGE_LOCK:
        if _STORAGE is None:
            _STORAGE = PackStorage()
        return _STORAGE


class PackStore(PackStorage):
    """``PackStorage`` with the two operations ``eng.store`` adds: delete and keys."""

    def delete(self, tenant: str, pack_id: str, key: str) -> bool:
        key = _bounded_key(key)
        path = self._record_path(self._scope(tenant, pack_id), key)
        with self._lock:
            record = self._read_record(path)
            if record is None:
                return False
            if record[0] != key:
                raise RuntimeError("secure storage key digest collision")
            path.unlink()
            return True

    def keys(self, tenant: str, pack_id: str) -> list:
        scope = self._scope(tenant, pack_id)
        with self._lock:
            if not scope.is_dir():
                return []
            found = []
            for candidate in scope.iterdir():
                if candidate.suffix == ".json":
                    record = self._read_record(candidate)
                    if record is not None:
                        found.append(record[0])
            return sorted(found)


_STORE: PackStore | None = None


def pack_store() -> PackStore:
    global _STORE
    with _STORAGE_LOCK:
        if _STORE is None:
            _STORE = PackStore()
        return _STORE
