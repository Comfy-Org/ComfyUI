import builtins
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from blake3 import blake3

from app.assets.services.snapshot_hash import snapshot_hash


class _MutatingReader:
    def __init__(self, file, mutate: Callable[[], None]) -> None:
        self._file = file
        self._mutate = mutate
        self._did_mutate = False

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, *args):
        return self._file.__exit__(*args)

    def fileno(self) -> int:
        return self._file.fileno()

    def read(self, size: int = -1) -> bytes:
        result = self._file.read(size)
        if not self._did_mutate:
            self._did_mutate = True
            self._mutate()
        return result


def test_snapshot_hash_returns_digest_for_quiescent_file(tmp_path: Path) -> None:
    payload = b"quiescent" * 1024
    path = tmp_path / "asset.bin"
    path.write_bytes(payload)

    result = snapshot_hash(str(path), chunk_size=64)

    assert result is not None
    digest, stat_result = result
    assert digest == blake3(payload).hexdigest()
    assert isinstance(stat_result, os.stat_result)
    assert stat_result.st_size == len(payload)


@pytest.mark.parametrize("mutation", ["bytes", "replace", "unlink", "truncate", "append"])
def test_snapshot_hash_returns_none_when_file_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"original" * 1024)
    original_open = builtins.open

    def mutate() -> None:
        match mutation:
            case "bytes":
                path.write_bytes(b"changed" * 1024)
            case "replace":
                replacement = tmp_path / "replacement.bin"
                replacement.write_bytes(b"replacement")
                os.replace(replacement, path)
            case "unlink":
                path.unlink()
            case "truncate":
                path.write_bytes(b"")
            case "append":
                with original_open(path, "ab") as output:
                    output.write(b"more")
            case unreachable:
                raise AssertionError(f"unexpected mutation {unreachable}")

    def open_with_mutation(*args, **kwargs):
        return _MutatingReader(original_open(*args, **kwargs), mutate)

    monkeypatch.setattr(builtins, "open", open_with_mutation)

    assert snapshot_hash(str(path), chunk_size=64) is None
