from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset
from app.assets.database.queries import create_record as create_record_query
from app.assets.scanner import SeedAssetSpec, seed_asset_specs
from app.assets.services.snapshot_hash import snapshot_hash


def _spec(path: Path) -> SeedAssetSpec:
    stat_result = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "info_name": path.name,
        "tags": ["input"],
        "fname": path.name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def _specs_with_vanished_path(root: Path) -> tuple[list[SeedAssetSpec], Path]:
    paths = [root / name for name in ("first.bin", "vanished.bin", "last.bin")]
    for path in paths:
        _ = path.write_bytes(path.name.encode())
    return [_spec(path) for path in paths], paths[1]


def _record_count(session: Session) -> int:
    return len(session.scalars(select(Asset)).all())


def test_seed_persists_remaining_specs_when_path_vanishes_before_restat(
    session: Session, temp_dir: Path
) -> None:
    specs, vanished_path = _specs_with_vanished_path(temp_dir)
    vanished_path.unlink()

    created = seed_asset_specs(session, specs)
    session.commit()

    assert created == 2
    assert _record_count(session) == 2


def test_seed_persists_remaining_specs_when_path_vanishes_during_recovery_hash(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, vanished_path = _specs_with_vanished_path(temp_dir)

    def _hash_or_raise(path: str) -> str | None:
        if path == str(vanished_path):
            vanished_path.unlink()
            raise OSError("file vanished during recovery")
        return snapshot_hash(path)

    monkeypatch.setattr("app.assets.scanner_changes.snapshot_hash", _hash_or_raise)

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, specs)
    session.commit()

    assert created == 2
    assert _record_count(session) == 2


def _delete_before_restat(_monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    path.unlink()


def _delete_during_recovery(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    def _hash_or_raise(candidate_path: str) -> str | None:
        if candidate_path == str(path):
            path.unlink()
            raise OSError("file vanished during recovery")
        return snapshot_hash(candidate_path)

    monkeypatch.setattr("app.assets.scanner_changes.snapshot_hash", _hash_or_raise)


@pytest.mark.parametrize(
    "delete_path",
    [_delete_before_restat, _delete_during_recovery],
    ids=["before-restat", "during-recovery"],
)
def test_seed_logs_once_for_each_vanished_path(
    session: Session,
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    delete_path: Callable[[pytest.MonkeyPatch, Path], None],
) -> None:
    specs, vanished_path = _specs_with_vanished_path(temp_dir)
    delete_path(monkeypatch, vanished_path)

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        _ = seed_asset_specs(session, specs)
    session.commit()

    messages = [
        record.getMessage()
        for record in caplog.records
        if str(vanished_path) in record.getMessage()
    ]
    assert messages == [f"Skipping vanished asset during scan: {vanished_path}"]


def test_seed_propagates_integrity_error_and_aborts_batch(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, vanished_path = _specs_with_vanished_path(temp_dir)

    def _create_record_or_raise(
        session: Session,
        content_id: str,
        name: str,
        mime_type: str | None = None,
        job_id: str | None = None,
        loader_path: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> Asset:
        if name == vanished_path.name:
            raise IntegrityError("insert asset content", {}, Exception("forced failure"))
        return create_record_query(
            session,
            content_id,
            name,
            mime_type,
            job_id,
            loader_path,
            tags,
        )

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    with pytest.raises(IntegrityError):
        _ = seed_asset_specs(session, specs)
    session.rollback()

    assert _record_count(session) == 0
