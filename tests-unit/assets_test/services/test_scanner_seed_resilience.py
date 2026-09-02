from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_content, create_record, delete_record
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


def test_seed_isolates_a_poisoned_spec_and_persists_the_specs_around_it(
    session: Session, temp_dir: Path
) -> None:
    specs, poisoned_path = _specs_with_vanished_path(temp_dir)
    specs[1]["size_bytes"] = -1

    created = seed_asset_specs(session, specs)
    session.commit()

    assert created == 2
    assert _record_count(session) == 2
    assert {record.name for record in session.scalars(select(Asset))} == {
        "first.bin",
        "last.bin",
    }, (
        "the batch shares one transaction, so a bare rollback would erase the spec BEFORE "
        f"the poisoned one; both neighbours of {poisoned_path.name} must survive"
    )


def test_seed_record_failure_preserves_retained_live_content(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "retained.bin"
    path.write_bytes(b"retained-live-content")
    spec = _spec(path)
    content = create_content(
        session,
        path=str(path),
        size_bytes=spec["size_bytes"],
        mtime_ns=spec["mtime_ns"],
    )
    record = create_record(session, content.id, path.name)
    session.commit()
    retained_content_id = content.id
    delete_record(session, record.id)
    session.commit()

    def _raise_record_creation(*_args, **_kwargs):
        raise RuntimeError("forced record creation failure")

    monkeypatch.setattr("app.assets.scanner.create_record", _raise_record_creation)

    with pytest.raises(RuntimeError, match="forced record creation failure"):
        seed_asset_specs(session, [spec])
    session.rollback()

    assert session.get(AssetContent, retained_content_id) is not None
