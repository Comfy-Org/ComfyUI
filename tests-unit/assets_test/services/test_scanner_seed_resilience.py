import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import (
    create_content,
    create_content_reporting_insert,
    create_record,
    delete_record,
)
from app.assets.scanner import SeedAssetSpec, insert_asset_specs, seed_asset_specs
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

    created, error = seed_asset_specs(session, specs)
    session.commit()

    assert error is None
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

    monkeypatch.setattr("app.assets.scanner.snapshot_hash", _hash_or_raise)

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created, error = seed_asset_specs(session, specs)
    session.commit()

    assert error is None
    assert created == 2
    assert _record_count(session) == 2


def _delete_before_restat(_monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    path.unlink()


def _delete_during_recovery(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    def _hash_or_raise(candidate_path: str) -> str | None:
        if candidate_path == str(path):
            path.unlink()
            raise FileNotFoundError("file vanished during recovery")
        return snapshot_hash(candidate_path)

    monkeypatch.setattr("app.assets.scanner.snapshot_hash", _hash_or_raise)


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


def test_seed_absorbs_live_path_conflict_and_persists_the_specs_around_it(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, conflicted_path = _specs_with_vanished_path(temp_dir)

    def _create_content_or_conflict(
        session_arg: Session,
        path: str,
        *,
        hash: str | None = None,
        size_bytes: int = 0,
        mtime_ns: int | None = None,
    ) -> tuple[AssetContent, bool]:
        if path != str(conflicted_path):
            return create_content_reporting_insert(
                session_arg,
                path=path,
                hash=hash,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
            )
        session_arg.add(
            AssetContent(
                path=path,
                hash=hash,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
            )
        )
        session_arg.flush()
        session_arg.add(
            AssetContent(
                path=path,
                hash=hash,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
            )
        )
        session_arg.flush()
        raise AssertionError("duplicate live paths must violate the unique index")

    monkeypatch.setattr(
        "app.assets.scanner.create_content_reporting_insert",
        _create_content_or_conflict,
    )

    created, error = seed_asset_specs(session, specs)
    session.commit()

    assert error is None
    assert created == 2
    assert _record_count(session) == 2
    assert {record.name for record in session.scalars(select(Asset))} == {
        "first.bin",
        "last.bin",
    }


def test_seed_propagates_unrelated_integrity_error(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "unrelated-integrity-error.bin"
    path.write_bytes(b"content")
    unrelated_error = IntegrityError(
        "forced record creation failure",
        {},
        ValueError("unrelated integrity failure"),
    )

    def _create_record_or_raise(
        session_arg: Session,
        *,
        content_id: str,
        name: str,
        mime_type: str | None,
        job_id: str | None,
        loader_path: str | None,
        tags: list[str],
    ) -> Asset:
        raise unrelated_error

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    _created, error = seed_asset_specs(session, [_spec(path)])

    assert error is unrelated_error


def test_seed_raises_memory_error_instead_of_attempting_later_specs(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for name in ("first.bin", "second.bin"):
        path = temp_dir / name
        path.write_bytes(b"content")
        paths.append(path)
    attempted: list[str] = []

    def _create_record_or_exhaust(
        session_arg: Session,
        *,
        content_id: str,
        name: str,
        mime_type: str | None,
        job_id: str | None,
        loader_path: str | None,
        tags: list[str],
    ) -> Asset:
        attempted.append(name)
        raise MemoryError("out of memory")

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_exhaust)

    with pytest.raises(MemoryError):
        seed_asset_specs(session, [_spec(path) for path in paths])

    assert attempted == ["first.bin"]


def test_seed_attempts_remaining_specs_before_propagating_integrity_error(
    session: Session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [temp_dir / name for name in ("first.bin", "broken.bin", "last.bin")]
    for path in paths:
        path.write_bytes(path.name.encode())
    attempted: list[str] = []
    unrelated_error = IntegrityError(
        "forced record creation failure",
        {},
        ValueError("unrelated integrity failure"),
    )

    def _create_record_or_raise(
        session_arg: Session,
        *,
        content_id: str,
        name: str,
        mime_type: str | None,
        job_id: str | None,
        loader_path: str | None,
        tags: list[str],
    ) -> Asset:
        attempted.append(name)
        if name == "broken.bin":
            raise unrelated_error
        return create_record(
            session_arg,
            content_id=content_id,
            name=name,
            mime_type=mime_type,
            job_id=job_id,
            loader_path=loader_path,
            tags=tags,
        )

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    created, error = seed_asset_specs(session, [_spec(path) for path in paths])
    session.commit()

    assert error is unrelated_error
    assert created == 2
    assert attempted == ["first.bin", "broken.bin", "last.bin"]
    assert {record.name for record in session.scalars(select(Asset))} == {
        "first.bin",
        "last.bin",
    }


def test_insert_commits_successful_specs_before_propagating_batch_fault(
    db_engine, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [temp_dir / name for name in ("first.bin", "broken.bin", "last.bin")]
    for path in paths:
        path.write_bytes(path.name.encode())

    @contextmanager
    def _create_session():
        with Session(db_engine) as session:
            yield session

    def _create_record_or_raise(
        session_arg: Session,
        *,
        content_id: str,
        name: str,
        mime_type: str | None,
        job_id: str | None,
        loader_path: str | None,
        tags: list[str],
    ) -> Asset:
        if name == "broken.bin":
            raise RuntimeError("forced record creation failure")
        return create_record(
            session_arg,
            content_id=content_id,
            name=name,
            mime_type=mime_type,
            job_id=job_id,
            loader_path=loader_path,
            tags=tags,
        )

    monkeypatch.setattr("app.assets.scanner.create_session", _create_session)
    monkeypatch.setattr("app.assets.scanner.create_write_session", _create_session)
    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    created, error = insert_asset_specs([_spec(path) for path in paths], set())

    with Session(db_engine) as session:
        assert {record.name for record in session.scalars(select(Asset))} == {
            "first.bin",
            "last.bin",
        }
    assert created == 2
    assert isinstance(error, RuntimeError)
    assert str(error) == "forced record creation failure"


def test_seed_skips_negative_fresh_mtime_with_warning_and_telemetry(
    session: Session,
    temp_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = [temp_dir / name for name in ("first.bin", "old.bin", "last.bin")]
    for path in paths:
        path.write_bytes(path.name.encode())
    pre_epoch_ns = -315_547_200_000_000_000
    os.utime(paths[1], ns=(pre_epoch_ns, pre_epoch_ns))

    with caplog.at_level(logging.INFO):
        created, error = seed_asset_specs(session, [_spec(path) for path in paths])
    session.commit()

    assert error is None
    assert created == 2
    assert {record.name for record in session.scalars(select(Asset))} == {
        "first.bin",
        "last.bin",
    }
    assert any(
        record.getMessage() == f"Skipping asset with invalid mtime during scan: {paths[1]}"
        for record in caplog.records
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[assets-event] scanner.invalid_mtime")
    ] == ["[assets-event] scanner.invalid_mtime count=1"]


def test_seed_emits_one_invalid_mtime_event_for_a_whole_batch_of_pre_epoch_files(
    session: Session,
    temp_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = [temp_dir / f"restored-{index}.bin" for index in range(5)]
    pre_epoch_ns = -315_547_200_000_000_000
    for path in paths:
        _ = path.write_bytes(path.name.encode())
        os.utime(path, ns=(pre_epoch_ns, pre_epoch_ns))

    with caplog.at_level(logging.INFO):
        created, error = seed_asset_specs(session, [_spec(path) for path in paths])
    session.commit()

    assert error is None
    assert created == 0
    assert _record_count(session) == 0
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[assets-event] scanner.invalid_mtime")
    ] == ["[assets-event] scanner.invalid_mtime count=5"]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Skipping asset with invalid mtime")
    ] == [f"Skipping asset with invalid mtime during scan: {path}" for path in paths]


def test_seed_persists_fresh_stat_after_spec_was_built(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "changed-after-walk.bin"
    path.write_bytes(b"walk-time")
    spec = _spec(path)

    path.write_bytes(b"fresh-seed-time-content")
    fresh_mtime_ns = spec["mtime_ns"] + 1_000_000
    os.utime(path, ns=(fresh_mtime_ns, fresh_mtime_ns))
    fresh_stat = path.stat()

    created, error = seed_asset_specs(session, [spec])
    session.commit()

    persisted = session.scalar(
        select(AssetContent).where(AssetContent.path == str(path))
    )
    assert error is None
    assert created == 1
    assert fresh_stat.st_size != spec["size_bytes"]
    assert fresh_stat.st_mtime_ns != spec["mtime_ns"]
    assert persisted is not None
    assert persisted.size_bytes == fresh_stat.st_size
    assert persisted.mtime_ns == fresh_stat.st_mtime_ns


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

    created, error = seed_asset_specs(session, [spec])
    session.rollback()

    assert created == 0
    assert isinstance(error, RuntimeError)
    assert str(error) == "forced record creation failure"
    assert session.get(AssetContent, retained_content_id) is not None
