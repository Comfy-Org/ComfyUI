import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import mark_content_missing
from app.assets.helpers import to_stored_hash
from app.assets.scanner import (
    clear_pending_verifications,
    drain_pending_verifications,
    apply_reference_observations,
    observe_references_on_filesystem,
)
from app.assets.scanner_changes import queue_pending_verification
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def _clear_pending_verifications():
    clear_pending_verifications()
    yield
    clear_pending_verifications()


def _seed_content(session, path: Path, hash_value: str | None) -> tuple[AssetContent, Asset]:
    stat = path.stat()
    content = AssetContent(
        path=str(path),
        hash=hash_value,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name=path.name)
    session.add(record)
    session.commit()
    return content, record


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def _sync_references(session, root: Path) -> None:
    observations, _ = observe_references_on_filesystem(session, [str(root)])
    apply_reference_observations(session, observations)


def _stored_hash(path: Path) -> str:
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    return to_stored_hash(digest)


def test_off_mode_same_size_touch_does_not_split(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "touched.bin"
    path.write_bytes(b"same bytes")
    old_content, _ = _seed_content(session, path, hash_value="historical")
    _bump_mtime(path)

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=False),
    ):
        _sync_references(session, input_root)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    assert len(contents) == 1
    assert session.get(AssetContent, old_content.id).is_missing is False


def test_off_mode_size_change_splits(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "grown.bin"
    path.write_bytes(b"small")
    previous_target_ns = path.stat().st_mtime_ns
    old_content, _ = _seed_content(session, path, hash_value="historical")
    path.write_bytes(b"a decidedly larger set of bytes")
    target_ns = max(path.stat().st_mtime_ns, previous_target_ns) + 1_000_000
    os.utime(path, ns=(target_ns, target_ns))

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=False),
    ):
        _sync_references(session, input_root)
    session.commit()

    contents = list(session.scalars(select(AssetContent).order_by(AssetContent.created_at)))
    assert len(contents) == 2
    assert session.get(AssetContent, old_content.id).is_missing is True
    assert [content.is_missing for content in contents] == [True, False]


def test_hash_mode_touch_refreshes_mtime(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "touched.bin"
    path.write_bytes(b"same bytes")
    old_content, _ = _seed_content(session, path, _stored_hash(path))
    _bump_mtime(path)

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
    ):
        _sync_references(session, input_root)
        processed = drain_pending_verifications(session)
    session.commit()

    refreshed = session.get(AssetContent, old_content.id)
    assert processed == 1
    assert refreshed.is_missing is False
    assert refreshed.mtime_ns == path.stat().st_mtime_ns
    assert len(session.scalars(select(AssetContent)).all()) == 1


def test_hash_mode_real_edit_splits(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "edited.bin"
    path.write_bytes(b"old bytes")
    previous_target_ns = path.stat().st_mtime_ns
    old_content, _ = _seed_content(session, path, _stored_hash(path))
    path.write_bytes(b"new bytes with a different length")
    target_ns = max(path.stat().st_mtime_ns, previous_target_ns) + 1_000_000
    os.utime(path, ns=(target_ns, target_ns))

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
    ):
        _sync_references(session, input_root)
        drain_pending_verifications(session)
    session.commit()

    contents = list(session.scalars(select(AssetContent).order_by(AssetContent.created_at)))
    assert len(contents) == 2
    assert session.get(AssetContent, old_content.id).is_missing is True
    assert next(content for content in contents if not content.is_missing).hash == _stored_hash(path)


def test_old_record_id_resolves_to_missing_content_after_split(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "edited.bin"
    path.write_bytes(b"old bytes")
    previous_target_ns = path.stat().st_mtime_ns
    old_content, old_record = _seed_content(session, path, _stored_hash(path))
    path.write_bytes(b"replacement bytes")
    target_ns = max(path.stat().st_mtime_ns, previous_target_ns) + 1_000_000
    os.utime(path, ns=(target_ns, target_ns))

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
    ):
        _sync_references(session, input_root)
        drain_pending_verifications(session)
    session.commit()

    session.expire_all()
    original_record = session.get(Asset, old_record.id)
    assert original_record is not None
    assert original_record.content_id == old_content.id
    assert original_record.content.is_missing is True


def test_hash_mode_split_uses_stat_from_the_verified_snapshot(session, temp_dir: Path, monkeypatch):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "changed.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(input_root))
    path.write_bytes(b"old")
    content, _ = _seed_content(session, path, _stored_hash(path))
    queue_pending_verification(content.id)
    new_payload = b"new bytes with a different size"
    real_snapshot_hash = snapshot_hash

    def mutate_then_hash(candidate_path: str):
        path.write_bytes(new_payload)
        return real_snapshot_hash(candidate_path)

    monkeypatch.setattr("app.assets.scanner_changes.snapshot_hash", mutate_then_hash)

    processed = drain_pending_verifications(session)

    live_content = session.scalar(
        select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    assert processed == 1
    assert live_content is not None
    assert live_content.size_bytes == len(new_payload)
    assert live_content.mtime_ns == path.stat().st_mtime_ns


def test_observation_is_skipped_when_the_row_changed_before_it_was_applied(
    session, temp_dir: Path
):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "raced.bin"
    path.write_bytes(b"v1")
    content, _ = _seed_content(session, path, hash_value=None)
    path.write_bytes(b"v2 is longer")
    _bump_mtime(path)

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=False),
    ):
        observations, _ = observe_references_on_filesystem(session, [str(input_root)])
        assert len(observations) == 1
        # The file changes again and another writer records that before the observation lands.
        path.write_bytes(b"v3 is longer still")
        _bump_mtime(path)
        content.size_bytes = path.stat().st_size
        content.mtime_ns = path.stat().st_mtime_ns
        session.commit()
        apply_reference_observations(session, observations)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    assert len(contents) == 1
    assert contents[0].is_missing is False
    assert contents[0].size_bytes == len(b"v3 is longer still")


def test_drain_commits_each_entry_before_hashing_the_next(session, temp_dir: Path, monkeypatch):
    gone = temp_dir / "gone.bin"
    kept = temp_dir / "kept.bin"
    gone.write_bytes(b"gone")
    kept.write_bytes(b"kept")
    gone_content, _ = _seed_content(session, gone, None)
    kept_content, _ = _seed_content(session, kept, None)
    gone.unlink()
    queue_pending_verification(gone_content.id)
    queue_pending_verification(kept_content.id)
    in_transaction_while_hashing = []

    def recording_snapshot_hash(path: str):
        in_transaction_while_hashing.append(session.connection().connection.driver_connection.in_transaction)
        return snapshot_hash(path)

    monkeypatch.setattr("app.assets.scanner_changes.snapshot_hash", recording_snapshot_hash)

    assert drain_pending_verifications(session) == 2
    assert in_transaction_while_hashing == [False]


def _retire_during_hash(monkeypatch, session, content_id: str, replace: bool) -> None:
    def competing_write_then_hash(path: str):
        with Session(session.get_bind()) as other:
            mark_content_missing(other, content_id)
            if replace:
                other.add(AssetContent(path=path, hash="blake3:" + "f" * 64, size_bytes=1, mtime_ns=1))
            other.commit()
        return snapshot_hash(path)

    monkeypatch.setattr("app.assets.scanner_changes.snapshot_hash", competing_write_then_hash)


@pytest.mark.parametrize(
    ("replace", "seeded_hash"),
    [(False, None), (True, "blake3:" + "0" * 64)],
    ids=["retired", "replaced"],
)
def test_drain_skips_a_row_retired_while_hashing(
    session, temp_dir: Path, monkeypatch, replace: bool, seeded_hash: str | None
):
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path = temp_dir / "raced.bin"
    path.write_bytes(b"raced bytes")
    content, _ = _seed_content(session, path, seeded_hash)
    content_id = content.id
    queue_pending_verification(content_id)
    _retire_during_hash(monkeypatch, session, content_id, replace)

    processed = drain_pending_verifications(session)
    session.commit()

    session.expire_all()
    assert session.get(AssetContent, content_id).hash == seeded_hash
    live = session.scalars(select(AssetContent).where(AssetContent.is_missing.is_(False))).all()
    assert len(live) == (1 if replace else 0)
    assert len(session.scalars(select(Asset)).all()) == 1
    assert processed == 0
