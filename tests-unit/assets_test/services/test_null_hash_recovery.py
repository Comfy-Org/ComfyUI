import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.scanner import SeedAssetSpec, clear_pending_verifications, seed_asset_specs
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
    record_transition_intent,
    write_stored_mode,
)
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def _clear_queues():
    clear_transition_queue()
    clear_pending_verifications()
    yield
    clear_transition_queue()
    clear_pending_verifications()


def _stored_hash(path: Path) -> str:
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    return to_stored_hash(digest)


def _spec(path: Path) -> SeedAssetSpec:
    stat = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "info_name": path.name,
        "tags": ["input"],
        "fname": path.name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def test_deleted_null_hash_row_recovers_via_scanner_after_restore(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "recoverable.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    original_bytes = b"the exact bytes that come back"
    path.write_bytes(original_bytes)
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    record = create_record(session, content_id, "recoverable.bin", tags=["input"])
    record_id = record.id
    session.commit()

    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    path.unlink()
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()
    assert session.get(AssetContent, content_id).is_missing is True, (
        "precondition: A1's drain marked the row missing on delete"
    )

    path.write_bytes(original_bytes)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == stat.st_mtime_ns, "setup: mtime must round-trip exactly"

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 0, "the original row must recover — no fresh content row minted"
    recovered = session.get(AssetContent, content_id)
    assert recovered.is_missing is False
    assert recovered.hash == _stored_hash(path)
    assert len(session.scalars(select(AssetContent)).all()) == 1, "no duplicate content row"
    assert len(session.scalars(select(Asset)).all()) == 1, "no duplicate record"
    assert session.get(Asset, record_id) is not None, "the original record's identity survives"
    assert session.get(AssetTag, {"asset_id": record_id, "tag_name": "missing"}) is None


def test_different_bytes_restored_at_same_path_does_not_recover_old_row(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "replaced.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path.write_bytes(b"original bytes")
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    create_record(session, content_id, "replaced.bin")
    session.commit()

    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    path.unlink()
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()
    assert session.get(AssetContent, content_id).is_missing is True

    path.write_bytes(b"a completely different, much longer payload than the original")

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 1, "a genuinely different file must take the normal new-content path"
    assert session.get(AssetContent, content_id).is_missing is True, (
        "the old row must stay missing — recovering it here would hand the wrong record's "
        "identity, tags, and metadata to unrelated bytes"
    )
    assert len(session.scalars(select(AssetContent)).all()) == 2


def test_same_size_different_mtime_restored_at_same_path_does_not_recover_old_row(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "retimed.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    original_bytes = b"identical length, different moment in time"
    path.write_bytes(original_bytes)
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    create_record(session, content_id, "retimed.bin")
    session.commit()

    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    path.unlink()
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()
    assert session.get(AssetContent, content_id).is_missing is True

    path.write_bytes(original_bytes)
    shifted_ns = stat.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(stat.st_atime_ns, shifted_ns))
    assert path.stat().st_mtime_ns != stat.st_mtime_ns, "setup: mtime must actually differ"
    assert path.stat().st_size == stat.st_size, "setup: size must match so only mtime disambiguates"

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 1, "a same-size-but-different-mtime restore must take the new-content path"
    assert session.get(AssetContent, content_id).is_missing is True, (
        "a matching size with a mismatched mtime is not proof the old row's bytes are back — "
        "recovering here would hand the wrong record's identity to different bytes"
    )
    assert len(session.scalars(select(AssetContent)).all()) == 2


def test_two_missing_null_hash_candidates_at_same_path_do_not_recover(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "ambiguous_null.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path.write_bytes(b"bytes shared by two missing generations")
    stat = path.stat()
    first = AssetContent(
        path=str(path), hash=None, is_missing=True,
        size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
    )
    second = AssetContent(
        path=str(path), hash=None, is_missing=True,
        size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
    )
    session.add_all([first, second])
    session.flush()
    first_id, second_id = first.id, second.id
    session.commit()

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 1, "ambiguous candidates must fall through to the normal new-content path"
    assert session.get(AssetContent, first_id).is_missing is True
    assert session.get(AssetContent, second_id).is_missing is True
