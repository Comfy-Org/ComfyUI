from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_content, create_record
from app.assets.database.queries.records import (
    RecordPageSpec,
    fetch_record_tags,
    list_records_page,
)
from app.assets.helpers import to_stored_hash
from app.assets.scanner import get_unenriched_assets_for_roots
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
)
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
)
from app.assets.services.lookup import lookup_for_view
from app.assets.services.snapshot_hash import snapshot_hash


@dataclass(frozen=True, slots=True)
class _FakeStat:

    st_size: int
    st_mtime_ns: int


@contextmanager
def _reuse_session(session: Session) -> Iterator[Session]:
    yield session


def _raw_system_metadata(session: Session, record_id: str) -> object:
    return session.execute(
        sa.text("SELECT system_metadata FROM assets WHERE id = :id"),
        {"id": record_id},
    ).scalar()


def _candidates_under(session: Session, temp_dir: Path, *, compute_hashes: bool) -> set[str]:
    with (
        patch("app.assets.scanner.create_session", lambda: _reuse_session(session)),
        patch(
            "app.assets.scanner.get_scan_prefixes_for_root",
            return_value=[str(temp_dir)],
        ),
    ):
        rows = get_unenriched_assets_for_roots(("models",), compute_hashes=compute_hashes)
    return {row.record_id for row in rows}


@pytest.fixture(autouse=True)
def _transition_queue_isolation() -> Iterator[None]:
    clear_transition_queue()
    yield
    clear_transition_queue()


@pytest.fixture(autouse=True)
def _input_base_is_temp_dir(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))


@pytest.fixture(autouse=True)
def _pending_verification_isolation() -> Iterator[None]:
    clear_pending_verifications()
    yield
    clear_pending_verifications()


def _seed_hashed_content(session: Session, path: Path, data: bytes) -> AssetContent:
    path.write_bytes(data)
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, verified_stat = snapshot
    return create_content(
        session,
        str(path),
        to_stored_hash(digest),
        verified_stat.st_size,
        verified_stat.st_mtime_ns,
    )


def _bump_mtime(path: Path) -> os.stat_result:
    before = path.stat()
    bumped = before.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(bumped, bumped))
    after = path.stat()
    assert after.st_mtime_ns != before.st_mtime_ns
    assert after.st_size == before.st_size
    return after


def _rewrite_same_size(path: Path, data: bytes) -> os.stat_result:
    before = path.stat()
    assert len(data) == before.st_size
    assert data != path.read_bytes()
    path.write_bytes(data)
    bumped = before.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(bumped, bumped))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns != before.st_mtime_ns
    return after


def test_split_record_is_enrich_candidate_in_hash_mode(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "split.safetensors"
    content = create_content(session, str(path), hash="blake3:deadbeef")
    record = create_record(session, content.id, path.name)
    session.commit()
    record_id = record.id

    candidates = _candidates_under(session, temp_dir, compute_hashes=True)

    assert record_id in candidates


def test_enriched_record_is_not_enrich_candidate_in_hash_mode(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "enriched.safetensors"
    content = create_content(session, str(path), hash="blake3:deadbeef")
    record = create_record(
        session, content.id, path.name, system_metadata={"architecture": "flux"}
    )
    session.commit()
    record_id = record.id

    candidates = _candidates_under(session, temp_dir, compute_hashes=True)

    assert record_id not in candidates


def test_same_size_mtime_bump_does_not_split(session: Session, temp_dir: Path) -> None:
    path = temp_dir / "touched.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    record = create_record(
        session, content.id, path.name, tags=["keepme"], system_metadata={"k": "v"}
    )
    session.commit()
    content_id, record_id = content.id, record.id

    detect_content_change(
        session, content, _FakeStat(st_size=100, st_mtime_ns=2000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    live = session.get(AssetContent, content_id)
    assert live is not None and live.is_missing is False
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1
    surviving = session.get(Asset, record_id)
    assert surviving.system_metadata == {"k": "v"}
    tags = fetch_record_tags(session, record_id)
    assert "keepme" in tags
    assert "missing" not in tags

    assert live.mtime_ns == 2000
    assert live.size_bytes == 100


def test_accepted_mtime_bump_drops_the_unverifiable_hash(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "synced.safetensors"
    content = _seed_hashed_content(session, path, b"rsynced bytes")
    record = create_record(
        session, content.id, path.name, tags=["keepme"], system_metadata={"k": "v"}
    )
    session.commit()
    content_id, record_id, stored_hash = content.id, record.id, content.hash
    assert lookup_for_view(session, stored_hash) is not None

    observed = _bump_mtime(path)
    detect_content_change(session, content, observed, hashing_is_enabled=False)
    session.commit()
    session.expire_all()

    live = session.get(AssetContent, content_id)
    assert live.hash is None
    assert lookup_for_view(session, stored_hash) is None

    assert live.is_missing is False
    assert live.mtime_ns == observed.st_mtime_ns
    assert live.size_bytes == observed.st_size

    surviving = session.get(Asset, record_id)
    assert surviving is not None and surviving.content_id == content_id
    assert surviving.system_metadata == {"k": "v"}
    assert "keepme" in fetch_record_tags(session, record_id)
    listed, _, _ = list_records_page(session, RecordPageSpec(limit=100))
    assert record_id in {row.id for row in listed}


def test_same_size_content_change_is_never_served_under_the_old_hash(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "overwritten.safetensors"
    content = _seed_hashed_content(session, path, b"AAAA")
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    old_hash = content.hash
    assert lookup_for_view(session, old_hash) is not None

    observed = _rewrite_same_size(path, b"BBBB")
    detect_content_change(session, content, observed, hashing_is_enabled=False)
    session.commit()
    session.expire_all()

    assert path.read_bytes() == b"BBBB"
    assert lookup_for_view(session, old_hash) is None


def test_accepted_mtime_bump_is_not_re_detected_by_the_next_scan(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "resynced.safetensors"
    content = _seed_hashed_content(session, path, b"cloud synced bytes")
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    content_id = content.id
    detect_content_change(session, content, _bump_mtime(path), hashing_is_enabled=False)
    session.commit()

    detect_content_change(session, content, path.stat(), hashing_is_enabled=True)

    assert drain_pending_verifications(session) == 0

    detect_content_change(session, content, path.stat(), hashing_is_enabled=False)
    session.commit()
    session.expire_all()
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1
    assert rows_at_path[0].id == content_id
    assert rows_at_path[0].is_missing is False


def test_dropped_hash_is_refilled_in_place_by_a_later_hash_mode_pass(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "refilled.safetensors"
    content = _seed_hashed_content(session, path, b"cloud synced bytes")
    record = create_record(
        session, content.id, path.name, tags=["keepme"], system_metadata={"k": "v"}
    )
    session.commit()
    content_id, record_id = content.id, record.id
    assert record_id not in _candidates_under(session, temp_dir, compute_hashes=True)

    detect_content_change(session, content, _bump_mtime(path), hashing_is_enabled=False)
    session.commit()
    session.expire_all()
    assert session.get(AssetContent, content_id).hash is None
    assert record_id in _candidates_under(session, temp_dir, compute_hashes=True)

    enqueue_transition_work(session, "off_to_on")
    drain_transition_queue(session)
    session.commit()
    session.expire_all()

    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    live = session.get(AssetContent, content_id)
    assert live.is_missing is False
    assert live.hash == to_stored_hash(snapshot[0])
    assert lookup_for_view(session, live.hash).id == content_id
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1
    assert "keepme" in fetch_record_tags(session, record_id)
    assert session.get(Asset, record_id).system_metadata == {"k": "v"}
    assert record_id not in _candidates_under(session, temp_dir, compute_hashes=True)


def test_mtime_and_size_change_splits_with_null_metadata(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "grown.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    create_record(
        session, content.id, path.name, tags=["oldtag"], system_metadata={"k": "v"}
    )
    session.commit()
    old_content_id = content.id

    detect_content_change(
        session, content, _FakeStat(st_size=200, st_mtime_ns=2000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    assert session.get(AssetContent, old_content_id).is_missing is True
    live = session.scalar(
        select(AssetContent).where(
            AssetContent.path == str(path), AssetContent.is_missing.is_(False)
        )
    )
    assert live is not None and live.id != old_content_id
    assert live.size_bytes == 200 and live.mtime_ns == 2000
    new_record = session.scalar(select(Asset).where(Asset.content_id == live.id))
    assert new_record is not None

    assert new_record.system_metadata is None
    assert _raw_system_metadata(session, new_record.id) is None
    assert "oldtag" not in fetch_record_tags(session, new_record.id)
    assert new_record.id in _candidates_under(session, temp_dir, compute_hashes=True)


def test_mtime_unchanged_size_changed_does_not_split(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "weird.safetensors"
    content = create_content(session, str(path), hash=None, size_bytes=100, mtime_ns=1000)
    create_record(session, content.id, path.name, tags=["keepme"])
    session.commit()
    content_id = content.id

    detect_content_change(
        session, content, _FakeStat(st_size=999, st_mtime_ns=1000), hashing_is_enabled=False
    )
    session.commit()
    session.expire_all()

    assert session.get(AssetContent, content_id).is_missing is False
    rows_at_path = list(
        session.scalars(select(AssetContent).where(AssetContent.path == str(path)))
    )
    assert len(rows_at_path) == 1


def test_transition_drain_split_replacement_has_null_metadata(
    session: Session, temp_dir: Path
) -> None:
    path = temp_dir / "changed.bin"
    path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = path.stat()
    old_content = create_content(
        session, str(path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(
        session, old_content_id, "changed.bin", tags=["oldtag"], system_metadata={"k": "v"}
    )
    path.write_bytes(b"different new bytes")

    enqueue_transition_work(session, "off_to_on")
    drain_transition_queue(session)
    session.commit()
    session.expire_all()

    assert session.get(AssetContent, old_content_id).is_missing is True
    live = session.scalar(
        select(AssetContent).where(
            AssetContent.path == str(path), AssetContent.is_missing.is_(False)
        )
    )
    assert live is not None and live.id != old_content_id
    new_record = session.scalar(select(Asset).where(Asset.content_id == live.id))
    assert new_record is not None
    assert new_record.system_metadata is None
    assert _raw_system_metadata(session, new_record.id) is None
    assert "oldtag" not in fetch_record_tags(session, new_record.id)
