import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.assets import scanner as asset_scanner
from app.assets import scanner_changes as asset_scanner_changes
from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import to_stored_hash
from app.assets.scanner import (
    clear_pending_verifications,
    drain_pending_verifications,
    sync_prefixes_with_filesystem,
)
from app.assets.scanner_changes import (
    is_path_under_prefixes,
    queue_pending_verification,
)
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
        sync_prefixes_with_filesystem(session, [str(input_root)])
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
        sync_prefixes_with_filesystem(session, [str(input_root)])
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
        sync_prefixes_with_filesystem(session, [str(input_root)])
        processed = drain_pending_verifications(session)
    session.commit()

    refreshed = session.get(AssetContent, old_content.id)
    assert processed == 1
    assert refreshed.is_missing is False
    assert refreshed.mtime_ns == path.stat().st_mtime_ns
    assert len(session.scalars(select(AssetContent)).all()) == 1


def test_hash_mode_real_edit_splits(session, temp_dir: Path, monkeypatch):
    input_root = temp_dir / "input"
    input_root.mkdir()
    path = input_root / "edited.bin"
    path.write_bytes(b"old bytes")
    old_content, _ = _seed_content(session, path, _stored_hash(path))
    path.write_bytes(b"new bytes with a different length")
    prefixes = [str(input_root)]
    trace_lines = [
        f"PATHS stored={old_content.path!r} disk={str(path)!r} prefixes={prefixes!r} "
        f"python_match={is_path_under_prefixes(old_content.path, prefixes)!r}"
    ]
    real_live = asset_scanner.live_contents_under_prefixes
    real_detect = asset_scanner.detect_content_change

    def traced_live(candidate_session, candidate_prefixes):
        rows = real_live(candidate_session, candidate_prefixes)
        trace_lines.append(
            "LIVE rows="
            + repr(
                [
                    (
                        row.id,
                        row.path,
                        row.mtime_ns,
                        row.size_bytes,
                        row.is_missing,
                    )
                    for row in rows
                ]
            )
        )
        return rows

    def traced_detect(
        candidate_session,
        content,
        stat_result,
        hashing_is_enabled,
    ):
        trace_lines.append(
            f"DETECT mtimes=({content.mtime_ns!r}, {stat_result.st_mtime_ns!r}) "
            f"equal={content.mtime_ns == stat_result.st_mtime_ns!r} "
            f"sizes=({content.size_bytes!r}, {stat_result.st_size!r}) "
            f"hashing={hashing_is_enabled!r}"
        )
        return real_detect(
            candidate_session,
            content,
            stat_result,
            hashing_is_enabled,
        )

    monkeypatch.setattr(asset_scanner, "live_contents_under_prefixes", traced_live)
    monkeypatch.setattr(asset_scanner, "detect_content_change", traced_detect)

    with (
        patch("folder_paths.get_input_directory", return_value=str(input_root)),
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
    ):
        sync_prefixes_with_filesystem(session, prefixes)
        queued_before_drain = list(asset_scanner_changes._pending_verification_ids)
        processed = drain_pending_verifications(session)
        queued_after_drain = list(asset_scanner_changes._pending_verification_ids)
    session.commit()

    drain_rows = [
        (
            content.id,
            content.path,
            content.hash,
            content.mtime_ns,
            content.size_bytes,
            content.is_missing,
        )
        for content in session.scalars(
            select(AssetContent).order_by(AssetContent.created_at)
        )
    ]
    trace_lines.append(
        f"DRAIN queued_before={queued_before_drain!r} processed={processed!r} "
        f"queued_after={queued_after_drain!r} rows={drain_rows!r}"
    )
    pytest.fail("WIN12_DIAG " + "\n".join(trace_lines))

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
        sync_prefixes_with_filesystem(session, [str(input_root)])
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
