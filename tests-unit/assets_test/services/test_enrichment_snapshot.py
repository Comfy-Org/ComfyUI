import os
from pathlib import Path
from unittest.mock import patch

from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import to_stored_hash
from app.assets.scanner import enrich_asset


def _create_unhashed_record(session, path: Path) -> tuple[AssetContent, Asset]:
    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name=path.name)
    session.add(record)
    session.commit()
    return content, record


def test_enrichment_uses_snapshot_hash_not_direct_blake3(session, temp_dir: Path):
    path = temp_dir / "stable.bin"
    path.write_bytes(b"stable content")
    content, record = _create_unhashed_record(session, path)

    with patch(
        "app.assets.scanner.snapshot_hash", return_value=("snapshot-digest", path.stat())
    ) as mocked_snapshot_hash:
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=False,
            compute_hash=True,
        )

    assert enriched is True
    assert session.get(AssetContent, content.id).hash == to_stored_hash("snapshot-digest")
    mocked_snapshot_hash.assert_called_once_with(str(path))


def test_enrichment_discards_unstable_hash(session, temp_dir: Path):
    path = temp_dir / "unstable.bin"
    path.write_bytes(b"unstable content")
    content, record = _create_unhashed_record(session, path)

    with patch("app.assets.scanner.snapshot_hash", return_value=None):
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=False,
            compute_hash=True,
        )

    assert enriched is False
    assert session.get(AssetContent, content.id).hash is None


def test_enrichment_discards_metadata_read_from_a_different_file_than_the_hash(
    session, temp_dir: Path
):
    path = temp_dir / "swapped.bin"
    path.write_bytes(b"original bytes")
    content, record = _create_unhashed_record(session, path)

    def _replace_file_then_hash(_file_path: str):
        path.write_bytes(b"replacement bytes, a different length entirely")
        return "replacement-digest", path.stat()

    with patch("app.assets.scanner.snapshot_hash", side_effect=_replace_file_then_hash):
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=True,
            compute_hash=True,
        )

    assert enriched is False
    session.expire_all()
    assert session.get(AssetContent, content.id).hash is None
    assert session.get(Asset, record.id).system_metadata is None, (
        "a mismatched hash observation must discard already-computed metadata too, "
        "not just the hash"
    )


def test_enrichment_discards_result_when_only_the_hashed_mtime_disagrees(
    session, temp_dir: Path
):
    path = temp_dir / "touched.bin"
    path.write_bytes(b"same length bytes")
    content, record = _create_unhashed_record(session, path)
    later_mtime_ns = path.stat().st_mtime_ns + 5_000_000_000

    def _touch_file_then_hash(_file_path: str):
        os.utime(path, ns=(later_mtime_ns, later_mtime_ns))
        return "rewritten-digest", path.stat()

    with patch("app.assets.scanner.snapshot_hash", side_effect=_touch_file_then_hash):
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=True,
            compute_hash=True,
        )

    assert enriched is False
    session.expire_all()
    assert session.get(AssetContent, content.id).hash is None
    assert session.get(Asset, record.id).system_metadata is None, (
        "size matching alone is not proof the file is unchanged; an mtime "
        "disagreement alone must also discard the result"
    )


def test_enrichment_lands_metadata_and_hash_from_one_stable_observation(
    session, temp_dir: Path
):
    path = temp_dir / "stable-both.bin"
    path.write_bytes(b"stable content for both")
    content, record = _create_unhashed_record(session, path)

    with patch(
        "app.assets.scanner.snapshot_hash", return_value=("both-digest", path.stat())
    ):
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=True,
            compute_hash=True,
        )

    assert enriched is True
    session.expire_all()
    assert session.get(AssetContent, content.id).hash == to_stored_hash("both-digest")
    assert session.get(Asset, record.id).system_metadata == {
        "filename": "stable-both.bin",
        "file_path": str(path),
        "format": "bin",
        "content_type": "application/octet-stream",
        "content_length": path.stat().st_size,
    }


def test_off_mode_enrichment_still_lands_metadata_without_a_hash(
    session, temp_dir: Path
):
    path = temp_dir / "off-mode.bin"
    path.write_bytes(b"metadata only")
    content, record = _create_unhashed_record(session, path)

    with patch("app.assets.scanner.snapshot_hash") as never_hashed:
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=True,
            compute_hash=False,
        )

    never_hashed.assert_not_called()
    assert enriched is True
    session.expire_all()
    assert session.get(AssetContent, content.id).hash is None
    assert session.get(Asset, record.id).system_metadata == {
        "filename": "off-mode.bin",
        "file_path": str(path),
        "format": "bin",
        "content_type": "application/octet-stream",
        "content_length": path.stat().st_size,
    }
