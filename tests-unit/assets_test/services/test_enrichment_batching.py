import os
from pathlib import Path
from unittest.mock import patch

import app.database.db as db_mod
from app.assets import scanner
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_content, create_record


def _seed_enrichment_row(
    path: Path,
    *,
    stored_size: int,
    stored_mtime_ns: int,
    stored_hash: str | None = None,
) -> scanner.UnenrichedContent:
    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=stored_size,
            mtime_ns=stored_mtime_ns,
        )
        content.hash = stored_hash
        record = create_record(session, content.id, path.name)
        return content.id, record.id

    content_id, record_id = db_mod.run_write_txn(seed)
    return scanner.UnenrichedContent(
        content_id,
        record_id,
        str(path),
        needs_hash=stored_hash is None,
        observed_size_bytes=stored_size,
        observed_mtime_ns=stored_mtime_ns,
    )


def test_enrichment_applies_when_unchanged_row_is_behind_disk(
    tmp_path: Path,
    session,
) -> None:
    path = tmp_path / "row-behind-disk.bin"
    path.write_bytes(b"current disk bytes")
    disk_stat = path.stat()
    row = _seed_enrichment_row(
        path,
        stored_size=len(b"old bytes"),
        stored_mtime_ns=disk_stat.st_mtime_ns - 5_000_000_000,
    )

    enriched, failed_ids = scanner.enrich_assets_batch(
        [row], extract_metadata=True, compute_hash=False
    )

    assert enriched == 1
    assert failed_ids == []
    session.expire_all()
    assert session.get(Asset, row.record_id).system_metadata is not None


def test_enrichment_skips_row_changed_after_prepare(
    tmp_path: Path,
    session,
    monkeypatch,
) -> None:
    path = tmp_path / "row-changed-after-prepare.bin"
    path.write_bytes(b"stable bytes")
    disk_stat = path.stat()
    row = _seed_enrichment_row(
        path,
        stored_size=disk_stat.st_size,
        stored_mtime_ns=disk_stat.st_mtime_ns,
    )
    prepared = scanner._prepare_enrichment(
        row, extract_metadata=True, compute_hash=False, progress=None
    )
    assert prepared is not None
    changed_mtime_ns = disk_stat.st_mtime_ns + 1

    def change_row(write_session) -> None:
        content = write_session.get(AssetContent, row.content_id)
        assert content is not None
        content.mtime_ns = changed_mtime_ns

    db_mod.run_write_txn(change_row)
    monkeypatch.setattr(scanner, "_prepare_enrichment", lambda *_args: prepared)

    enriched, failed_ids = scanner.enrich_assets_batch(
        [row], extract_metadata=True, compute_hash=False
    )

    assert enriched == 0
    assert failed_ids == [row.record_id]
    session.expire_all()
    content = session.get(AssetContent, row.content_id)
    assert content is not None
    assert content.mtime_ns == changed_mtime_ns
    assert content.hash is None
    assert session.get(Asset, row.record_id).system_metadata is None


def test_root_sync_reselects_metadata_stale_after_file_changes_post_prepare(
    tmp_path: Path,
    session,
) -> None:
    path = tmp_path / "metadata-stale-after-prepare.bin"
    path.write_bytes(b"old bytes")
    initial_stat = path.stat()
    row = _seed_enrichment_row(
        path,
        stored_size=initial_stat.st_size,
        stored_mtime_ns=initial_stat.st_mtime_ns,
        stored_hash="sha256:stale",
    )
    prepared = scanner._prepare_enrichment(
        row, extract_metadata=True, compute_hash=False, progress=None
    )
    assert prepared is not None

    path.write_bytes(b"new bytes")
    rewritten_mtime_ns = initial_stat.st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(rewritten_mtime_ns, rewritten_mtime_ns))

    applied = db_mod.run_write_txn(
        lambda write_session: scanner._apply_enrichments(write_session, [prepared])
    )
    assert applied == [row.record_id]
    session.expire_all()
    assert session.get(AssetContent, row.content_id).hash == "sha256:stale"
    assert session.get(Asset, row.record_id).system_metadata is not None

    with patch("folder_paths.get_input_directory", return_value=str(tmp_path)):
        survivors = scanner.sync_root_safely("input")
        selected = scanner.get_unenriched_assets_for_roots(
            ("input",), compute_hashes=False
        )

    assert survivors == {str(path.resolve())}
    session.expire_all()
    content = session.get(AssetContent, row.content_id)
    assert content is not None
    assert content.hash is None
    assert session.get(Asset, row.record_id).system_metadata is None
    assert row.record_id in {candidate.record_id for candidate in selected}
