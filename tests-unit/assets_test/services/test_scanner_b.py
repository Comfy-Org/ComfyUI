import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.helpers import to_stored_hash
from app.assets.scanner import (
    build_asset_specs,
    enrich_asset,
    mark_contents_missing_outside_prefixes,
    seed_asset_specs,
    sync_prefixes_with_filesystem,
)
from app.assets.services.snapshot_hash import snapshot_hash


@dataclass(frozen=True, slots=True)
class _ExtractedMetadata:
    content_type: str | None
    system_metadata: Mapping[str, int | str]

    def to_user_metadata(self) -> dict[str, int | str]:
        return dict(self.system_metadata)


def _build_seed_specs(root: Path) -> list:
    return build_asset_specs(
        [str(path) for path in sorted(root.iterdir())],
        existing_paths=set(),
        enable_metadata_extraction=False,
    )[0]


def _create_enrichment_target(
    session: Session,
    path: Path,
    system_metadata: dict[str, int | str] | None = None,
) -> tuple[AssetContent, Asset]:
    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )
    session.add(content)
    session.flush()
    record = Asset(
        content_id=content.id,
        name=path.name,
        system_metadata=system_metadata,
    )
    session.add(record)
    session.commit()
    return content, record


def test_enrichment_retains_absent_system_metadata_keys(session: Session, temp_dir: Path):
    path = temp_dir / "metadata.bin"
    path.write_bytes(b"metadata")
    content, record = _create_enrichment_target(session, path)

    with patch(
        "app.assets.scanner.extract_file_metadata",
        side_effect=[
            _ExtractedMetadata(None, {"a": 1, "b": 2}),
            _ExtractedMetadata(None, {"b": 3}),
        ],
    ):
        enrich_asset(session, str(path), content.id, record.id)
        enrich_asset(session, str(path), content.id, record.id)

    assert record.system_metadata == {"a": 1, "b": 3}


def test_enrichment_retains_dimensions_when_image_extraction_degrades(
    session: Session, temp_dir: Path
):
    path = temp_dir / "image.png"
    path.write_bytes(b"not a complete image")
    content, record = _create_enrichment_target(
        session,
        path,
        {"kind": "image", "width": 64, "height": 64},
    )

    with (
        patch(
            "app.assets.scanner.extract_file_metadata",
            return_value=_ExtractedMetadata("image/png", {"filename": "image.png"}),
        ),
        patch("app.assets.scanner.extract_image_dimensions", return_value=None),
    ):
        enrich_asset(session, str(path), content.id, record.id)

    assert record.system_metadata == {
        "filename": "image.png",
        "kind": "image",
        "width": 64,
        "height": 64,
    }


def test_enrichment_overrides_content_length_with_zero(
    session: Session, temp_dir: Path
):
    path = temp_dir / "empty.bin"
    path.write_bytes(b"")
    content, record = _create_enrichment_target(
        session,
        path,
        {"content_length": 1024},
    )

    with patch(
        "app.assets.scanner.extract_file_metadata",
        return_value=_ExtractedMetadata(None, {"content_length": 0}),
    ):
        enrich_asset(session, str(path), content.id, record.id)

    assert record.system_metadata == {"content_length": 0}


def test_seed_creates_content_and_record(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    (input_root / "first.png").write_bytes(b"first")
    (input_root / "second.png").write_bytes(b"second")

    with patch("folder_paths.get_input_directory", return_value=str(input_root)):
        created = seed_asset_specs(session, _build_seed_specs(input_root))
    session.commit()

    contents = list(session.scalars(select(AssetContent).order_by(AssetContent.path)))
    records = list(session.scalars(select(Asset).order_by(Asset.name)))

    assert created == 2
    assert len(contents) == 2
    assert len(records) == 2
    assert [record.loader_path for record in records] == ["first.png", "second.png"]
    assert [link.tag_name for link in session.scalars(select(AssetTag).order_by(AssetTag.asset_id))] == [
        "input",
        "input",
    ]


def test_prune_marks_missing_not_deletes(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    file_path = input_root / "removed-from-registry.png"
    file_path.write_bytes(b"content")

    with patch("folder_paths.get_input_directory", return_value=str(input_root)):
        seed_asset_specs(session, _build_seed_specs(input_root))
    session.commit()

    marked = mark_contents_missing_outside_prefixes(session, prefixes=[])
    session.commit()

    content = session.scalar(select(AssetContent))
    record = session.scalar(select(Asset))
    missing_tag = session.get(AssetTag, {"asset_id": record.id, "tag_name": "missing"})

    assert marked == 1
    assert content is not None and content.is_missing is True
    assert record is not None
    assert missing_tag is not None and missing_tag.origin == "automatic"


def test_unhashed_missing_content_gets_tagged(session, temp_dir: Path):
    missing_path = os.path.abspath(temp_dir / "missing.bin")
    content = AssetContent(path=missing_path, hash=None, size_bytes=7, mtime_ns=1)
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name="missing.bin")
    session.add(record)
    session.commit()

    sync_prefixes_with_filesystem(session, prefixes=[str(temp_dir)])
    session.commit()

    session.expire_all()
    content = session.get(AssetContent, content.id)
    missing_tag = session.get(AssetTag, {"asset_id": record.id, "tag_name": "missing"})

    assert content is not None and content.is_missing is True
    assert missing_tag is not None and missing_tag.origin == "automatic"


def test_enrichment_keeps_equal_hash_contents_distinct(session, temp_dir: Path):
    path = temp_dir / "new.bin"
    path.write_bytes(b"same bytes")
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot

    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )
    existing = AssetContent(
        path=str(temp_dir / "existing.bin"),
        hash=to_stored_hash(digest),
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )
    session.add_all((content, existing))
    session.flush()
    record = Asset(content_id=content.id, name=path.name)
    existing_record = Asset(content_id=existing.id, name="existing.bin")
    session.add_all((record, existing_record))
    session.commit()

    enriched = enrich_asset(
        session,
        file_path=str(path),
        content_id=content.id,
        record_id=record.id,
        extract_metadata=False,
        compute_hash=True,
    )

    assert enriched is True
    assert session.get(AssetContent, content.id).hash == to_stored_hash(digest)
    assert session.get(Asset, record.id).content_id == content.id
    assert session.get(Asset, existing_record.id).content_id == existing.id
