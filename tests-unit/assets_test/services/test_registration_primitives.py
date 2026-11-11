import os
from datetime import datetime

import folder_paths
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.services.ingest import (
    register_cached_output,
    register_executed_output,
    register_file_in_place,
)


def _output_path(name: str) -> str:
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


def test_executed_creates_metadata_and_null_hash(mock_create_session):
    body = b"pixels"
    path = _output_path("primitives_executed_meta.png")
    with open(path, "wb") as fh:
        fh.write(body)

    try:
        record = register_executed_output(path, job_id="job-exec")

        assert record is not None
        with mock_create_session() as session:
            content = session.scalar(
                select(AssetContent).where(
                    AssetContent.path == os.path.abspath(path)
                )
            )
            asset = session.get(Asset, record.id)

            assert content.hash is None
            assert content.is_missing is False
            assert asset.system_metadata is not None
            assert asset.system_metadata.get("format") == "png"
            assert asset.system_metadata.get("content_length") == len(body)
            assert "filename" in asset.system_metadata
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_executed_over_live_path_marks_old_missing(mock_create_session):
    path = _output_path("primitives_executed_overwrite.png")

    try:
        with open(path, "wb") as fh:
            fh.write(b"v1")
        first = register_executed_output(path, job_id="job1")

        with open(path, "wb") as fh:
            fh.write(b"v2")
        second = register_executed_output(path, job_id="job2")

        assert first is not None
        assert second is not None
        assert second.id != first.id
        with mock_create_session() as session:
            old_content = session.get(AssetContent, first.content_id)
            old_record = session.get(Asset, first.id)
            new_record = session.get(Asset, second.id)

            assert old_content.is_missing is True
            assert old_record.job_id == "job1"
            assert new_record.job_id == "job2"
            assert new_record.content_id != first.content_id
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_cached_copies_sibling_metadata(mock_create_session):
    sibling_metadata = {"provenance": "sibling", "width": 4, "kind": "image"}
    path = _output_path("primitives_cached_copy.png")
    with open(path, "wb") as fh:
        fh.write(b"pixels")

    try:
        with mock_create_session() as session:
            content = create_content(session, os.path.abspath(path), None, 6, 0)
            create_record(
                session,
                content.id,
                "sibling",
                system_metadata=dict(sibling_metadata),
            )
            session.commit()
            content_id = content.id

        cached = register_cached_output(path, job_id="delivery-job")

        assert cached is not None
        assert cached.content_id == content_id
        with mock_create_session() as session:
            record = session.get(Asset, cached.id)
            assert record.system_metadata == sibling_metadata
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_cached_earliest_sibling_wins(mock_create_session):
    earliest = datetime(2020, 1, 1, 0, 0, 0)
    later = datetime(2020, 1, 2, 0, 0, 0)
    winner_metadata = {"winner": "earliest-low-id"}
    path = _output_path("primitives_cached_earliest.png")
    with open(path, "wb") as fh:
        fh.write(b"pixels")

    try:
        with mock_create_session() as session:
            content = create_content(session, os.path.abspath(path), None, 6, 0)
            session.add_all(
                [
                    Asset(
                        id="00000000-0000-0000-0000-000000000001",
                        content_id=content.id,
                        name="earliest-low-id",
                        created_at=earliest,
                        system_metadata=dict(winner_metadata),
                    ),
                    Asset(
                        id="00000000-0000-0000-0000-000000000002",
                        content_id=content.id,
                        name="earliest-high-id",
                        created_at=earliest,
                        system_metadata={"winner": "earliest-high-id"},
                    ),
                    Asset(
                        id="00000000-0000-0000-0000-000000000000",
                        content_id=content.id,
                        name="later-lowest-id",
                        created_at=later,
                        system_metadata={"winner": "later-lowest-id"},
                    ),
                ]
            )
            session.commit()
            content_id = content.id

        cached = register_cached_output(path, job_id="delivery-job")

        assert cached is not None
        assert cached.content_id == content_id
        with mock_create_session() as session:
            record = session.get(Asset, cached.id)
            assert record.system_metadata == winner_metadata
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_cached_orphan_content_extracts_fresh(mock_create_session):
    body = b"pixels"
    path = _output_path("primitives_cached_orphan.png")
    with open(path, "wb") as fh:
        fh.write(body)

    try:
        with mock_create_session() as session:
            content = create_content(
                session, os.path.abspath(path), None, len(body), 0
            )
            session.commit()
            content_id = content.id

        cached = register_cached_output(path, job_id="delivery-job")

        assert cached is not None
        assert cached.content_id == content_id
        with mock_create_session() as session:
            record = session.get(Asset, cached.id)
            assert record.system_metadata is not None
            assert record.system_metadata.get("format") == "png"
            assert record.system_metadata.get("content_length") == len(body)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_cached_missing_content_is_nonevent(mock_create_session):
    path = _output_path("primitives_cached_missing.png")
    with open(path, "wb") as fh:
        fh.write(b"pixels")

    try:
        cached = register_cached_output(path, job_id="delivery-job")

        assert cached is None
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            records = list(
                session.scalars(
                    select(Asset).where(Asset.job_id == "delivery-job")
                )
            )
            assert contents == []
            assert records == []
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_registration_failure_never_raises_and_leaves_no_rows(
    mock_create_session, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated create_record failure")

    monkeypatch.setattr(
        "app.assets.services.ingest.create_record", _boom
    )

    path = _output_path("primitives_reg_fail.png")
    with open(path, "wb") as fh:
        fh.write(b"pixels")

    try:
        result = register_executed_output(path, job_id="job-boom")

        assert result is None
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == os.path.abspath(path)
                    )
                )
            )
            records = list(
                session.scalars(
                    select(Asset).where(Asset.job_id == "job-boom")
                )
            )
            assert contents == []
            assert records == []
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_upload_record_has_metadata_and_hash(mock_create_session):
    body = b"pixels-content"
    path = _output_path("primitives_upload_in_place.png")
    with open(path, "wb") as fh:
        fh.write(body)

    try:
        result = register_file_in_place(
            path, name="pic.png", tags=["demo"], mime_type="image/png"
        )

        assert result.asset.hash is not None
        assert result.ref.system_metadata is not None
        assert result.ref.system_metadata.get("format") == "png"
        assert result.ref.system_metadata.get("content_length") == len(body)
    finally:
        if os.path.exists(path):
            os.unlink(path)
