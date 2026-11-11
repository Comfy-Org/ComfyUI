import os

import pytest
from sqlalchemy import select

import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.services.ingest import register_executed_output


@pytest.fixture(autouse=True)
def hashing_off():
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def test_new_path_save_off_mode_hash_null(mock_create_session):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    f = os.path.join(output_dir, "test_ingest_b_new.png")
    with open(f, "wb") as fh:
        fh.write(b"pixels")

    try:
        record = register_executed_output(f, job_id="job1")
        record_id = record.id
        with mock_create_session() as session:
            content = session.execute(
                select(AssetContent).where(AssetContent.path == os.path.abspath(f))
            ).scalar_one()
            assert content.hash is None
            assert content.is_missing is False
            asset = session.execute(
                select(Asset).where(Asset.id == record_id)
            ).scalar_one()
            assert asset.job_id == "job1"
    finally:
        if os.path.exists(f):
            os.unlink(f)


def test_overwrite_at_live_path_marks_old_missing(mock_create_session):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    f = os.path.join(output_dir, "test_ingest_b_overwrite.png")

    try:
        with open(f, "wb") as fh:
            fh.write(b"v1")
        r1 = register_executed_output(f, job_id="job1")
        old_content_id = r1.content_id
        old_record_id = r1.id

        with open(f, "wb") as fh:
            fh.write(b"v2")
        r2 = register_executed_output(f, job_id="job2")
        new_record_id = r2.id

        with mock_create_session() as session:
            old_content = session.execute(
                select(AssetContent).where(AssetContent.id == old_content_id)
            ).scalar_one()
            assert old_content.is_missing is True, "Old content should be marked missing"

            old_record = session.execute(
                select(Asset).where(Asset.id == old_record_id)
            ).scalar_one()
            assert old_record.job_id == "job1", "Old record's job_id must not be mutated"

            new_record = session.execute(
                select(Asset).where(Asset.id == new_record_id)
            ).scalar_one()
            assert new_record.job_id == "job2"
            assert new_record_id != old_record_id
    finally:
        if os.path.exists(f):
            os.unlink(f)
