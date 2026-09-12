import os

from sqlalchemy import event, select

import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import mark_content_missing
from app.assets.services.ingest import register_cached_output, register_executed_output


def test_cached_save_creates_delivery_record(mock_create_session, db_engine):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "test_cached_save.png")
    update_count = 0

    def count_updates(_, __, statement, ___, ____, _____):
        nonlocal update_count
        if statement.lstrip().upper().startswith("UPDATE"):
            update_count += 1

    event.listen(db_engine, "before_cursor_execute", count_updates)
    try:
        with open(path, "wb") as file:
            file.write(b"pixels")
        original = register_executed_output(path, job_id="executed-job")
        update_count = 0

        cached = register_cached_output(path, job_id="delivery-job")

        with mock_create_session() as session:
            records = list(
                session.scalars(select(Asset).where(Asset.content_id == original.content_id))
            )
            assert {record.id for record in records} == {original.id, cached.id}
            assert {record.job_id for record in records} == {"executed-job", "delivery-job"}
        assert update_count == 0
    finally:
        event.remove(db_engine, "before_cursor_execute", count_updates)
        if os.path.exists(path):
            os.unlink(path)


def test_cached_save_against_missing_content_is_nonevent(mock_create_session):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "test_cached_save_missing.png")

    try:
        with open(path, "wb") as file:
            file.write(b"pixels")
        original = register_executed_output(path, job_id="executed-job")
        with mock_create_session() as session:
            mark_content_missing(session, original.content_id)
            session.commit()

        cached = register_cached_output(path, job_id="delivery-job")

        assert cached is None
        with mock_create_session() as session:
            contents = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.path == os.path.abspath(path))
                )
            )
            assert len(contents) == 1
            assert contents[0].id == original.content_id
            assert contents[0].is_missing is True
            records = list(
                session.scalars(select(Asset).where(Asset.job_id == "delivery-job"))
            )
            assert records == []
    finally:
        if os.path.exists(path):
            os.unlink(path)
