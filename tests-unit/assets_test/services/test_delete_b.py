import os

from sqlalchemy import update

import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import (
    create_content,
    create_record,
    delete_record,
    get_record_by_id,
    mark_content_missing,
)
from app.assets.services.asset_management import delete_asset_reference
from app.assets.services.ingest import register_executed_output


def test_hard_delete_record_content_and_file_remain(mock_create_session, session, temp_dir):
    content = create_content(session, path=str(temp_dir / "file.bin"), size_bytes=4)
    record = create_record(session, content_id=content.id, name="file.bin")
    temp_dir.joinpath("file.bin").write_bytes(b"data")
    session.commit()
    record_id = record.id
    content_id = content.id
    path = content.path

    assert delete_asset_reference(record_id) is True

    session.expire_all()
    assert get_record_by_id(session, record_id) is None
    assert session.get(AssetContent, content_id) is not None
    assert os.path.isfile(path)


def test_two_records_one_content_other_unaffected(mock_create_session, session):
    content = create_content(session, path="/tmp/shared.bin")
    r1 = create_record(session, content_id=content.id, name="a")
    r2 = create_record(session, content_id=content.id, name="b")
    session.commit()
    r1_id = r1.id
    r2_id = r2.id

    assert delete_asset_reference(r1_id) is True

    session.expire_all()
    assert get_record_by_id(session, r1_id) is None
    assert get_record_by_id(session, r2_id) is not None
    assert session.get(AssetContent, content.id) is not None


def test_delete_record_never_deletes_pointed_at_preview(mock_create_session, session):
    b_content = create_content(session, path="/tmp/preview_target.png")
    b = create_record(session, content_id=b_content.id, name="B")
    a_content = create_content(session, path="/tmp/owner.png")
    a = create_record(session, content_id=a_content.id, name="A")
    session.execute(update(Asset).where(Asset.id == a.id).values(preview_id=b.id))
    session.commit()
    a_id = a.id
    b_id = b.id

    assert delete_asset_reference(a_id) is True

    session.expire_all()
    assert get_record_by_id(session, a_id) is None
    assert get_record_by_id(session, b_id) is not None


def test_missing_content_record_deletable(mock_create_session, session):
    content = create_content(session, path="/tmp/missing.bin")
    record = create_record(session, content_id=content.id, name="missing.bin")
    mark_content_missing(session, content.id)
    session.commit()
    record_id = record.id

    assert delete_asset_reference(record_id) is True
    session.expire_all()
    assert get_record_by_id(session, record_id) is None


def test_reregister_same_path_fresh_ids(mock_create_session, monkeypatch):
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    f = os.path.join(output_dir, "test_delete_reregister.png")

    try:
        with open(f, "wb") as fh:
            fh.write(b"v1")
        first = register_executed_output(f, job_id="job1")
        first_id = first.id
        assert delete_asset_reference(first_id) is True

        with open(f, "wb") as fh:
            fh.write(b"v2")
        second = register_executed_output(f, job_id="job2")
        assert second.id != first_id
    finally:
        if os.path.exists(f):
            os.unlink(f)


def test_delete_nonexistent_returns_false(mock_create_session):
    assert delete_asset_reference("00000000-0000-0000-0000-000000000000") is False


def test_delete_order_record_before_content(mock_create_session, session):
    content = create_content(session, path="/tmp/order.bin")
    record = create_record(session, content_id=content.id, name="order.bin")
    session.commit()
    content_id = content.id

    delete_record(session, record.id)
    session.flush()

    assert session.get(Asset, record.id) is None
    assert session.get(AssetContent, content_id) is not None
