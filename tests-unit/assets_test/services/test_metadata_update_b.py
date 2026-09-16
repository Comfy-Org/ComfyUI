import pytest

from app.assets.database.models import Asset, AssetTag, Tag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    fetch_record_tags,
)
from app.assets.services.asset_management import update_asset_metadata
from app.assets.services.tagging import apply_tags, remove_tags


def _create_record(session, path: str, tags: list[str] | None = None) -> Asset:
    content = create_content(session, path)
    record = create_record(session, content.id, "original", tags=tags)
    session.commit()
    return record


def test_rename_via_update_asset_metadata(session, mock_create_session):
    record = _create_record(session, "/output/rename.png")

    update_asset_metadata(record.id, name="renamed")

    session.expire_all()
    assert session.get(Asset, record.id).name == "renamed"


def test_apply_tags_adds_tags(session, mock_create_session):
    record = _create_record(session, "/output/apply-tags.png")

    apply_tags(record.id, ["foo", "bar"])

    session.expire_all()
    assert fetch_record_tags(session, record.id) == ["bar", "foo"]


def test_remove_tags_removes_non_automatic(session, mock_create_session):
    record = _create_record(session, "/output/remove-tags.png", tags=["foo"])
    session.add(Tag(name="automatic"))
    session.add(AssetTag(asset_id=record.id, tag_name="automatic", origin="automatic"))
    session.commit()

    remove_tags(record.id, ["foo"])

    session.expire_all()
    assert fetch_record_tags(session, record.id) == ["automatic"]


def test_update_asset_metadata_unknown_preview_id_raises(session, mock_create_session):
    record = _create_record(session, "/output/preview-validate.png")

    with pytest.raises(ValueError):
        update_asset_metadata(
            record.id, preview_id="00000000-0000-0000-0000-000000000000"
        )

    session.expire_all()
    assert session.get(Asset, record.id).preview_id is None
