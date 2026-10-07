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


def test_update_asset_metadata_clear_preview_unlinks(session, mock_create_session):
    preview = _create_record(session, "/output/thumb.webp")
    record = _create_record(session, "/output/with-preview.exr")
    update_asset_metadata(record.id, preview_id=preview.id)

    update_asset_metadata(record.id, clear_preview=True)

    session.expire_all()
    assert session.get(Asset, record.id).preview_id is None
    assert session.get(Asset, preview.id) is not None, "clearing the link must not delete the preview"


def test_update_body_explicit_null_preview_id_is_a_clear():
    from pydantic import ValidationError

    from app.assets.api.schemas_in import UpdateAssetBody

    assert UpdateAssetBody.model_validate({"preview_id": None}).clears_preview is True
    assert UpdateAssetBody.model_validate({"name": "x"}).clears_preview is False
    with pytest.raises(ValidationError):
        UpdateAssetBody.model_validate({})


@pytest.mark.asyncio
async def test_put_rejects_an_asset_as_its_own_preview(session, mock_create_session):
    from unittest.mock import AsyncMock, patch

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from app.assets.api import routes

    record = _create_record(session, "/output/self.png")
    request = make_mocked_request("PUT", f"/api/assets/{record.id}", match_info={"id": record.id})

    with (
        patch.object(routes, "_ASSETS_ENABLED", True),
        patch.object(web.BaseRequest, "json", AsyncMock(return_value={"preview_id": record.id})),
    ):
        response = await routes.update_asset_route(request)

    assert response.status == 400, "a self-reference makes the record undeletable, so it is refused"
    session.expire_all()
    assert session.get(Asset, record.id).preview_id is None
