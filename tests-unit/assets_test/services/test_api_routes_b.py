import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.api import routes
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)


def _session_factory(engine) -> Callable[[], Session]:
    return lambda: Session(engine)


@pytest.mark.asyncio
async def test_unfiltered_listing_includes_missing_entity(
    db_engine, session, temp_dir, monkeypatch
):
    content = create_content(session, path=str(temp_dir / "missing.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="missing.png",
        mime_type="image/png",
        tags=["input"],
    )
    mark_content_missing(session, content.id)
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)

    response = await routes.list_assets_route(
        make_mocked_request("GET", "/api/assets")
    )

    assert isinstance(response, web.Response)
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    body = json.loads(response_body)
    assert record.id in {item["id"] for item in body["assets"]}
    assert body["total"] == 1
    listed = next(item for item in body["assets"] if item["id"] == record.id)
    assert "missing" in listed["tags"]


@pytest.mark.asyncio
async def test_exclude_tags_missing_hides_it(db_engine, session, temp_dir, monkeypatch):
    content = create_content(session, path=str(temp_dir / "missing.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="missing.png",
        mime_type="image/png",
    )
    mark_content_missing(session, content.id)
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)

    response = await routes.list_assets_route(
        make_mocked_request("GET", "/api/assets?exclude_tags=missing")
    )

    assert isinstance(response, web.Response)
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    body = json.loads(response_body)
    assert record.id not in {item["id"] for item in body["assets"]}


@pytest.mark.asyncio
async def test_listing_serves_is_immutable_as_false(
    db_engine, session, temp_dir, monkeypatch
):
    content = create_content(session, path=str(temp_dir / "mutable.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="mutable.png",
        mime_type="image/png",
    )
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)

    response = await routes.list_assets_route(
        make_mocked_request("GET", "/api/assets")
    )

    assert isinstance(response, web.Response)
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    listed = next(
        item
        for item in json.loads(response_body)["assets"]
        if item["id"] == record.id
    )
    assert listed["is_immutable"] is False, (
        "the field must be present in the serialized body, not dropped as a default"
    )


@pytest.mark.asyncio
async def test_detail_serves_is_immutable_as_false(
    mock_create_session, db_engine, session, temp_dir, monkeypatch
):
    content = create_content(session, path=str(temp_dir / "detail.png"))
    record = create_record(
        session,
        content_id=content.id,
        name="detail.png",
        mime_type="image/png",
    )
    session.commit()

    monkeypatch.setattr(routes, "create_session", _session_factory(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)

    response = await routes.get_asset_route(
        make_mocked_request(
            "GET", f"/api/assets/{record.id}", match_info={"id": record.id}
        )
    )

    assert isinstance(response, web.Response)
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    assert json.loads(response_body)["is_immutable"] is False


@pytest.mark.asyncio
async def test_from_hash_off_mode_returns_400(monkeypatch):
    monkeypatch.setattr(mode, "hashing_enabled", lambda: False)
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    request = AsyncMock(spec=web.Request)
    request.json.return_value = {"hash": f"blake3:{'a' * 64}"}

    response = await routes.create_asset_from_hash_route(request)

    assert isinstance(response, web.Response)
    assert response.status == 400
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    assert json.loads(response_body)["error"]["code"] == "FEATURE_DISABLED"
