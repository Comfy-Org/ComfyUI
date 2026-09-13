from __future__ import annotations

import urllib.parse
from datetime import datetime

import pytest
from sqlalchemy import event

from . import helpers
from .helpers import (
    RouteDatabase,
    RecordSeed,
    error_body as _error_body,
    asset_list_body as _asset_list_body,
    seed_record as _seed_record,
    request_assets as _request_assets,
)
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.services.cursor import encode_cursor

autoclean_unit_test_assets = helpers.autoclean_unit_test_assets
route_database = helpers.route_database
sortable_record_ids = helpers.sortable_record_ids


def test_record_list_filters_record_tags(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("filtered.png", ["output", "unit-tests", "chosen"], {}, make_asset_bytes("filtered"))

    response = http.get(f"{api_base}/api/assets", params={"include_tags": "chosen"})

    assert response.status_code == 200
    assert [asset["id"] for asset in response.json()["assets"]] == [record["id"]]


def test_record_list_rejects_metadata_filter(http, api_base):
    response = http.get(
        f"{api_base}/api/assets", params={"metadata_filter": '{"k":"v"}'}
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "UNSUPPORTED_PARAM",
            "message": "metadata_filter is no longer supported",
            "details": {},
        }
    }


@pytest.mark.asyncio
async def test_tags_any_is_accepted(route_database: RouteDatabase) -> None:
    _, session = route_database
    _seed_record(session, RecordSeed("any.png", ("a",)))
    session.commit()

    response = await _request_assets("tags_any=a")

    assert response.status == 200


@pytest.mark.asyncio
async def test_tags_any_matches_any_requested_tag(route_database: RouteDatabase) -> None:
    _, session = route_database
    a_record = _seed_record(session, RecordSeed("a.png", ("a",)))
    b_record = _seed_record(session, RecordSeed("b.png", ("b",)))
    _seed_record(session, RecordSeed("neither.png", ("c",)))
    session.commit()

    response = await _request_assets("tags_any=a,b")

    body = _asset_list_body(response)
    assert {asset["id"] for asset in body["assets"]} == {a_record.id, b_record.id}


@pytest.mark.parametrize(("tag_count", "expected_status"), ((100, 200), (101, 400)))
@pytest.mark.asyncio
async def test_tags_any_enforces_the_shared_tag_cap(
    route_database: RouteDatabase,
    tag_count: int,
    expected_status: int,
) -> None:
    del route_database
    tags = ",".join(f"tag-{index}" for index in range(tag_count))

    response = await _request_assets(urllib.parse.urlencode({"tags_any": tags}))

    assert response.status == expected_status
    if expected_status == 400:
        assert _error_body(response)["error"]["code"] == "INVALID_TAG_FILTER"


@pytest.mark.asyncio
async def test_name_contains_filters_records(route_database: RouteDatabase) -> None:
    _, session = route_database
    match = _seed_record(session, RecordSeed("alpha.png"))
    _seed_record(session, RecordSeed("beta.png"))
    session.commit()

    response = await _request_assets("name_contains=lph")

    body = _asset_list_body(response)
    assert [asset["id"] for asset in body["assets"]] == [match.id]


@pytest.mark.asyncio
async def test_offset_skips_records(route_database: RouteDatabase) -> None:
    _, session = route_database
    _seed_record(session, RecordSeed("a.png", ("offset-case",)))
    second = _seed_record(session, RecordSeed("b.png", ("offset-case",)))
    session.commit()

    response = await _request_assets(
        "tags_all=offset-case&sort=name&order=asc&offset=1&limit=1"
    )

    body = _asset_list_body(response)
    assert [asset["id"] for asset in body["assets"]] == [second.id]


@pytest.mark.asyncio
async def test_default_order_is_newest_first(route_database: RouteDatabase) -> None:
    _, session = route_database
    older = _seed_record(session, RecordSeed("older.png"))
    newer = _seed_record(session, RecordSeed("newer.png"))
    older.created_at = datetime(2026, 1, 1)
    newer.created_at = datetime(2026, 1, 2)
    session.commit()

    response = await _request_assets()

    body = _asset_list_body(response)
    assert [asset["id"] for asset in body["assets"]] == [newer.id, older.id]


@pytest.mark.parametrize(
    ("sort", "order"),
    (
        ("name", "asc"),
        ("created_at", "desc"),
        ("updated_at", "asc"),
        ("size", "desc"),
        ("last_access_time", "desc"),
    ),
)
@pytest.mark.asyncio
async def test_each_sort_field_orders_records(
    route_database: RouteDatabase,
    sortable_record_ids: tuple[str, str],
    sort: str,
    order: str,
) -> None:
    del route_database
    query = urllib.parse.urlencode(
        {"tags_all": "sort-case", "sort": sort, "order": order}
    )

    response = await _request_assets(query)

    body = _asset_list_body(response)
    assert response.status == 200
    assert [asset["id"] for asset in body["assets"]] == list(sortable_record_ids)


@pytest.mark.asyncio
async def test_total_counts_all_filtered_records(route_database: RouteDatabase) -> None:
    _, session = route_database
    for index in range(5):
        _seed_record(session, RecordSeed(f"total-{index}.png", ("total-case",)))
    session.commit()

    response = await _request_assets("tags_all=total-case&limit=2")

    body = _asset_list_body(response)
    assert len(body["assets"]) == 2
    assert body["total"] == 5


@pytest.mark.parametrize(("offset", "expected"), ((0, True), (4, False)))
@pytest.mark.asyncio
async def test_offset_mode_has_more_uses_total(
    route_database: RouteDatabase,
    offset: int,
    expected: bool,
) -> None:
    _, session = route_database
    for index in range(5):
        _seed_record(session, RecordSeed(f"more-{index}.png", ("more-case",)))
    session.commit()

    response = await _request_assets(
        f"tags_all=more-case&sort=name&order=asc&offset={offset}&limit=1"
    )

    body = _asset_list_body(response)
    assert body["has_more"] is expected


@pytest.mark.asyncio
async def test_bad_cursor_returns_400(route_database: RouteDatabase) -> None:
    del route_database

    response = await _request_assets("after=not-a-cursor")

    assert response.status == 400
    assert _error_body(response)["error"]["code"] == "INVALID_CURSOR"


@pytest.mark.asyncio
async def test_cursor_rejects_last_access_time_sort(route_database: RouteDatabase) -> None:
    del route_database

    response = await _request_assets("after=not-a-cursor&sort=last_access_time")

    assert response.status == 400
    assert _error_body(response)["error"]["code"] == "INVALID_CURSOR"


@pytest.mark.parametrize(
    "query",
    (
        "sort=created_at&order=asc",
        "sort=name&order=desc",
    ),
)
@pytest.mark.asyncio
async def test_cursor_rejects_sort_or_order_mismatch(
    route_database: RouteDatabase,
    query: str,
) -> None:
    del route_database
    cursor = encode_cursor("name", "a.png", "cursor-id", order="asc")

    response = await _request_assets(f"{query}&after={cursor}")

    assert response.status == 400
    assert _error_body(response)["error"]["code"] == "INVALID_CURSOR"


@pytest.mark.asyncio
async def test_terminal_page_has_no_cursor(route_database: RouteDatabase) -> None:
    _, session = route_database
    only_record = _seed_record(session, RecordSeed("only.png", ("terminal-case",)))
    session.commit()

    response = await _request_assets(
        "tags_all=terminal-case&sort=name&order=asc&limit=1"
    )

    body = _asset_list_body(response)
    assert [asset["id"] for asset in body["assets"]] == [only_record.id]
    assert body["has_more"] is False
    assert "next_cursor" not in body


@pytest.mark.asyncio
async def test_page_query_budget_is_four_statements(
    route_database: RouteDatabase,
) -> None:
    engine, session = route_database
    preview = _seed_record(session, RecordSeed("preview.png", ("preview",)))
    for index in range(3):
        record = _seed_record(
            session,
            RecordSeed(f"page-{index}.png", ("budget-case",)),
        )
        record.preview_id = preview.id
    session.commit()
    statements: list[str] = []

    def count_statements(_, __, statement, ___, ____, _____) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", count_statements)
    try:
        response = await _request_assets("tags_all=budget-case&limit=3")
    finally:
        event.remove(engine, "before_cursor_execute", count_statements)

    body = _asset_list_body(response)
    assert len(body["assets"]) == 3
    assert {asset.get("preview_id") for asset in body["assets"]} == {preview.id}
    assert len(statements) == 4


@pytest.mark.asyncio
async def test_missing_content_is_listed_with_missing_tag(
    route_database: RouteDatabase,
) -> None:
    _, session = route_database
    live = _seed_record(session, RecordSeed("live.png", ("live-case",)))
    missing_content = create_content(session, "/output/missing.png")
    missing = create_record(
        session,
        content_id=missing_content.id,
        name="missing.png",
        tags=("live-case",),
    )
    mark_content_missing(session, missing_content.id)
    session.commit()

    response = await _request_assets("tags_all=live-case&sort=name&order=asc")

    body = _asset_list_body(response)
    assert {asset["id"] for asset in body["assets"]} == {live.id, missing.id}
    assert body["total"] == 2
    missing_asset = next(a for a in body["assets"] if a["id"] == missing.id)
    assert "missing" in missing_asset["tags"]


@pytest.mark.asyncio
async def test_missing_content_is_reachable_via_tags_all_missing(
    route_database: RouteDatabase,
) -> None:
    _, session = route_database
    _seed_record(session, RecordSeed("live.png", ("live-case",)))
    missing_content = create_content(session, "/output/missing.png")
    missing = create_record(
        session,
        content_id=missing_content.id,
        name="missing.png",
        tags=("live-case",),
    )
    mark_content_missing(session, missing_content.id)
    session.commit()

    response = await _request_assets("tags_all=missing")

    body = _asset_list_body(response)
    assert {asset["id"] for asset in body["assets"]} == {missing.id}
    assert body["total"] == 1
