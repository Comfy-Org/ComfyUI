"""Integration tests for cursor-based pagination on GET /api/assets.

These tests exercise the handler/service/query path end-to-end;
cursor-encoding-level tests live in
tests-unit/assets_test/services/test_cursor.py.
"""
import pytest
import requests


def _seed(asset_factory, make_asset_bytes, count: int, tag: str) -> list[str]:
    names = [f"cursor_{i:02d}.safetensors" for i in range(count)]
    for n in names:
        asset_factory(
            n,
            ["models", "checkpoints", "unit-tests", tag],
            {},
            make_asset_bytes(n, size=2048),
        )
    return sorted(names)


def test_cursor_pages_all_items_in_order(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    names = _seed(asset_factory, make_asset_bytes, count=5, tag="cursor-walk")

    params = {
        "include_tags": "unit-tests,cursor-walk",
        "sort": "name",
        "order": "asc",
        "limit": "2",
    }

    seen: list[str] = []
    after: str | None = None
    pages = 0
    while True:
        page_params = dict(params)
        if after is not None:
            page_params["after"] = after
        r = http.get(api_base + "/api/assets", params=page_params, timeout=120)
        assert r.status_code == 200, r.text
        body = r.json()
        seen.extend(a["name"] for a in body["assets"])
        pages += 1
        after = body.get("next_cursor")
        if after is None:
            break
        assert body["has_more"] is True
        assert pages < 10, "guard against runaway cursor loop"

    assert seen == names, f"expected {names}, got {seen}"
    # Last page should have has_more False
    assert body["has_more"] is False
    assert "next_cursor" not in body


def test_cursor_invalid_returns_400(http: requests.Session, api_base: str):
    r = http.get(
        api_base + "/api/assets",
        params={"after": "not-a-real-cursor", "sort": "created_at"},
        timeout=120,
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "INVALID_CURSOR"


def test_cursor_sort_mismatch_returns_400(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    _seed(asset_factory, make_asset_bytes, count=2, tag="cursor-mismatch")

    # Take a real cursor minted for sort=name.
    r = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-mismatch",
            "sort": "name",
            "order": "asc",
            "limit": "1",
        },
        timeout=120,
    )
    assert r.status_code == 200
    cursor = r.json()["next_cursor"]
    assert cursor is not None

    # Replay against sort=created_at — should fail with INVALID_CURSOR.
    r2 = http.get(
        api_base + "/api/assets",
        params={"after": cursor, "sort": "created_at"},
        timeout=120,
    )
    assert r2.status_code == 400, r2.text
    assert r2.json()["error"]["code"] == "INVALID_CURSOR"


def test_cursor_wins_over_offset(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    names = _seed(asset_factory, make_asset_bytes, count=4, tag="cursor-vs-offset")

    # Take a cursor that points past the first item.
    r = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-vs-offset",
            "sort": "name",
            "order": "asc",
            "limit": "1",
        },
        timeout=120,
    )
    assert r.status_code == 200, r.text
    cursor = r.json()["next_cursor"]
    assert cursor is not None

    # Pass both 'after' and a large offset. Cursor must win; offset is ignored.
    r2 = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-vs-offset",
            "sort": "name",
            "order": "asc",
            "limit": "1",
            "after": cursor,
            "offset": "999",
        },
        timeout=120,
    )
    assert r2.status_code == 200
    body = r2.json()
    # Should land on the second name in sorted order — not skip ahead by 999.
    assert [a["name"] for a in body["assets"]] == [names[1]]


def test_next_cursor_absent_when_no_more_results(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    _seed(asset_factory, make_asset_bytes, count=2, tag="cursor-exhaust")

    r = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-exhaust",
            "sort": "name",
            "order": "asc",
            "limit": "50",
        },
        timeout=120,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_more"] is False
    assert "next_cursor" not in body


def test_cursor_pagination_first_page_mints_cursor(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    """First-page request (no `after`) must still return `next_cursor` when
    more rows exist, or pagination is unreachable from a cold start.
    """
    _seed(asset_factory, make_asset_bytes, count=3, tag="cursor-first-page")
    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,cursor-first-page", "sort": "name", "order": "asc", "limit": "2"},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_more"] is True
    assert body.get("next_cursor"), "first page must mint a cursor when more rows exist"


def test_cursor_no_spurious_cursor_when_page_size_equals_remainder(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    """When `total` is an exact multiple of `limit`, the final page must
    NOT carry a next_cursor — there is nothing past it.
    """
    _seed(asset_factory, make_asset_bytes, count=4, tag="cursor-exact-multiple")
    # Page 1
    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,cursor-exact-multiple", "sort": "name", "order": "asc", "limit": "2"},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    cursor = r.json()["next_cursor"]
    assert cursor is not None
    # Page 2 — should exhaust the set with no cursor for a phantom page 3
    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,cursor-exact-multiple", "sort": "name", "order": "asc", "limit": "2", "after": cursor},
        timeout=120,
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert len(body["assets"]) == 2
    assert body["has_more"] is False
    assert "next_cursor" not in body


@pytest.mark.parametrize("sort_field", ["created_at", "updated_at", "size"])
def test_cursor_walks_for_non_name_sorts(sort_field, http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    """Cursor pagination must work for every sort field the contract claims.

    Without this, the `created_at` / `updated_at` (time-encoded micros) and
    `size` (int-encoded) cursor paths go entirely unexercised end-to-end.
    """
    # Sizes increase strictly by index, so `size desc` has a deterministic
    # expected order. Time-based sorts (created_at / updated_at) can tie when
    # rows are inserted faster than the DB's timestamp resolution; for those
    # we check coverage and no-duplicates and let the keyset tiebreaker do
    # the rest, instead of sleeping between inserts and asserting an order
    # that depends on clock granularity.
    names = []
    for i in range(4):
        n = f"cursor_{sort_field}_{i:02d}.safetensors"
        asset_factory(n, ["models", "checkpoints", "unit-tests", f"cursor-{sort_field}"], {}, make_asset_bytes(n, size=2048 + i))
        names.append(n)

    params = {
        "include_tags": f"unit-tests,cursor-{sort_field}",
        "sort": sort_field,
        "order": "desc",
        "limit": "2",
    }
    seen: list[str] = []
    after: str | None = None
    pages = 0
    while True:
        page_params = dict(params)
        if after is not None:
            page_params["after"] = after
        r = http.get(api_base + "/api/assets", params=page_params, timeout=120)
        assert r.status_code == 200, r.text
        body = r.json()
        seen.extend(a["name"] for a in body["assets"])
        after = body.get("next_cursor")
        pages += 1
        if after is None:
            break
        assert pages < 10, "guard against runaway cursor loop"

    # No duplicates: a faulty keyset boundary that returns the same row across
    # two pages must fail this check.
    assert len(seen) == len(set(seen)), (
        f"cursor walk repeated rows for sort={sort_field}: {seen}"
    )
    # Full coverage: every seeded asset reached exactly once.
    assert set(seen) == set(names), (
        f"missing items for sort={sort_field}: expected {set(names)}, got {set(seen)}"
    )
    # Strict order check for the only field with a clock-independent ordering.
    if sort_field == "size":
        assert seen == list(reversed(names)), (
            f"size cursor walked out of order: got {seen}, expected {list(reversed(names))}"
        )


def test_cursor_order_mismatch_returns_400(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    """A cursor minted under desc order replayed against asc must 400, not
    silently walk the wrong direction."""
    _seed(asset_factory, make_asset_bytes, count=3, tag="cursor-order-flip")

    r = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-order-flip",
            "sort": "name",
            "order": "desc",
            "limit": "1",
        },
        timeout=120,
    )
    assert r.status_code == 200, r.text
    cursor = r.json()["next_cursor"]
    assert cursor is not None

    # Replay with order flipped to asc — server must reject the cursor.
    r2 = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-order-flip",
            "sort": "name",
            "order": "asc",
            "limit": "1",
            "after": cursor,
        },
        timeout=120,
    )
    assert r2.status_code == 400, r2.text
    assert r2.json()["error"]["code"] == "INVALID_CURSOR"


def test_cursor_invalid_cursor_at_microsecond_boundary(http: requests.Session, api_base: str):
    """A cursor carrying an out-of-range microsecond timestamp must map to
    400 INVALID_CURSOR, not 500."""
    import base64
    import json
    # 10^18 microseconds ≈ year 33658, well past datetime.MAX_YEAR.
    # `o` and `order=` must be set; otherwise decode fails earlier on the
    # missing-order branch and the µs-overflow path is never exercised.
    payload = {"s": "created_at", "o": "desc", "v": "999999999999999999999", "id": "asset-x"}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    cursor = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    r = http.get(
        api_base + "/api/assets",
        params={"after": cursor, "sort": "created_at", "order": "desc"},
        timeout=120,
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "INVALID_CURSOR"


def test_cursor_pagination_stable_after_delete(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    names = _seed(asset_factory, make_asset_bytes, count=4, tag="cursor-delete")

    # Page 1.
    r = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-delete",
            "sort": "name",
            "order": "asc",
            "limit": "2",
        },
        timeout=120,
    )
    assert r.status_code == 200
    body = r.json()
    page1_names = [a["name"] for a in body["assets"]]
    cursor = body["next_cursor"]
    assert cursor is not None
    assert page1_names == names[:2]

    # Delete an item from page 1 (already returned) — cursor should still
    # locate the next page from where it was minted, not re-index.
    target_id = body["assets"][0]["id"]
    d = http.delete(api_base + f"/api/assets/{target_id}", timeout=120)
    assert d.status_code in (200, 204), d.text

    # Page 2 via cursor.
    r2 = http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,cursor-delete",
            "sort": "name",
            "order": "asc",
            "limit": "2",
            "after": cursor,
        },
        timeout=120,
    )
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert [a["name"] for a in body2["assets"]] == names[2:]
