import json
import uuid

import requests


def test_tags_present(http: requests.Session, api_base: str, seeded_asset: dict):
    # Include zero-usage tags by default
    r1 = http.get(api_base + "/api/tags", params={"limit": "50"}, timeout=120)
    body1 = r1.json()
    assert r1.status_code == 200
    names = [t["name"] for t in body1["tags"]]
    # A few system tags from migration should exist:
    assert "models" in names
    assert "checkpoints" in names

    # Only used tags before we add anything new from this test cycle
    r2 = http.get(api_base + "/api/tags", params={"include_zero": "false"}, timeout=120)
    body2 = r2.json()
    assert r2.status_code == 200
    # We already seeded one asset via fixture, so used tags must be non-empty
    used_names = [t["name"] for t in body2["tags"]]
    assert "models" in used_names
    assert "checkpoints" in used_names

    # Prefix filter should refine the list
    r3 = http.get(api_base + "/api/tags", params={"include_zero": "false", "prefix": "uni"}, timeout=120)
    b3 = r3.json()
    assert r3.status_code == 200
    names3 = [t["name"] for t in b3["tags"]]
    assert "unit-tests" in names3
    assert "models" not in names3  # filtered out by prefix

    # Order by name ascending should be stable
    r4 = http.get(api_base + "/api/tags", params={"include_zero": "false", "order": "name_asc"}, timeout=120)
    b4 = r4.json()
    assert r4.status_code == 200
    names4 = [t["name"] for t in b4["tags"]]
    assert names4 == sorted(names4)


def test_tags_empty_usage(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    # Baseline: system tags exist when include_zero (default) is true
    r1 = http.get(api_base + "/api/tags", params={"limit": "500"}, timeout=120)
    body1 = r1.json()
    assert r1.status_code == 200
    names = [t["name"] for t in body1["tags"]]
    assert "models" in names and "checkpoints" in names

    # Create a short-lived asset under input with a unique custom tag
    scope = f"tags-empty-usage-{uuid.uuid4().hex[:6]}"
    custom_tag = f"temp-{uuid.uuid4().hex[:8]}"
    name = "tag_seed.bin"
    _asset = asset_factory(
        name,
        ["input", "unit-tests", scope, custom_tag],
        {},
        make_asset_bytes(name, 512),
    )

    # While the asset exists, the custom tag must appear when include_zero=false
    r2 = http.get(
        api_base + "/api/tags",
        params={"include_zero": "false", "prefix": custom_tag, "limit": "50"},
        timeout=120,
    )
    body2 = r2.json()
    assert r2.status_code == 200
    used_names = [t["name"] for t in body2["tags"]]
    assert custom_tag in used_names

    # Delete the asset reference so the tag usage drops to zero
    rd = http.delete(f"{api_base}/api/assets/{_asset['id']}", timeout=120)
    assert rd.status_code == 204

    # Now the custom tag must not be returned when include_zero=false
    r3 = http.get(
        api_base + "/api/tags",
        params={"include_zero": "false", "prefix": custom_tag, "limit": "50"},
        timeout=120,
    )
    body3 = r3.json()
    assert r3.status_code == 200
    names_after = [t["name"] for t in body3["tags"]]
    assert custom_tag not in names_after
    assert not names_after  # filtered view should be empty now


def test_add_and_remove_tags(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # Add tags with duplicates and mixed case
    payload_add = {"tags": ["NewTag", "unit-tests", "newtag", "BETA"]}
    r1 = http.post(f"{api_base}/api/assets/{aid}/tags", json=payload_add, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 200, b1
    # normalized, deduplicated; 'unit-tests' was already present from the seed
    assert set(b1["added"]) == {"newtag", "beta"}
    assert set(b1["already_present"]) == {"unit-tests"}
    assert "newtag" in b1["total_tags"] and "beta" in b1["total_tags"]

    rg = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    g = rg.json()
    assert rg.status_code == 200
    tags_now = set(g["tags"])
    assert {"newtag", "beta"}.issubset(tags_now)

    # Remove a tag and a non-existent tag
    payload_del = {"tags": ["newtag", "does-not-exist"]}
    r2 = http.delete(f"{api_base}/api/assets/{aid}/tags", json=payload_del, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200
    assert set(b2["removed"]) == {"newtag"}
    assert set(b2["not_present"]) == {"does-not-exist"}

    # Verify remaining tags after deletion
    rg2 = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    g2 = rg2.json()
    assert rg2.status_code == 200
    tags_later = set(g2["tags"])
    assert "newtag" not in tags_later
    assert "beta" in tags_later  # still present


def test_tags_list_order_and_prefix(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]
    h = seeded_asset["asset_hash"]

    # Add both tags to the seeded asset (usage: orderaaa=1, orderbbb=1)
    r_add = http.post(f"{api_base}/api/assets/{aid}/tags", json={"tags": ["orderaaa", "orderbbb"]}, timeout=120)
    add_body = r_add.json()
    assert r_add.status_code == 200, add_body

    # Create another AssetInfo from the same content but tagged ONLY with 'orderbbb'.
    payload = {
        "hash": h,
        "name": "order_only_bbb.safetensors",
        "tags": ["input", "unit-tests", "orderbbb"],
        "user_metadata": {},
    }
    r_copy = http.post(f"{api_base}/api/assets/from-hash", json=payload, timeout=120)
    copy_body = r_copy.json()
    assert r_copy.status_code == 201, copy_body

    # 1) Default order (count_desc): 'orderbbb' should come before 'orderaaa'
    #    because it has higher usage (2 vs 1).
    r1 = http.get(api_base + "/api/tags", params={"prefix": "order", "include_zero": "false"}, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 200, b1
    names1 = [t["name"] for t in b1["tags"]]
    counts1 = {t["name"]: t["count"] for t in b1["tags"]}
    # Both must be present within the prefix subset
    assert "orderaaa" in names1 and "orderbbb" in names1
    # Usage of 'orderbbb' must be >= 'orderaaa'; in our setup it's 2 vs 1
    assert counts1["orderbbb"] >= counts1["orderaaa"]
    # And with count_desc, 'orderbbb' appears earlier than 'orderaaa'
    assert names1.index("orderbbb") < names1.index("orderaaa")

    # 2) name_asc: lexical order should flip the relative order
    r2 = http.get(
        api_base + "/api/tags",
        params={"prefix": "order", "include_zero": "false", "order": "name_asc"},
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200, b2
    names2 = [t["name"] for t in b2["tags"]]
    assert "orderaaa" in names2 and "orderbbb" in names2
    assert names2.index("orderaaa") < names2.index("orderbbb")

    # 3) invalid limit rejected (existing negative case retained)
    r3 = http.get(api_base + "/api/tags", params={"limit": "1001"}, timeout=120)
    b3 = r3.json()
    assert r3.status_code == 400
    assert b3["error"]["code"] == "INVALID_QUERY"


def test_tags_endpoints_invalid_bodies(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # Add with empty list
    r1 = http.post(f"{api_base}/api/assets/{aid}/tags", json={"tags": []}, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 400
    assert b1["error"]["code"] == "INVALID_BODY"

    # Remove with wrong type
    r2 = http.delete(f"{api_base}/api/assets/{aid}/tags", json={"tags": [123]}, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 400
    assert b2["error"]["code"] == "INVALID_BODY"

    # metadata_filter provided as JSON array should be rejected (must be object)
    r3 = http.get(
        api_base + "/api/assets",
        params={"metadata_filter": json.dumps([{"x": 1}])},
        timeout=120,
    )
    b3 = r3.json()
    assert r3.status_code == 400
    assert b3["error"]["code"] == "INVALID_QUERY"


def test_tags_prefix_treats_underscore_literal(
    http,
    api_base,
    asset_factory,
    make_asset_bytes,
):
    """'prefix' for /api/tags must treat '_' literally, not as a wildcard."""
    base = f"pref_{uuid.uuid4().hex[:6]}"
    tag_ok = f"{base}_ok"   # should match prefix=f"{base}_"
    tag_bad = f"{base}xok"  # must NOT match if '_' is escaped
    scope = f"tags-underscore-{uuid.uuid4().hex[:6]}"

    asset_factory("t1.bin", ["input", "unit-tests", scope, tag_ok], {}, make_asset_bytes("t1", 512))
    asset_factory("t2.bin", ["input", "unit-tests", scope, tag_bad], {}, make_asset_bytes("t2", 512))

    r = http.get(api_base + "/api/tags", params={"include_zero": "false", "prefix": f"{base}_"}, timeout=120)
    body = r.json()
    assert r.status_code == 200, body
    names = [t["name"] for t in body["tags"]]
    assert tag_ok in names, f"Expected {tag_ok} to be returned for prefix '{base}_'"
    assert tag_bad not in names, f"'{tag_bad}' must not match — '_' is not a wildcard"
    assert body["total"] == 1
