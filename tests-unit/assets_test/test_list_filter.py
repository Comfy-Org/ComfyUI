import time
import uuid

import pytest
import requests
from helpers import assert_hash_fields_consistent


def test_list_assets_paging_and_sort(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    names = ["a1_u.safetensors", "a2_u.safetensors", "a3_u.safetensors"]
    for n in names:
        asset_factory(
            n,
            ["models", "checkpoints", "unit-tests", "paging"],
            {"epoch": 1},
            make_asset_bytes(n, size=2048),
        )

    # name ascending for stable order
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,paging", "sort": "name", "order": "asc", "limit": "2", "offset": "0"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200
    got1 = [a["name"] for a in b1["assets"]]
    assert got1 == sorted(names)[:2]
    assert b1["has_more"] is True
    # Populated assets in list responses must carry both `hash` and `asset_hash` consistently
    for asset in b1["assets"]:
        assert_hash_fields_consistent(asset)
        assert "hash" in asset, "populated asset must emit hash on list endpoint"

    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,paging", "sort": "name", "order": "asc", "limit": "2", "offset": "2"},
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200
    got2 = [a["name"] for a in b2["assets"]]
    assert got2 == sorted(names)[2:]
    assert b2["has_more"] is False


def test_list_assets_include_exclude_and_name_contains(http: requests.Session, api_base: str, asset_factory):
    a = asset_factory("inc_a.safetensors", ["models", "checkpoints", "unit-tests", "alpha"], {}, b"X" * 1024)
    b = asset_factory("inc_b.safetensors", ["models", "checkpoints", "unit-tests", "beta"], {}, b"Y" * 1024)

    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,alpha", "exclude_tags": "beta", "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200
    names = [x["name"] for x in body["assets"]]
    assert a["name"] in names
    assert b["name"] not in names

    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests", "name_contains": "inc_"},
        timeout=120,
    )
    body2 = r2.json()
    assert r2.status_code == 200
    names2 = [x["name"] for x in body2["assets"]]
    assert a["name"] in names2
    assert b["name"] in names2

    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "non-existing-tag"},
        timeout=120,
    )
    body3 = r2.json()
    assert r2.status_code == 200
    assert not body3["assets"]


def test_list_assets_sort_by_size_both_orders(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-size"]
    n1, n2, n3 = "sz1.safetensors", "sz2.safetensors", "sz3.safetensors"
    asset_factory(n1, t, {}, make_asset_bytes(n1, 1024))
    asset_factory(n2, t, {}, make_asset_bytes(n2, 2048))
    asset_factory(n3, t, {}, make_asset_bytes(n3, 3072))

    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-size", "sort": "size", "order": "asc"},
        timeout=120,
    )
    b1 = r1.json()
    names = [a["name"] for a in b1["assets"]]
    assert names[:3] == [n1, n2, n3]

    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-size", "sort": "size", "order": "desc"},
        timeout=120,
    )
    b2 = r2.json()
    names2 = [a["name"] for a in b2["assets"]]
    assert names2[:3] == [n3, n2, n1]



def test_list_assets_sort_by_updated_at_desc(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-upd"]
    a1 = asset_factory("upd_a.safetensors", t, {}, make_asset_bytes("upd_a", 1200))
    a2 = asset_factory("upd_b.safetensors", t, {}, make_asset_bytes("upd_b", 1200))

    # Rename the second asset to bump updated_at
    rp = http.put(f"{api_base}/api/assets/{a2['id']}", json={"name": "upd_b_renamed.safetensors"}, timeout=120)
    upd = rp.json()
    assert rp.status_code == 200, upd

    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-upd", "sort": "updated_at", "order": "desc"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200
    names = [x["name"] for x in body["assets"]]
    assert names[0] == "upd_b_renamed.safetensors"
    assert a1["name"] in names



def test_list_assets_sort_by_last_access_time_desc(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-access"]
    asset_factory("acc_a.safetensors", t, {}, make_asset_bytes("acc_a", 1100))
    time.sleep(0.02)
    a2 = asset_factory("acc_b.safetensors", t, {}, make_asset_bytes("acc_b", 1100))

    # Touch last_access_time of b by downloading its content
    time.sleep(0.02)
    dl = http.get(f"{api_base}/api/assets/{a2['id']}/content", timeout=120)
    assert dl.status_code == 200
    dl.content

    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-access", "sort": "last_access_time", "order": "desc"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200
    names = [x["name"] for x in body["assets"]]
    assert names[0] == a2["name"]


def test_list_assets_include_tags_variants_and_case(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-include"]
    a = asset_factory("incvar_alpha.safetensors", [*t, "alpha"], {}, make_asset_bytes("iva"))
    asset_factory("incvar_beta.safetensors", [*t, "beta"], {}, make_asset_bytes("ivb"))

    # CSV + case-insensitive
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "UNIT-TESTS,LF-INCLUDE,alpha"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200
    names1 = [x["name"] for x in b1["assets"]]
    assert a["name"] in names1
    assert not any("beta" in x for x in names1)

    # Repeated query params for include_tags
    params_multi = [
        ("include_tags", "unit-tests"),
        ("include_tags", "lf-include"),
        ("include_tags", "alpha"),
    ]
    r2 = http.get(api_base + "/api/assets", params=params_multi, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200
    names2 = [x["name"] for x in b2["assets"]]
    assert a["name"] in names2
    assert not any("beta" in x for x in names2)

    # Duplicates and spaces in CSV
    r3 = http.get(
        api_base + "/api/assets",
        params={"include_tags": " unit-tests , lf-include , alpha , alpha "},
        timeout=120,
    )
    b3 = r3.json()
    assert r3.status_code == 200
    names3 = [x["name"] for x in b3["assets"]]
    assert a["name"] in names3


def test_list_assets_exclude_tags_dedup_and_case(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-exclude"]
    a = asset_factory("ex_a_alpha.safetensors", [*t, "alpha"], {}, make_asset_bytes("exa", 900))
    asset_factory("ex_b_beta.safetensors", [*t, "beta"], {}, make_asset_bytes("exb", 900))

    # Exclude uppercase should work
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-exclude", "exclude_tags": "BETA"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200
    names1 = [x["name"] for x in b1["assets"]]
    assert a["name"] in names1
    # Repeated excludes with duplicates
    params_multi = [
        ("include_tags", "unit-tests"),
        ("include_tags", "lf-exclude"),
        ("exclude_tags", "beta"),
        ("exclude_tags", "beta"),
    ]
    r2 = http.get(api_base + "/api/assets", params=params_multi, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200
    names2 = [x["name"] for x in b2["assets"]]
    assert all("beta" not in x for x in names2)


def test_list_assets_name_contains_case_and_specials(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-name"]
    a1 = asset_factory("CaseMix.SAFE", t, {}, make_asset_bytes("cm", 800))
    a2 = asset_factory("case-other.safetensors", t, {}, make_asset_bytes("co", 800))

    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-name", "name_contains": "casemix"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200
    names1 = [x["name"] for x in b1["assets"]]
    assert a1["name"] in names1

    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-name", "name_contains": ".SAFE"},
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200
    names2 = [x["name"] for x in b2["assets"]]
    assert a1["name"] in names2

    r3 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-name", "name_contains": "case-"},
        timeout=120,
    )
    b3 = r3.json()
    assert r3.status_code == 200
    names3 = [x["name"] for x in b3["assets"]]
    assert a2["name"] in names3


def test_list_assets_offset_beyond_total_and_limit_boundary(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "lf-pagelimits"]
    asset_factory("pl1.safetensors", t, {}, make_asset_bytes("pl1", 600))
    asset_factory("pl2.safetensors", t, {}, make_asset_bytes("pl2", 600))
    asset_factory("pl3.safetensors", t, {}, make_asset_bytes("pl3", 600))

    # Offset far beyond total
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-pagelimits", "limit": "2", "offset": "10"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200
    assert not b1["assets"]
    assert b1["has_more"] is False

    # Boundary large limit (<=500 is valid)
    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-pagelimits", "limit": "500"},
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200
    assert len(b2["assets"]) == 3
    assert b2["has_more"] is False


@pytest.mark.parametrize(
    "params,error_code",
    [
        ({"offset": "-1"}, "INVALID_QUERY"),
        ({"limit": "abc"}, "INVALID_QUERY"),
        ({"limit": "0"}, "INVALID_QUERY"),
        ({"metadata_filter": "{not json"}, "INVALID_QUERY"),
    ],
    ids=["negative_offset", "non_int_limit", "zero_limit", "invalid_metadata_json"],
)
def test_list_assets_invalid_query_rejected(http: requests.Session, api_base: str, params, error_code):
    r = http.get(api_base + "/api/assets", params=params, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == error_code


def test_list_assets_name_contains_literal_underscore(
    http,
    api_base,
    asset_factory,
    make_asset_bytes,
):
    """'name_contains' must treat '_' literally, not as a SQL wildcard.
    We create:
      - foo_bar.safetensors      (should match)
      - fooxbar.safetensors      (must NOT match if '_' is escaped)
      - foobar.safetensors       (must NOT match)
    """
    scope = f"lf-underscore-{uuid.uuid4().hex[:6]}"
    tags = ["models", "checkpoints", "unit-tests", scope]

    a = asset_factory("foo_bar.safetensors", tags, {}, make_asset_bytes("a", 700))
    b = asset_factory("fooxbar.safetensors", tags, {}, make_asset_bytes("b", 700))
    c = asset_factory("foobar.safetensors", tags, {}, make_asset_bytes("c", 700))

    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "name_contains": "foo_bar"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [x["name"] for x in body["assets"]]
    assert a["name"] in names, f"Expected literal underscore match to include {a['name']}"
    assert b["name"] not in names, "Underscore must be escaped — should not match 'fooxbar'"
    assert c["name"] not in names, "Underscore must be escaped — should not match 'foobar'"
    assert body["total"] == 1
