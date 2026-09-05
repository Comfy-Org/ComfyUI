import time
import uuid
import warnings

import pytest
import requests
from helpers import assert_hash_fields_consistent, get_asset_filename

from app.assets.api import routes as assets_routes
from app.assets.api import schemas_in


def test_list_assets_paging_and_sort(http: requests.Session, api_base: str, asset_factory, make_asset_bytes):
    names = ["a1_u.safetensors", "a2_u.safetensors", "a3_u.safetensors"]
    for n in names:
        asset_factory(
            n,
            ["models", "model_type:checkpoints", "unit-tests", "paging"],
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
    a = asset_factory("inc_a.safetensors", ["models", "model_type:checkpoints", "unit-tests", "alpha"], {}, b"X" * 1024)
    b = asset_factory("inc_b.safetensors", ["models", "model_type:checkpoints", "unit-tests", "beta"], {}, b"Y" * 1024)

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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-size"]
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-upd"]
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-access"]
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-include"]
    a = asset_factory("incvar_alpha.safetensors", [*t, "alpha"], {}, make_asset_bytes("iva"))
    asset_factory("incvar_beta.safetensors", [*t, "beta"], {}, make_asset_bytes("ivb"))

    # CSV tag filters are whitespace-trimmed and case-sensitive.
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-include,alpha"},
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-exclude"]
    a = asset_factory("ex_a_alpha.safetensors", [*t, "alpha"], {}, make_asset_bytes("exa", 900))
    asset_factory("ex_b_beta.safetensors", [*t, "beta"], {}, make_asset_bytes("exb", 900))

    # Exclude filters are case-sensitive.
    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,lf-exclude", "exclude_tags": "beta"},
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-name"]
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
    t = ["models", "model_type:checkpoints", "unit-tests", "lf-pagelimits"]
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


def test_list_assets_display_name_emitted(http, api_base, asset_factory, make_asset_bytes):
    """`display_name` is emitted for every populated asset in list responses,
    derived from the storage path."""
    scope = f"lf-dispname-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    asset_factory("dn_a.safetensors", tags, {}, make_asset_bytes("dn_a", 700))
    asset_factory("dn_b.safetensors", tags, {}, make_asset_bytes("dn_b", 700))

    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["assets"], "expected at least one asset"
    for asset in body["assets"]:
        assert "display_name" in asset, "populated asset must emit display_name"
        expected = "checkpoints/" + get_asset_filename(asset["asset_hash"], ".safetensors")
        assert asset["display_name"] == expected


def test_list_assets_hash_filter_exact_match(http, api_base, asset_factory, make_asset_bytes):
    """`hash` filters to assets whose content hash matches exactly."""
    scope = f"lf-hash-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    a = asset_factory("hf_a.safetensors", tags, {}, make_asset_bytes("hf_a", 1024))
    b = asset_factory("hf_b.safetensors", tags, {}, make_asset_bytes("hf_b", 2048))

    target = a["hash"]
    assert target and a["hash"] != b["hash"], "fixtures must have distinct content hashes"

    r = http.get(
        api_base + "/api/assets",
        params={"hash": target, "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [x["name"] for x in body["assets"]]
    assert names == [a["name"]]
    assert body["total"] == 1


def test_list_assets_hash_filter_no_match(http, api_base, asset_factory, make_asset_bytes):
    """A well-formed but unknown hash returns an empty page (200)."""
    scope = f"lf-hash-none-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    asset_factory("hn_a.safetensors", tags, {}, make_asset_bytes("hn_a", 800))

    unknown = "blake3:" + ("0" * 64)
    r = http.get(
        api_base + "/api/assets",
        params={"hash": unknown, "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["assets"] == []
    assert body["total"] == 0


def test_list_assets_hash_filter_normalizes_case_and_whitespace(
    http, api_base, asset_factory, make_asset_bytes
):
    """An upper-cased, space-padded `hash` still matches the stored hash."""
    scope = f"lf-hashnorm-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    a = asset_factory("hnorm_a.safetensors", tags, {}, make_asset_bytes("hnorm_a", 1024))

    target = a["hash"]
    assert target == target.lower(), "stored hash is expected to be lowercase"
    messy = f"  {target.upper()}  "

    r = http.get(
        api_base + "/api/assets",
        params={"hash": messy, "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [x["name"] for x in body["assets"]]
    assert names == [a["name"]]
    assert body["total"] == 1


def test_list_assets_hash_filter_empty_returns_empty_page(
    http, api_base, asset_factory, make_asset_bytes
):
    """An empty `hash` matches nothing and returns an empty page, rather than
    disabling the filter."""
    scope = f"lf-hashempty-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    asset_factory("he_a.safetensors", tags, {}, make_asset_bytes("he_a", 800))

    r = http.get(
        api_base + "/api/assets",
        params={"hash": "", "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["assets"] == []
    assert body["total"] == 0


def test_list_assets_include_public_accepted(http, api_base, asset_factory, make_asset_bytes):
    """`include_public` is accepted and does not change which assets come back."""
    scope = f"lf-incpub-{uuid.uuid4().hex[:6]}"
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]
    a = asset_factory("ip_a.safetensors", tags, {}, make_asset_bytes("ip_a", 900))

    for value in ("false", "true"):
        r = http.get(
            api_base + "/api/assets",
            params={"include_tags": f"unit-tests,{scope}", "include_public": value, "limit": "50"},
            timeout=120,
        )
        body = r.json()
        assert r.status_code == 200, body
        names = [x["name"] for x in body["assets"]]
        assert a["name"] in names, f"asset must be returned (include_public={value})"


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
    tags = ["models", "model_type:checkpoints", "unit-tests", scope]

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


def test_list_assets_tags_any_alone(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-any-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    a = asset_factory("any_a.safetensors", [*t, f"{scope}-alpha"], {}, make_asset_bytes("any_a"))
    b = asset_factory("any_b.safetensors", [*t, f"{scope}-beta"], {}, make_asset_bytes("any_b"))
    c = asset_factory("any_c.safetensors", [*t, f"{scope}-gamma"], {}, make_asset_bytes("any_c"))

    r = http.get(
        api_base + "/api/assets",
        params={"tags_any": f"{scope}-alpha,{scope}-beta", "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [x["name"] for x in body["assets"]]
    assert a["name"] in names
    assert b["name"] in names
    assert c["name"] not in names


def test_list_assets_tags_any_with_tags_all(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-anyall-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    x = asset_factory("aa_x.safetensors", [*t, alpha], {}, make_asset_bytes("aa_x"))
    y = asset_factory("aa_y.safetensors", [*t, beta], {}, make_asset_bytes("aa_y"))
    w = asset_factory("aa_w.safetensors", t, {}, make_asset_bytes("aa_w"))
    d = asset_factory(
        "aa_d.safetensors",
        ["models", "model_type:checkpoints", "unit-tests", f"{scope}-other", alpha],
        {},
        make_asset_bytes("aa_d"),
    )

    r = http.get(
        api_base + "/api/assets",
        params={"tags_all": f"unit-tests,{scope}", "tags_any": f"{alpha},{beta}", "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [a["name"] for a in body["assets"]]
    assert x["name"] in names
    assert y["name"] in names
    assert w["name"] not in names, "asset matching tags_all but not tags_any must be excluded"
    assert d["name"] not in names, "asset matching tags_any but not tags_all must be excluded"


def test_list_assets_tags_none_wins_over_tags_any(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-nonewins-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    x = asset_factory("nw_x.safetensors", [*t, alpha], {}, make_asset_bytes("nw_x"))
    y = asset_factory("nw_y.safetensors", [*t, alpha, beta], {}, make_asset_bytes("nw_y"))

    r = http.get(
        api_base + "/api/assets",
        params={"tags_any": alpha, "tags_none": beta, "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [a["name"] for a in body["assets"]]
    assert x["name"] in names
    assert y["name"] not in names, "tags_none must exclude an asset even when it matches tags_any"


def test_list_assets_empty_tag_filter_lists_behave_as_absent(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-empty-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    a = asset_factory("em_a.safetensors", t, {}, make_asset_bytes("em_a"))
    b = asset_factory("em_b.safetensors", t, {}, make_asset_bytes("em_b"))
    expected = {a["name"], b["name"]}

    # Empty new-name lists impose no constraint.
    r1 = http.get(
        api_base + "/api/assets",
        params={"tags_all": f"unit-tests,{scope}", "tags_any": "", "tags_none": ""},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200, b1
    assert {x["name"] for x in b1["assets"]} == expected

    # An empty new-name param alongside old names must not trigger validation.
    r2 = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "tags_any": ""},
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200, b2
    assert {x["name"] for x in b2["assets"]} == expected

    # An empty tags_all next to include_tags is not a mixed-spelling conflict.
    r3 = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "tags_all": ""},
        timeout=120,
    )
    b3 = r3.json()
    assert r3.status_code == 200, b3
    assert {x["name"] for x in b3["assets"]} == expected


def test_list_assets_old_names_match_new_names(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-alias-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    asset_factory("al_a.safetensors", [*t, alpha], {}, make_asset_bytes("al_a"))
    asset_factory("al_b.safetensors", [*t, beta], {}, make_asset_bytes("al_b"))

    def names_for(params: dict) -> tuple[list, int]:
        r = http.get(api_base + "/api/assets", params={**params, "sort": "name", "order": "asc"}, timeout=120)
        body = r.json()
        assert r.status_code == 200, body
        return [x["name"] for x in body["assets"]], body["total"]

    # include_tags ≡ tags_all
    old_names, old_total = names_for({"include_tags": f"unit-tests,{scope}"})
    new_names, new_total = names_for({"tags_all": f"unit-tests,{scope}"})
    assert old_names == new_names
    assert old_total == new_total

    # exclude_tags ≡ tags_none (and old/new spellings mix across slots)
    old_names, old_total = names_for({"include_tags": f"unit-tests,{scope}", "exclude_tags": alpha})
    new_names, new_total = names_for({"tags_all": f"unit-tests,{scope}", "tags_none": alpha})
    mixed_names, mixed_total = names_for({"include_tags": f"unit-tests,{scope}", "tags_none": alpha})
    assert old_names == new_names == mixed_names == ["al_b.safetensors"]
    assert old_total == new_total == mixed_total == 1


@pytest.mark.parametrize(
    "params,expected_parameters",
    [
        ({"include_tags": "mx-x", "tags_all": "mx-y"}, ["include_tags", "tags_all"]),
        ({"exclude_tags": "mx-x", "tags_none": "mx-y"}, ["exclude_tags", "tags_none"]),
    ],
    ids=["include_tags_with_tags_all", "exclude_tags_with_tags_none"],
)
def test_list_assets_mixed_tag_spellings_rejected(http, api_base, params, expected_parameters):
    r = http.get(api_base + "/api/assets", params=params, timeout=120)
    body = r.json()
    assert r.status_code == 400, body
    assert body["error"]["code"] == "INVALID_TAG_FILTER"
    assert body["error"]["details"]["parameters"] == expected_parameters


@pytest.mark.parametrize(
    "params,conflicting,parameters",
    [
        (
            {"tags_all": "cf-x", "tags_none": "cf-x"},
            ["cf-x"],
            ["tags_all", "tags_none"],
        ),
        (
            {"include_tags": "cf-x", "tags_none": "cf-x"},
            ["cf-x"],
            ["include_tags", "tags_none"],
        ),
        (
            {"tags_all": "cf-a,cf-b", "tags_none": "cf-b,cf-c"},
            ["cf-b"],
            ["tags_all", "tags_none"],
        ),
    ],
    ids=["new_names", "include_tags_remapped", "partial_overlap"],
)
def test_list_assets_all_none_conflict_rejected(http, api_base, params, conflicting, parameters):
    r = http.get(api_base + "/api/assets", params=params, timeout=120)
    body = r.json()
    assert r.status_code == 400, body
    assert body["error"]["code"] == "INVALID_TAG_FILTER"
    assert body["error"]["details"]["conflicting_tags"] == conflicting
    assert body["error"]["details"]["parameters"] == parameters


def test_list_assets_any_none_overlap_accepted(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-deadterm-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    x = asset_factory("dt_x.safetensors", [*t, alpha], {}, make_asset_bytes("dt_x"))
    y = asset_factory("dt_y.safetensors", [*t, beta], {}, make_asset_bytes("dt_y"))

    # alpha is a dead term (in both tags_any and tags_none) but the query is valid.
    r = http.get(
        api_base + "/api/assets",
        params={"tags_any": f"{alpha},{beta}", "tags_none": alpha, "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = [a["name"] for a in body["assets"]]
    assert y["name"] in names
    assert x["name"] not in names


def test_list_assets_legacy_include_exclude_conflict_still_200(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-legacy-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    asset_factory("lg_a.safetensors", t, {}, make_asset_bytes("lg_a"))

    # Old names only: the self-contradictory query stays an empty 200, never a 400.
    r = http.get(
        api_base + "/api/assets",
        params={"include_tags": scope, "exclude_tags": scope},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["assets"] == []


def test_tags_refine_new_tag_filters(http, api_base, asset_factory, make_asset_bytes):
    scope = f"rf-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    asset_factory("rf_a.safetensors", [*t, alpha], {}, make_asset_bytes("rf_a"))
    asset_factory("rf_b.safetensors", [*t, beta], {}, make_asset_bytes("rf_b"))

    r = http.get(
        api_base + "/api/assets/tags/refine",
        params={"tags_any": f"{alpha},{beta}", "tags_none": alpha},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    counts = body["tag_counts"]
    assert counts.get(beta) == 1
    assert alpha not in counts

    r2 = http.get(
        api_base + "/api/assets/tags/refine",
        params={"tags_all": "rf-x", "tags_none": "rf-x"},
        timeout=120,
    )
    body2 = r2.json()
    assert r2.status_code == 400, body2
    assert body2["error"]["code"] == "INVALID_TAG_FILTER"
    assert body2["error"]["details"]["conflicting_tags"] == ["rf-x"]


def test_list_assets_cross_slot_old_new_combinations(http, api_base, asset_factory, make_asset_bytes):
    """Old and new spellings of *different* slots combine freely; only
    same-slot mixing is rejected."""
    scope = f"lf-cross-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    a = asset_factory("cs_a.safetensors", [*t, alpha], {}, make_asset_bytes("cs_a"))
    b = asset_factory("cs_b.safetensors", [*t, beta], {}, make_asset_bytes("cs_b"))

    def names_for(params: dict) -> set:
        r = http.get(api_base + "/api/assets", params=params, timeout=120)
        body = r.json()
        assert r.status_code == 200, body
        return {x["name"] for x in body["assets"]}

    assert names_for(
        {"include_tags": f"unit-tests,{scope}", "tags_any": alpha}
    ) == {a["name"]}
    assert names_for(
        {"tags_all": f"unit-tests,{scope}", "exclude_tags": alpha}
    ) == {b["name"]}
    assert names_for(
        {"tags_any": f"{alpha},{beta}", "exclude_tags": alpha}
    ) == {b["name"]}


def test_list_assets_repeated_query_keys_concatenate(http, api_base, asset_factory, make_asset_bytes):
    """Repeated occurrences of a tag param concatenate before the CSV split
    (Core-local behavior, not a cross-platform guarantee)."""
    scope = f"lf-repeat-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha, beta = f"{scope}-alpha", f"{scope}-beta"
    a = asset_factory("rp_a.safetensors", [*t, alpha], {}, make_asset_bytes("rp_a"))
    b = asset_factory("rp_b.safetensors", [*t, beta], {}, make_asset_bytes("rp_b"))

    # requests encodes a list value as repeated keys: tags_any=<alpha>&tags_any=<beta>
    r = http.get(
        api_base + "/api/assets",
        params={"tags_any": [alpha, beta], "limit": "50"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    names = {x["name"] for x in body["assets"]}
    assert {a["name"], b["name"]} <= names


def test_list_assets_tags_any_cursor_pagination_consistent(http, api_base, asset_factory, make_asset_bytes):
    scope = f"lf-anypage-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    alpha = f"{scope}-alpha"
    expected = set()
    for i in range(3):
        made = asset_factory(f"pg_{i}.safetensors", [*t, alpha], {}, make_asset_bytes(f"pg_{i}"))
        expected.add(made["name"])

    r1 = http.get(
        api_base + "/api/assets",
        params={"tags_any": alpha, "limit": "2", "sort": "name", "order": "asc"},
        timeout=120,
    )
    b1 = r1.json()
    assert r1.status_code == 200, b1
    assert b1["total"] == 3
    assert b1["has_more"] is True
    assert b1.get("next_cursor"), "expected a keyset cursor on the first page"

    r2 = http.get(
        api_base + "/api/assets",
        params={
            "tags_any": alpha,
            "limit": "2",
            "sort": "name",
            "order": "asc",
            "after": b1["next_cursor"],
        },
        timeout=120,
    )
    b2 = r2.json()
    assert r2.status_code == 200, b2
    assert b2["has_more"] is False

    page1 = {x["name"] for x in b1["assets"]}
    page2 = {x["name"] for x in b2["assets"]}
    assert not page1 & page2, "cursor pages must not overlap"
    assert page1 | page2 == expected


def test_tags_refine_mixed_spellings_rejected_and_legacy_conflict_kept(http, api_base):
    r = http.get(
        api_base + "/api/assets/tags/refine",
        params={"include_tags": "rfmx-x", "tags_all": "rfmx-y"},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 400, body
    assert body["error"]["code"] == "INVALID_TAG_FILTER"
    assert body["error"]["details"]["parameters"] == ["include_tags", "tags_all"]

    # Old names only: the refine route keeps legacy behaviour too.
    r2 = http.get(
        api_base + "/api/assets/tags/refine",
        params={"include_tags": "rfmx-z", "exclude_tags": "rfmx-z"},
        timeout=120,
    )
    body2 = r2.json()
    assert r2.status_code == 200, body2
    assert body2["tag_counts"] == {}


def test_list_assets_tag_values_case_sensitive(http, api_base, asset_factory, make_asset_bytes):
    """Case-distinct tags are distinct; the all/none conflict check is byte-exact."""
    scope = f"lf-case-{uuid.uuid4().hex[:6]}"
    t = ["models", "model_type:checkpoints", "unit-tests", scope]
    upper, lower = f"{scope}-ALPHA", f"{scope}-alpha"
    a = asset_factory("cx_a.safetensors", [*t, upper], {}, make_asset_bytes("cx_a"))
    b = asset_factory("cx_b.safetensors", [*t, lower], {}, make_asset_bytes("cx_b"))

    def names_for(params: dict) -> set:
        r = http.get(api_base + "/api/assets", params=params, timeout=120)
        body = r.json()
        assert r.status_code == 200, body
        return {x["name"] for x in body["assets"]}

    assert names_for({"tags_all": f"unit-tests,{scope},{upper}"}) == {a["name"]}
    assert names_for({"tags_any": lower, "limit": "50"}) == {b["name"]}
    # Case-distinct all/none pair is NOT a conflict — byte-exact comparison.
    assert names_for({"tags_all": f"unit-tests,{scope},{upper}", "tags_none": lower}) == {a["name"]}


def test_tag_list_cap_applies_to_all_spellings(http, api_base):
    """The cap covers the legacy spellings too."""
    big = ",".join(f"cap-{i}" for i in range(101))
    for param in ("tags_any", "include_tags"):
        r = http.get(api_base + "/api/assets", params={param: big}, timeout=120)
        body = r.json()
        assert r.status_code == 400, body
        assert body["error"]["code"] == "INVALID_TAG_FILTER"
        assert body["error"]["details"]["parameter"] == param
        assert body["error"]["details"]["max"] == 100

    exact = ",".join(f"cap-{i}" for i in range(100))
    r = http.get(api_base + "/api/assets", params={"tags_any": exact}, timeout=120)
    assert r.status_code == 200, r.json()

    # The cap counts normalized (deduped) tags, not raw CSV items.
    dups = ",".join("cap-dup" for _ in range(150))
    r = http.get(api_base + "/api/assets", params={"tags_any": dups}, timeout=120)
    assert r.status_code == 200, r.json()


def test_resolve_tag_filters_no_deprecation_warning():
    """The deprecated-field warning is for API clients; the server's own remap
    shim must not fire it on every request."""
    for q in (
        schemas_in.ListAssetsQuery(tags_all="a", tags_none="b"),
        schemas_in.TagsRefineQuery(tags_any="c"),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assets_routes._resolve_tag_filters(q)


def test_tag_filter_alias_fields_marked_deprecated():
    for model in (schemas_in.ListAssetsQuery, schemas_in.TagsRefineQuery):
        props = model.model_json_schema()["properties"]
        for field in ("include_tags", "exclude_tags"):
            assert props[field].get("deprecated") is True, (model.__name__, field)
        for field in ("tags_all", "tags_any", "tags_none"):
            assert "deprecated" not in props[field], (model.__name__, field)
