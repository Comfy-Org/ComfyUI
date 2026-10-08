def test_tag_listing_includes_record_tag(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("tagged.png", ["output", "unit-tests", "visible"], {}, make_asset_bytes("tagged"))

    response = http.get(f"{api_base}/api/assets", params={"include_tags": "visible"})

    assert response.status_code == 200
    assert [asset["id"] for asset in response.json()["assets"]] == [record["id"]]


def test_tag_refine_rejects_metadata_filter(http, api_base):
    response = http.get(
        f"{api_base}/api/assets/tags/refine",
        params={"metadata_filter": '{"k":"v"}'},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "UNSUPPORTED_PARAM",
            "message": "metadata_filter is no longer supported",
            "details": {},
        }
    }


def _expected_system_tag_envelope() -> dict:
    return {
        "error": {
            "code": "SYSTEM_TAG_FORBIDDEN",
            "message": "Tag 'missing' is system-managed and cannot be modified via the API",
            "details": {"tag": "missing"},
        }
    }


def test_add_system_tag_returns_the_error_envelope(
    http, api_base, asset_factory, make_asset_bytes
):
    record = asset_factory("sys-add.png", ["output"], {}, make_asset_bytes("sys-add"))

    response = http.post(
        f"{api_base}/api/assets/{record['id']}/tags",
        json={"tags": ["missing"]},
        timeout=120,
    )

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json"), (
        "a rejection clients must parse cannot be text/plain while every neighbouring "
        "error on this route is the JSON envelope"
    )
    assert response.json() == _expected_system_tag_envelope()


def test_delete_system_tag_returns_the_error_envelope(
    http, api_base, asset_factory, make_asset_bytes
):
    record = asset_factory("sys-del.png", ["output"], {}, make_asset_bytes("sys-del"))

    response = http.delete(
        f"{api_base}/api/assets/{record['id']}/tags",
        json={"tags": ["missing"]},
        timeout=120,
    )

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.json() == _expected_system_tag_envelope()
