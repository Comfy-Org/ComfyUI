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
