def test_record_detail_keeps_record_metadata(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("metadata.png", ["output", "unit-tests"], {"source": "test"}, make_asset_bytes("metadata"))

    response = http.get(f"{api_base}/api/assets/{record['id']}")

    assert response.status_code == 200
    assert response.json()["user_metadata"] == {"source": "test"}
