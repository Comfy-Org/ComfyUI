def test_record_content_read_serves_live_content(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("download.png", ["output", "unit-tests"], {}, make_asset_bytes("download"))

    response = http.get(f"{api_base}/api/assets/{record['id']}/content")

    assert response.status_code == 200
