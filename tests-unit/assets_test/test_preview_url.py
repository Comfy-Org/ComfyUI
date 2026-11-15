def test_record_detail_exposes_a_stable_id(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("preview.png", ["output", "unit-tests"], {}, make_asset_bytes("preview"))

    detail = http.get(f"{api_base}/api/assets/{record['id']}").json()

    assert detail["id"] == record["id"]
