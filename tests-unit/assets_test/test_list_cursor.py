def test_record_list_returns_created_record(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("listed.png", ["output", "unit-tests"], {}, make_asset_bytes("listed"))

    response = http.get(f"{api_base}/api/assets", params={"limit": 50})

    assert response.status_code == 200
    assert record["id"] in {asset["id"] for asset in response.json()["assets"]}
