def test_asset_routes_reject_invalid_record_ids(http, api_base):
    response = http.get(f"{api_base}/api/assets/not-a-uuid")

    assert response.status_code == 404
