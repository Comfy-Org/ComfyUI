def test_asset_listing_is_available_for_missing_state_projection(http, api_base):
    response = http.get(f"{api_base}/api/assets")

    assert response.status_code == 200
    assert "assets" in response.json()
