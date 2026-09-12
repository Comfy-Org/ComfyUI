def test_record_list_exposes_no_legacy_reference_shape(http, api_base):
    response = http.get(f"{api_base}/api/assets")

    assert response.status_code == 200
    assert "asset_references" not in response.json()
