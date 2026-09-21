def test_seed_endpoint_is_available_for_content_reconciliation(http, api_base):
    response = http.post(f"{api_base}/api/assets/seed?wait=true", json={"roots": []})

    assert response.status_code in (200, 400)
