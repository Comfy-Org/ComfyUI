import time as _time

import pytest


@pytest.mark.hashing_on
def test_missing_content_remains_in_list_after_rm_with_missing_tag(
    http, api_base, asset_factory, comfy_tmp_base_dir, make_asset_bytes
):
    record = asset_factory(
        "missing.png", ["output", "unit-tests"], {}, make_asset_bytes("missing")
    )
    next((comfy_tmp_base_dir / "output").glob("*.png")).unlink()

    response = None
    for _attempt in range(5):
        response = http.post(
            f"{api_base}/api/assets/seed?wait=true", json={"roots": ["output"]}
        )
        if response.status_code != 409:
            break
        _time.sleep(1.0)
    assert response is not None
    assert response.status_code == 200
    assets = http.get(f"{api_base}/api/assets", timeout=120).json()["assets"]
    listed = {asset["id"]: asset for asset in assets}
    assert record["id"] in listed
    assert "missing" in listed[record["id"]]["tags"]


def test_record_crud_hard_deletes_the_record(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("crud.png", ["output", "unit-tests"], {}, make_asset_bytes("crud"))

    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 200
    assert http.delete(f"{api_base}/api/assets/{record['id']}").status_code == 204
    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 404
