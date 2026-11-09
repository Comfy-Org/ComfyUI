import json

import pytest


@pytest.mark.hashing_on
def test_hash_on_upload_mints_a_record_per_upload(asset_factory, make_asset_bytes):
    data = make_asset_bytes("hash-on-dedup")

    first = asset_factory("dedup.png", ["output", "unit-tests"], {}, data)
    second = asset_factory("dedup.png", ["output", "unit-tests"], {}, data)

    assert second["id"] != first["id"], "every upload is its own delivery record"
    assert second["created_new"] is True


def test_repeat_upload_of_identical_bytes_returns_201(http, api_base, make_asset_bytes):
    data = make_asset_bytes("route-201-on-repeat")
    form = {
        "tags": json.dumps(["output", "unit-tests"]),
        "name": "repeat.png",
        "user_metadata": json.dumps({}),
    }
    created: list[str] = []
    try:
        for _ in range(2):
            response = http.post(
                f"{api_base}/api/assets",
                files={"file": ("repeat.png", data, "application/octet-stream")},
                data=form,
                timeout=120,
            )
            body = response.json()
            created.append(body["id"])
            assert response.status_code == 201, body
            assert body["created_new"] is True

        assert created[0] != created[1], "every upload is its own delivery record"
    finally:
        for asset_id in created:
            http.delete(f"{api_base}/api/assets/{asset_id}", timeout=30)


def test_upload_returns_a_record_identifier(http, api_base, asset_factory, make_asset_bytes):
    record = asset_factory("upload.png", ["output", "unit-tests"], {}, make_asset_bytes("upload"))

    assert record["id"]
    assert http.get(f"{api_base}/api/assets/{record['id']}").status_code == 200
