"""Helper functions for assets integration tests."""
import time

import requests


def trigger_sync_seed_assets(session: requests.Session, base_url: str) -> None:
    """Force a synchronous sync/seed pass by calling the seed endpoint with wait=true.

    Retries on 409 (already running) until the previous scan finishes.
    """
    deadline = time.monotonic() + 60
    while True:
        r = session.post(
            base_url + "/api/assets/seed?wait=true",
            json={"roots": ["models", "input", "output"]},
            timeout=60,
        )
        if r.status_code != 409:
            assert r.status_code == 200, f"seed endpoint returned {r.status_code}: {r.text}"
            return
        if time.monotonic() > deadline:
            raise TimeoutError("seed endpoint stuck in 409 (already running)")
        time.sleep(0.25)


def get_asset_filename(asset_hash: str, extension: str) -> str:
    return asset_hash.removeprefix("blake3:") + extension


def assert_hash_fields_consistent(body: dict, expected_hash: str | None = None) -> None:
    """Assert hash and asset_hash invariants on an Asset response.

    Both must be present or both absent (so a regression that drops only one
    is caught). When present, they must equal each other and, if expected_hash
    is provided, must equal that value.
    """
    hash_present = "hash" in body
    asset_hash_present = "asset_hash" in body
    assert hash_present == asset_hash_present, (
        f"hash and asset_hash must both be present or both absent: "
        f"hash present={hash_present}, asset_hash present={asset_hash_present}"
    )
    if hash_present:
        h = body["hash"]
        ah = body["asset_hash"]
        assert h == ah, f"hash and asset_hash must match: hash={h!r}, asset_hash={ah!r}"
        if expected_hash is not None:
            assert h == expected_hash, (
                f"hash must equal expected: got {h!r}, expected {expected_hash!r}"
            )
