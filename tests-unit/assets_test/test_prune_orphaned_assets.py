import uuid
from pathlib import Path

import pytest
import requests
from helpers import get_asset_filename, trigger_sync_seed_assets


@pytest.fixture
def create_seed_file(comfy_tmp_base_dir: Path):
    """Create a file on disk that will become a seed asset after sync."""
    created: list[Path] = []

    def _create(root: str, scope: str, name: str | None = None, data: bytes = b"TEST") -> Path:
        name = name or f"seed_{uuid.uuid4().hex[:8]}.bin"
        path = comfy_tmp_base_dir / root / "unit-tests" / scope / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        created.append(path)
        return path

    yield _create

    for p in created:
        p.unlink(missing_ok=True)


@pytest.fixture
def find_asset(http: requests.Session, api_base: str):
    """Query API for assets matching scope and optional name."""
    def _find(scope: str, name: str | None = None) -> list[dict]:
        params = {"include_tags": f"unit-tests,{scope}"}
        if name:
            params["name_contains"] = name
        r = http.get(f"{api_base}/api/assets", params=params, timeout=120)
        assert r.status_code == 200
        assets = r.json().get("assets", [])
        if name:
            return [a for a in assets if a.get("name") == name]
        return assets

    return _find


@pytest.mark.parametrize("root", ["input", "output"])
def test_orphaned_seed_asset_is_pruned(
    root: str,
    create_seed_file,
    find_asset,
    http: requests.Session,
    api_base: str,
):
    """Seed asset with deleted file is removed; with file present, it survives."""
    scope = f"prune-{uuid.uuid4().hex[:6]}"
    fp = create_seed_file(root, scope)
    name = fp.name

    trigger_sync_seed_assets(http, api_base)
    assert find_asset(scope, name), "Seed asset should exist"

    fp.unlink()
    trigger_sync_seed_assets(http, api_base)
    assert not find_asset(scope, name), "Orphaned seed should be pruned"


def test_seed_asset_with_file_survives_prune(
    create_seed_file,
    find_asset,
    http: requests.Session,
    api_base: str,
):
    """Seed asset with file still on disk is NOT pruned."""
    scope = f"keep-{uuid.uuid4().hex[:6]}"
    fp = create_seed_file("input", scope)

    trigger_sync_seed_assets(http, api_base)
    trigger_sync_seed_assets(http, api_base)

    assert find_asset(scope, fp.name), "Seed with valid file should survive"


def test_hashed_asset_not_pruned_when_file_missing(
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
):
    """Hashed assets are never deleted by prune, even without file."""
    scope = f"hashed-{uuid.uuid4().hex[:6]}"
    data = make_asset_bytes("test", 2048)
    a = asset_factory("test.bin", ["input", "unit-tests", scope], {}, data)

    path = comfy_tmp_base_dir / "input" / "unit-tests" / scope / get_asset_filename(a["asset_hash"], ".bin")
    path.unlink()

    trigger_sync_seed_assets(http, api_base)

    r = http.get(f"{api_base}/api/assets/{a['id']}", timeout=120)
    assert r.status_code == 200, "Hashed asset should NOT be pruned"


def test_prune_across_multiple_roots(
    create_seed_file,
    find_asset,
    http: requests.Session,
    api_base: str,
):
    """Prune correctly handles assets across input and output roots."""
    scope = f"multi-{uuid.uuid4().hex[:6]}"
    input_fp = create_seed_file("input", scope, "input.bin")
    create_seed_file("output", scope, "output.bin")

    trigger_sync_seed_assets(http, api_base)
    assert len(find_asset(scope)) == 2

    input_fp.unlink()
    trigger_sync_seed_assets(http, api_base)

    remaining = find_asset(scope)
    assert len(remaining) == 1
    assert remaining[0]["name"] == "output.bin"


@pytest.mark.parametrize("dirname", ["100%_done", "my_folder_name", "has spaces"])
def test_special_chars_in_path_escaped_correctly(
    dirname: str,
    create_seed_file,
    find_asset,
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    """SQL LIKE wildcards (%, _) and spaces in paths don't cause false matches."""
    scope = f"special-{uuid.uuid4().hex[:6]}/{dirname}"
    fp = create_seed_file("input", scope)

    trigger_sync_seed_assets(http, api_base)
    trigger_sync_seed_assets(http, api_base)

    assert find_asset(scope.split("/")[0], fp.name), "Asset with special chars should survive"
