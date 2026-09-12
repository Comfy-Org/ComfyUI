
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference, Base
from app.assets.database.queries.asset_reference import (
    mark_references_missing_outside_prefixes,
)
from app.assets.scanner import (
    collect_paths_for_roots,
    get_owned_prefixes,
    get_temp_prefixes,
    sync_prefixes_with_filesystem,
)
from app.assets.services.file_utils import get_mtime_ns


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override parent autouse fixture - temp asset tests don't need server cleanup."""
    yield


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def comfy_dirs():
    with tempfile.TemporaryDirectory() as base:
        dirs = {
            name: Path(base) / name
            for name in ("models", "input", "output", "temp", "elsewhere")
        }
        for d in dirs.values():
            d.mkdir()
        with (
            patch("folder_paths.get_input_directory", return_value=str(dirs["input"])),
            patch("folder_paths.get_output_directory", return_value=str(dirs["output"])),
            patch("folder_paths.get_temp_directory", return_value=str(dirs["temp"])),
            patch(
                "app.assets.scanner.get_comfy_models_folders",
                return_value=[("checkpoints", [str(dirs["models"])], set())],
            ),
        ):
            yield dirs


def _write(directory: Path, name: str) -> str:
    p = directory / name
    p.write_bytes(b"\x00" * 100)
    return str(p)


def _register(
    session: Session,
    file_path: str,
    ref_id: str,
    *,
    mtime_ns: int,
    asset_hash: str | None = "",
) -> None:
    if asset_hash == "":
        asset_hash = f"blake3:{ref_id}"
    session.add(Asset(id=f"asset-{ref_id}", hash=asset_hash, size_bytes=100))
    session.flush()
    session.add(
        AssetReference(
            id=ref_id,
            asset_id=f"asset-{ref_id}",
            name=os.path.basename(file_path),
            owner_id="",
            file_path=file_path,
            mtime_ns=mtime_ns,
        )
    )
    session.flush()


def _mtime(path: str) -> int:
    return get_mtime_ns(os.stat(path, follow_symlinks=True))


def test_owned_prefixes_include_temp(comfy_dirs):
    owned = get_owned_prefixes()
    assert str(comfy_dirs["temp"]) in owned, (
        "temp must be owned, or the prune disowns assets whose files are present"
    )
    for name in ("models", "input", "output"):
        assert str(comfy_dirs[name]) in owned, f"{name} must stay owned"


def test_discovery_does_not_walk_temp(comfy_dirs):
    temp_file = _write(comfy_dirs["temp"], "preview.png")
    output_file = _write(comfy_dirs["output"], "render.png")

    with patch("app.assets.scanner.collect_models_files", return_value=[]):
        paths = collect_paths_for_roots(("models", "input", "output"))

    assert output_file in paths, "scan roots must still be walked"
    assert temp_file not in paths, (
        "temp is wiped before every scan, so walking it only ever finds nothing"
    )


def test_prune_keeps_live_temp_reference(session, comfy_dirs):
    temp_file = _write(comfy_dirs["temp"], "preview.png")
    stray_file = _write(comfy_dirs["elsewhere"], "stray.png")
    _register(session, temp_file, "temp-ref", mtime_ns=_mtime(temp_file))
    _register(session, stray_file, "stray-ref", mtime_ns=_mtime(stray_file))
    session.commit()

    marked = mark_references_missing_outside_prefixes(session, get_owned_prefixes())
    session.commit()

    session.expire_all()
    assert marked == 1, "only the reference outside every owned directory is disowned"
    assert session.get(AssetReference, "temp-ref").is_missing is False, (
        "a temp file on disk is not missing, however often the prune runs"
    )
    assert session.get(AssetReference, "stray-ref").is_missing is True, (
        "owning temp must not stop the prune disowning files elsewhere"
    )


def test_temp_sync_marks_deleted_file_missing(session, comfy_dirs):
    temp_file = _write(comfy_dirs["temp"], "preview.png")
    _register(session, temp_file, "temp-ref", mtime_ns=_mtime(temp_file))
    session.commit()
    os.remove(temp_file)

    sync_prefixes_with_filesystem(session, get_temp_prefixes())
    session.commit()

    session.expire_all()
    assert session.get(AssetReference, "temp-ref").is_missing is True, (
        "nothing else stats temp, so this pass is what retires a wiped file"
    )


def test_temp_sync_drops_unhashed_asset_whose_file_is_gone(session, comfy_dirs):
    temp_file = _write(comfy_dirs["temp"], "preview.png")
    _register(session, temp_file, "temp-ref", mtime_ns=_mtime(temp_file), asset_hash=None)
    session.commit()
    os.remove(temp_file)

    sync_prefixes_with_filesystem(session, get_temp_prefixes())
    session.commit()

    session.expire_all()
    assert session.get(AssetReference, "temp-ref") is None, (
        "an unhashed asset with no surviving reference is retired, not kept as missing"
    )
    assert session.get(Asset, "asset-temp-ref") is None, (
        "the orphaned asset row goes with its last reference"
    )


def test_temp_sync_keeps_live_file(session, comfy_dirs):
    temp_file = _write(comfy_dirs["temp"], "preview.png")
    _register(session, temp_file, "temp-ref", mtime_ns=_mtime(temp_file))
    session.commit()

    sync_prefixes_with_filesystem(session, get_temp_prefixes())
    session.commit()

    session.expire_all()
    ref = session.get(AssetReference, "temp-ref")
    assert ref.is_missing is False, "the file is still there"
    assert ref.needs_verify is False, "an unchanged file needs no re-verification"
