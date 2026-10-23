from pathlib import Path

import pytest

import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.lifecycle import wipe_temp_db_rows
from app.assets.services.asset_management import delete_asset_reference, update_asset_metadata


@pytest.fixture
def db_engine(db_engine_fk):
    """Foreign keys on, as in production: deletes rely on RESTRICT, SET NULL and cascades."""
    return db_engine_fk


def _record(session, path: Path, *, tags=(), preview_id=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(path.name.encode())
    content = session.query(AssetContent).filter_by(path=str(path), is_missing=False).one_or_none() or create_content(session, str(path))
    record = create_record(session, content.id, path.name, tags=list(tags))
    record.preview_id = preview_id
    session.commit()
    return record.id


def test_previews_are_reclaimed_with_their_last_user_and_never_before(session, mock_create_session, roots, tmp_path):
    previews = roots / "previews"

    # Shared by two outputs: survives the first delete, goes with the second.
    shared = _record(session, previews / "shared.webp", tags=["preview"])
    first = _record(session, roots / "output" / "a.exr", preview_id=shared)
    second = _record(session, roots / "output" / "b.exr", preview_id=shared)
    delete_asset_reference(first)
    assert (previews / "shared.webp").exists() and session.get(Asset, shared) is not None
    delete_asset_reference(second)
    session.expire_all()
    assert not (previews / "shared.webp").exists() and session.get(Asset, shared) is None

    # A file another record also uses stays when one of them goes.
    twin_a = _record(session, previews / "twin.webp", tags=["preview"])
    _record(session, previews / "twin.webp", tags=["preview"])
    delete_asset_reference(_record(session, roots / "output" / "d.exr", preview_id=twin_a))
    assert (previews / "twin.webp").exists(), "the other record still uses the file"

    # Deleting a preview itself reclaims its file.
    direct = _record(session, previews / "direct.webp", tags=["preview"])
    _record(session, roots / "output" / "e.exr", preview_id=direct)
    delete_asset_reference(direct)
    assert not (previews / "direct.webp").exists()

    # A preview replaced through PUT goes once nothing links it.
    generated = _record(session, previews / "generated.webp", tags=["preview"])
    chosen = _record(session, roots / "output" / "chosen.png")
    update_asset_metadata(_record(session, roots / "output" / "f.exr", preview_id=generated), preview_id=chosen)
    session.expire_all()
    assert session.get(Asset, generated) is None and not (previews / "generated.webp").exists()

    # A missing row's path that a live preview now uses keeps its file.
    old = _record(session, previews / "again.webp", tags=["preview"])
    parent = _record(session, roots / "output" / "c.exr", preview_id=old)
    mark_content_missing(session, session.get(Asset, old).content_id)
    session.commit()
    _record(session, previews / "again.webp", tags=["preview"])
    delete_asset_reference(parent)
    assert (previews / "again.webp").exists(), "the newer preview still uses this file"

    # A temp upload's preview goes with the restart temp wipe.
    saved = folder_paths.get_temp_directory()
    folder_paths.set_temp_directory(str(tmp_path / "temp"))
    try:
        temp_preview = _record(session, previews / "temp.webp", tags=["preview"])
        _record(session, tmp_path / "temp" / "upload.exr", tags=["input"], preview_id=temp_preview)
        wipe_temp_db_rows(session)
        session.commit()
    finally:
        folder_paths.set_temp_directory(saved)
    session.expire_all()
    assert session.get(Asset, temp_preview) is None and not (previews / "temp.webp").exists()
