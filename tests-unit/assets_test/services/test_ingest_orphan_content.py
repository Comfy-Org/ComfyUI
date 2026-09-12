import os
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries import create_record as create_record_query
from app.assets.scanner import SeedAssetSpec, seed_asset_specs
from app.assets.services import ingest
from app.assets.services.ingest import register_file_in_place, upload_from_temp_path


@pytest.fixture
def hashing_off():
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _write_temp(content: bytes) -> str:
    uploads_root = os.path.join(
        folder_paths.get_temp_directory(), "uploads", uuid.uuid4().hex
    )
    os.makedirs(uploads_root, exist_ok=True)
    path = os.path.join(uploads_root, ".upload.part")
    with open(path, "wb") as file:
        file.write(content)
    return path


def _raise_create_record(*_args, **_kwargs):
    raise RuntimeError("forced create_record failure")


def _live_content_count(session: Session) -> int:
    return session.scalar(
        select(func.count())
        .select_from(AssetContent)
        .where(AssetContent.is_missing.is_(False))
    )


def test_upload_record_failure_discards_newly_inserted_content(
    mock_create_session, hashing_off, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"orphan-upload-" + uuid.uuid4().bytes
    temp_path = _write_temp(content)
    monkeypatch.setattr(ingest, "create_record", _raise_create_record)

    with pytest.raises(RuntimeError, match="forced create_record failure"):
        upload_from_temp_path(
            temp_path=temp_path,
            name="orphan.bin",
            tags=["output"],
            client_filename="orphan.bin",
        )

    with mock_create_session() as session:
        assert _live_content_count(session) == 0


def test_register_file_in_place_discards_content_on_record_failure(
    mock_create_session, hashing_off, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"orphan_inplace_{uuid.uuid4().hex}.png")
    with open(path, "wb") as file:
        file.write(b"orphan-inplace-" + uuid.uuid4().bytes)
    monkeypatch.setattr(ingest, "create_record", _raise_create_record)

    try:
        with pytest.raises(RuntimeError, match="forced create_record failure"):
            register_file_in_place(abs_path=path, name="orphan_inplace.png", tags=["output"])

        with mock_create_session() as session:
            assert _live_content_count(session) == 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _seed_spec(path: str, size_bytes: int, mtime_ns: int, name: str) -> SeedAssetSpec:
    return {
        "abs_path": path,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "info_name": name,
        "tags": ["input"],
        "fname": name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def _content_at(session: Session, path: str) -> AssetContent | None:
    return session.scalar(select(AssetContent).where(AssetContent.path == path))


def _reference_count(session: Session, content_id: str) -> int:
    return session.scalar(
        select(func.count()).select_from(Asset).where(Asset.content_id == content_id)
    )


def _orphaned_content_paths(session: Session) -> list[str]:
    return [
        content.path
        for content in session.scalars(select(AssetContent))
        if _reference_count(session, content.id) == 0
    ]


def test_seed_asset_specs_orphans_nothing_and_keeps_earlier_specs_on_record_failure(
    session: Session, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs: list[SeedAssetSpec] = []
    paths: dict[str, str] = {}
    fail_name = "vanished.bin"
    for name in ("first.bin", fail_name, "last.bin"):
        file_path = tmp_path / name
        file_path.write_bytes(name.encode())
        stat_result = file_path.stat()
        paths[name] = str(file_path)
        specs.append(
            _seed_spec(str(file_path), stat_result.st_size, stat_result.st_mtime_ns, name)
        )

    attempted: list[str] = []

    def _create_record_or_raise(session_arg, content_id, name, *args, **kwargs):
        attempted.append(name)
        if name == fail_name:
            raise RuntimeError("forced create_record failure")
        return create_record_query(session_arg, content_id, name, *args, **kwargs)

    monkeypatch.setattr("app.assets.scanner.create_record", _create_record_or_raise)

    with pytest.raises(RuntimeError, match="forced create_record failure"):
        seed_asset_specs(session, specs)
    session.rollback()

    assert _content_at(session, paths[fail_name]) is None, (
        "the failed spec's content must not outlive the record that would have referenced it"
    )
    assert session.scalar(select(Asset).where(Asset.name == fail_name)) is None
    assert _orphaned_content_paths(session) == []

    survivor = _content_at(session, paths["first.bin"])
    assert survivor is not None, (
        "a spec that already succeeded must not be retroactively erased by a later "
        "unrelated failure; its savepoint was released before that failure happened"
    )
    assert session.scalar(select(Asset).where(Asset.name == "first.bin")) is not None
    assert _reference_count(session, survivor.id) == 1

    assert attempted == ["first.bin", fail_name]
    assert _content_at(session, paths["last.bin"]) is None, (
        "the raise aborts the loop, so the spec after the failed one is never attempted"
    )
