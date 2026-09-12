import json
import os
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp.test_utils import make_mocked_request
from blake3 import blake3
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.api import routes
from app.assets.database.models import Asset, AssetContent, Base
from app.assets.database.queries import (
    create_content,
    create_record,
    fetch_record_tags,
    get_record_by_id,
    mark_content_missing,
    rename_record,
)
from app.assets.database.queries.records import (
    get_preview_file_paths_by_ids,
    get_record_by_path_or_none,
)
from app.assets.helpers import to_stored_hash
from app.assets.lifecycle import wipe_temp_db_rows
from app.assets.scanner import (
    build_asset_specs,
    seed_asset_specs,
    sync_prefixes_with_filesystem,
)
from app.assets.scanner_admission import _should_skip_extension
from app.assets.scanner_changes import (
    clear_pending_verifications,
    detect_content_change,
    drain_pending_verifications,
    queue_pending_verification,
)
from app.assets.services.asset_management import (
    asset_exists,
    delete_asset_reference,
    resolve_hash_to_path,
)
from app.assets.services.file_utils import list_files_recursively
from app.assets.services.ingest import register_cached_output, upload_from_temp_path
from app.assets.services.lookup import (
    lookup_for_from_hash,
    lookup_for_view,
)
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _record(session, path: Path, name: str, hash_value: str | None = None):
    content = create_content(session, str(path), hash=hash_value, size_bytes=path.stat().st_size if path.exists() else 0)
    return create_record(session, content.id, name)


@contextmanager
def _sandbox_asset_roots(root: Path, model_category: Path | None = None):
    with patch("app.assets.services.path_utils.folder_paths") as folder_paths_mock:
        folder_paths_mock.get_input_directory.return_value = str(root / "input")
        folder_paths_mock.get_output_directory.return_value = str(root / "output")
        folder_paths_mock.get_temp_directory.return_value = str(root / "temp")
        folder_paths_mock.models_dir = str(root / "models")
        categories = (
            [("checkpoints", [str(model_category)], {".safetensors"})]
            if model_category is not None
            else []
        )
        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=categories,
        ):
            yield


_WRITER_UPDATE_CAP = 16


def _differently_sized(payload: bytes, avoid_size: int) -> bytes:
    return payload if len(payload) != avoid_size else payload + b"!"


def _seed_content_row(session, path: Path, hash_value: str | None = None):
    seed_stat = path.stat()
    return create_content(
        session,
        str(path),
        hash=hash_value,
        size_bytes=seed_stat.st_size,
        mtime_ns=seed_stat.st_mtime_ns,
    )


def _scan_pass(session, root: Path) -> int:
    survivors = sync_prefixes_with_filesystem(
        session, [str(root)], collect_existing_paths=True
    )
    specs, _tag_pool, _skipped = build_asset_specs(
        list_files_recursively(str(root)), survivors or set()
    )
    return seed_asset_specs(session, specs)


@contextmanager
def _writer_lands_mid_hash(path: Path, replacement: bytes):
    real_blake3 = blake3

    class _WriterHasher:
        def __init__(self) -> None:
            self._inner = real_blake3()
            self._updates = 0

        def update(self, chunk: bytes) -> None:
            self._updates += 1
            assert self._updates <= _WRITER_UPDATE_CAP, (
                f"simulated writer never converged after {_WRITER_UPDATE_CAP} "
                "chunks — it is feeding the read loop instead of ending it"
            )
            if self._updates == 1:
                path.write_bytes(_differently_sized(replacement, path.stat().st_size))
            self._inner.update(chunk)

        def hexdigest(self) -> str:
            return self._inner.hexdigest()

    with patch("app.assets.services.snapshot_hash.blake3", _WriterHasher):
        yield _WriterHasher


def test_scenario_1_rm_missing_and_strict_recovery(session, tmp_path):
    """Ruling 1: missing is projected from content to every record."""
    record = _record(session, tmp_path / "missing.bin", "missing")
    mark_content_missing(session, record.content_id)
    assert fetch_record_tags(session, record.id) == ["missing"]


def test_scenario_2_edit_split(session, tmp_path):
    """Ruling 2: edits split content and the old content read is unavailable.

    The split is the production verdict, not the fixture's: an in-place edit is
    handed to ``detect_content_change`` with hashing off and both halves of a
    genuine change (size AND mtime moved), which is the one OFF-mode shape that
    is allowed to retire content.
    """
    input_root = tmp_path / "input"
    input_root.mkdir()
    path = input_root / "edit.bin"
    path.write_bytes(b"the-original-bytes")
    seed_stat = path.stat()
    old_content = _seed_content_row(session, path)
    old_record = create_record(session, old_content.id, "edit.bin")

    path.write_bytes(b"a replacement of a decidedly different length")
    moved_mtime = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(moved_mtime, moved_mtime))
    edited_stat = path.stat()
    assert edited_stat.st_size != seed_stat.st_size
    assert edited_stat.st_mtime_ns != seed_stat.st_mtime_ns

    with _sandbox_asset_roots(tmp_path):
        detect_content_change(session, old_content, edited_stat, hashing_is_enabled=False)

    new_record = get_record_by_path_or_none(session, str(path))
    assert new_record is not None
    assert new_record.id != old_record.id
    assert new_record.content_id != old_content.id
    assert new_record.content.is_missing is False
    assert new_record.content.size_bytes == edited_stat.st_size
    assert new_record.content.mtime_ns == edited_stat.st_mtime_ns
    assert session.get(Asset, old_record.id).content_id == old_content.id
    assert old_content.is_missing is True
    assert fetch_record_tags(session, old_record.id) == ["missing"]
    assert create_content(session, str(path)).id == new_record.content_id


def test_scenario_3_path_reuse_convergence(session, tmp_path):
    """Ruling 3: path reuse converges on missing old content plus a new record."""
    old = _record(session, tmp_path / "same.bin", "old")
    mark_content_missing(session, old.content_id)
    new = _record(session, tmp_path / "same.bin", "new")
    assert old.content_id != new.content_id
    assert old.content.is_missing is True
    assert new.content.is_missing is False
    assert create_content(session, str(tmp_path / "same.bin")).id == new.content_id


def test_scenario_4_delete_no_revival(session, tmp_path):
    """Ruling 4: delete is hard, spares the content, and cannot be undone."""
    record = _record(session, tmp_path / "deleted.bin", "deleted")
    record_id, content_id = record.id, record.content_id

    with patch(
        "app.assets.services.asset_management.create_session",
        lambda: nullcontext(session),
    ):
        assert delete_asset_reference(record_id) is True
        assert delete_asset_reference(record_id) is False

    assert get_record_by_id(session, record_id) is None
    assert session.get(AssetContent, content_id) is not None


def test_scenario_5_rename_always(session):
    """Ruling 5: names are labels and duplicate names are allowed."""
    first = create_record(session, create_content(session, "/one").id, "same")
    second = create_record(session, create_content(session, "/two").id, "other")
    assert rename_record(session, second.id, "same").name == first.name


def test_scenario_6_upload_reuses_content_never_the_record(session, tmp_path):
    """Ruling 6: every upload is a delivery event, so it always mints its own
    record; dedup is content-level only, and runs in both hashing modes.

    Ratified 2026-08-28, superseding the "same name + same bytes returns the
    existing record" behavior. Content dedup is unchanged: identical bytes
    still share one ``AssetContent`` row, in BOTH modes — uploads hash
    unconditionally (``upload_from_temp_path`` calls
    ``_snapshot_hash_with_retry`` before it consults anything) and
    ``lookup_for_view`` never asks ``mode.hashing_enabled``, unlike
    ``lookup_for_from_hash``. What changed is record identity: a re-upload is a
    new delivery, so it gets a new record carrying the attributes THAT request
    supplied, instead of silently handing back an older record that never saw
    them.
    """
    (tmp_path / "output").mkdir(parents=True)
    payload = b"the-uploaded-bytes"
    expected_hash = to_stored_hash(blake3(payload).hexdigest())

    def upload(
        uploaded: bytes,
        name: str,
        *,
        hashing: bool,
        user_metadata: dict | None = None,
    ):
        staging = tmp_path / "temp" / "uploads" / uuid.uuid4().hex
        staging.mkdir(parents=True)
        staged = staging / ".upload.part"
        staged.write_bytes(uploaded)
        with (
            _sandbox_asset_roots(tmp_path),
            patch(
                "folder_paths.get_temp_directory", return_value=str(tmp_path / "temp")
            ),
            patch(
                "app.assets.services.ingest.create_session",
                lambda: nullcontext(session),
            ),
            patch.object(mode, "hashing_enabled", return_value=hashing),
        ):
            return upload_from_temp_path(
                temp_path=str(staged),
                name=name,
                tags=["output"],
                client_filename=name,
                user_metadata=user_metadata,
            )

    first = upload(payload, "upload.bin", hashing=False)
    again = upload(
        payload, "upload.bin", hashing=False, user_metadata={"note": "second delivery"}
    )
    renamed = upload(payload, "renamed.bin", hashing=False)
    in_hash_mode = upload(payload, "upload.bin", hashing=True)
    other = upload(b"a wholly different payload", "upload.bin", hashing=False)

    assert first.created_new is True
    assert first.asset.hash == expected_hash

    assert again.created_new is True
    assert again.ref.id != first.ref.id, "a re-upload is its own delivery"
    assert again.ref.user_metadata == {"note": "second delivery"}, (
        "the new record carries the attributes THIS request supplied"
    )
    assert (
        session.get(Asset, again.ref.id).content_id
        == session.get(Asset, first.ref.id).content_id
    ), "identical bytes still share one content row"

    assert in_hash_mode.created_new is True
    assert in_hash_mode.ref.id != first.ref.id
    assert (
        session.get(Asset, in_hash_mode.ref.id).content_id
        == session.get(Asset, first.ref.id).content_id
    ), "content dedup is mode-independent"

    assert renamed.created_new is True
    assert renamed.ref.id != first.ref.id
    assert renamed.ref.file_path == first.ref.file_path
    assert (
        session.get(Asset, renamed.ref.id).content_id
        == session.get(Asset, first.ref.id).content_id
    )

    assert other.created_new is True
    assert other.asset.hash != expected_hash
    assert other.ref.file_path != first.ref.file_path

    live = list(
        session.scalars(select(AssetContent).where(AssetContent.is_missing.is_(False)))
    )
    assert {content.hash for content in live} == {expected_hash, other.asset.hash}


def test_scenario_7_same_bytes_new_name(session, tmp_path):
    """Ruling 7: a new name creates a new record."""
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"bytes")
    content = create_content(session, str(path), hash="digest")
    one = create_record(session, content.id, "one")
    two = create_record(session, content.id, "two")
    assert one.id != two.id
    assert one.content_id == two.content_id == content.id
    assert {one.name, two.name} == {"one", "two"}


def test_scenario_8_diff_bytes_same_name(session, tmp_path):
    """Ruling 8: different bytes may share a display name."""
    a = _record(session, tmp_path / "one", "same")
    b = _record(session, tmp_path / "two", "same")
    assert a.id != b.id
    assert a.name == b.name == "same"
    assert a.content_id != b.content_id


def test_scenario_9_equal_hashes_no_merge(session, tmp_path):
    """Ruling 9: equal hashes never impose content uniqueness."""
    a = _record(session, tmp_path / "one", "one", "digest")
    b = _record(session, tmp_path / "two", "two", "digest")
    assert a.content_id != b.content_id
    assert a.content.hash == b.content.hash == "digest"
    assert a.content.is_missing is False
    assert b.content.is_missing is False


def test_scenario_10_cached_delivery_record(session, tmp_path):
    """Ruling 10: cached delivery creates another record for existing content."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    path = output_root / "cached.png"
    path.write_bytes(b"pixels")
    content = _seed_content_row(session, path, hash_value="digest")
    original = create_record(
        session,
        content.id,
        "cached.png",
        job_id="executed-job",
        system_metadata={"inherited": "from-the-earliest-sibling"},
    )
    content_before = (
        content.hash,
        content.size_bytes,
        content.mtime_ns,
        content.is_missing,
    )

    with (
        _sandbox_asset_roots(tmp_path),
        patch(
            "app.assets.services.ingest.create_session",
            lambda: nullcontext(session),
        ),
    ):
        delivered = register_cached_output(str(path), job_id="delivery-job")

    assert delivered is not None
    assert delivered.id != original.id
    assert delivered.content_id == content.id
    assert delivered.job_id == "delivery-job"
    assert session.get(Asset, delivered.id).system_metadata == original.system_metadata

    assert (
        content.hash,
        content.size_bytes,
        content.mtime_ns,
        content.is_missing,
    ) == content_before
    assert (original.job_id, original.name) == ("executed-job", "cached.png")
    assert [row.id for row in session.scalars(select(AssetContent))] == [content.id]
    assert {row.id for row in session.scalars(select(Asset))} == {
        original.id,
        delivered.id,
    }

    mark_content_missing(session, content.id)
    with (
        _sandbox_asset_roots(tmp_path),
        patch(
            "app.assets.services.ingest.create_session",
            lambda: nullcontext(session),
        ),
    ):
        assert register_cached_output(str(path), job_id="second-delivery") is None
    assert {row.id for row in session.scalars(select(Asset))} == {
        original.id,
        delivered.id,
    }


def test_scenario_11_restart_survival(tmp_path):
    """Ruling 11: non-temp records survive reopening the database."""
    database = tmp_path / "assets.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        record = create_record(session, create_content(session, "/durable").id, "durable")
        session.commit()
        record_id = record.id
    with Session(engine) as session:
        assert session.get(type(record), record_id) is not None


def test_scenario_12_temp_wipe_both_layers(session, tmp_path):
    """Ruling 12: temp removal deletes records before their content."""
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    keep_root = tmp_path / "output"
    keep_root.mkdir()
    doomed = _record(session, temp_root / "render.png", "render")
    survivor = _record(session, keep_root / "final.png", "final")
    doomed_id, doomed_content_id = doomed.id, doomed.content_id
    survivor_id, survivor_content_id = survivor.id, survivor.content_id

    with patch("folder_paths.get_temp_directory", return_value=str(temp_root)):
        deleted = wipe_temp_db_rows(session)
    session.commit()

    assert deleted == (1, 1)
    assert session.get(Asset, doomed_id) is None
    assert session.get(AssetContent, doomed_content_id) is None
    assert session.get(Asset, survivor_id) is not None
    assert session.get(AssetContent, survivor_content_id) is not None


def test_scenario_15_two_locations_hash_relation(session, tmp_path):
    """Ruling 15: locations retain separate rows even with equal hashes."""
    a = _record(session, tmp_path / "a", "a", "digest")
    b = _record(session, tmp_path / "b", "b", "digest")
    assert a.content_id != b.content_id
    assert a.content.hash == b.content.hash == "digest"
    assert a.content.path != b.content.path


def test_scenario_17_move_is_missing_plus_new(session, tmp_path):
    """Ruling 17: a move is missing old content plus a new record.

    Driven as a real ``os.replace`` observed by a real scanner pass: nothing
    tells the scanner a move happened, it only sees one path gone and another
    arrived, and the missing-plus-new outcome is entirely its own.
    """
    input_root = tmp_path / "input"
    input_root.mkdir()
    old_path = input_root / "old.bin"
    old_path.write_bytes(b"the-bytes-that-move")
    old_content = _seed_content_row(session, old_path)
    old_record = create_record(session, old_content.id, "old.bin")

    new_path = input_root / "new.bin"
    os.replace(old_path, new_path)

    with (
        _sandbox_asset_roots(tmp_path),
        patch.object(mode, "hashing_enabled", return_value=False),
    ):
        created = _scan_pass(session, input_root)

    assert created == 1
    new_record = get_record_by_path_or_none(session, str(new_path))
    assert new_record is not None
    assert new_record.id != old_record.id
    assert new_record.content_id != old_content.id
    assert new_record.content.path == str(new_path)
    assert new_record.content.is_missing is False
    assert old_content.is_missing is True
    assert "missing" in fetch_record_tags(session, old_record.id)
    assert session.get(Asset, old_record.id).content_id == old_content.id
    assert get_record_by_path_or_none(session, str(old_path)) is None


def test_scenario_18_edit_during_hash_discard(session, tmp_path):
    """Ruling 18: unstable hashing must not overwrite a content identity."""
    path = tmp_path / "unstable.bin"
    committed = b"the-committed-bytes"
    path.write_bytes(committed)
    committed_hash = to_stored_hash(blake3(committed).hexdigest())
    seed_stat = path.stat()
    content = create_content(
        session,
        str(path),
        hash=committed_hash,
        size_bytes=seed_stat.st_size,
        mtime_ns=seed_stat.st_mtime_ns,
    )
    record = create_record(session, content.id, "unstable.bin")

    clear_pending_verifications()
    try:
        queue_pending_verification(content.id)
        with _writer_lands_mid_hash(path, b"a-concurrent-writer-was-here"):
            assert snapshot_hash(str(path)) is None
            assert drain_pending_verifications(session) == 0

        assert content.hash == committed_hash
        assert content.is_missing is False
        assert [row.id for row in session.scalars(select(Asset))] == [record.id]
        assert [row.id for row in session.scalars(select(AssetContent))] == [content.id]

        path.write_bytes(committed)
        assert drain_pending_verifications(session) == 1
    finally:
        clear_pending_verifications()

    assert content.hash == committed_hash
    assert content.mtime_ns == path.stat().st_mtime_ns


def test_writer_simulation_terminates_and_is_capped(tmp_path):
    path = tmp_path / "bounded.bin"
    path.write_bytes(b"0123456789")

    with _writer_lands_mid_hash(path, b"a-replacement-of-another-length") as hasher:
        assert snapshot_hash(str(path)) is None
        assert snapshot_hash(str(path)) is None
        assert path.stat().st_size != 10

        runaway = hasher()
        with pytest.raises(AssertionError, match="never converged"):
            for _ in range(_WRITER_UPDATE_CAP + 1):
                runaway.update(b"chunk")


def test_scenario_20_partial_download(tmp_path):
    """Ruling 20: partial-download admission is separate from content creation."""
    partial = tmp_path / "model.safetensors.part"
    partial.write_bytes(b"partial")
    complete = tmp_path / "model.safetensors"
    complete.write_bytes(b"complete")
    assert _should_skip_extension(str(partial)) is True
    assert _should_skip_extension(str(complete)) is False


def test_scenario_21_symlink_two_rows(session, tmp_path):
    """Ruling 21: lexical locations always retain separate content rows.

    One inode, two names, one real scanner pass. Lexical means the scanner must
    not resolve the link: if it took the realpath, the second admission would
    collide on the live-path index and yield a single row.
    """
    input_root = tmp_path / "input"
    input_root.mkdir()
    real_path = input_root / "real.bin"
    real_path.write_bytes(b"one-set-of-bytes-answering-to-two-names")
    link_path = input_root / "link.bin"
    os.symlink(real_path, link_path)
    assert link_path.is_symlink()
    assert os.path.samefile(real_path, link_path)

    with (
        _sandbox_asset_roots(tmp_path),
        patch.object(mode, "hashing_enabled", return_value=False),
    ):
        created = _scan_pass(session, input_root)

    assert created == 2
    real_record = get_record_by_path_or_none(session, str(real_path))
    link_record = get_record_by_path_or_none(session, str(link_path))
    assert real_record is not None
    assert link_record is not None
    assert real_record.content_id != link_record.content_id
    assert real_record.content.path == str(real_path)
    assert link_record.content.path == str(link_path)
    assert real_record.content.is_missing is False
    assert link_record.content.is_missing is False
    assert real_record.content.size_bytes == link_record.content.size_bytes


def test_scenario_25_registry_birth_fact(session, tmp_path):
    """Ruling 25: loader classification is stamped at record birth."""
    category = tmp_path / "models" / "checkpoints"
    (category / "family").mkdir(parents=True)
    path = category / "family" / "model.safetensors"
    path.write_bytes(b"first")
    seed_stat = path.stat()
    content = create_content(
        session, str(path), size_bytes=seed_stat.st_size, mtime_ns=seed_stat.st_mtime_ns
    )
    original = create_record(session, content.id, "model.safetensors")
    assert original.loader_path is None

    path.write_bytes(b"second-bytes-of-a-different-length")
    with _sandbox_asset_roots(tmp_path, category):
        detect_content_change(session, content, path.stat(), hashing_is_enabled=False)

    born = get_record_by_path_or_none(session, str(path))
    assert born.id != original.id
    assert born.loader_path == "family/model.safetensors"
    assert original.loader_path is None


@pytest.mark.asyncio
async def test_scenario_26_view_forms(tmp_path):
    """Ruling 26: record identity is stable for the canonical view form."""
    nested = tmp_path / "output" / "nested"
    nested.mkdir(parents=True)
    file_path = nested / "view.png"
    file_path.write_bytes(b"pixels")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        content = create_content(setup, str(file_path))
        record_id = create_record(
            setup, content.id, "original-name.png", mime_type="image/png"
        ).id
        setup.commit()

    async def view_form() -> tuple[str, str]:
        response = await routes.list_assets_route(
            make_mocked_request("GET", "/api/assets")
        )
        listed = json.loads(response.body)["assets"]
        assert len(listed) == 1
        return listed[0]["id"], listed[0]["preview_url"]

    with (
        patch.object(routes, "create_session", lambda: Session(engine)),
        patch.object(routes, "_ASSETS_ENABLED", True),
        _sandbox_asset_roots(tmp_path),
    ):
        before = await view_form()
        with Session(engine) as renaming:
            rename_record(renaming, record_id, "renamed-entirely.png")
            renaming.commit()
        after = await view_form()
    engine.dispose()

    assert before == (
        record_id,
        "/api/view?type=output&filename=view.png&subfolder=nested",
    )
    assert after == before


def test_scenario_27_fail_closed_previews_fromhash(session, tmp_path):
    """Ruling 27: missing content is never a serving candidate."""
    path = tmp_path / "servable.png"
    payload = b"pixels"
    path.write_bytes(payload)
    seed_stat = path.stat()
    digest = to_stored_hash(blake3(payload).hexdigest())
    content = create_content(
        session,
        str(path),
        hash=digest,
        size_bytes=seed_stat.st_size,
        mtime_ns=seed_stat.st_mtime_ns,
    )
    record_id = create_record(session, content.id, "servable.png").id

    def previews() -> dict[str, str]:
        return get_preview_file_paths_by_ids(session, [record_id])

    def from_hash():
        with patch.object(mode, "hashing_enabled", return_value=True):
            return lookup_for_from_hash(session, digest)

    def serving() -> tuple[bool, object]:
        with patch(
            "app.assets.services.asset_management.create_session",
            lambda: nullcontext(session),
        ):
            return asset_exists(digest), resolve_hash_to_path(digest)

    with patch("folder_paths.get_temp_directory", return_value=str(tmp_path / "temp")):
        assert previews() == {record_id: str(path)}
        assert from_hash().id == content.id
        assert lookup_for_view(session, digest).id == content.id
        exists, resolved = serving()
        assert exists is True
        assert resolved is not None

        mark_content_missing(session, content.id)

        assert previews() == {}
        assert from_hash() is None
        assert lookup_for_view(session, digest) is None
        assert serving() == (False, None)


def test_scenario_28_temp_exclusion(session, tmp_path):
    """Ruling 28: a temporary location cannot become permanent shared content."""
    path = tmp_path / "temp.bin"
    path.write_bytes(b"bytes")
    record = _record(session, path, "temp", "digest")
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        assert lookup_for_view(session, "digest") is None
    with patch("app.assets.services.lookup.is_temp_path", return_value=False):
        assert lookup_for_view(session, "digest").id == record.content_id
