"""Tests for the asset semantics reset (app/assets/semantics)."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.assets.semantics as semantics
from app.assets.database.models import (
    Asset,
    AssetReference,
    AssetReferenceTag,
    AssetSemanticsVersion,
    Base,
    Tag,
)
from app.assets.database.queries.semantics import get_semantics_version
from app.assets.semantics import run_pending_semantics_steps
from app.assets.semantics.reproject_derived import reproject_derived_state
from app.assets.semantics.step import (
    SemanticsStep,
    SemanticsStepInterrupted,
    StepResult,
)
from app.assets.services.file_utils import get_mtime_ns


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override parent autouse fixture - these tests don't need server cleanup."""
    yield


def _engine_shared_across_sessions():
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_factory():
    factory = sessionmaker(bind=_engine_shared_across_sessions())
    with (
        patch("app.assets.semantics.reproject_derived.create_session", factory),
        patch("app.assets.semantics.create_session", factory),
        patch("app.assets.semantics.can_create_session", return_value=True),
    ):
        yield factory


@pytest.fixture
def session(session_factory) -> Session:
    with session_factory() as sess:
        yield sess


@pytest.fixture
def comfy_dirs():
    with tempfile.TemporaryDirectory() as base:
        dirs = {
            name: Path(base) / name
            for name in ("checkpoints", "loras", "input", "output", "temp", "elsewhere")
        }
        for directory in dirs.values():
            directory.mkdir()
        with (
            patch("folder_paths.get_input_directory", return_value=str(dirs["input"])),
            patch(
                "folder_paths.get_output_directory", return_value=str(dirs["output"])
            ),
            patch("folder_paths.get_temp_directory", return_value=str(dirs["temp"])),
            patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[
                    ("checkpoints", [str(dirs["checkpoints"])], {".safetensors"}),
                    ("loras", [str(dirs["loras"])], {".safetensors"}),
                ],
            ),
        ):
            yield dirs


ANY_UNIQUE_HASH = "<any unique hash>"


def _write(directory: Path, name: str, content: bytes = b"\x00" * 100) -> str:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return os.path.abspath(str(path))


def _register(
    session: Session,
    file_path: str,
    ref_id: str,
    *,
    loader_path: str | None = None,
    asset_hash: str | None = ANY_UNIQUE_HASH,
    size_bytes: int = 100,
    mtime_ns: int | None = None,
    is_missing: bool = False,
    needs_verify: bool = False,
    user_metadata: dict | None = None,
    preview_id: str | None = None,
    deleted_at: datetime | None = None,
    job_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> AssetReference:
    if asset_hash == ANY_UNIQUE_HASH:
        asset_hash = f"blake3:{ref_id}"
    if mtime_ns is None:
        mtime_ns = get_mtime_ns(os.stat(file_path, follow_symlinks=True))

    session.add(Asset(id=f"asset-{ref_id}", hash=asset_hash, size_bytes=size_bytes))
    session.flush()
    ref = AssetReference(
        id=ref_id,
        asset_id=f"asset-{ref_id}",
        name=os.path.basename(file_path),
        owner_id="",
        file_path=file_path,
        loader_path=loader_path,
        mtime_ns=mtime_ns,
        is_missing=is_missing,
        needs_verify=needs_verify,
        user_metadata=user_metadata,
        preview_id=preview_id,
        deleted_at=deleted_at,
        job_id=job_id,
    )
    session.add(ref)
    session.flush()

    for tag_name, origin in (tags or {}).items():
        session.merge(Tag(name=tag_name))
        session.add(
            AssetReferenceTag(
                asset_reference_id=ref_id, tag_name=tag_name, origin=origin
            )
        )
    session.commit()
    return ref


def _tags(session: Session, ref_id: str) -> dict[str, str]:
    rows = session.execute(
        select(AssetReferenceTag.tag_name, AssetReferenceTag.origin).where(
            AssetReferenceTag.asset_reference_id == ref_id
        )
    ).all()
    return {name: origin for name, origin in rows}


def _state_the_reset_could_touch(session: Session) -> list[tuple]:
    session.expire_all()
    refs = session.execute(select(AssetReference).order_by(AssetReference.id)).scalars()
    rows = [
        (
            ref.id,
            ref.file_path,
            ref.loader_path,
            ref.is_missing,
            ref.needs_verify,
            ref.mtime_ns,
            ref.name,
            ref.preview_id,
            str(ref.user_metadata),
            ref.deleted_at,
            ref.job_id,
            tuple(sorted(_tags(session, ref.id).items())),
        )
        for ref in refs
    ]
    assets = session.execute(select(Asset).order_by(Asset.id)).scalars()
    rows.extend((asset.id, asset.hash, asset.size_bytes) for asset in assets)
    return rows


class TestLoaderPathReprojection:
    def test_null_loader_path_is_backfilled(self, session, comfy_dirs):
        path = _write(comfy_dirs["checkpoints"], "flux/model.safetensors")
        _register(session, path, "ref-1", loader_path=None)

        reproject_derived_state()

        session.expire_all()
        ref = session.get(AssetReference, "ref-1")
        assert ref.loader_path == "flux/model.safetensors", (
            "0006 added the column with no backfill, and nothing has filled it since"
        )

    def test_stale_loader_path_is_rewritten(self, session, comfy_dirs):
        path = _write(comfy_dirs["input"], "sub/photo.png")
        _register(session, path, "ref-1", loader_path="checkpoints/sub/photo.png")

        summary = reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").loader_path == "sub/photo.png"
        assert summary.loader_paths_rewritten == 1

    def test_correct_loader_path_is_left_alone(self, session, comfy_dirs):
        path = _write(comfy_dirs["loras"], "style.safetensors")
        _register(session, path, "ref-1", loader_path="style.safetensors")

        summary = reproject_derived_state()

        assert summary.loader_paths_rewritten == 0

    def test_unloadable_extension_loses_its_loader_path(self, session, comfy_dirs):
        path = _write(comfy_dirs["checkpoints"], "notes.txt")
        _register(session, path, "ref-1", loader_path="notes.txt")

        reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").loader_path is None, (
            "a file its category cannot load must not advertise a loader path"
        )

    def test_reference_without_file_path_is_skipped(self, session, comfy_dirs):
        session.add(Asset(id="asset-api", hash="blake3:abc", size_bytes=10))
        session.flush()
        session.add(
            AssetReference(
                id="ref-api", asset_id="asset-api", name="api.png", owner_id=""
            )
        )
        session.commit()

        summary = reproject_derived_state()

        assert summary.scanned == 0
        session.expire_all()
        assert session.get(AssetReference, "ref-api").loader_path is None


class TestHashPreservation:
    def test_unchanged_file_keeps_its_hash_and_is_not_read(self, session, comfy_dirs):
        path = _write(comfy_dirs["checkpoints"], "big.safetensors")
        _register(session, path, "ref-1", asset_hash="blake3:original", size_bytes=100)

        with patch(
            "app.assets.services.hashing.compute_blake3_hash",
            side_effect=AssertionError("the reset must never re-hash"),
        ):
            summary = reproject_derived_state()

        session.expire_all()
        assert session.get(Asset, "asset-ref-1").hash == "blake3:original"
        assert summary.unchanged_files == 1
        assert summary.changed_files == 0

    def test_changed_file_keeps_its_hash_and_is_flagged_for_verify(
        self, session, comfy_dirs
    ):
        path = _write(comfy_dirs["checkpoints"], "big.safetensors")
        _register(
            session,
            path,
            "ref-1",
            asset_hash="blake3:original",
            mtime_ns=1,
            size_bytes=100,
        )

        summary = reproject_derived_state()

        session.expire_all()
        assert session.get(Asset, "asset-ref-1").hash == "blake3:original"
        assert session.get(Asset, "asset-ref-1").size_bytes == 100
        assert session.get(AssetReference, "ref-1").needs_verify is True, (
            "a file that moved on goes to the verify path rather than being re-read"
        )
        assert summary.changed_files == 1

    def test_changed_file_still_gets_its_path_state_reprojected(
        self, session, comfy_dirs
    ):
        path = _write(comfy_dirs["loras"], "style.safetensors")
        _register(session, path, "ref-1", loader_path=None, mtime_ns=1)

        reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").loader_path == "style.safetensors"


class TestIntentIsPreserved:
    def test_manual_tags_metadata_preview_and_deletion_survive(
        self, session, comfy_dirs
    ):
        path = _write(comfy_dirs["checkpoints"], "curated.safetensors")
        other = _write(comfy_dirs["input"], "thumb.png")
        _register(session, other, "ref-preview")
        deleted = datetime(2026, 1, 2, 3, 4, 5)
        _register(
            session,
            path,
            "ref-1",
            loader_path=None,
            user_metadata={"note": "hand written", "filename": "kept.safetensors"},
            preview_id="ref-preview",
            deleted_at=deleted,
            job_id="job-42",
            tags={"favourite": "manual", "uploaded": "upload"},
        )

        reproject_derived_state()

        session.expire_all()
        ref = session.get(AssetReference, "ref-1")
        assert ref.user_metadata == {
            "note": "hand written",
            "filename": "kept.safetensors",
        }
        assert ref.preview_id == "ref-preview"
        assert ref.deleted_at == deleted
        assert ref.job_id == "job-42"
        assert ref.name == "curated.safetensors"
        tags = _tags(session, "ref-1")
        assert tags["favourite"] == "manual"
        assert tags["uploaded"] == "upload"
        assert ref.loader_path == "curated.safetensors", (
            "preserving intent must not cost the reprojection around it"
        )

    def test_manual_tag_inside_the_derived_vocabulary_is_not_removed(
        self, session, comfy_dirs
    ):
        path = _write(comfy_dirs["input"], "photo.png")
        _register(session, path, "ref-1", tags={"models": "manual"})

        reproject_derived_state()

        assert _tags(session, "ref-1")["models"] == "manual", (
            "a person may tag an input file 'models'; that is their business"
        )


class TestTagReprojection:
    def test_missing_derived_tags_are_added(self, session, comfy_dirs):
        path = _write(comfy_dirs["checkpoints"], "model.safetensors")
        _register(session, path, "ref-1", tags={})

        reproject_derived_state()

        assert _tags(session, "ref-1") == {
            "models": "automatic",
            "model_type:checkpoints": "automatic",
        }

    def test_superseded_automatic_tag_is_removed(self, session, comfy_dirs):
        path = _write(comfy_dirs["checkpoints"], "model.safetensors")
        _register(
            session,
            path,
            "ref-1",
            tags={
                "models": "automatic",
                "model_type:checkpoints": "automatic",
                "model_type:loras": "automatic",
            },
        )

        reproject_derived_state()

        assert "model_type:loras" not in _tags(session, "ref-1"), (
            "an older rule tagged by directory alone; extensions now gate the tag"
        )
        assert "model_type:checkpoints" in _tags(session, "ref-1")

    @pytest.mark.parametrize(
        ("position", "registered"),
        [("leading", " loras"), ("trailing", "loras ")],
    )
    def test_vocabulary_matches_tags_as_the_database_stores_them(
        self, session, comfy_dirs, position, registered
    ):
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "model.safetensors"),
            "ref-1",
            tags={"model_type:loras": "automatic"},
        )

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("checkpoints", [str(comfy_dirs["checkpoints"])], {".safetensors"}),
                (registered, [str(comfy_dirs["loras"])], {".safetensors"}),
            ],
        ):
            reproject_derived_state()

        assert "model_type:loras" not in _tags(session, "ref-1"), (
            f"{position} whitespace in {registered!r} must not smuggle an entry "
            "past the vocabulary and leave the stale tag in place"
        )

    def test_automatic_tag_outside_the_vocabulary_is_left_alone(
        self, session, comfy_dirs
    ):
        path = _write(comfy_dirs["checkpoints"], "model.safetensors")
        _register(session, path, "ref-1", tags={"missing": "automatic"})

        reproject_derived_state()

        assert "missing" in _tags(session, "ref-1"), (
            "'missing' is automatic but not path-derived, so the scanner owns it"
        )


class TestFileState:
    def test_present_file_is_unflagged_as_missing(self, session, comfy_dirs):
        path = _write(comfy_dirs["temp"], "preview.png")
        _register(session, path, "ref-1", is_missing=True)

        summary = reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").is_missing is False
        assert summary.missing_flags_cleared == 1

    def test_absent_file_is_left_to_the_scanner(self, session, comfy_dirs):
        path = os.path.join(str(comfy_dirs["checkpoints"]), "gone.safetensors")
        _register(
            session,
            path,
            "ref-1",
            mtime_ns=1,
            loader_path="stale/gone.safetensors",
            is_missing=True,
        )

        summary = reproject_derived_state()

        session.expire_all()
        ref = session.get(AssetReference, "ref-1")
        assert ref.is_missing is True
        assert ref.loader_path == "stale/gone.safetensors"
        assert summary.absent_files == 1
        assert summary.complete, (
            "a file that is simply gone is the scanner's business, not unfinished work"
        )

    def test_unclassified_row_is_not_flagged_for_verify(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["elsewhere"], "model.safetensors"),
            "ref-1",
            mtime_ns=1,
        )

        summary = reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").needs_verify is False, (
            "the one write that did not skip made 'leaves it alone' untrue"
        )
        assert summary.unclassified_paths == 1
        assert summary.changed_files == 0

    def test_path_outside_every_known_root_is_left_alone(self, session, comfy_dirs):
        path = _write(comfy_dirs["elsewhere"], "model.safetensors")
        _register(
            session,
            path,
            "ref-1",
            loader_path="model.safetensors",
            tags={"models": "automatic", "model_type:checkpoints": "automatic"},
        )

        summary = reproject_derived_state()

        session.expire_all()
        assert session.get(AssetReference, "ref-1").loader_path == "model.safetensors"
        assert _tags(session, "ref-1") == {
            "models": "automatic",
            "model_type:checkpoints": "automatic",
        }, "a root missing from the config must not strip tags off its assets"
        assert summary.unclassified_paths == 1
        assert not summary.complete, (
            "the row is unrepaired, so the step has not finished its work"
        )


class TestIdempotence:
    def test_empty_database_is_a_no_op(self, session, comfy_dirs):
        summary = reproject_derived_state()

        assert summary.scanned == 0
        assert summary.loader_paths_rewritten == 0
        assert summary.tags_added == 0

    def test_second_run_changes_nothing(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "a/model.safetensors"),
            "ref-1",
            loader_path=None,
            tags={"model_type:loras": "automatic", "favourite": "manual"},
        )
        _register(
            session,
            _write(comfy_dirs["input"], "pasted/clip.png"),
            "ref-2",
            loader_path="wrong.png",
            is_missing=True,
        )
        _register(
            session,
            os.path.join(str(comfy_dirs["output"]), "gone.png"),
            "ref-3",
            mtime_ns=1,
        )

        reproject_derived_state()
        after_first = _state_the_reset_could_touch(session)

        second = reproject_derived_state()
        assert _state_the_reset_could_touch(session) == after_first, (
            "a second pass must be indistinguishable from one"
        )
        assert second.loader_paths_rewritten == 0
        assert second.tags_added == 0
        assert second.tags_removed == 0
        assert second.missing_flags_cleared == 0

    def test_statements_chunked_by_bind_param_limit_stay_correct(
        self, session, comfy_dirs
    ):
        for index in range(4):
            _register(
                session,
                _write(comfy_dirs["checkpoints"], f"m{index}.safetensors"),
                f"ref-{index}",
                tags={"model_type:loras": "automatic"},
            )

        with (
            patch("app.assets.database.queries.semantics.MAX_BIND_PARAMS", 2),
            patch("app.assets.database.queries.common.MAX_BIND_PARAMS", 2),
        ):
            reproject_derived_state()

        for index in range(4):
            assert _tags(session, f"ref-{index}") == {
                "models": "automatic",
                "model_type:checkpoints": "automatic",
            }, "chunking by bind-param limit must not drop or duplicate a tag"

    def test_walk_crosses_batch_boundaries(self, session, comfy_dirs):
        for index in range(7):
            _register(
                session,
                _write(comfy_dirs["checkpoints"], f"m{index:02d}.safetensors"),
                f"ref-{index:02d}",
                loader_path=None,
            )

        with patch("app.assets.semantics.reproject_derived._ROWS_PER_TRANSACTION", 2):
            summary = reproject_derived_state()

        assert summary.scanned == 7
        session.expire_all()
        assert all(
            session.get(AssetReference, f"ref-{index:02d}").loader_path
            == f"m{index:02d}.safetensors"
            for index in range(7)
        )


class TestRunner:
    def test_pending_step_runs_and_stamps(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "model.safetensors"),
            "ref-1",
            loader_path=None,
        )

        assert run_pending_semantics_steps() == 1

        session.expire_all()
        assert get_semantics_version(session) == semantics.CURRENT_SEMANTICS_VERSION
        assert session.get(AssetReference, "ref-1").loader_path == "model.safetensors"

    def test_unclassified_row_withholds_the_stamp(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["elsewhere"], "unconfigured.safetensors"),
            "ref-outside",
            loader_path=None,
        )
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "configured.safetensors"),
            "ref-inside",
            loader_path=None,
        )

        assert run_pending_semantics_steps() == 0

        session.expire_all()
        assert get_semantics_version(session) == 0, (
            "skipping a row must not stamp past it -- nothing else repairs one"
        )
        assert (
            session.get(AssetReference, "ref-inside").loader_path
            == "configured.safetensors"
        ), "the rows it could classify are still repaired"

    def test_restoring_a_root_repairs_the_rows_it_had_hidden(
        self, session, comfy_dirs
    ):
        hidden = _write(comfy_dirs["elsewhere"], "unconfigured.safetensors")
        _register(session, hidden, "ref-outside", loader_path=None)

        run_pending_semantics_steps()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("checkpoints", [str(comfy_dirs["checkpoints"])], {".safetensors"}),
                ("loras", [str(comfy_dirs["loras"])], {".safetensors"}),
                ("vae", [str(comfy_dirs["elsewhere"])], {".safetensors"}),
            ],
        ):
            assert run_pending_semantics_steps() == 1

        session.expire_all()
        assert get_semantics_version(session) == 1
        assert (
            session.get(AssetReference, "ref-outside").loader_path
            == "unconfigured.safetensors"
        ), "withholding the stamp is what lets a re-added root get reprojected"

    def test_fully_classified_pass_stamps(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "model.safetensors"),
            "ref-1",
            loader_path=None,
        )

        assert run_pending_semantics_steps() == 1
        session.expire_all()
        assert get_semantics_version(session) == 1

    def test_stamped_database_does_not_walk_again(self, session, comfy_dirs):
        run_pending_semantics_steps()

        with patch(
            "app.assets.semantics.reproject_derived.get_file_backed_references_page",
            side_effect=AssertionError("a stamped database must not be walked"),
        ):
            assert run_pending_semantics_steps() == 0

    def test_stamp_failure_does_not_escape_to_the_caller(self, session, comfy_dirs):
        _register(
            session,
            _write(comfy_dirs["checkpoints"], "model.safetensors"),
            "ref-1",
            loader_path=None,
        )

        with patch(
            "app.assets.semantics.set_semantics_version",
            side_effect=RuntimeError("database is locked"),
        ):
            assert run_pending_semantics_steps() == 0, (
                "the seeder calls this, so a transient lock here must return "
                "rather than abort its whole scan"
            )

        session.expire_all()
        assert get_semantics_version(session) == 0
        assert (
            session.get(AssetReference, "ref-1").loader_path == "model.safetensors"
        ), "the step's own committed work survives a failed stamp"

    def test_stamp_is_not_advanced_when_a_step_raises(self, session, comfy_dirs):
        def _explode(_interrupt_check):
            raise RuntimeError("step failed")

        with patch.object(
            semantics,
            "SEMANTICS_STEPS",
            (SemanticsStep(version=1, description="explodes", apply=_explode),),
        ):
            assert run_pending_semantics_steps() == 0

        session.expire_all()
        assert get_semantics_version(session) == 0
        assert session.get(AssetSemanticsVersion, 1) is None

    def test_stamp_is_not_advanced_when_a_step_is_interrupted(
        self, session, comfy_dirs
    ):
        def _interrupted(_interrupt_check):
            raise SemanticsStepInterrupted("stopped")

        with patch.object(
            semantics,
            "SEMANTICS_STEPS",
            (SemanticsStep(version=1, description="interrupts", apply=_interrupted),),
        ):
            assert run_pending_semantics_steps() == 0

        session.expire_all()
        assert get_semantics_version(session) == 0

    def test_earlier_steps_stay_stamped_when_a_later_one_fails(
        self, session, comfy_dirs
    ):
        def _ok(_interrupt_check):
            return StepResult()

        def _explode(_interrupt_check):
            raise RuntimeError("step failed")

        with patch.object(
            semantics,
            "SEMANTICS_STEPS",
            (
                SemanticsStep(version=1, description="ok", apply=_ok),
                SemanticsStep(version=2, description="explodes", apply=_explode),
            ),
        ):
            assert run_pending_semantics_steps() == 1

        session.expire_all()
        assert get_semantics_version(session) == 1

    def test_steps_below_the_stored_version_are_skipped(self, session, comfy_dirs):
        applied: list[int] = []

        def _record(version):
            def _apply(_interrupt_check):
                applied.append(version)
                return StepResult()

            return _apply

        steps = (
            SemanticsStep(version=1, description="one", apply=_record(1)),
            SemanticsStep(version=2, description="two", apply=_record(2)),
        )
        with patch.object(semantics, "SEMANTICS_STEPS", steps[:1]):
            run_pending_semantics_steps()
        with patch.object(semantics, "SEMANTICS_STEPS", steps):
            run_pending_semantics_steps()

        assert applied == [1, 2]

    def test_interrupted_walk_leaves_committed_work_and_resumes(
        self, session, comfy_dirs
    ):
        for index in range(4):
            _register(
                session,
                _write(comfy_dirs["checkpoints"], f"m{index}.safetensors"),
                f"ref-{index}",
                loader_path=None,
            )

        calls = {"n": 0}

        def _stop_after_first_batch() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        with (
            patch("app.assets.semantics.reproject_derived._ROWS_PER_TRANSACTION", 2),
            patch.object(
                semantics,
                "SEMANTICS_STEPS",
                (
                    SemanticsStep(
                        version=1,
                        description="reproject",
                        apply=reproject_derived_state,
                    ),
                ),
            ),
        ):
            assert run_pending_semantics_steps(_stop_after_first_batch) == 0
            session.expire_all()
            assert get_semantics_version(session) == 0
            assert session.get(AssetReference, "ref-0").loader_path is not None
            assert session.get(AssetReference, "ref-3").loader_path is None

            assert run_pending_semantics_steps() == 1

        session.expire_all()
        assert get_semantics_version(session) == 1
        assert session.get(AssetReference, "ref-3").loader_path == "m3.safetensors"
