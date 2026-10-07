import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest


from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.manager import AssetsEnabled
from app.assets.scanner import content_ids_outside_prefixes, get_owned_prefixes
from app.assets.services.asset_management import delete_asset_reference
from app.assets.services.ingest import register_cached_output, register_file_in_place
from comfy_execution.asset_enrichment import generate_output_previews, register_executed_outputs

from .preview_helpers import write_exr




def _record(session, path: Path, *, tags=(), preview_id=None, mime_type=None) -> Asset:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(path.name.encode())
    content = session.query(AssetContent).filter_by(path=str(path)).one_or_none() or create_content(session, str(path))
    record = create_record(session, content.id, path.name, mime_type=mime_type, tags=list(tags))
    record.preview_id = preview_id
    session.commit()
    return record


# --- startup prune ---


def test_the_startup_prune_never_marks_previews_missing(session, roots):
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    owned_without_previews = [p for p in get_owned_prefixes() if not p.endswith("previews")]

    assert preview.content_id in content_ids_outside_prefixes(session, owned_without_previews), "control"
    assert preview.content_id not in content_ids_outside_prefixes(session, get_owned_prefixes())


# --- deleting a parent ---


def test_deleting_a_parent_removes_its_unshared_preview(session, mock_create_session, roots):
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    preview_id = preview.id
    parent = _record(session, roots / "output" / "a.exr", preview_id=preview_id)

    assert delete_asset_reference(parent.id)

    session.expire_all()
    assert session.get(Asset, preview_id) is None
    assert not (roots / "previews" / "p.webp").exists()
    assert session.query(AssetContent).filter_by(path=str(roots / "previews" / "p.webp")).count() == 0


def test_a_preview_another_asset_links_is_kept(session, mock_create_session, roots):
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    parent = _record(session, roots / "output" / "a.exr", preview_id=preview.id)
    _record(session, roots / "output" / "b.exr", preview_id=preview.id)

    delete_asset_reference(parent.id)

    session.expire_all()
    assert session.get(Asset, preview.id) is not None
    assert (roots / "previews" / "p.webp").exists()


def test_a_preview_file_another_record_shares_is_kept(session, mock_create_session, roots):
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    preview_id = preview.id
    twin = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    parent = _record(session, roots / "output" / "a.exr", preview_id=preview_id)
    _record(session, roots / "output" / "b.exr", preview_id=twin.id)

    delete_asset_reference(parent.id)

    session.expire_all()
    assert session.get(Asset, preview_id) is None
    assert (roots / "previews" / "p.webp").exists(), "the twin record still uses the file"


def test_a_nominated_preview_that_is_not_a_preview_asset_is_kept(session, mock_create_session, roots):
    thumb = _record(session, roots / "output" / "thumb.png", tags=["output"])
    parent = _record(session, roots / "output" / "model.glb", preview_id=thumb.id)

    delete_asset_reference(parent.id)

    session.expire_all()
    assert session.get(Asset, thumb.id) is not None
    assert (roots / "output" / "thumb.png").exists(), "Core never deletes a user's own file"


# --- reusing a preview for the same bytes ---


def test_a_cached_replay_reuses_a_live_preview_from_any_sibling(session, mock_create_session, roots):
    frame = roots / "output" / "f.exr"
    _record(session, frame)  # oldest: made before previews existed
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    _record(session, frame, preview_id=preview.id)

    replayed = register_cached_output(str(frame), "job-2")

    assert replayed.preview_id == preview.id


def test_a_dead_preview_is_not_reused(session, mock_create_session, roots):
    frame = roots / "output" / "f.exr"
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    _record(session, frame, preview_id=preview.id)
    mark_content_missing(session, preview.content_id)
    session.commit()

    replayed = register_cached_output(str(frame), "job-2")

    assert replayed.preview_id is None


def test_a_repeat_upload_of_the_same_bytes_reuses_the_preview(session, mock_create_session, roots):
    upload = roots / "input" / "f.exr"
    upload.write_bytes(b"exr bytes")
    first = register_file_in_place(str(upload), "f.exr", ["input"], content_written=True)
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    session.get(Asset, first.ref.id).preview_id = preview.id
    session.commit()

    second = register_file_in_place(str(upload), "f.exr", ["input"], content_written=False)

    assert second.ref.id != first.ref.id
    assert second.ref.preview_id == preview.id


# --- the output hook ---


class _Args:
    enable_assets = True
    enable_asset_hashing = False


def _ui(*names: str) -> dict:
    return {"images": [{"filename": n, "subfolder": "", "type": "output"} for n in names]}


def test_outputs_get_their_preview_id(session, mock_create_session, roots):
    from utils.mime_types import init_mime_types

    init_mime_types()
    write_exr(roots / "output" / "frame.exr", 64, 48)
    (roots / "output" / "still.png").write_bytes(b"png")
    (roots / "output" / "notes.txt").write_bytes(b"txt")

    enriched = register_executed_outputs(_ui("frame.exr", "still.png", "notes.txt"), "job", AssetsEnabled(_Args()))
    asyncio.run(generate_output_previews(enriched))

    exr, png, txt = enriched["images"]
    session.expire_all()
    assert exr["preview_id"] == session.get(Asset, exr["id"]).preview_id != exr["id"], "a generated, linked preview"
    assert png["preview_id"] == png["id"], "a displayable image is its own preview"
    assert "preview_id" not in txt, "only images carry preview_id"


def test_an_exr_whose_preview_fails_has_no_preview_id(session, mock_create_session, roots):
    from utils.mime_types import init_mime_types

    init_mime_types()
    (roots / "output" / "broken.exr").write_bytes(b"not an exr")

    enriched = register_executed_outputs(_ui("broken.exr"), "job", AssetsEnabled(_Args()))
    asyncio.run(generate_output_previews(enriched))

    assert "preview_id" not in enriched["images"][0], "never itself, and nothing generated"


def test_with_assets_off_entries_are_unchanged(roots):
    from app.assets.manager import NoAssets

    class _Off:
        enable_assets = False
        enable_asset_hashing = False

    (roots / "output" / "still.png").write_bytes(b"png")
    ui = _ui("still.png")

    assert register_executed_outputs(ui, "job", NoAssets(_Off())) == ui


def test_generation_is_only_asked_for_types_with_a_generator(session, mock_create_session, roots):
    (roots / "output" / "still.png").write_bytes(b"png")
    enriched = register_executed_outputs(_ui("still.png"), "job", AssetsEnabled(_Args()))
    del enriched["images"][0]["preview_id"]

    with patch("app.assets.previews.generate_previews") as generate:
        asyncio.run(generate_output_previews(enriched))

    generate.assert_not_called()


def test_a_preview_tagged_asset_outside_previews_is_not_cascaded(session, mock_create_session, roots):
    user_file = roots / "output" / "my-preview.png"
    tagged = _record(session, user_file, tags=["preview"])
    tagged_id = tagged.id
    parent = _record(session, roots / "output" / "a.exr", preview_id=tagged_id)

    delete_asset_reference(parent.id)

    session.expire_all()
    assert session.get(Asset, tagged_id) is not None, "a user's own asset, whatever its tags"
    assert user_file.exists()


def test_a_self_linked_sibling_is_not_reused(session, mock_create_session, roots):
    frame = roots / "output" / "f.exr"
    sibling = _record(session, frame)
    sibling.preview_id = sibling.id
    session.commit()

    assert register_cached_output(str(frame), "job-2").preview_id is None


def test_a_hand_picked_preview_is_not_reused(session, mock_create_session, roots):
    frame = roots / "output" / "f.exr"
    picked = _record(session, roots / "output" / "thumb.png", tags=["output"])
    _record(session, frame, preview_id=picked.id)

    assert register_cached_output(str(frame), "job-2").preview_id is None


def test_a_preview_whose_file_was_removed_is_not_reused(session, mock_create_session, roots):
    frame = roots / "output" / "f.exr"
    preview = _record(session, roots / "previews" / "p.webp", tags=["preview"])
    _record(session, frame, preview_id=preview.id)
    (roots / "previews" / "p.webp").unlink()

    assert register_cached_output(str(frame), "job-2").preview_id is None


def test_a_cached_replay_drops_a_stale_preview_id(session, mock_create_session, roots):
    from comfy_execution.asset_enrichment import register_cached_outputs

    write_exr(roots / "output" / "frame.exr", 8, 8)
    wrapper = {"meta": {}, "output": {"images": [
        {"filename": "frame.exr", "subfolder": "", "type": "output", "id": "old", "preview_id": "old-preview"},
    ]}}
    register_executed_outputs(_ui("frame.exr"), "seed", AssetsEnabled(_Args()))

    entry = register_cached_outputs(wrapper, "replay", AssetsEnabled(_Args()))["output"]["images"][0]

    assert entry["id"] != "old"
    assert "preview_id" not in entry, "no reusable preview, and a replay never generates"


def test_a_non_file_ui_entry_with_an_id_does_not_fail_the_node(session, mock_create_session, roots):
    ui = {"items": [{"id": "row-1", "label": "Result"}]}

    asyncio.run(generate_output_previews(ui))

    assert ui == {"items": [{"id": "row-1", "label": "Result"}]}


@pytest.mark.parametrize(
    ("tags", "root"),
    [(["preview"], "previews"), (["input", "preview"], "input"), (["output", "preview"], "output")],
)
def test_preview_is_a_destination_only_on_its_own(roots, tags, root):
    from app.assets.services.path_utils import resolve_destination_from_tags

    base, _ = resolve_destination_from_tags(tags)

    assert base == str(roots / root)


def test_preview_is_a_reserved_tag():
    from app.assets.api.routes import SystemTagForbiddenError, _reject_system_tags

    with pytest.raises(SystemTagForbiddenError):
        _reject_system_tags(["preview"])
