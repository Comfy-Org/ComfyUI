from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.api.routes import _build_asset_response, _build_record_response
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.services.schemas import AssetData, AssetDetailResult, ReferenceData

_TS = datetime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def sandboxed_comfy_roots(tmp_path: Path):
    with patch("app.assets.services.path_utils.folder_paths") as fp:
        fp.get_input_directory.return_value = str(tmp_path / "input")
        fp.get_output_directory.return_value = str(tmp_path / "output")
        fp.get_temp_directory.return_value = str(tmp_path / "temp")
        fp.models_dir = str(tmp_path / "models")
        yield tmp_path


def _make_result(
    *,
    ref_id: str = "ref-1",
    name: str = "ComfyUI_temp_abcde_00001_.png",
    file_path: str | None = None,
    mime_type: str | None = "image/png",
    preview_id: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    with_asset: bool = True,
    is_missing: bool = False,
) -> AssetDetailResult:
    ref = ReferenceData(
        id=ref_id,
        name=name,
        file_path=file_path,
        loader_path=None,
        user_metadata=user_metadata,
        preview_id=preview_id,
        created_at=_TS,
        updated_at=_TS,
        last_access_time=_TS,
    )
    asset = (
        AssetData(
            hash="blake3:abc",
            size_bytes=1024,
            mime_type=mime_type,
            is_missing=is_missing,
        )
        if with_asset
        else None
    )
    return AssetDetailResult(ref=ref, asset=asset, tags=tags or [])


def _url(asset_id: str) -> str:
    return f"/api/assets/{asset_id}/content"


@pytest.mark.parametrize(
    "relative",
    [
        "temp/ComfyUI_temp_abcde_00001_.png",
        "output/ComfyUI_00001_.png",
        "input/example.png",
        "models/checkpoints/m.png",
        "output/runs & takes/my shot.png",
    ],
)
def test_preview_url_is_the_content_route_by_id_wherever_the_file_lives(
    sandboxed_comfy_roots: Path, relative: str
):
    resp = _build_asset_response(
        _make_result(name=Path(relative).name, file_path=str(sandboxed_comfy_roots / relative)),
        {},
    )

    assert resp.preview_url == _url("ref-1"), (
        "the URL names the asset, not its root or subfolder, so clients never parse a path"
    )
    assert "?" not in resp.preview_url, "clients tell this form from /api/view?type=... by the query"


def test_a_file_outside_every_root_still_previews(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(_make_result(file_path="/elsewhere/a.png"), {})

    assert resp.preview_url == _url("ref-1"), "the content route serves by id, not by root"


@pytest.mark.parametrize(
    "tags",
    [[], ["input"], ["output"], ["models", "model_type:checkpoints"]],
)
def test_preview_url_does_not_depend_on_tags(
    sandboxed_comfy_roots: Path, tags: list[str]
):
    resp = _build_asset_response(
        _make_result(file_path=str(sandboxed_comfy_roots / "temp" / "a.png"), tags=tags), {}
    )

    assert resp.preview_url == _url("ref-1"), (
        "tags are user-editable; removing one must not destroy the preview"
    )


def test_an_image_is_its_own_preview_id(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(file_path=str(sandboxed_comfy_roots / "output" / "a.png")), {}
    )

    assert (resp.preview_id, resp.preview_url) == ("ref-1", _url("ref-1"))


@pytest.mark.parametrize(
    ("name", "mime_type"),
    [
        ("frame.exr", "image/x-exr"),
        ("frame.exr", None),
        ("frame.exr", "image/png"),  # a client-supplied type
        ("sky.hdr", "image/vnd.radiance"),
        ("sky.hdr", None),
        ("sky.HDR", "image/x-whatever-this-host-says"),
        ("blake3-named-upload", "image/x-exr"),
    ],
)
def test_exr_and_hdr_are_never_their_own_preview(
    sandboxed_comfy_roots: Path, name: str, mime_type: str | None
):
    resp = _build_asset_response(
        _make_result(name=name, file_path=str(sandboxed_comfy_roots / "output" / name), mime_type=mime_type),
        {},
    )

    assert (resp.preview_id, resp.preview_url) == (None, None), (
        "browsers can't display these, so the bytes must never be offered as a preview"
    )


def test_an_exr_with_a_generated_preview_shows_it(sandboxed_comfy_roots: Path):
    result = _make_result(
        name="frame.exr",
        file_path=str(sandboxed_comfy_roots / "output" / "frame.exr"),
        mime_type="image/x-exr",
        preview_id="preview-ref",
    )

    resp = _build_asset_response(result, {"preview-ref": str(sandboxed_comfy_roots / "previews" / "p.webp")})

    assert (resp.preview_id, resp.preview_url) == ("preview-ref", _url("preview-ref"))


def test_preview_id_resolves_through_the_page_lookup(sandboxed_comfy_roots: Path):
    result = _make_result(
        file_path=str(sandboxed_comfy_roots / "models" / "checkpoints" / "m.safetensors"),
        mime_type="application/safetensors",
        preview_id="preview-ref",
    )

    resp = _build_asset_response(
        result, {"preview-ref": str(sandboxed_comfy_roots / "output" / "thumb.png")}
    )

    assert resp.preview_url == _url("preview-ref"), (
        "a nominated preview stands in for content with no visual form"
    )
    assert resp.preview_id == "preview-ref"


@pytest.mark.parametrize(
    ("name", "mime_type"),
    [
        ("notes.txt", "text/plain"),
        ("notes.md", "text/markdown"),
        ("rows.csv", "text/csv"),
        ("page.html", "text/html"),
    ],
)
def test_text_is_previewable(sandboxed_comfy_roots: Path, name: str, mime_type: str):
    resp = _build_asset_response(
        _make_result(
            name=name,
            file_path=str(sandboxed_comfy_roots / "output" / name),
            mime_type=mime_type,
        ),
        {},
    )

    assert resp.preview_url == _url("ref-1"), (
        "text assets are rendered as a snippet from preview_url, so withholding "
        "it leaves that with nothing to fetch; the dangerous members stay safe "
        "because the content route forces them to download, not because they get no URL"
    )


@pytest.mark.parametrize(
    "mime_type",
    ["application/safetensors", "application/gguf", "application/octet-stream"],
)
def test_no_preview_url_for_content_a_browser_cannot_render(
    sandboxed_comfy_roots: Path, mime_type: str
):
    resp = _build_asset_response(
        _make_result(
            name="model.safetensors",
            file_path=str(sandboxed_comfy_roots / "input" / "model.safetensors"),
            mime_type=mime_type,
        ),
        {},
    )

    assert resp.preview_url is None, (
        "content a browser cannot render must not advertise itself as a preview"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.png", "/api/assets/ref-1/content"),
        ("clip.mp4", "/api/assets/ref-1/content"),
        ("model.safetensors", None),
    ],
)
def test_missing_mime_type_falls_back_to_the_path(
    sandboxed_comfy_roots: Path, name: str, expected: str | None
):
    resp = _build_asset_response(
        _make_result(
            name=name, file_path=str(sandboxed_comfy_roots / "temp" / name), mime_type=None
        ),
        {},
    )

    assert resp.preview_url == expected, (
        "a previewable file must not lose its preview just because the scan "
        "that found it recorded no mime type"
    )


@pytest.mark.parametrize(
    ("name", "stored_filename", "expected"),
    [
        ("untitled", "shot.png", "/api/assets/ref-1/content"),
        ("shot.png", "weights.safetensors", None),
    ],
)
def test_previewability_follows_the_path_not_the_editable_name(
    sandboxed_comfy_roots: Path, name: str, stored_filename: str, expected: str | None
):
    resp = _build_asset_response(
        _make_result(
            name=name,
            file_path=str(sandboxed_comfy_roots / "temp" / stored_filename),
            mime_type=None,
        ),
        {},
    )

    assert resp.preview_url == expected, (
        "name is editable through PUT /api/assets/{id}, so deriving "
        "previewability from it would let a rename create or destroy a preview "
        "without the bytes changing"
    )


def test_mime_type_parameters_do_not_defeat_the_media_check(
    sandboxed_comfy_roots: Path,
):
    resp = _build_asset_response(
        _make_result(
            file_path=str(sandboxed_comfy_roots / "temp" / "a.png"),
            mime_type="IMAGE/PNG; charset=binary",
        ),
        {},
    )

    assert resp.preview_url == _url("ref-1")


def test_no_preview_url_without_a_file_path(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(_make_result(file_path=None), {})

    assert resp.preview_url is None, (
        "an API-created reference has no path, so it has no bytes to preview"
    )


def test_no_preview_url_without_content(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(
            file_path=str(sandboxed_comfy_roots / "temp" / "a.png"), with_asset=False
        ),
        {},
    )

    assert resp.preview_url is None, "no asset row means there is nothing to preview"


def test_no_self_preview_url_when_content_is_missing(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(
            name="gone.png",
            file_path=str(sandboxed_comfy_roots / "output" / "gone.png"),
            mime_type="image/png",
            is_missing=True,
        ),
        {},
    )

    assert (resp.preview_id, resp.preview_url) == (None, None), (
        "a missing-content record must not advertise a preview of its own bytes; "
        "those bytes are gone, so the content route would 404"
    )


def test_missing_content_still_shows_a_nominated_preview(sandboxed_comfy_roots: Path):
    result = _make_result(
        name="gone.safetensors",
        file_path=str(sandboxed_comfy_roots / "models" / "checkpoints" / "gone.safetensors"),
        mime_type="application/safetensors",
        preview_id="preview-ref",
        is_missing=True,
    )

    resp = _build_asset_response(
        result, {"preview-ref": str(sandboxed_comfy_roots / "output" / "thumb.png")}
    )

    assert resp.preview_url == _url("preview-ref"), (
        "suppression targets self-content only; a nominated (live) preview still "
        "stands in for a missing record"
    )


def test_record_response_shows_self_preview_for_live_content(
    sandboxed_comfy_roots: Path, session
):
    content = create_content(
        session, path=str(sandboxed_comfy_roots / "output" / "live.png")
    )
    record = create_record(
        session, content_id=content.id, name="live.png", mime_type="image/png"
    )
    session.commit()

    resp = _build_record_response(record, [], {})

    assert (resp.preview_id, resp.preview_url) == (record.id, _url(record.id)), (
        "a live record still previews its own bytes — the positive control that "
        "proves the missing-case suppression is what withholds the URL"
    )


def test_record_response_falls_back_to_the_path_when_mime_type_is_empty(
    sandboxed_comfy_roots: Path, session
):
    content = create_content(
        session, path=str(sandboxed_comfy_roots / "output" / "live.png")
    )
    record = create_record(session, content_id=content.id, name="untitled", mime_type=None)
    session.commit()

    resp = _build_record_response(record, [], {})

    assert resp.preview_url == _url(record.id)


def test_record_response_has_no_self_preview_url_when_content_is_missing(
    sandboxed_comfy_roots: Path, session
):
    content = create_content(
        session, path=str(sandboxed_comfy_roots / "output" / "gone.png")
    )
    record = create_record(
        session, content_id=content.id, name="gone.png", mime_type="image/png"
    )
    mark_content_missing(session, content.id)
    session.commit()

    resp = _build_record_response(record, ["missing"], {})

    assert resp.preview_url is None, (
        "the list surface must also withhold a missing record's self-preview URL"
    )
