from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.api.routes import _build_asset_response
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
        AssetData(hash="blake3:abc", size_bytes=1024, mime_type=mime_type)
        if with_asset
        else None
    )
    return AssetDetailResult(ref=ref, asset=asset, tags=tags or [])


@pytest.mark.parametrize(
    ("root", "relative"),
    [
        ("temp", "ComfyUI_temp_abcde_00001_.png"),
        ("output", "ComfyUI_00001_.png"),
        ("input", "example.png"),
    ],
)
def test_every_view_root_gets_a_preview_url(
    sandboxed_comfy_roots: Path, root: str, relative: str
):
    resp = _build_asset_response(
        _make_result(name=relative, file_path=str(sandboxed_comfy_roots / root / relative)),
        {},
    )

    assert resp.preview_url == f"/api/view?type={root}&filename={relative}", (
        f"a file in {root} must get a preview URL; temp is the one the old "
        f"tag chain fell off the end of"
    )


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

    assert resp.preview_url == "/api/view?type=temp&filename=a.png", (
        "tags are user-editable; removing one must not destroy the preview"
    )


def test_preview_url_does_not_depend_on_the_metadata_filename(
    sandboxed_comfy_roots: Path,
):
    resp = _build_asset_response(
        _make_result(
            file_path=str(sandboxed_comfy_roots / "output" / "a.png"), user_metadata=None
        ),
        {},
    )

    assert resp.preview_url == "/api/view?type=output&filename=a.png", (
        "a reference carrying no metadata filename must still get a preview"
    )


def test_subfolder_is_split_out_and_both_halves_encoded(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(
            name="my shot.png",
            file_path=str(sandboxed_comfy_roots / "output" / "runs & takes" / "my shot.png"),
        ),
        {},
    )

    assert resp.preview_url == (
        "/api/view?type=output&filename=my%20shot.png&subfolder=runs%20%26%20takes"
    ), "an unencoded & or space in the path would break the query string"


def test_preview_id_resolves_through_the_page_lookup(sandboxed_comfy_roots: Path):
    result = _make_result(
        file_path=str(sandboxed_comfy_roots / "models" / "checkpoints" / "m.safetensors"),
        mime_type="application/safetensors",
        preview_id="preview-ref",
    )

    resp = _build_asset_response(
        result, {"preview-ref": str(sandboxed_comfy_roots / "output" / "thumb.png")}
    )

    assert resp.preview_url == "/api/view?type=output&filename=thumb.png", (
        "a nominated preview stands in for content with no visual form"
    )
    assert resp.preview_id == "preview-ref"


def test_unresolvable_preview_id_yields_no_url(sandboxed_comfy_roots: Path):
    result = _make_result(
        file_path=str(sandboxed_comfy_roots / "output" / "a.png"), preview_id="gone"
    )

    resp = _build_asset_response(result, {})

    assert resp.preview_url is None, (
        "a preview absent from the lookup is soft-deleted, invisible or "
        "path-less, so advertising it would promise a URL that 404s - and the "
        "asset's own bytes are a different picture, not a degraded one"
    )


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

    assert resp.preview_url == f"/api/view?type=output&filename={name}", (
        "text assets are rendered as a snippet from preview_url, so withholding "
        "it leaves that with nothing to fetch; the dangerous members stay safe "
        "because /api/view forces them to download, not because they get no URL"
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
        ("shot.png", "/api/view?type=temp&filename=shot.png"),
        ("clip.mp4", "/api/view?type=temp&filename=clip.mp4"),
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
        ("untitled", "shot.png", "/api/view?type=temp&filename=shot.png"),
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

    assert resp.preview_url == "/api/view?type=temp&filename=a.png"


def test_no_preview_url_for_a_model(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(
            name="m.png",
            file_path=str(sandboxed_comfy_roots / "models" / "checkpoints" / "m.png"),
        ),
        {},
    )

    assert resp.preview_url is None, (
        "models is not a root /api/view can address, whatever the file is"
    )


def test_no_preview_url_for_a_path_outside_every_root(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(_make_result(file_path="/elsewhere/a.png"), {})

    assert resp.preview_url is None, (
        "/api/view cannot address a file outside the roots it serves"
    )


def test_no_preview_url_without_a_file_path(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(_make_result(file_path=None), {})

    assert resp.preview_url is None, (
        "an API-created reference has no path, so no view URL can be derived"
    )


def test_no_preview_url_without_content(sandboxed_comfy_roots: Path):
    resp = _build_asset_response(
        _make_result(
            file_path=str(sandboxed_comfy_roots / "temp" / "a.png"), with_asset=False
        ),
        {},
    )

    assert resp.preview_url is None, "no asset row means there is nothing to preview"
