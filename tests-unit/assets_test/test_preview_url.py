import contextlib
import json
import re
import uuid

import requests


def test_preview_url_serves_the_asset(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-url-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.png"
    data = make_asset_bytes(name, 2048)

    body = asset_factory(name, ["output", "unit-tests", scope], {}, data)

    assert re.fullmatch(r"/api/view\?type=output&filename=[^&]+", body["preview_url"]), (
        f"unexpected preview URL shape: {body['preview_url']!r}"
    )

    r = http.get(api_base + body["preview_url"], timeout=120)
    assert r.status_code == 200, r.text
    assert r.content == data, "the preview URL must serve the asset's own bytes"


def test_preview_url_honours_range_requests(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-range-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.png"
    data = make_asset_bytes(name, 2048)

    body = asset_factory(name, ["output", "unit-tests", scope], {}, data)

    r = http.get(
        api_base + body["preview_url"], headers={"Range": "bytes=10-109"}, timeout=120
    )
    assert r.status_code == 206, (
        f"expected a partial response, got {r.status_code}: native <video>/<audio> "
        f"seek by byte range and will not play a source that ignores it"
    )
    assert r.content == data[10:110], "the served range must be the requested one"


def test_preview_url_needs_no_user_header(
    api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-anon-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.png"
    data = make_asset_bytes(name, 1024)

    body = asset_factory(name, ["input", "unit-tests", scope], {}, data)

    with requests.Session() as bare:
        r = bare.get(api_base + body["preview_url"], timeout=120)

    assert r.status_code == 200, (
        f"a browser fetching <img src> cannot attach a Comfy-User header, so the "
        f"preview URL must resolve without one; got {r.status_code}: {r.text}"
    )
    assert r.content == data


def test_preview_url_survives_tag_removal(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-tags-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.png"
    data = make_asset_bytes(name, 2048)

    body = asset_factory(name, ["input", "unit-tests", scope], {}, data)
    aid = body["id"]
    preview_url = body["preview_url"]
    assert preview_url, "an uploaded image starts out with a preview"

    r = http.delete(
        f"{api_base}/api/assets/{aid}/tags", json={"tags": ["input"]}, timeout=120
    )
    assert r.status_code == 200, r.text

    after = http.get(f"{api_base}/api/assets/{aid}", timeout=120).json()
    assert "input" not in after["tags"]
    assert after["preview_url"] == preview_url, (
        "dropping the tag that used to select the view type must not take the "
        "preview with it"
    )
    assert http.get(api_base + preview_url, timeout=120).status_code == 200


def test_preview_url_is_the_nominated_preview_when_one_is_set(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-id-{uuid.uuid4().hex[:6]}"
    thumb_name = f"{scope}_thumb.png"
    thumb_data = make_asset_bytes(thumb_name, 1024)
    thumb = asset_factory(thumb_name, ["input", "unit-tests", scope], {}, thumb_data)

    model_name = f"{scope}.safetensors"
    files = {"file": (model_name, make_asset_bytes(model_name, 2048), "application/octet-stream")}
    form_data = {
        "tags": json.dumps(["models", "model_type:checkpoints", "unit-tests", scope]),
        "name": model_name,
        "preview_id": thumb["id"],
    }
    r = http.post(api_base + "/api/assets", files=files, data=form_data, timeout=120)
    model = r.json()
    assert r.status_code in (200, 201), model

    try:
        assert model["preview_id"] == thumb["id"]
        assert model["preview_url"] == thumb["preview_url"], (
            "a nominated preview stands in for content with no visual form"
        )
        got = http.get(api_base + model["preview_url"], timeout=120)
        assert got.status_code == 200, got.text
        assert got.content == thumb_data
    finally:
        with contextlib.suppress(Exception):
            http.delete(f"{api_base}/api/assets/{model['id']}", timeout=30)


def test_soft_deleted_preview_is_not_advertised(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-gone-{uuid.uuid4().hex[:6]}"
    thumb_name = f"{scope}_thumb.png"
    thumb = asset_factory(
        thumb_name, ["input", "unit-tests", scope], {}, make_asset_bytes(thumb_name, 1024)
    )

    parent_name = f"{scope}_parent.png"
    parent = asset_factory(
        parent_name, ["input", "unit-tests", scope], {}, make_asset_bytes(parent_name, 1024)
    )
    r = http.put(
        f"{api_base}/api/assets/{parent['id']}",
        json={"preview_id": thumb["id"]},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    assert r.json()["preview_url"] == thumb["preview_url"]

    assert http.delete(f"{api_base}/api/assets/{thumb['id']}", timeout=30).status_code in (200, 204)

    after = http.get(f"{api_base}/api/assets/{parent['id']}", timeout=120).json()
    assert after["preview_id"] == thumb["id"]
    assert after.get("preview_url") is None, (
        "soft delete leaves the parent's preview_id pointing at it, so the "
        "response has to drop the URL rather than promise one that 404s"
    )


def test_no_preview_url_for_content_a_browser_cannot_render(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-model-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.safetensors"

    body = asset_factory(
        name,
        ["models", "model_type:checkpoints", "unit-tests", scope],
        {},
        make_asset_bytes(name, 2048),
    )

    assert body.get("preview_url") is None, (
        "model weights have no preview of their own"
    )

    listed = http.get(
        api_base + "/api/assets", params={"include_tags": scope}, timeout=120
    ).json()["assets"]
    assert [a.get("preview_url") for a in listed] == [None], (
        "the list route must withhold it too, not just the detail route"
    )


def test_text_asset_gets_a_preview_url_that_serves_its_content(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    scope = f"preview-text-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.txt"
    data = b"line one\nline two\n" + make_asset_bytes(name, 256)

    body = asset_factory(name, ["output", "unit-tests", scope], {}, data)

    assert body.get("preview_url"), (
        "text assets are rendered as a snippet fetched from preview_url"
    )
    r = http.get(api_base + body["preview_url"], timeout=120)
    assert r.status_code == 200, r.text
    assert r.content == data


def test_dangerous_text_preview_is_still_forced_to_download(
    http: requests.Session, api_base: str, asset_factory
):
    scope = f"preview-html-{uuid.uuid4().hex[:6]}"
    name = f"{scope}.html"
    data = f"<html><script>alert('{scope}')</script></html>".encode()

    body = asset_factory(name, ["output", "unit-tests", scope], {}, data)

    assert body.get("preview_url"), "text/html matches the previewable prefix"

    r = http.get(api_base + body["preview_url"], timeout=120)
    r.content
    assert r.status_code == 200
    ct = r.headers.get("Content-Type", "").lower()
    cd = r.headers.get("Content-Disposition", "").lower()
    assert ct.startswith("application/octet-stream"), (
        f"admitting text/ to the previewable set must not let HTML render "
        f"inline in the app origin; got {ct!r}"
    )
    assert "attachment" in cd, f"expected a forced download, got {cd!r}"
