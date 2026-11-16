"""End-to-end tests of /api/v2 against a real `main.py` server (see api_v2_server.py)."""
import io
import os
import uuid

from blake3 import blake3
from PIL import Image
import pytest
import requests

from api_v2_server import TIMEOUT, V2Server, asset_ref, png_bytes, running, sse_frames, upload, wait_job

pytestmark = pytest.mark.execution

ID = str(uuid.uuid4())
HASH = "blake3:" + "0" * 64
CONTRACT_REQUESTS = [  # the 15 (method, path) pairs of openapi-v2.yaml, placeholders filled
    ("GET", "/api/v2/health"),
    ("POST", "/api/v2/assets"),
    ("POST", "/api/v2/assets/from-hash"),
    ("POST", "/api/v2/assets/from-path"),
    ("HEAD", f"/api/v2/assets/by-hash/{HASH}"),
    ("GET", f"/api/v2/assets/{ID}"),
    ("GET", f"/api/v2/assets/{ID}/content"),
    ("DELETE", f"/api/v2/assets/{ID}"),
    ("GET", "/api/v2/jobs"),
    ("POST", "/api/v2/jobs"),
    ("GET", f"/api/v2/jobs/{ID}"),
    ("POST", f"/api/v2/jobs/{ID}/cancel"),
    ("GET", f"/api/v2/jobs/{ID}/workflow"),
    ("GET", f"/api/v2/jobs/{ID}/logs"),
    ("GET", f"/api/v2/jobs/{ID}/events"),
]
SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-store, no-transform", "X-Accel-Buffering": "no"}


@pytest.fixture(scope="module", params=[(), ("--enable-asset-hashing",)], ids=["hashing-off", "hashing-on"])
def v2(request, tmp_path_factory):
    with running(tmp_path_factory.mktemp("v2"), "--enable-assets", *request.param) as server:
        yield server


@pytest.fixture(scope="module")
def disabled(tmp_path_factory):
    with running(tmp_path_factory.mktemp("v2-disabled"), "--disable-assets") as server:
        yield server


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    with running(tmp_path_factory.mktemp("v2-multi"), "--enable-assets", "--multi-user") as server:
        yield server


def hashing(server: V2Server) -> bool:
    return "--enable-asset-hashing" in server.extra_args


def unique_png() -> bytes:
    return png_bytes(tuple(os.urandom(3)))


def call(method: str, server: V2Server, path: str, **kwargs) -> requests.Response:
    return requests.request(method, server.url(path), timeout=TIMEOUT, **kwargs)


def error_code(resp: requests.Response) -> str:
    return resp.json()["error"]["code"]


def submit(server: V2Server, workflow: dict, headers: dict | None = None, **body) -> requests.Response:
    return call("POST", server, "/api/v2/jobs", json={"workflow": workflow, **body}, headers=headers)


def sleep_graph(asset_id: str, seconds: float) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": asset_ref(asset_id)}},
        "2": {"class_type": "TestSleep", "inputs": {"value": ["1", 0], "seconds": seconds}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "e2e"}},
    }


def preview_graph(asset_id: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": asset_ref(asset_id)}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "e2e"}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }


@pytest.fixture
def asset(v2) -> dict:
    resp = upload(v2, unique_png(), "input/e2e/source.png")
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
def finished_job(v2, asset) -> dict:
    resp = submit(v2, preview_graph(asset["id"]), extra_data={"api_key_comfy_org": "comfyui-test"})
    assert resp.status_code == 201, resp.text
    return wait_job(v2, resp.json()["id"])


def test_disabled_refuses_every_contract_route(disabled):
    for method, path in CONTRACT_REQUESTS:
        resp = call(method, disabled, path)
        sibling = call(method, disabled, "/api/v2/__nope__")
        # No v2 route is registered when assets are disabled, so ComfyUI's own routing answers exactly as for an unknown sibling path (404 for GET/HEAD, 405 for POST/DELETE via the static '/' GET catch-all) and no v2 handler emits its JSON envelope.
        assert resp.status_code == sibling.status_code, (method, path)
        assert resp.status_code in {404, 405}, (method, path)
        assert '"error"' not in resp.text, (method, path, resp.text)
    assert len(CONTRACT_REQUESTS) == 15


def test_health(v2):
    resp = call("GET", v2, "/api/v2/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_from_path_is_not_implemented_and_registers_nothing(v2):
    data = unique_png()
    (v2.input_dir / "from-path.png").write_bytes(data)
    before = call("GET", v2, "/api/assets", params={"limit": 500}).json()["total"]
    resp = call("POST", v2, "/api/v2/assets/from-path", json={"path": str(v2.input_dir / "from-path.png")})
    assert resp.status_code == 501
    assert error_code(resp) == "not_implemented"
    assert call("GET", v2, "/api/assets", params={"limit": 500}).json()["total"] == before
    by_hash = call("HEAD", v2, f"/api/v2/assets/by-hash/blake3:{blake3(data).hexdigest()}")
    assert by_hash.status_code == 404


def test_asset_upload_dedupe_hash_and_content_ranges(v2):
    data = unique_png()
    first = upload(v2, data, "input/e2e/upload.png")
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["created_new"] is True
    assert body["hash"] == f"blake3:{blake3(data).hexdigest()}"
    again = upload(v2, data, "input/e2e/upload.png")
    if hashing(v2):
        assert again.status_code == 200, again.text
        assert again.json()["created_new"] is False
    else:
        assert again.status_code == 201, again.text
        assert again.json()["created_new"] is True
    assert again.json()["hash"] == body["hash"] and again.json()["id"] != body["id"]
    assert again.json()["file_path"] == body["file_path"] == f"{blake3(data).hexdigest()}.png"
    assert (v2.input_dir / body["file_path"]).read_bytes() == data
    by_hash = call("HEAD", v2, f"/api/v2/assets/by-hash/{body['hash']}")
    from_hash = call("POST", v2, "/api/v2/assets/from-hash", json={"hash": body["hash"]})
    if hashing(v2):
        assert by_hash.status_code == 200
        assert from_hash.status_code == 200, from_hash.text
        assert from_hash.json()["hash"] == body["hash"] and from_hash.json()["created_new"] is False
    else:
        assert by_hash.status_code == 404
        assert from_hash.status_code == 404 and error_code(from_hash) == "blob_not_found"
    for asset_id in (body["id"], again.json()["id"]):
        content = call("GET", v2, f"/api/v2/assets/{asset_id}/content")
        assert content.status_code == 200 and content.content == data
    ranged = call("GET", v2, f"/api/v2/assets/{body['id']}/content", headers={"Range": "bytes=2-5"})
    assert ranged.status_code == 206 and ranged.content == data[2:6]
    assert ranged.headers["Content-Range"] == f"bytes 2-5/{len(data)}"
    beyond = call("GET", v2, f"/api/v2/assets/{body['id']}/content", headers={"Range": f"bytes={len(data) + 10}-"})
    assert beyond.status_code == 416


@pytest.mark.parametrize("file_path", ["models/checkpoints/e2e/model.safetensors", "checkpoints/model.safetensors"])
def test_model_path_upload_lands_in_checkpoints(v2, file_path):
    data = os.urandom(64)
    resp = upload(v2, data, file_path)
    assert resp.status_code == 201, resp.text
    stored = resp.json()["file_path"]
    assert stored == f"{blake3(data).hexdigest()}.safetensors"
    assert (v2.root / "models" / "checkpoints" / stored).read_bytes() == data
    assert stored in call("GET", v2, "/models/checkpoints").json()


def test_output_bytes_uploaded_as_checkpoint_get_their_own_copy(v2):
    data = os.urandom(64)
    stored = f"{blake3(data).hexdigest()}.safetensors"
    output = upload(v2, data, "output/e2e.safetensors")
    assert output.status_code == 201, output.text
    model = upload(v2, data, "checkpoints/e2e.safetensors")
    assert model.status_code == 201 and model.json()["created_new"] is True, model.text
    assert (v2.root / "out" / stored).read_bytes() == (v2.root / "models" / "checkpoints" / stored).read_bytes() == data
    assert model.json()["file_path"] == stored and stored in call("GET", v2, "/models/checkpoints").json()


def test_job_outputs_are_downloadable_assets(v2, finished_job):
    assert finished_job["status"] == "succeeded", finished_job
    by_node = {o["node_id"]: o for o in finished_job["outputs"]}
    saved, preview = by_node["2"], by_node["3"]
    assert saved["type"] == preview["type"] == "image"
    png = call("GET", v2, f"/api/v2/assets/{saved['id']}/content")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG\r\n\x1a\n")
    Image.open(io.BytesIO(png.content)).verify()
    for output in (saved, preview):
        detail = call("GET", v2, f"/api/v2/assets/{output['id']}")
        assert detail.status_code == 200
        assert detail.json()["job_id"] == finished_job["id"]
    assert (v2.root / "temp" / "temp" / preview["name"]).is_file()  # ComfyUI's temp dir is <--temp-directory>/temp
    assert (v2.root / "out" / saved["name"]).is_file()


def test_workflow_endpoint_redacts_secrets(v2, finished_job):
    resp = call("GET", v2, f"/api/v2/jobs/{finished_job['id']}/workflow")
    assert resp.status_code == 200
    body = resp.json()
    assert body["format"] == "api"
    image = body["workflow"]["1"]["inputs"]["image"]
    assert isinstance(image, str) and image.endswith(".png")
    assert "api_key_comfy_org" not in resp.text and "comfyui-test" not in resp.text


def test_logs_finished(v2, finished_job):
    resp = call("GET", v2, f"/api/v2/jobs/{finished_job['id']}/logs")
    assert resp.status_code == 204
    assert resp.headers["No-Log-Reason"] == "finished"


def test_sse_live_stream_then_reconnect(v2, asset):
    job = submit(v2, sleep_graph(asset["id"], 3)).json()
    events = f"/api/v2/jobs/{job['id']}/events"
    with requests.get(v2.url(events), stream=True, timeout=TIMEOUT) as live:
        assert live.status_code == 200
        for name, value in SSE_HEADERS.items():
            assert live.headers[name] == value
        frames = sse_frames(live)
    names = [name for name, _ in frames]
    assert names[0] == "status", frames
    assert frames[-1] == ("status", {"status": "succeeded"}), frames
    assert "output" in names[:-1], frames
    assert not any(n == "status" and d["status"] in {"succeeded", "failed", "canceled", "expired"} for n, d in frames[:-1])
    assert "error" not in names
    with requests.get(v2.url(events), stream=True, timeout=TIMEOUT) as again:
        assert again.status_code == 200
        assert sse_frames(again) == [("status", {"status": "succeeded"})]


def test_cancel(v2, asset):
    job = submit(v2, sleep_graph(asset["id"], 5)).json()
    wait_job(v2, job["id"], {"running"}, timeout=30)
    resp = call("POST", v2, f"/api/v2/jobs/{job['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"canceling", "canceled"}
    assert wait_job(v2, job["id"], timeout=30)["status"] == "canceled"
    assert call("POST", v2, f"/api/v2/jobs/{uuid.uuid4()}/cancel").status_code == 404


def test_cancel_succeeded_job_is_noop(v2, finished_job):
    resp = call("POST", v2, f"/api/v2/jobs/{finished_job['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "succeeded"


def test_idempotency_key_reuse(v2, asset):
    key = {"Idempotency-Key": f"e2e-{uuid.uuid4()}"}
    assert submit(v2, preview_graph(asset["id"]), headers=key).status_code == 201
    again = submit(v2, preview_graph(asset["id"]), headers=key)
    assert again.status_code == 422
    assert error_code(again) == "idempotency_key_reuse"


def test_labels_echoed_and_filterable(v2, asset):
    resp = submit(v2, preview_graph(asset["id"]), metadata={"client": "acme"})
    assert resp.status_code == 201
    job = resp.json()
    assert job["metadata"] == {"client": "acme"}
    assert wait_job(v2, job["id"])["metadata"] == {"client": "acme"}
    listed = call("GET", v2, "/api/v2/jobs", params={"metadata[client]": "acme"})
    assert listed.status_code == 200
    assert job["id"] in [j["id"] for j in listed.json()["jobs"]]
    other = call("GET", v2, "/api/v2/jobs", params={"metadata[client]": "nobody"}).json()["jobs"]
    assert job["id"] not in [j["id"] for j in other]


def test_restart_drops_jobs_keeps_assets(v2, asset):
    key = {"Idempotency-Key": f"restart-{uuid.uuid4()}"}
    job = submit(v2, preview_graph(asset["id"]), headers=key).json()
    wait_job(v2, job["id"])
    v2.restart()
    assert call("GET", v2, f"/api/v2/jobs/{job['id']}").status_code == 404
    assert call("GET", v2, "/api/v2/jobs").json()["jobs"] == []
    assert submit(v2, preview_graph(asset["id"]), headers=key).status_code == 201
    kept = call("GET", v2, f"/api/v2/assets/{asset['id']}")
    assert kept.status_code == 200 and kept.json()["hash"] == asset["hash"]
    assert call("HEAD", v2, f"/api/v2/assets/by-hash/{asset['hash']}").status_code == (200 if hashing(v2) else 404)


def test_multi_user_visibility(multi):
    def create_user(name: str) -> dict:
        resp = call("POST", multi, "/users", json={"username": f"{name}-{uuid.uuid4().hex[:8]}"})
        assert resp.status_code == 200, resp.text
        return {"comfy-user": resp.json()}

    alice, bob = create_user("alice"), create_user("bob")
    unknown = call("GET", multi, "/api/v2/jobs", headers={"comfy-user": "nobody"})
    assert unknown.status_code == 403 and error_code(unknown) == "forbidden"
    image = upload(multi, unique_png(), "input/multi.png", headers=alice)
    assert image.status_code == 201, image.text
    resp = submit(multi, preview_graph(image.json()["id"]), headers=alice, metadata={"owner": "alice"})
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]
    wait_job(multi, job_id, headers=alice)
    assert [j["id"] for j in call("GET", multi, "/api/v2/jobs", headers=alice).json()["jobs"]] == [job_id]
    assert call("GET", multi, "/api/v2/jobs", headers=bob).json()["jobs"] == []
    as_bob = call("GET", multi, f"/api/v2/jobs/{job_id}", headers=bob)
    assert as_bob.status_code == 200
    assert "metadata" not in as_bob.json()
    assert call("GET", multi, f"/api/v2/jobs/{job_id}", headers=alice).json()["metadata"] == {"owner": "alice"}


def test_sse_frame_parser():
    class Fake:
        def iter_content(self, chunk_size=None):
            yield b": keepalive\n\nevent: status\ndata: {\"status\": \"queued\"}\n\n"
            yield b"event: status\r\ndata: {\"status\": \"succeeded\"}\r\n\r\n"

    assert sse_frames(Fake()) == [("status", {"status": "queued"}), ("status", {"status": "succeeded"})]
