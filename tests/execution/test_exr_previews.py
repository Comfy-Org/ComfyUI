"""EXR outputs register already linked to the preview their save node wrote: a real server with assets enabled."""

import io
import json
import socket
import subprocess
import threading
import time
import urllib.request
import uuid

import pytest
import websocket
from PIL import Image

from comfy_execution.graph_utils import GraphBuilder


@pytest.fixture(scope="module")
def server(args_pytest, tmp_path_factory):
    work = tmp_path_factory.mktemp("exr-previews")
    log = (work / "server.log").open("w")
    # Its own port: a server another module hasn't finished killing must not answer for this one.
    with socket.socket() as probe:
        probe.bind((args_pytest["listen"], 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([
        "python", "main.py",
        "--output-directory", args_pytest["output_dir"],
        "--listen", args_pytest["listen"],
        "--port", str(port),
        "--extra-model-paths-config", "tests/execution/extra_model_paths.yaml",
        "--cpu",
        "--enable-assets",
        "--database-url", f"sqlite:///{work / 'assets.db'}",
        "--previews-directory", str(work / "previews"),
    ], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://{args_pytest['listen']}:{port}"
    try:
        for _ in range(120):
            if process.poll() is not None:
                pytest.fail(f"server exited with {process.returncode}:\n{(work / 'server.log').read_text()[-4000:]}")
            try:
                urllib.request.urlopen(f"{base}/system_stats", timeout=2)
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail(f"server never answered:\n{(work / 'server.log').read_text()[-4000:]}")
        yield base, work / "previews"
    finally:
        process.kill()
        process.wait(timeout=10)
        log.close()


def _run(base: str, prompt: dict) -> tuple[dict, str]:
    """Queue a prompt and return its executed messages by node id, and its prompt id."""
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.settimeout(120)
    ws.connect(f"ws://{base.removeprefix('http://')}/ws?clientId={client_id}")
    request = urllib.request.Request(f"{base}/prompt", data=json.dumps({"prompt": prompt, "client_id": client_id}).encode())
    prompt_id = json.loads(urllib.request.urlopen(request).read())["prompt_id"]
    executed = {}
    while True:
        message = ws.recv()
        if not isinstance(message, str):
            continue
        message = json.loads(message)
        data = message.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue
        if message["type"] == "execution_error":
            raise AssertionError(data)
        if message["type"] == "executed":
            executed[data["node"]] = data["output"]
        if message["type"] == "executing" and data["node"] is None:
            ws.close()
            return executed, prompt_id


def _exr_save_graph(prefix: str, batch_size: int) -> tuple[dict, str]:
    g = GraphBuilder(prefix=prefix)
    # Small frames: decode speed is the unit tests' business, not this one's.
    image = g.node("StubImage", content="WHITE", height=60, width=200, batch_size=batch_size)
    save = g.node(
        "SaveImageAdvanced",
        images=image.out(0),
        filename_prefix=f"exr_previews/{prefix}",
        format="exr",
        **{"format.bit_depth": "16-bit float", "format.input_color_space": "sRGB"},
    )
    return g.finalize(), save.id


def _get(base: str, path: str):
    with urllib.request.urlopen(f"{base}{path}") as response:
        return json.loads(response.read())


@pytest.mark.execution
def test_saved_exr_frames_carry_their_preview(server):
    base, previews_dir = server
    prefix = f"exr_{uuid.uuid4().hex[:8]}"
    prompt, save_id = _exr_save_graph(prefix, batch_size=3)
    unlinked, done = [], threading.Event()

    def poll_listing():
        # Every listing while the job runs: an output is never visible without its preview.
        while not done.is_set():
            for asset in _get(base, "/api/assets?include_tags=output&limit=500")["assets"]:
                if asset["name"].startswith(prefix) and not asset.get("preview_id"):
                    unlinked.append(asset["name"])
            time.sleep(0.01)

    poller = threading.Thread(target=poll_listing)
    poller.start()
    try:
        executed, prompt_id = _run(base, prompt)
    finally:
        done.set()
        poller.join()
    entries = executed[save_id]["images"]
    assert unlinked == [], "an output was listed before its preview was linked"
    listed = [a for a in _get(base, "/api/assets?include_tags=output&limit=500")["assets"] if a["name"].startswith(prefix)]
    assert len(listed) == 3 and all(a.get("preview_id") for a in listed), "the poll's query finds these outputs"

    assert len(entries) == 3
    for entry in entries:
        assert entry["filename"].endswith(".exr")
        assert entry["preview_id"] != entry["id"], "an EXR is never its own preview"
        with urllib.request.urlopen(f"{base}/api/assets/{entry['preview_id']}/content") as response:
            preview = Image.open(io.BytesIO(response.read()))
        assert preview.format == "JPEG"
        assert preview.size == (200, 60)
        asset = _get(base, f"/api/assets/{entry['id']}")
        assert asset["preview_id"] == entry["preview_id"]
        assert asset["metadata"]["width"] == 200 and asset["metadata"]["height"] == 60
    assert len({entry["preview_id"] for entry in entries}) == 1, "identical frames share one preview"
    assert len(list(previews_dir.glob("*.jpg"))) >= 1
    for value in (executed, _get(base, f"/history/{prompt_id}"), _get(base, f"/api/jobs/{prompt_id}")):
        assert "asset_preview" not in json.dumps(value)


@pytest.mark.execution
def test_a_cached_rerun_keeps_the_preview(server):
    base, _ = server
    prompt, save_id = _exr_save_graph(f"exr_{uuid.uuid4().hex[:8]}", batch_size=1)

    first = _run(base, prompt)[0][save_id]["images"][0]
    replay, _ = _run(base, prompt)
    replay = replay[save_id]["images"][0]

    assert replay["filename"] == first["filename"], "served from cache, not saved again"
    assert replay["id"] != first["id"], "a replay registers its own record"
    assert replay["preview_id"] == first["preview_id"], "and reuses the preview of the same bytes"
    assert "asset_preview" not in replay
