"""EXR outputs get a generated preview end to end: a real server with assets enabled."""

import io
import json
import subprocess
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
    process = subprocess.Popen([
        "python", "main.py",
        "--output-directory", args_pytest["output_dir"],
        "--listen", args_pytest["listen"],
        "--port", str(args_pytest["port"]),
        "--extra-model-paths-config", "tests/execution/extra_model_paths.yaml",
        "--cpu",
        "--enable-assets",
        "--database-url", f"sqlite:///{work / 'assets.db'}",
        "--previews-directory", str(work / "previews"),
    ], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://{args_pytest['listen']}:{args_pytest['port']}"
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


def _run(base: str, prompt: dict) -> dict:
    """Queue a prompt and return its executed messages by node id."""
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
            return executed


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


@pytest.mark.execution
def test_saved_exr_frames_carry_a_generated_preview(server):
    base, previews_dir = server
    prompt, save_id = _exr_save_graph(f"exr_{uuid.uuid4().hex[:8]}", batch_size=3)

    entries = _run(base, prompt)[save_id]["images"]

    assert len(entries) == 3
    for entry in entries:
        assert entry["filename"].endswith(".exr")
        assert entry["preview_id"] != entry["id"], "an EXR is never its own preview"
        with urllib.request.urlopen(f"{base}/api/assets/{entry['preview_id']}/content") as response:
            preview = Image.open(io.BytesIO(response.read()))
        assert preview.format == "WEBP"
        assert preview.size == (200, 60)
        with urllib.request.urlopen(f"{base}/api/assets/{entry['id']}") as response:
            asset = json.loads(response.read())
        assert asset["preview_id"] == entry["preview_id"]
        assert asset["metadata"]["width"] == 200 and asset["metadata"]["height"] == 60
    assert len(list(previews_dir.glob("*.webp"))) >= 3


@pytest.mark.execution
def test_a_cached_rerun_keeps_the_preview(server):
    base, _ = server
    prompt, save_id = _exr_save_graph(f"exr_{uuid.uuid4().hex[:8]}", batch_size=1)

    first = _run(base, prompt)[save_id]["images"][0]
    replay = _run(base, prompt)[save_id]["images"][0]

    assert replay["id"] != first["id"], "a replay registers its own record"
    assert replay["preview_id"] == first["preview_id"], "and reuses the preview of the same bytes"
