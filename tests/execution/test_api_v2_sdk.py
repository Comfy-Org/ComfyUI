"""Opt-in: the Comfy Python SDK (`comfy_sdk`) against a real `main.py` /api/v2 server. Skipped unless the SDK is installed."""
import pytest

comfy_sdk = pytest.importorskip("comfy_sdk")

from api_v2_server import TERMINAL, png_bytes, running  # noqa: E402
from comfy_sdk.events import StatusChange  # noqa: E402

pytestmark = pytest.mark.execution


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with running(tmp_path_factory.mktemp("v2-sdk"), "--enable-assets") as server, pytest.MonkeyPatch.context() as mp:
        mp.setenv("COMFY_BASE_URL", server.base)
        mp.delenv("COMFY_API_KEY", raising=False)
        with comfy_sdk.Comfy() as client:
            yield client


def submit(client, **kwargs):
    asset = client.assets.from_bytes(png_bytes((10, 200, 30)), filename="sdk.png", content_type="image/png")
    workflow = client.workflows.from_json({
        "1": {"class_type": "LoadImage", "inputs": {"image": asset}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "sdk"}},
    })
    return client.submit(workflow, **kwargs)


def test_upload_submit_wait_download(client, tmp_path):
    job = submit(client).wait(timeout=60)
    assert job.status == "succeeded", job.error
    path = job.outputs[0].to_file(tmp_path / "out.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_events_labels_list_and_workflow(client):
    job = submit(client, metadata={"client": "acme"})
    assert job.metadata == {"client": "acme"}
    events = list(job.events())
    assert isinstance(events[-1], StatusChange) and events[-1].status in TERMINAL
    assert job.wait(timeout=60).status == "succeeded", job.error
    assert job.id in [j.id for j in client.list_jobs(metadata={"client": "acme"})]
    workflow = job.get_workflow()
    assert workflow.format == "api"
    assert workflow.graph["1"]["class_type"] == "LoadImage"
