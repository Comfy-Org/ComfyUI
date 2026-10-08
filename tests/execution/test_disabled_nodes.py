from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Iterator
import urllib.error
import urllib.request

import pytest


COMFYUI_ROOT = Path(__file__).parents[2]


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def _post_prompt(base_url: str, prompt: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/prompt",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def _wait_for_server(process: subprocess.Popen, base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"server exited during startup:\n{log_path.read_text(encoding='utf-8')}")
        try:
            _get_json(f"{base_url}/system_stats")
            return
        except urllib.error.URLError:
            time.sleep(0.25)
    pytest.fail(f"server readiness timed out:\n{log_path.read_text(encoding='utf-8')}")


def _wait_for_history(base_url: str, prompt_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        history = _get_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.1)
    raise AssertionError(f"prompt {prompt_id} did not finish")


@contextmanager
def _running_server(tmp_path: Path, disabled_node: str, testing_nodes: bool = False) -> Iterator[str]:
    config_path = tmp_path / "disabled_nodes.yaml"
    config_path.write_text(f"disabled_nodes:\n  - {disabled_node}\n", encoding="utf-8")
    log_path = tmp_path / "server.log"
    port = _unused_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--cpu",
        "--cache-none",
        "--disable-api-nodes",
        "--output-directory",
        str(tmp_path / "output"),
        "--temp-directory",
        str(tmp_path),
        "--disabled-nodes-config",
        str(config_path),
    ]
    if testing_nodes:
        command += ["--extra-model-paths-config", str(Path(__file__).parent / "extra_model_paths.yaml")]
    else:
        command.append("--disable-all-custom-nodes")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=COMFYUI_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_for_server(process, base_url, log_path)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.execution
def test_disabled_old_id_forwards_and_executes_allowed_target(tmp_path: Path) -> None:
    # Given
    prompt = {
        "source-a": {
            "class_type": "EmptyImage",
            "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0},
        },
        "source-b": {
            "class_type": "EmptyImage",
            "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0},
        },
        "legacy": {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["source-a", 0], "image2": ["source-b", 0]},
        },
        "output": {"class_type": "PreviewImage", "inputs": {"images": ["legacy", 0]}},
    }

    # When
    with _running_server(tmp_path, "ImageBatch") as base_url:
        object_info = _get_json(f"{base_url}/object_info")
        status, response = _post_prompt(base_url, prompt)
        history = _wait_for_history(base_url, response["prompt_id"])

    # Then
    assert "ImageBatch" not in object_info
    assert status == 200
    assert history["status"]["status_str"] == "success"
    assert history["prompt"][2]["legacy"]["class_type"] == "BatchImagesNode"


@pytest.mark.execution
def test_replacement_pointing_to_disabled_target_is_refused(tmp_path: Path) -> None:
    # Given
    prompt = {"1": {"class_type": "ConditioningAverage ", "inputs": {}}}

    # When
    with _running_server(tmp_path, "ConditioningAverage") as base_url:
        object_info = _get_json(f"{base_url}/object_info")
        status, response = _post_prompt(base_url, prompt)

    # Then
    assert "ConditioningAverage" not in object_info
    assert status == 400
    assert response["error"]["type"] == "missing_node_type"
    assert response["error"]["extra_info"]["class_type"] == "ConditioningAverage "


@pytest.mark.execution
def test_disabled_node_cannot_be_reached_through_expansion(tmp_path: Path) -> None:
    # Given
    prompt = {"1": {"class_type": "TestExpandsToDisabledNode", "inputs": {}}}

    # When
    with _running_server(tmp_path, "TestDisabledNode", testing_nodes=True) as base_url:
        object_info = _get_json(f"{base_url}/object_info")
        status, response = _post_prompt(base_url, prompt)
        history = _wait_for_history(base_url, response["prompt_id"])

    # Then
    assert "TestExpandsToDisabledNode" in object_info
    assert "TestDisabledNode" not in object_info
    assert status == 200
    assert history["status"]["status_str"] == "error"
