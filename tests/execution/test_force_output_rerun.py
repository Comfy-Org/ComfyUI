"""--force-output-rerun: output nodes re-execute on a cache hit, upstream nodes stay cached.

A long-lived server that deletes output files after each job (a multi-tenant
deployment) otherwise answers a repeated prompt by replaying the first run's
filename for a file that no longer exists.
"""
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


def _queue(base_url: str, prompt: dict) -> str:
    request = urllib.request.Request(
        f"{base_url}/prompt",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)["prompt_id"]


def _wait_for_history(base_url: str, prompt_id: str) -> dict:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        history = _get_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.1)
    raise AssertionError(f"prompt {prompt_id} did not finish")


def _run(base_url: str, prompt: dict) -> dict:
    history = _wait_for_history(base_url, _queue(base_url, prompt))
    assert history["status"]["status_str"] == "success", history["status"]
    return history


def _cached_nodes(history: dict) -> set:
    for event, data in history["status"]["messages"]:
        if event == "execution_cached":
            return set(data["nodes"])
    raise AssertionError("no execution_cached message in history")


@contextmanager
def _running_server(tmp_path: Path, extra_args: list) -> Iterator[str]:
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
        "--disable-assets",
        "--cache-classic",
        "--output-directory",
        str(tmp_path / "output"),
        "--temp-directory",
        str(tmp_path),
        "--extra-model-paths-config",
        str(Path(__file__).parent / "extra_model_paths.yaml"),
        *extra_args,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=COMFYUI_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    pytest.fail(f"server exited during startup:\n{log_path.read_text(encoding='utf-8')}")
                try:
                    _get_json(f"{base_url}/system_stats")
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        pytest.fail(f"server readiness timed out:\n{log_path.read_text(encoding='utf-8')}")
                    time.sleep(0.25)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


PROMPT = {
    "source": {
        "class_type": "EmptyImage",
        "inputs": {"width": 16, "height": 16, "batch_size": 1, "color": 0},
    },
    "save": {
        "class_type": "SaveImage",
        "inputs": {"images": ["source", 0], "filename_prefix": "force_rerun"},
    },
    "paid": {"class_type": "TestPaidOutputNode", "inputs": {"seed": 1}},
}


def _saved_files(history: dict) -> list:
    return [image["filename"] for image in history["outputs"]["save"]["images"]]


@pytest.mark.execution
def test_output_node_rewrites_its_file_on_a_cache_hit(tmp_path: Path) -> None:
    # Given a server started with the flag, and a prompt that has already run once
    with _running_server(tmp_path, ["--force-output-rerun"]) as base_url:
        first = _run(base_url, PROMPT)
        for name in _saved_files(first):  # what a multi-tenant server does after upload
            (tmp_path / "output" / name).unlink()

        # When the identical prompt runs again
        second = _run(base_url, PROMPT)

    # Then the save node ran again and wrote a file that exists. (The name can
    # repeat: the first file was deleted, so the counter starts over.)
    assert _saved_files(second)
    for name in _saved_files(second):
        assert (tmp_path / "output" / name).is_file()
    cached = _cached_nodes(second)
    assert "save" not in cached
    # and its upstream stayed cached
    assert "source" in cached
    # and the API output node was not re-run (that would repeat a paid call)
    assert "paid" in cached
    assert second["outputs"]["paid"] == first["outputs"]["paid"]


@pytest.mark.execution
def test_without_the_flag_a_cache_hit_replays_the_first_filename(tmp_path: Path) -> None:
    # Control: the default behaviour this flag exists to change
    with _running_server(tmp_path, []) as base_url:
        first = _run(base_url, PROMPT)
        second = _run(base_url, PROMPT)

    assert _saved_files(second) == _saved_files(first)
    assert {"source", "save", "paid"} <= _cached_nodes(second)
