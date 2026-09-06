import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest
import websocket


@pytest.fixture(scope="module", autouse=True)
def server(args_pytest):
    process = subprocess.Popen([
        sys.executable,
        "main.py",
        "--output-directory", args_pytest["output_dir"],
        "--listen", args_pytest["listen"],
        "--port", str(args_pytest["port"]),
        "--extra-model-paths-config", "tests/execution/extra_model_paths.yaml",
        "--cpu",
    ])

    try:
        server_url = f"http://{args_pytest['listen']}:{args_pytest['port']}/system_stats"
        for _ in range(90):
            try:
                with urllib.request.urlopen(server_url, timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                if process.poll() is not None:
                    raise RuntimeError("ComfyUI exited before accepting connections")
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI did not accept connections within 90 seconds")

        yield
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def connect(args_pytest, client_id):
    ws = websocket.WebSocket()
    ws.settimeout(5)
    ws.connect(f"ws://{args_pytest['listen']}:{args_pytest['port']}/ws?clientId={client_id}")
    return ws


def receive_message(ws, expected_type):
    message = json.loads(ws.recv())
    assert message["type"] == expected_type
    return message


@pytest.mark.execution
def test_reconnect_retires_stale_socket_and_keeps_replacement(args_pytest):
    client_id = "reconnecting-client"
    first = connect(args_pytest, client_id)
    receive_message(first, "status")

    second = connect(args_pytest, client_id)
    assert first.recv() == ""
    assert not first.connected
    receive_message(second, "status")

    second.send(json.dumps({
        "type": "feature_flags",
        "data": {"supports_preview_metadata": True},
    }))
    response = receive_message(second, "feature_flags")
    assert response["data"]["supports_preview_metadata"] is True

    second.close()
