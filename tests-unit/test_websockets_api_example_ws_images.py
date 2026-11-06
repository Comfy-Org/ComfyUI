import io
import json
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("event, error", [
    ({"type": "execution_error", "data": {"prompt_id": "prompt", "node_id": "3", "node_type": "KSampler",
                                          "exception_type": "OutOfMemoryError", "exception_message": "CUDA out of memory"}}, "CUDA out of memory"),
    ({"type": "execution_interrupted", "data": {"prompt_id": "prompt", "node_id": "3", "node_type": "KSampler"}}, "interrupted"),
    ({"type": "execution_error", "data": {"prompt_id": "other", "node_id": "3", "node_type": "KSampler",
                                          "exception_type": "RuntimeError", "exception_message": "other failure"}}, None),
])
def test_websocket_image_example_execution_failure(event, error):
    closed = []
    events = [event, {"type": "executing", "data": {"prompt_id": "prompt", "node": None}}]

    class FakeWebSocket:
        def connect(self, url):
            pass

        def recv(self):
            return json.dumps(events.pop(0))

        def close(self):
            closed.append(True)

    websocket = types.ModuleType("websocket")
    websocket.WebSocket = FakeWebSocket
    source = Path(__file__).resolve().parents[1] / "script_examples/websockets_api_example_ws_images.py"
    with patch.dict(sys.modules, {"websocket": websocket}), patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"prompt_id": "prompt"}')):
        if error:
            with pytest.raises(RuntimeError, match=error):
                runpy.run_path(str(source))
        else:
            assert runpy.run_path(str(source))["images"] == {}
    assert closed == [True]
