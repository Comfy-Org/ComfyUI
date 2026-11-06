import io
import json
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("events, expected_error", [
    ([{"type": "execution_error", "data": {"prompt_id": "other", "exception_message": "other failure"}},
      {"type": "executing", "data": {"prompt_id": "prompt", "node": None}}], None),
    ([{"type": "execution_error", "data": {"prompt_id": "prompt", "exception_message": "example failure"}},
      {"type": "executing", "data": {"prompt_id": "prompt", "node": None}}], "example failure"),
    ([{"type": "executing", "data": {"prompt_id": "prompt", "node": None}}], None),
])
def test_websocket_example_execution_error(events, expected_error):
    closed = []

    class FakeWebSocket:
        def connect(self, url):
            pass

        def recv(self):
            return json.dumps(events.pop(0))

        def close(self):
            closed.append(True)

    def urlopen(request):
        if not isinstance(request, str):
            return io.BytesIO(b'{}')
        assert '/history/prompt' in request
        return io.BytesIO(json.dumps({'prompt': {'outputs': {}}}).encode())

    websocket = types.ModuleType('websocket')
    websocket.WebSocket = FakeWebSocket
    source = Path(__file__).resolve().parents[1] / 'script_examples/websockets_api_example.py'
    with patch.dict(sys.modules, {'websocket': websocket}), patch('urllib.request.urlopen', urlopen), patch('uuid.uuid4', return_value='prompt'):
        if expected_error:
            with pytest.raises(RuntimeError, match=expected_error):
                runpy.run_path(str(source))
        else:
            assert runpy.run_path(str(source))['images'] == {}
    assert closed == [True]
