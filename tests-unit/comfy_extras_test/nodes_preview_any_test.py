import sys
from unittest.mock import MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

# Stub only `nodes` and `server`, and put back only those two keys.
#
# `patch.dict("sys.modules", ...)` restores the mapping to its state on entry,
# which also *evicts* every module this import pulled in for the first time. A
# later import of one of those re-initialises an already-loaded C extension --
# numpy raises "cannot load module more than once per process" -- aborting the
# whole pytest run rather than failing one test.
_MISSING = object()
_stubs = {"nodes": mock_nodes, "server": mock_server}
_originals = {name: sys.modules.get(name, _MISSING) for name in _stubs}
sys.modules.update(_stubs)
try:
    from comfy_extras.nodes_preview_any import PreviewAny
finally:
    for _name, _original in _originals.items():
        if _original is _MISSING:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _original


class TestPreviewAnyMain:
    @staticmethod
    def _exec(source) -> dict:
        return PreviewAny().main(source)

    def test_dict_keeps_non_ascii(self):
        result = self._exec({"greeting": "你好"})
        assert "你好" in result["ui"]["text"][0]
        assert "\\u" not in result["ui"]["text"][0]
        assert result["result"][0] == result["ui"]["text"][0]

    def test_list_keeps_non_ascii(self):
        result = self._exec(["你好", "こんにちは"])
        assert "こんにちは" in result["result"][0]
        assert "\\u" not in result["result"][0]

    def test_string_passthrough(self):
        result = self._exec("你好")
        assert result["ui"]["text"][0] == "你好"
        assert result["result"][0] == "你好"
