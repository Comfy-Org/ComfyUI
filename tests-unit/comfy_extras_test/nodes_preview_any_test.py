from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_preview_any import PreviewAny


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
