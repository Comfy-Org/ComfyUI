import asyncio
import logging

from comfy.cli_args import args as cli_args

cli_args.cpu = True

import nodes  # noqa: E402
import execution  # noqa: E402


class StubNodeForUnknownInputTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {})}}

    RETURN_TYPES = ()
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, a):
        return ()


def test_unknown_input_is_warned_and_ignored(monkeypatch, caplog):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StubNodeForUnknownInputTest", StubNodeForUnknownInputTest)
    prompt = {
        "1": {
            "class_type": "StubNodeForUnknownInputTest",
            "inputs": {"a": 1, "ref_audios": {"ref_audio_0": ["2", 0]}},
        }
    }

    with caplog.at_level(logging.WARNING):
        valid, errors, node_id = asyncio.run(execution.validate_inputs("test-prompt", prompt, "1", {}))

    assert valid
    assert errors == []
    assert any(
        "Node 1 (StubNodeForUnknownInputTest)" in record.message
        and "ref_audios" in record.message
        for record in caplog.records
    )
