import asyncio

import pytest

from comfy.cli_args import args

args.cpu = True

import execution  # noqa: E402
import nodes  # noqa: E402


class _Producer:
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _Consumer:
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}


@pytest.fixture
def execution_module(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LinkValidationProducer", _Producer)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LinkValidationConsumer", _Consumer)
    return execution


def _prompt(output_index):
    return {
        "producer": {
            "class_type": "LinkValidationProducer",
            "inputs": {},
        },
        "consumer": {
            "class_type": "LinkValidationConsumer",
            "inputs": {"images": ["producer", output_index]},
        },
    }


@pytest.mark.parametrize("bad_index", [False, 0.0, -1])
def test_validate_inputs_rejects_invalid_link_index(execution_module, bad_index):
    valid, errors, node_id = asyncio.run(
        execution_module.validate_inputs(
            "prompt",
            _prompt(bad_index),
            "consumer",
            {},
        )
    )

    assert node_id == "consumer"
    assert valid is False
    assert [error["type"] for error in errors] == ["bad_linked_input"]
    assert errors[0]["extra_info"]["received_value"] == ["producer", bad_index]


def test_validate_inputs_accepts_nonnegative_integer_link_index(execution_module):
    valid, errors, node_id = asyncio.run(
        execution_module.validate_inputs(
            "prompt",
            _prompt(0),
            "consumer",
            {},
        )
    )

    assert node_id == "consumer"
    assert valid is True
    assert errors == []
