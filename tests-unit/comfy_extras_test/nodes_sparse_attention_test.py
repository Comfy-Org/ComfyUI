import pytest
from unittest.mock import patch, MagicMock

mock_modules = {name: MagicMock() for name in (
    "comfy_kitchen",
    "torch",
    "comfy.model_management",
    "comfy.model_prefetch",
    "comfy.patcher_extension",
    "comfy.ldm.minimax.model",
    "comfy_api.latest",
)}

with patch.dict("sys.modules", mock_modules):
    from comfy_extras.nodes_sparse_attention import parse_block_list


def test_single_blocks_and_ranges():
    assert parse_block_list("0, 1, 47-49") == {0, 1, 47, 48, 49}


def test_empty_input():
    assert parse_block_list("") == set()
    assert parse_block_list(None) == set()


def test_negative_entry_raises_instead_of_silently_misparsing():
    # A bare "-1" used to be silently read as block 1 (the leading "-" was
    # dropped by the old regex), which is not the "last block" a user would
    # expect from other sparse-attention implementations' conventions.
    with pytest.raises(ValueError):
        parse_block_list("38-45,-1")
