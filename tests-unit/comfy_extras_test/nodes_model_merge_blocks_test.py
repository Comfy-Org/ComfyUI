from unittest.mock import patch

import pytest
import torch
from torch import nn

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_model_merging import ModelMergeBlocks
from comfy_extras.nodes_model_merging_model_specific import ModelMergeSDXL


def make_patcher(value):
    model = nn.Module()
    model.diffusion_model = nn.ModuleDict({
        "input_blocks": nn.ModuleDict({
            "1": nn.Linear(1, 1, bias=False),
            "10": nn.Linear(1, 1, bias=False),
            "2": nn.Linear(1, 1, bias=False),
        }),
        "middle_block": nn.ModuleDict({"0": nn.Linear(1, 1, bias=False)}),
        "out": nn.Linear(1, 1, bias=False),
        "other": nn.Linear(1, 1, bias=False),
    })
    model.unrelated = nn.Linear(1, 1, bias=False)
    for weight in model.parameters():
        nn.init.constant_(weight, value)
    device = torch.device("cpu")
    patcher = ModelPatcher(model, device, device)
    patcher.model_size()
    return patcher


@pytest.mark.parametrize("node_class", [ModelMergeBlocks, ModelMergeSDXL])
def test_batches_block_patches_by_ratio_without_changing_weights(node_class):
    destination = make_patcher(2.0)
    source = make_patcher(10.0)
    existing = "diffusion_model.input_blocks.10.weight"
    destination.add_patches({existing: (torch.tensor([[4.0]]),)})
    source.add_patches({existing: (torch.tensor([[2.0]]),)})

    with patch.object(destination.model, "state_dict", wraps=destination.model.state_dict) as state_dict:
        merged, = node_class().merge(destination, source, **{
            "input_blocks.1": 0.25,
            "input_blocks.10": 0.75,
            "middle_block.0": 0.25,
            "out": 1.0,
        })

    expected = {
        "input_blocks.1.weight": 8.0,
        "input_blocks.10.weight": 7.5,
        "input_blocks.2.weight": 8.0,
        "middle_block.0.weight": 8.0,
        "out.weight": 2.0,
        "other.weight": 8.0,
    }
    assert set(merged.patches) == {"diffusion_model." + key for key in expected}
    for key, value in expected.items():
        weight = merged.patch_weight_to_device("diffusion_model." + key, return_weight=True)
        torch.testing.assert_close(weight, torch.tensor([[value]]))
    assert len(destination.patches[existing]) == len(source.patches[existing]) == 1
    assert len(merged.patches[existing]) == 2
    assert state_dict.call_count == 3
