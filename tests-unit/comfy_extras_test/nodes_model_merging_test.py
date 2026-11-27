from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_model_merging import ModelAdd, ModelMergeSimple, ModelSubtract


def make_patcher(value):
    model = torch.nn.Module()
    model.diffusion_model = torch.nn.Linear(2, 2)
    model.diffusion_model.weight = torch.nn.Parameter(torch.full((2, 2), value))
    model.diffusion_model.bias = torch.nn.Parameter(torch.full((2,), value + 1))
    model.register_buffer("other", torch.tensor([value]))
    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)


@pytest.mark.parametrize(
    "node,parameters,strength_patch,strength_model",
    [
        (ModelMergeSimple, {"ratio": 0.0}, 1.0, 0.0),
        (ModelMergeSimple, {"ratio": 0.25}, 0.75, 0.25),
        (ModelMergeSimple, {"ratio": 1.0}, 0.0, 1.0),
        (ModelSubtract, {"multiplier": -2.0}, 2.0, -2.0),
        (ModelSubtract, {"multiplier": 0.0}, 0.0, 0.0),
        (ModelSubtract, {"multiplier": 1.5}, -1.5, 1.5),
        (ModelAdd, {}, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize("existing_patches", [False, True])
def test_model_merge_batches_patches(node, parameters, strength_patch, strength_model, existing_patches):
    model1 = make_patcher(2.0)
    model2 = make_patcher(6.0)
    key = "diffusion_model.weight"
    if existing_patches:
        model1.add_patches({key: (torch.ones(2, 2),)}, 0.5)
        model2.add_patches({key: (torch.ones(2, 2),)}, -0.25)

    original_patches = {k: v[:] for k, v in model1.patches.items()}
    source_patches = {k: v[:] for k, v in model2.patches.items()}
    original_uuid = model1.patches_uuid
    kp = model2.get_key_patches("diffusion_model.")
    reference = model1.clone()
    for k in kp:
        reference.add_patches({k: kp[k]}, strength_patch, strength_model)

    with patch.object(model1.model, "state_dict", wraps=model1.model.state_dict) as state_dict:
        with patch.object(ModelPatcher, "add_patches", autospec=True, side_effect=ModelPatcher.add_patches) as add_patches:
            merged, = node().merge(model1, model2, **parameters)
        assert add_patches.call_count == 1
        assert state_dict.call_count == 1

    assert merged is not model1
    assert merged.patches_uuid != original_uuid
    assert model1.patches_uuid == original_uuid
    assert model1.patches == original_patches
    assert model2.patches == source_patches
    assert list(merged.patches) == list(reference.patches)
    for k in reference.patches:
        assert len(merged.patches[k]) == len(reference.patches[k])
        for actual_patch, expected_patch in zip(merged.patches[k], reference.patches[k]):
            assert actual_patch[0] == expected_patch[0]
            assert actual_patch[2:] == expected_patch[2:]
        assert merged.patches[k][-1][1][0][0] is kp[k][0][0]
        assert merged.patches[k][-1][1][1:] == kp[k][1:]
        actual = merged.patch_weight_to_device(k, return_weight=True)
        expected = reference.patch_weight_to_device(k, return_weight=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        base = model1.model.state_dict()[k]
        source = model2.model.state_dict()[k]
        if existing_patches and k == key:
            base = base + 0.5
            source = source - 0.25
        torch.testing.assert_close(actual, base * strength_model + source * strength_patch)
    assert "other" not in merged.patches
    torch.testing.assert_close(model1.model.diffusion_model.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(model2.model.diffusion_model.weight, torch.full((2, 2), 6.0))
