import copy
from types import SimpleNamespace

import pytest
import torch

import comfy.minimax_sequence_parallel as sp
from comfy.ldm.minimax.model import DiTBlock, MiniMaxH3Model, rope_rotation_table


def test_partition_segments_slices_modality_rows():
    indices = torch.tensor([1, 2, 0, 1])
    segments = [(0, 3, 0), (3, 7, indices), (7, 9, torch.tensor(2))]
    actual = sp.partition_segments(segments, 4, 8, torch.device("cpu"))
    assert [(a, b) for a, b, _ in actual] == [(0, 3), (3, 4)]
    torch.testing.assert_close(actual[0][2], torch.tensor([2, 0, 1]))
    assert actual[1][2].item() == 2
    assert sp.partition_segments(segments, 9, 9, torch.device("cpu")) == []


@pytest.fixture
def sampling(monkeypatch):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    base = torch.nn.Module()
    base.diffusion_model = model
    patcher = sp.comfy.model_patcher.ModelPatcher(base, torch.device("cuda:0"), torch.device("cpu"))
    secondary = patcher.clone(model_override=(copy.deepcopy(base), ({}, {}, {}, set())))
    secondary.load_device = torch.device("cuda:1")
    patcher.set_additional_models("minimax_sequence_parallel", [secondary])
    guider = SimpleNamespace(model_patcher=patcher, model_options={"transformer_options": {}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setattr(sp.mm, "in_training", False)
    return guider


@pytest.mark.parametrize("fail", [False, True])
def test_sampling_reuses_managed_weights_and_restores_options(sampling, fail):
    original = sampling.model_options
    stored = sampling.model_patcher.get_additional_models_with_key("minimax_sequence_parallel")[0]

    class Executor:
        class_obj = sampling

        def __call__(self):
            replacement = sampling.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
            replica = replacement.args[1]
            assert replica is stored.get_model_object("diffusion_model").blocks[0]
            if fail:
                raise RuntimeError("sampling failed")
            return "sample"

    wrapper = sp.SequenceParallelSampling(1)
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="sampling failed"):
                wrapper(Executor())
        else:
            assert wrapper(Executor()) == "sample"
        assert sampling.model_options is original
        refreshed = sampling.model_patcher.get_additional_models_with_key("minimax_sequence_parallel")[0]
        assert refreshed.model is stored.model
        assert refreshed.load_device == torch.device("cuda:1")
        assert not refreshed.get_additional_models_with_key("minimax_sequence_parallel")


def test_managed_replica_inherits_later_weight_patches(sampling):
    key = "diffusion_model.blocks.0.weight"
    patcher = sampling.model_patcher
    delta = torch.ones_like(patcher.model.diffusion_model.blocks[0].weight)
    patcher.add_patches({key: (delta,)})

    class Executor:
        class_obj = sampling

        def __call__(self):
            secondary = patcher.get_additional_models_with_key("minimax_sequence_parallel")[0]
            assert secondary.patches_uuid == patcher.patches_uuid
            assert secondary.patches[key] == patcher.patches[key]
            assert secondary.model is not patcher.model

    sp.SequenceParallelSampling(1)(Executor())


@pytest.mark.parametrize("device", [0, 2, -1])
def test_rejects_invalid_secondary_device(sampling, device):
    executor = SimpleNamespace(class_obj=sampling)
    with pytest.raises(ValueError, match="second available GPU"):
        sp.SequenceParallelSampling(device)(executor)


def test_rejects_missing_peer_access(sampling, monkeypatch):
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: a == 0)
    with pytest.raises(ValueError, match="both directions"):
        sp.SequenceParallelSampling(1)(SimpleNamespace(class_obj=sampling))


def test_preserves_existing_transformer_patch(sampling):
    patch = {"dit": {("double_block", 0): object()}}
    sampling.model_options["transformer_options"]["patches_replace"] = patch
    with pytest.raises(ValueError, match="transformer patches"):
        sp.SequenceParallelSampling(1)(SimpleNamespace(class_obj=sampling))
    assert sampling.model_options["transformer_options"]["patches_replace"] is patch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA/ROCm GPUs required")
@pytest.mark.parametrize("rows", [255, 257])
@pytest.mark.parametrize("with_rope", [False, True])
def test_two_device_block_matches_full_attention(rows, with_rope):
    if not torch.cuda.can_device_access_peer(0, 1) or not torch.cuda.can_device_access_peer(1, 0):
        pytest.skip("peer access required")
    torch.manual_seed(42)
    source, target = torch.device("cuda:0"), torch.device("cuda:1")
    block = DiTBlock(64, 2, 32, 96, 32, 1e-6, 1e-6, dtype=torch.float32, device=source, operations=torch.nn)
    block.requires_grad_(False)
    x = torch.randn(rows, 64, device=source)
    t = torch.randn(1, 32, device=source)
    segments = [(0, 17, 0), (17, rows, torch.arange(rows - 17, device=source) % 3)]
    rope = rope_rotation_table(torch.randn(rows, 32, device=source), x.dtype) if with_rope else None
    with torch.inference_mode():
        replica = copy.deepcopy(block).to(target)
        expected = block(x.clone(), t, segments, rope)
        actual = sp.parallel_block(block, replica, target, x.clone(), t, segments, rope, {})
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
