from unittest import mock

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
import comfy.model_patcher
import comfy.ops
from comfy_extras.nodes_lora_extract import LORAType, calc_lora_model, extract_lora


class _ToyDiffusionModel(nn.Module):
    def __init__(self, norm_val, small_val, big_val):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.norm = comfy.ops.manual_cast.RMSNorm(1024, dtype=torch.float32, device="cpu")
        self.diffusion_model.small_lin = comfy.ops.manual_cast.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu")
        self.diffusion_model.big_lin = comfy.ops.manual_cast.Linear(256, 256, bias=True, dtype=torch.float32, device="cpu")
        self.diffusion_model.norm.weight = nn.Parameter(torch.full((1024,), norm_val, dtype=torch.float32))
        self.diffusion_model.small_lin.weight = nn.Parameter(torch.full((32, 32), small_val, dtype=torch.float32))
        self.diffusion_model.small_lin.bias = nn.Parameter(torch.full((32,), small_val, dtype=torch.float32))
        self.diffusion_model.big_lin.weight = nn.Parameter(torch.full((256, 256), big_val, dtype=torch.float32))
        self.diffusion_model.big_lin.bias = nn.Parameter(torch.full((256,), norm_val, dtype=torch.float32))


def _make_dynamic_patcher(monkeypatch, model):
    monkeypatch.setattr(comfy.model_patcher.comfy_aimdo.host_buffer, "HostBuffer", mock.MagicMock())
    cpu = torch.device("cpu")
    # CPU construction reroutes to ModelPatcher, so build the dynamic patcher directly.
    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcher.__init__(patcher, model, cpu, cpu)
    comfy.model_patcher.ModelPatcherDynamic.__init__(patcher, model, cpu, cpu)
    return patcher


def _subtract_patcher(monkeypatch, m_turbo, m_base):
    m_diff = _make_dynamic_patcher(monkeypatch, m_turbo.model)
    kp = m_base.get_key_patches("diffusion_model.")
    for k in kp:
        m_diff.add_patches({k: kp[k]}, -1.0, 1.0)
    return m_diff


def test_dynamic_load_marks_force_loaded_modules_and_restore_clears_flag(monkeypatch):
    model = _ToyDiffusionModel(1.1, 1.2, 2.0)
    orig_norm = model.diffusion_model.norm.weight.clone()
    orig_small = model.diffusion_model.small_lin.weight.clone()

    base = _make_dynamic_patcher(monkeypatch, _ToyDiffusionModel(1.0, 1.0, 1.5))
    diff = _subtract_patcher(monkeypatch, _make_dynamic_patcher(monkeypatch, model), base)

    diff.load(device_to=torch.device("cpu"))

    assert model.diffusion_model.norm.comfy_patched_weights is True
    assert model.diffusion_model.small_lin.comfy_patched_weights is True
    assert model.diffusion_model.big_lin.comfy_patched_weights is False
    assert torch.allclose(model.diffusion_model.norm.weight, torch.full((1024,), 0.1), atol=1e-5)
    assert torch.allclose(model.diffusion_model.small_lin.weight, torch.full((32, 32), 0.2), atol=1e-5)

    diff.restore_loaded_backups()

    assert not hasattr(model.diffusion_model.norm, "comfy_patched_weights")
    assert not hasattr(model.diffusion_model.small_lin, "comfy_patched_weights")
    assert not hasattr(model.diffusion_model.big_lin, "comfy_patched_weights")
    assert torch.equal(model.diffusion_model.norm.weight, orig_norm)
    assert torch.equal(model.diffusion_model.small_lin.weight, orig_small)


def test_dynamic_lora_extract_does_not_double_patch_force_loaded_modules(monkeypatch):
    # Regression test for #15907: ModelPatcherDynamic force-loads small modules (<= 16 KB,
    # such as RMSNorm / LayerNorm) and patches them in place during load(). Without
    # comfy_patched_weights = True, calc_lora_model patched them a second time.
    turbo = _make_dynamic_patcher(monkeypatch, _ToyDiffusionModel(1.1, 1.2, 2.0))
    base = _make_dynamic_patcher(monkeypatch, _ToyDiffusionModel(1.0, 1.0, 1.5))
    diff = _subtract_patcher(monkeypatch, turbo, base)

    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: [m.load(device_to=m.load_device) for m in models],
    )

    full_sd = calc_lora_model(diff, 8, "diffusion_model.", "diffusion_model.", {}, LORAType.FULL_DIFF, bias_diff=True)
    assert torch.allclose(full_sd["diffusion_model.norm.diff"].float(), torch.full((1024,), 0.1), atol=1e-3)
    assert torch.allclose(full_sd["diffusion_model.small_lin.diff"].float(), torch.full((32, 32), 0.2), atol=1e-3)
    assert torch.allclose(full_sd["diffusion_model.small_lin.diff_b"].float(), torch.full((32,), 0.2), atol=1e-3)
    assert torch.allclose(full_sd["diffusion_model.big_lin.diff"].float(), torch.full((256, 256), 0.5), atol=1e-3)
    assert torch.allclose(full_sd["diffusion_model.big_lin.diff_b"].float(), torch.full((256,), 0.1), atol=1e-3)

    std_sd = calc_lora_model(diff, 8, "diffusion_model.", "diffusion_model.", {}, LORAType.STANDARD, bias_diff=True)
    assert torch.allclose(std_sd["diffusion_model.norm.diff"].float(), torch.full((1024,), 0.1), atol=1e-3)
    expected_small_up, expected_small_down = extract_lora(torch.full((32, 32), 0.2), 8)
    expected_big_up, expected_big_down = extract_lora(torch.full((256, 256), 0.5), 8)
    reconstructed_small = std_sd["diffusion_model.small_lin.lora_up.weight"].float() @ std_sd["diffusion_model.small_lin.lora_down.weight"].float()
    reconstructed_big = std_sd["diffusion_model.big_lin.lora_up.weight"].float() @ std_sd["diffusion_model.big_lin.lora_down.weight"].float()
    assert torch.allclose(reconstructed_small, expected_small_up.half().float() @ expected_small_down.half().float(), atol=1e-2)
    assert torch.allclose(reconstructed_big, expected_big_up.half().float() @ expected_big_down.half().float(), atol=1e-2)


def test_dynamic_model_state_dict_for_saving_before_and_after_restore(monkeypatch):
    turbo = _make_dynamic_patcher(monkeypatch, _ToyDiffusionModel(1.1, 1.2, 2.0))
    base = _make_dynamic_patcher(monkeypatch, _ToyDiffusionModel(1.0, 1.0, 1.5))
    diff = _subtract_patcher(monkeypatch, turbo, base)

    diff.load(device_to=torch.device("cpu"))
    saved_loaded = diff.model_state_dict_for_saving(diff.model.diffusion_model, "diffusion_model.")
    assert torch.allclose(saved_loaded["norm.weight"].to("cpu"), torch.full((1024,), 0.1), atol=1e-5)
    assert torch.allclose(saved_loaded["small_lin.weight"].to("cpu"), torch.full((32, 32), 0.2), atol=1e-5)
    assert torch.allclose(saved_loaded["big_lin.weight"].to("cpu"), torch.full((256, 256), 0.5), atol=1e-5)

    # After restoring backups (e.g. VRAM eviction), saving must patch once again, not skip patching.
    diff.restore_loaded_backups()
    saved_restored = diff.model_state_dict_for_saving(diff.model.diffusion_model, "diffusion_model.")
    assert torch.allclose(saved_restored["norm.weight"].to("cpu"), torch.full((1024,), 0.1), atol=1e-5)
    assert torch.allclose(saved_restored["small_lin.weight"].to("cpu"), torch.full((32, 32), 0.2), atol=1e-5)
    assert torch.allclose(saved_restored["big_lin.weight"].to("cpu"), torch.full((256, 256), 0.5), atol=1e-5)
