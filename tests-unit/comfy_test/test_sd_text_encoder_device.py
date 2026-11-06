from unittest.mock import Mock, call

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.sd as sd  # noqa: E402


class FakeTokenizer:
    def __init__(self, **kwargs):
        pass


class FakePatcher:
    def __init__(self, model, load_device, offload_device, fast_disk):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device

    def set_model_compute_dtype(self, dtype):
        pass


def make_target(module):
    class FakeClip:
        dtypes = []

        def __init__(self, device, **kwargs):
            self.device = device

        def to(self, device):
            self.device = device

    FakeClip.__module__ = module
    target = Mock()
    target.params = {}
    target.clip = FakeClip
    target.tokenizer = FakeTokenizer
    return target


def patch_clip_dependencies(
    monkeypatch, load_device, offload_device, initial_device, capability
):
    monkeypatch.setattr(sd.model_management, "text_encoder_device", lambda: load_device)
    monkeypatch.setattr(
        sd.model_management, "text_encoder_offload_device", lambda: offload_device
    )
    monkeypatch.setattr(
        sd.model_management, "text_encoder_dtype", lambda device: torch.float32
    )
    monkeypatch.setattr(
        sd.model_management, "text_encoder_initial_device", lambda *args: initial_device
    )
    monkeypatch.setattr(sd.model_management, "supports_cast", lambda *args: True)
    monkeypatch.setattr(sd.model_management, "archive_model_dtypes", lambda *args: None)
    monkeypatch.setattr(
        sd.model_management, "load_models_gpu", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(sd.comfy.model_patcher, "ModelPatcher", FakePatcher)
    monkeypatch.setattr(sd.comfy.model_patcher, "CoreModelPatcher", FakePatcher)
    capability_mock = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_mock)
    return capability_mock


def test_z_image_text_encoder_uses_cpu_on_blackwell(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(monkeypatch, cuda, cuda, cuda, (12, 0))
    options = {"marker": "preserve"}
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target, model_options=options)

    assert clip.cond_stage_model.device == torch.device("cpu")
    assert clip.patcher.load_device == torch.device("cpu")
    assert clip.patcher.offload_device == torch.device("cpu")
    assert options == {"marker": "preserve"}
    capability.assert_called_once_with(cuda)


def test_z_image_text_encoder_fallback_covers_load_device(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(
        monkeypatch, cuda, torch.device("cpu"), torch.device("cpu"), (12, 0)
    )
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == torch.device("cpu")
    assert clip.patcher.load_device == torch.device("cpu")
    assert clip.patcher.offload_device == torch.device("cpu")
    capability.assert_called_once_with(cuda)


def test_z_image_text_encoder_fallback_covers_offload_device(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(
        monkeypatch, torch.device("cpu"), cuda, torch.device("cpu"), (12, 0)
    )
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == torch.device("cpu")
    assert clip.patcher.load_device == torch.device("cpu")
    assert clip.patcher.offload_device == torch.device("cpu")
    capability.assert_called_once_with(cuda)


def test_z_image_text_encoder_fallback_covers_explicit_initial_device(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(
        monkeypatch,
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cpu"),
        (12, 0),
    )
    target = make_target("comfy.text_encoders.z_image")
    options = {"initial_device": cuda}

    clip = sd.CLIP(target=target, model_options=options)

    assert clip.cond_stage_model.device == torch.device("cpu")
    assert clip.patcher.load_device == torch.device("cpu")
    assert clip.patcher.offload_device == torch.device("cpu")
    capability.assert_called_once_with(cuda)


def test_z_image_text_encoder_keeps_other_cuda_device(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(monkeypatch, cuda, cuda, cuda, (12, 1))
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == cuda
    assert clip.patcher.load_device == cuda
    assert clip.patcher.offload_device == cuda
    assert capability.call_count == 3
    assert capability.call_args_list == [call(cuda)] * 3


def test_non_z_image_text_encoder_keeps_blackwell_cuda_device(monkeypatch):
    cuda = torch.device("cuda:0")
    capability = patch_clip_dependencies(monkeypatch, cuda, cuda, cuda, (12, 0))
    target = make_target("comfy.text_encoders.other")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == cuda
    assert clip.patcher.load_device == cuda
    assert clip.patcher.offload_device == cuda
    capability.assert_not_called()


def test_z_image_text_encoder_does_not_query_cuda_for_cpu(monkeypatch):
    cpu = torch.device("cpu")
    capability = patch_clip_dependencies(monkeypatch, cpu, cpu, cpu, (12, 0))
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == cpu
    assert clip.patcher.load_device == cpu
    assert clip.patcher.offload_device == cpu
    capability.assert_not_called()


def test_z_image_text_encoder_does_not_query_cuda_for_mps(monkeypatch):
    mps = torch.device("mps")
    capability = patch_clip_dependencies(monkeypatch, mps, mps, mps, (12, 0))
    target = make_target("comfy.text_encoders.z_image")

    clip = sd.CLIP(target=target)

    assert clip.cond_stage_model.device == mps
    assert clip.patcher.load_device == mps
    assert clip.patcher.offload_device == mps
    capability.assert_not_called()
