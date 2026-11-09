import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.sd
import comfy.supported_models
import comfy.ldm.seedvr.model as seedvr_model
import comfy.ldm.seedvr.vae as seedvr_vae


def test_seedvr2_fp16_manual_cast_only_for_bf16_device(monkeypatch):
    bf16_device = object()
    fp16_device = object()

    monkeypatch.setattr(
        comfy.supported_models.comfy.model_management,
        "should_use_bf16",
        lambda device=None: device is bf16_device,
    )

    bf16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    bf16_config.set_inference_dtype(torch.float16, None, device=bf16_device)
    assert bf16_config.manual_cast_dtype is torch.bfloat16

    fp16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    fp16_config.set_inference_dtype(torch.float16, None, device=fp16_device)
    assert fp16_config.manual_cast_dtype is None


def test_seedvr2_text_conditioning_accepts_cfg1_single_branch():
    context = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

    torch.testing.assert_close(txt, context.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3]], device=context.device))


def test_seedvr2_vae_encode_preserves_compute_dtype(monkeypatch):
    wrapper = seedvr_vae.VideoAutoencoderKLWrapper.__new__(seedvr_vae.VideoAutoencoderKLWrapper)
    nn.Module.__init__(wrapper)
    wrapper._dummy = nn.Parameter(torch.empty(1, dtype=torch.float16))
    input_dtype = None

    def slicing_encode(self, x):
        nonlocal input_dtype
        input_dtype = x.dtype
        return x

    monkeypatch.setattr(seedvr_vae.VideoAutoencoderKLWrapper, "slicing_encode", slicing_encode)

    x = torch.zeros((1, 3, 1, 8, 8), dtype=torch.float32)
    wrapper.encode(x)

    assert input_dtype == torch.float32


def test_seedvr2_vae_ops_cast_weights_to_compute_dtype():
    attention = seedvr_vae.Attention(query_dim=4, norm_num_groups=2, eps=1e-6).to(torch.float16)
    hidden_states = torch.zeros((1, 4, 2, 2), dtype=torch.float32)

    output = attention(hidden_states)

    assert output.dtype == torch.float32
