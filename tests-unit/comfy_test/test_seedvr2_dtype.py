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


def test_seedvr2_vae_decode_memory_covers_full_frame_lab_transfer():
    wrapper = seedvr_vae.VideoAutoencoderKLWrapper.__new__(seedvr_vae.VideoAutoencoderKLWrapper)
    latent_channels = seedvr_vae.SEEDVR2_LATENT_CHANNELS
    estimate = wrapper.comfy_memory_used_decode((1, latent_channels, 26, 120, 160))
    old_estimate = latent_channels * 120 * 160 * (4 * 8 * 8) * 2

    assert estimate == 101 * 960 * 1280 * 160
    assert estimate > 15 * 1024 ** 3
    assert estimate > old_estimate * 100


def test_seedvr2_vae_encode_preserves_compute_dtype(monkeypatch):
    wrapper = seedvr_vae.VideoAutoencoderKLWrapper.__new__(seedvr_vae.VideoAutoencoderKLWrapper)
    nn.Module.__init__(wrapper)
    wrapper._dummy = nn.Parameter(torch.empty(1, dtype=torch.float16))
    input_dtype = None

    def encode(self, x):
        nonlocal input_dtype
        input_dtype = x.dtype
        return x

    monkeypatch.setattr(seedvr_vae.VideoAutoencoderKL, "encode", encode)

    x = torch.zeros((1, 3, 1, 8, 8), dtype=torch.float32)
    wrapper._encode_with_raw_latent(x)

    assert input_dtype == torch.float32


def test_seedvr2_vae_ops_cast_weights_to_compute_dtype():
    attention = seedvr_vae.Attention(query_dim=4, heads=1, dim_head=4).to(torch.float16)
    hidden_states = torch.zeros((1, 2, 4), dtype=torch.float32)

    output = attention(hidden_states)

    assert output.dtype == torch.float32
