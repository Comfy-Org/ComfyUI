import hashlib
import os
from pathlib import Path

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.diffusers_convert
import comfy.model_management
import comfy.sd
import comfy.utils


@pytest.mark.parametrize("official_weights", (False, True), ids=("random", "official"))
def test_flux2_vae_load_encode_decode_matches_llada_reference(official_weights, monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    device = torch.device(os.environ.get("LLADA_IMAGE_PARITY_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("LLADA_IMAGE_PARITY_DEVICE=cuda requires a CUDA PyTorch build")
    if device.type == "cuda":
        # Core's 1x1 convolutions and Diffusers' linears need the same FP32 policy.
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        # Isolate the adapter from cuDNN convolution vs linear rounding differences.
        monkeypatch.setattr(torch.backends.cudnn, "enabled", False)
    weights = None
    if official_weights:
        weights_path = os.environ.get("LLADA_IMAGE_VAE_WEIGHTS")
        if not weights_path:
            pytest.skip("set LLADA_IMAGE_VAE_WEIGHTS to the pinned official VAE file")
        digest = hashlib.sha256()
        with Path(weights_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == "19874383f29bd4a716be716520647261e38aeaa50dcc29559e5f8d8186cf8f43"
        weights = comfy.utils.load_torch_file(weights_path, safe_load=True)
    torch.manual_seed(172)
    # Pinned LLaDA-Image VAE configuration; random weights isolate the adapter.
    reference = diffusers.AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=32,
        norm_num_groups=32,
        sample_size=1024,
        patch_size=(2, 2),
        batch_norm_eps=1e-4,
        batch_norm_momentum=0.1,
        use_quant_conv=True,
        use_post_quant_conv=True,
    ).eval()
    if weights is None:
        reference.bn.running_mean.copy_(torch.linspace(-0.3, 0.3, 128))
        reference.bn.running_var.copy_(torch.linspace(0.5, 1.5, 128))
    else:
        reference.load_state_dict(weights, strict=True)
    source_state = reference.state_dict()
    vae = comfy.sd.VAE(
        sd=dict(source_state), device=device, dtype=torch.float32
    )
    converted = comfy.diffusers_convert.convert_vae_state_dict(dict(source_state))
    actual_state = vae.first_stage_model.state_dict()
    assert set(actual_state) == set(converted)
    for key, expected in converted.items():
        assert torch.equal(actual_state[key], expected), key
    assert vae.latent_channels == 128
    assert vae.downscale_ratio == vae.upscale_ratio == 16
    reference.to(device)

    image = torch.rand(1, 32, 64, 3)
    try:
        with torch.inference_mode():
            unpatched = reference.encode(image.movedim(-1, 1).to(device) * 2 - 1).latent_dist.mode()
            batch, channels, height, width = unpatched.shape
            patched = (
                unpatched.reshape(batch, channels, height // 2, 2, width // 2, 2)
                .permute(0, 1, 3, 5, 2, 4)
                .reshape(batch, channels * 4, height // 2, width // 2)
            )
            mean = reference.bn.running_mean.view(1, -1, 1, 1)
            std = (reference.bn.running_var.view(1, -1, 1, 1) + 1e-4).sqrt()
            expected_latent = (patched - mean) / std
            actual_latent = vae.encode(image)
            torch.testing.assert_close(actual_latent.cpu(), expected_latent.cpu(), rtol=1e-4, atol=1e-5)

            # Decode identical normalized latents to isolate the reverse adapter.
            # Keep the fixture identical across CPU and CUDA execution.
            normalized = torch.randn(expected_latent.shape, dtype=torch.float32).to(device)
            patched = normalized * std + mean
            unpatched = (
                patched.reshape(batch, channels, 2, 2, height // 2, width // 2)
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(batch, channels, height, width)
            )
            expected_image = reference.decode(unpatched).sample
            expected_image = ((expected_image + 1) / 2).clamp(0, 1).movedim(1, -1)
            actual_image = vae.decode(normalized)
            torch.testing.assert_close(actual_image.cpu(), expected_image.cpu(), rtol=1e-4, atol=1e-5)
    finally:
        comfy.model_management.unload_all_models()
