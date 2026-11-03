import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_patcher
import comfy.ldm.lightricks.vae.na_diffusion_decoder as na_decoder

TINY_DECODER = dict(
    in_channels=8, out_channels=3, patch_size=2, head_dim=32,
    stage_channels=(256, 128, 64, 64, 32), stage_depths=(1, 1, 1, 1, 2),
    stage_kernels=((3, 3, 3),) * 5, stage5_kernel=(3, 3, 3), t_emb_dim=64,
)


def _tiny_decoder(dtype):
    torch.manual_seed(0)
    model = na_decoder.NADiffusionDecoder(**TINY_DECODER)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model.eval().requires_grad_(False).to(dtype)


def _decode(model, z):
    return model(z, generator=torch.Generator(device=z.device).manual_seed(0))


def test_ltx_diffusion_decoder_linears_use_comfy_ops():
    model = _tiny_decoder(torch.float32)
    plain = [name for name, m in model.named_modules() if type(m) is torch.nn.Linear]
    assert plain == []


def test_ltx_diffusion_decoder_vae_compute_dtype_fp32():
    device = torch.device("cpu")
    z = torch.randn(1, 8, 2, 2, 2)
    reference = _decode(_tiny_decoder(torch.bfloat16).float(), z)

    model = _tiny_decoder(torch.bfloat16)
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=device)
    patcher.set_model_compute_dtype(torch.float32)
    patcher.patch_model(device_to=device)
    try:
        out = _decode(model, z)
    finally:
        patcher.unpatch_model(device_to=device)

    assert model.conv_in.weight.dtype == torch.bfloat16
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference)
