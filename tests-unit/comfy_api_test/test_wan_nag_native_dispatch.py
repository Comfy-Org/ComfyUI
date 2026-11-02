import asyncio

import pytest
import torch

from comfy import model_management
from comfy.ldm.wan.model import WanI2VCrossAttention, WanT2VCrossAttention
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import _sdk
from comfy_api.latest._model_transforms import _wan_i2v_nag_forward, _wan_nag_forward


@pytest.mark.parametrize('image_conditioning', [False, True])
def test_wan_nag_accepts_native_cross_attention_dispatch(image_conditioning, monkeypatch):
    monkeypatch.setattr(model_management, 'get_torch_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(model_management, 'unet_dtype', lambda: torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        attention_type = WanI2VCrossAttention if image_conditioning else WanT2VCrossAttention
        attention = attention_type(
            4, 1, qk_norm=False,
            operation_settings={'operations': torch.nn, 'device': torch.device('cpu'),
                                'dtype': torch.float32})
        block = torch.nn.Module()
        block.cross_attn = attention
        diffusion = torch.nn.Module()
        diffusion.text_embedding = torch.nn.Identity()
        diffusion.blocks = torch.nn.ModuleList([block])
        model = torch.nn.Module()
        model.diffusion_model = diffusion
        original = ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
        negative = torch.randn(1, 2, 4)
        x = torch.randn(1, 3, 4)
        context = torch.randn(1, 3, 4)

    async def patch():
        refs = _sdk.InProcessRefResolver()
        model_ref = _sdk.ModelRef._wrap(await refs.create('MODEL', original))
        conditioning = _sdk.CondRef._wrap(
            await refs.create('CONDITIONING', [[negative, {}]]))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await model_ref.patch(
                'wan_video_nag', conditioning=conditioning,
                nag_scale=2.0, nag_alpha=0.25, nag_tau=2.5)
        return await refs.resolve(result)

    patched = asyncio.run(patch())
    forward = patched.object_patches['diffusion_model.blocks.0.cross_attn.forward']
    actual = forward(x, context, context_img_len=1, transformer_options={})
    if image_conditioning:
        expected = _wan_i2v_nag_forward(
            attention, x, context, 1, negative, 2.0, 0.25, 2.5, False, {})
    else:
        expected = _wan_nag_forward(
            attention, x, context, negative, 2.0, 0.25, 2.5, 'default', False, {})
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert patched is not original
    assert original.object_patches == {}
