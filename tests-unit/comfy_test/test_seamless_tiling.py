from unittest.mock import MagicMock

import pytest
import torch

from comfy_extras.nodes_seamless_tiling import ModelPatchSeamlessTiling, VAEDecodeSeamlessTiling, _periodic_pad_2d


def test_periodic_pad_2d_wraps_both_axes():
    x = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])

    output = _periodic_pad_2d(x, 1, 1)

    assert torch.equal(output, torch.tensor([[[[6, 4, 5, 6, 4], [3, 1, 2, 3, 1], [6, 4, 5, 6, 4], [3, 1, 2, 3, 1]]]]))


def test_periodic_pad_2d_preserves_video_latent_time_axis():
    x = torch.arange(12).reshape(1, 1, 2, 2, 3)

    output = _periodic_pad_2d(x, 1, 0)

    assert output.shape == (1, 1, 2, 4, 3)
    assert torch.equal(output[:, :, :, 1:3], x)
    assert torch.equal(output[:, :, :, 0], x[:, :, :, -1])
    assert torch.equal(output[:, :, :, -1], x[:, :, :, 0])


def test_model_patch_shifts_input_and_concat_conditioning_then_blends_output():
    patched_model = MagicMock()
    patched_model.model_options = {}
    model = MagicMock()
    model.clone.return_value = patched_model
    x = torch.arange(6).reshape(1, 1, 2, 3)
    c_concat = x + 10

    ModelPatchSeamlessTiling.execute(model, "x and y", 1)
    wrapper = patched_model.set_model_unet_function_wrapper.call_args.args[0]
    calls = []

    def apply_model(input, timestep, **c):
        calls.append(input)
        assert torch.equal(c["c_concat"], input + 10)
        return input

    output = wrapper(apply_model, {"input": x, "timestep": torch.ones(1), "c": {"c_concat": c_concat}})

    assert torch.equal(output, x)
    assert len(calls) == 4


def test_model_patch_handles_video_latent_spatial_axes():
    patched_model = MagicMock()
    patched_model.model_options = {}
    model = MagicMock()
    model.clone.return_value = patched_model
    x = torch.arange(12).reshape(1, 1, 2, 2, 3)
    c_concat = x + 20

    ModelPatchSeamlessTiling.execute(model, "x and y", 1)
    wrapper = patched_model.set_model_unet_function_wrapper.call_args.args[0]
    calls = []

    def apply_model(input, timestep, **c):
        calls.append(input)
        assert torch.equal(c["c_concat"], input + 20)
        return input

    output = wrapper(apply_model, {"input": x, "timestep": torch.ones(1), "c": {"c_concat": c_concat}})

    assert torch.equal(output, x)
    assert len(calls) == 4


def test_model_patch_rejects_precomputed_control_features():
    patched_model = MagicMock()
    patched_model.model_options = {}
    model = MagicMock()
    model.clone.return_value = patched_model
    ModelPatchSeamlessTiling.execute(model, "x", 1)
    wrapper = patched_model.set_model_unet_function_wrapper.call_args.args[0]

    with pytest.raises(ValueError, match="cannot be used with ControlNet"):
        wrapper(MagicMock(), {"input": torch.zeros(1, 1, 2, 3), "timestep": torch.ones(1), "c": {"control": {}}})


def test_vae_decode_crops_periodic_context():
    latent = torch.arange(6).reshape(1, 1, 2, 3)
    vae = MagicMock()
    vae.decode.side_effect = lambda x: x.repeat_interleave(2, -2).repeat_interleave(2, -1).movedim(1, -1)

    output = VAEDecodeSeamlessTiling.execute({"samples": latent}, vae, "x and y", 1).result[0]

    expected = latent.repeat_interleave(2, -2).repeat_interleave(2, -1).movedim(1, -1)
    assert torch.equal(output, expected)


def test_vae_decode_crops_video_latent_and_flattens_frames():
    latent = torch.arange(12).reshape(1, 1, 2, 2, 3)
    vae = MagicMock()
    vae.decode.side_effect = lambda x: x.repeat_interleave(2, -2).repeat_interleave(2, -1).movedim(1, -1)

    output = VAEDecodeSeamlessTiling.execute({"samples": latent}, vae, "x and y", 1).result[0]

    expected = latent.repeat_interleave(2, -2).repeat_interleave(2, -1).movedim(1, -1).reshape(2, 4, 6, 1)
    assert torch.equal(output, expected)
