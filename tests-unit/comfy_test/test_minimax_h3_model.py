from types import SimpleNamespace

import torch
from torch import nn

import comfy.ops
import comfy.ldm.minimax.model as minimax_model
from comfy.ldm.minimax.model import MLP, Attention, DiTBlock, MiniMaxH3Model, time_shift_sigma
from comfy.model_base import MiniMaxH3
from comfy.model_sampling import CONST


def make_model(video_output, audio_output):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    nn.Module.__init__(model)
    model.sigma_shift_video = 12.0
    model.sigma_shift_audio = 3.0
    model._forward = lambda *args, **kwargs: [video_output.clone(), audio_output.clone()]
    return model


def test_forward_scales_velocity_to_mask_timestep():
    video_output = torch.full((1, 2, 1, 2, 2), 2.0)
    audio_output = torch.full((1, 2, 2, 3), 3.0)
    video_mask = torch.tensor([[[[[1.0, 0.75], [0.5, 0.25]]]]])
    audio_mask = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    sigma = torch.tensor([0.5])
    clean = torch.arange(video_output.numel(), dtype=torch.float32).reshape_as(video_output)
    model_input = clean + sigma.reshape(1, 1, 1, 1, 1) * video_mask * video_output
    model = make_model(video_output, audio_output)

    out = model(
        [model_input, torch.zeros_like(audio_output)],
        sigma * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 1.0},
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    torch.testing.assert_close(out[0], video_output * video_mask)
    torch.testing.assert_close(out[1], audio_output * audio_mask)
    denoised = CONST.calculate_denoised(None, sigma, out[0], model_input)
    torch.testing.assert_close(denoised, clean)


def test_forward_scales_audio_velocity_before_carry_conversion():
    video_output = torch.ones((1, 1, 1, 1, 1))
    audio_output = torch.full((1, 1, 2, 2), 3.0)
    audio_src = torch.full_like(audio_output, 2.0)
    audio_mask = torch.tensor([[[[0.75, 0.5], [0.25, 0.0]]]])
    model = make_model(video_output, audio_output)
    sigma_v = torch.tensor(0.5)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry = sigma_a / sigma_v

    out = model(
        [torch.zeros_like(video_output), audio_src],
        sigma_v.reshape(1) * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 4.0},
        audio_denoise_mask=audio_mask,
    )

    expected = -3.0 * audio_src * carry + (1.0 + 3.0 * sigma_a) * audio_output * audio_mask
    torch.testing.assert_close(out[1], expected)


# wide enough that fc2's input leaves fp16 range at this activation scale
MLP_HIDDEN, MLP_FFN, MLP_INPUT_SCALE = 128, 256, 300.0


def make_mlp(dtype, reference=None):
    mlp = MLP(MLP_HIDDEN, MLP_FFN, dtype=dtype, device="cpu", operations=comfy.ops.disable_weight_init)
    if reference is None:
        torch.manual_seed(0)
        torch.nn.init.normal_(mlp.fc1.weight, std=0.05)
        torch.nn.init.normal_(mlp.fc2.weight, std=0.05)
    else:
        mlp.fc1.weight.data.copy_(reference.fc1.weight.data)
        mlp.fc2.weight.data.copy_(reference.fc2.weight.data)
    return mlp


def mlp_input(dtype):
    torch.manual_seed(1)
    return (torch.randn(8, MLP_HIDDEN) * MLP_INPUT_SCALE).to(dtype)


def unscaled_mlp(mlp, x):
    """What the projection returns without the fc2 rescale."""
    gate, up = mlp.fc1(x).chunk(2, dim=-1)
    return mlp.fc2(torch.nn.functional.silu(gate).mul_(up))


def make_block(dtype):
    block = DiTBlock(32, 2, 16, 64, 16, 1e-6, 1e-6, dtype=dtype, device="cpu",
                     operations=comfy.ops.disable_weight_init)
    block.requires_grad_(False)
    torch.manual_seed(0)
    for param in block.parameters():
        param.normal_(std=0.05)
    return block


def run_block(block, dtype):
    x = torch.randn(4, 32, dtype=torch.float32 if dtype == torch.float16 else dtype)
    return block(x, torch.randn(1, 16, dtype=dtype), [(0, 4, 0)],
                 torch.randn(1, 4, 1, 8, 2, 2, dtype=dtype),
                 transformer_options={"minimax_branch_dtype": dtype})


def test_mlp_fp16_rescale_keeps_fc2_output_in_range():
    mlp = make_mlp(torch.float16, make_mlp(torch.float32))
    x = mlp_input(torch.float16)

    assert not torch.isfinite(unscaled_mlp(mlp, x)).any()
    assert torch.isfinite(mlp(x)).all()


def test_mlp_fp16_rescale_matches_fp32_reference():
    reference = make_mlp(torch.float32)
    mlp = make_mlp(torch.float16, reference)

    expected = reference(mlp_input(torch.float32))
    error = (mlp(mlp_input(torch.float16)) - expected).abs().max()
    assert error <= 1e-3 * expected.abs().max()


def test_mlp_rescale_only_applies_to_fp16():
    # the rescale leaves fp32, every other dtype stays on the fused path
    assert make_mlp(torch.float16).forward(mlp_input(torch.float16)).dtype == torch.float32
    assert make_mlp(torch.bfloat16).forward(mlp_input(torch.bfloat16)).dtype == torch.bfloat16
    assert make_mlp(torch.float32).forward(mlp_input(torch.float32)).dtype == torch.float32


def test_dit_block_carries_residual_in_fp32_under_fp16():
    out = run_block(make_block(torch.float16), torch.float16)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_dit_block_residual_keeps_its_dtype_otherwise():
    assert run_block(make_block(torch.bfloat16), torch.bfloat16).dtype == torch.bfloat16
    assert run_block(make_block(torch.float32), torch.float32).dtype == torch.float32


def test_text_preprocessing_runs_in_fp32_against_fp16_weights():
    # matching weight and compute dtypes select the non-casting operations, so the fp32
    # preprocessing reaches condition_proj and the refiner while their weights are fp16
    model = MiniMaxH3Model(hidden_size=64, num_layers=1, token_refiner_num_layers=1,
                           num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=128,
                           text_dim=48, time_embed_hidden_size=64, time_embed_dim=32,
                           dtype=torch.float16, device="cpu",
                           operations=comfy.ops.disable_weight_init)
    model.requires_grad_(False)
    torch.manual_seed(0)
    for parameter in model.parameters():
        parameter.normal_(std=0.02)

    refined = model.preprocess_text_embeds(torch.randn(1, 6, 48, dtype=torch.float32))

    assert refined.shape == (1, 6, 64)
    assert torch.isfinite(refined).all()


def test_attention_out_proj_rescale_keeps_high_activations_in_range():
    attention = Attention(64, 2, 32, 1e-6, dtype=torch.float16, device="cpu",
                          operations=comfy.ops.disable_weight_init)
    attention.requires_grad_(False)
    for parameter in attention.parameters():
        parameter.fill_(1.0)
    # a unit projection sums its 64 inputs, so this lands at 1.28e5, past fp16's 65504
    activation = torch.full((1, 6, 64), 2000.0, dtype=torch.float16)
    original = minimax_model.optimized_attention
    minimax_model.optimized_attention = lambda *args, **kwargs: activation.clone()
    try:
        projected = attention(torch.zeros(6, 64, dtype=torch.float16))
    finally:
        minimax_model.optimized_attention = original

    assert not torch.isfinite(attention.out_proj(activation.squeeze(0))).any()
    assert projected.dtype == torch.float32
    torch.testing.assert_close(projected, torch.full((6, 64), 128000.0), rtol=0.0, atol=0.0)


def test_extra_conds_preprocesses_in_fp32_and_returns_inference_dtype():
    seen = {}

    def record(text_states):
        seen["dtype"] = text_states.dtype
        return torch.zeros(1, 4, 64, dtype=text_states.dtype)

    model = MiniMaxH3.__new__(MiniMaxH3)
    model.concat_keys = ()
    model.latent_shapes = None
    model.get_dtype_inference = lambda: torch.float16
    model.diffusion_model = SimpleNamespace(preprocess_text_embeds=record)

    out = model.extra_conds(cross_attn=torch.randn(1, 4, 48), device=torch.device("cpu"))

    assert seen["dtype"] == torch.float32
    assert out["c_crossattn"].cond.dtype == torch.float16
