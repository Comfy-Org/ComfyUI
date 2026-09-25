from comfy.cli_args import args

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as comfy_ops
from comfy.ldm.minimax.model import FinalLayer, MiniMaxH3Model


def _final_layer(heads=1, hidden=8, t_dim=8, video_dim=16, audio_dim=8):
    layer = FinalLayer(
        hidden, t_dim, video_dim, audio_dim, 1e-5,
        dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init,
    )
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_()
    if heads > 1:
        layer.video_out.weight = nn.Parameter(torch.randn(heads * video_dim, hidden, dtype=torch.float32))
        layer.video_out.bias = nn.Parameter(torch.randn(heads * video_dim, dtype=torch.float32))
        layer.audio_out.weight = nn.Parameter(torch.randn(heads * audio_dim, hidden, dtype=torch.float32))
        layer.audio_out.bias = nn.Parameter(torch.randn(heads * audio_dim, dtype=torch.float32))
    return layer, hidden, t_dim, video_dim, audio_dim


def _layer_inputs(hidden, t_dim):
    x = torch.randn(4, hidden)
    t_emb = torch.randn(1, t_dim)
    return x, t_emb, (0, 2, 0), (2, 4, 0)


def test_final_layer_four_arg_call_works_when_n_is_1():
    layer, hidden, t_dim, video_dim, audio_dim = _final_layer()
    v, a = layer(*_layer_inputs(hidden, t_dim))
    assert tuple(v.shape) == (2, video_dim)
    assert tuple(a.shape) == (2, audio_dim)


def test_final_layer_pdd_heads_without_schedule_raise():
    layer, hidden, t_dim, _, _ = _final_layer(heads=2)
    with pytest.raises(ValueError, match="PDD heads need the sampler's sigma schedule"):
        layer(*_layer_inputs(hidden, t_dim))


def test_final_layer_pdd_heads_with_schedule_return_output():
    layer, hidden, t_dim, video_dim, audio_dim = _final_layer(heads=2)
    x, t_emb, video_seg, audio_seg = _layer_inputs(hidden, t_dim)
    v, a = layer(
        x, t_emb, video_seg, audio_seg,
        torch.tensor(0.5), torch.tensor([1.0, 0.5, 0.0]), (12.0, 3.0),
    )
    assert tuple(v.shape) == (2, video_dim)
    assert tuple(a.shape) == (2, audio_dim)


def test_h3_forward_four_arg_wrapper_when_n_is_1():
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
        latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=8,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        rope_inv_freq_len=16, dtype=torch.float32, device="cpu",
        operations=comfy_ops.disable_weight_init,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
        model.rope.inv_freq.copy_(torch.ones_like(model.rope.inv_freq))

    orig = model.final_layer.forward

    def forward(x, t_emb, video_seg, audio_seg):
        return orig(x, t_emb, video_seg, audio_seg)

    model.final_layer.forward = forward
    video, audio = model._forward(
        [torch.randn(1, 24, 1, 2, 2), torch.randn(1, 32, 2, 1)],
        torch.tensor([500.0]),
        torch.randn(1, 2, 8),
        {},
    )
    assert tuple(video.shape) == (1, 24, 1, 2, 2)
    assert tuple(audio.shape) == (1, 32, 2, 1)
