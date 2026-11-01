import weakref

import pytest
import torch
import torch.utils.checkpoint as cp
from torch import nn

import comfy.model_management
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, _mod_gate, time_shift_sigma
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


@pytest.mark.parametrize("conditioning", ["none", "refs", "keyframes"])
def test_embed_and_pack_releases_intermediates(conditioning):
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, audio_latents_dim=3, text_dim=5,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    video = torch.randn(1, 2, 1, 4, 6)
    audio = torch.randn(1, 3, 2, 2)
    context = torch.randn(1, 4, 8)
    payload = {}
    if conditioning != "none":
        cond_video = torch.randn_like(video)
        cond_audio = torch.randn(1, 3, 2, 1)
        payload.update(cond_video_latents=[cond_video], cond_audio_latents=[cond_audio])
        if conditioning == "refs":
            payload["refs"] = [{"kind": "image", "latent_h": 4, "latent_w": 6},
                               {"kind": "audio", "ref_audio_t": 1}]
        else:
            payload["keyframes"] = [{"resolved_frame_index": 0, "latent": cond_video,
                                      "audio_latent": cond_audio}]
    layout = PackedLayout(4, 1, 4, 6, 2, refs=payload.get("refs"), keyframes=payload.get("keyframes"))
    outputs, refs = {}, []

    def remember(name):
        def hook(module, inputs, output):
            outputs[name] = output.clone()
            refs.extend([weakref.ref(inputs[0]), weakref.ref(output)])
        return hook

    handles = [model.video_patch_proj.register_forward_hook(remember("video")),
               model.audio_patch_proj.register_forward_hook(remember("audio"))]
    with torch.no_grad():
        packed = model._embed_and_pack(video, audio, context, layout, payload, {})
    for handle in handles:
        handle.remove()

    assert all(ref() is None for ref in refs)
    assert packed.shape == (layout.seq_len, 8)
    assert packed.dtype == torch.float32
    slices = {kind: torch.cat([packed[a:b] for a, b, name in layout.segments if name in kinds])
              for kind, kinds in {"video": {"video", "ref_img", "cond"},
                                  "audio": {"audio", "ref_audio", "cond_audio"},
                                  "text": {"text"}}.items()}
    assert torch.equal(slices["video"], outputs["video"])
    assert torch.equal(slices["audio"], outputs["audio"])
    assert torch.equal(slices["text"], context[0])


NUM_BLOCKS = 32
GATE_SEQ = 16
GATE_DIM = 32


def _upstream_mod_gate(x, gate, other, segments):
    """The pre-fix implementation, kept here to prove the fixture can fail."""
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


def _run_mod_gate_chain(monkeypatch, use_checkpoint, reentrant=False):
    """Thread a stream through NUM_BLOCKS gated residuals and backprop."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(GATE_SEQ, GATE_DIM, requires_grad=True)
    gates = [torch.full((1, GATE_DIM), 0.2 + 0.001 * i).requires_grad_() for i in range(NUM_BLOCKS)]
    others = [torch.randn(GATE_SEQ, GATE_DIM) for _ in range(NUM_BLOCKS)]
    segments = [(0, GATE_SEQ, 0)]

    h = base * 1.0
    for i in range(NUM_BLOCKS):
        if use_checkpoint:
            h = cp.checkpoint(_mod_gate, h, gates[i], others[i], segments, use_reentrant=reentrant)
        else:
            h = _mod_gate(h, gates[i], others[i], segments)

    h.square().mean().backward()
    return base.grad, [g.grad for g in gates]


def test_mod_gate_training_chain_backward_produces_finite_gradients(monkeypatch):
    base_grad, gate_grads = _run_mod_gate_chain(monkeypatch, use_checkpoint=False)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert all(torch.isfinite(g).all() for g in gate_grads)


@pytest.mark.parametrize("reentrant", [False, True])
def test_mod_gate_training_chain_survives_gradient_checkpointing(monkeypatch, reentrant):
    """The pre-fix in-place update raises here; the out-of-place path must not."""
    base_grad, gate_grads = _run_mod_gate_chain(monkeypatch, use_checkpoint=True, reentrant=reentrant)
    assert base_grad is not None
    assert torch.isfinite(base_grad).all()
    assert all(torch.isfinite(g).all() for g in gate_grads)


def test_mod_gate_pre_fix_implementation_actually_fails(monkeypatch):
    """Guards the guard: confirms the in-place update is the thing that breaks."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(0)
    base = torch.randn(GATE_SEQ, GATE_DIM, requires_grad=True)
    gates = [torch.full((1, GATE_DIM), 0.2).requires_grad_() for _ in range(NUM_BLOCKS)]
    others = [torch.randn(GATE_SEQ, GATE_DIM) for _ in range(NUM_BLOCKS)]
    segments = [(0, GATE_SEQ, 0)]

    h = base * 1.0
    for i in range(NUM_BLOCKS):
        h = cp.checkpoint(_upstream_mod_gate, h, gates[i], others[i], segments, use_reentrant=False)

    with pytest.raises(RuntimeError, match="inplace"):
        h.square().mean().backward()


def test_mod_gate_training_does_not_mutate_the_stream(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    x = torch.randn(GATE_SEQ, GATE_DIM)
    original = x.clone()
    gate = torch.full((2, GATE_DIM), 0.3)
    other = torch.randn(GATE_SEQ, GATE_DIM)

    out = _mod_gate(x, gate, other, [(0, 6, 0), (6, GATE_SEQ, 1)])

    assert torch.equal(x, original)
    assert out is not x


def test_mod_gate_training_matches_the_in_place_math(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    torch.manual_seed(1)
    x = torch.randn(GATE_SEQ, GATE_DIM)
    other = torch.randn(GATE_SEQ, GATE_DIM)
    gate = torch.stack([torch.full((GATE_DIM,), 0.1), torch.full((GATE_DIM,), 0.3)])
    segments = [(0, 6, 0), (6, GATE_SEQ, 1)]

    out = _mod_gate(x.clone(), gate, other, segments)

    expected = x.clone()
    expected[0:6] += other[0:6] * gate[0]
    expected[6:] += other[6:] * gate[1]
    assert torch.allclose(out, expected, atol=1e-6)


def test_mod_gate_training_keeps_every_row(monkeypatch):
    """Zero gates must return the stream: the reconstruction cannot drop segments."""
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    x = torch.randn(GATE_SEQ, GATE_DIM)
    gate = torch.zeros((3, GATE_DIM))
    other = torch.randn(GATE_SEQ, GATE_DIM)

    out = _mod_gate(x.clone(), gate, other, [(0, 5, 0), (5, 12, 1), (12, GATE_SEQ, 2)])

    assert torch.allclose(out, x, atol=1e-6)


def test_mod_gate_gradients_match_between_checkpointed_and_plain(monkeypatch):
    """Checkpointing must not change the gradients the update is based on."""
    plain_base, plain_gates = _run_mod_gate_chain(monkeypatch, use_checkpoint=False)
    ckpt_base, ckpt_gates = _run_mod_gate_chain(monkeypatch, use_checkpoint=True)

    assert torch.allclose(plain_base, ckpt_base, atol=1e-5)
    for a, b in zip(plain_gates, ckpt_gates):
        assert torch.allclose(a, b, atol=1e-5)
