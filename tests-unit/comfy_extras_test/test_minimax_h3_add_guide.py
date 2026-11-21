from copy import deepcopy

import pytest
import torch
from torch import nn

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import comfy.samplers
    from comfy.conds import is_equal
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_base import MiniMaxH3
    from comfy.model_sampling import ModelSamplingAV
    from comfy.sampler_helpers import convert_cond
    from comfy.utils import pack_latents
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide, MiniMaxH3ImageToVideo
    from nodes import ConditioningCombine, ConditioningSetTimestepRange
finally:
    args.cpu = _original_cpu


class FixedEncoder:
    """Substitute only the neural video/audio encoding boundary."""

    audio_sample_rate = 32000

    def __init__(self, encoded):
        self.encoded = encoded

    def encode(self, pixels):
        return self.encoded


class FixedTextEncoder:
    def tokenize(self, prompt, images):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.arange(16, dtype=torch.float32).reshape(1, 2, 8),
                 {"branch": tokens, "pooled_output": torch.tensor([0.25, 0.75])}]]


def video_marker(value):
    return torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2) / 128 + value


def image_branch(label, value, last=False):
    encoded = video_marker(value)
    image = torch.full((1, 32, 32, 3), 0.5)
    positive, latent = MiniMaxH3ImageToVideo.execute(
        FixedTextEncoder(), FixedEncoder(encoded), label, 32, 32, 39,
        first_frame=None if last else image, last_frame=image if last else None,
    ).result
    start, end = (0.5, 1.0) if label == "B" else (0.0, 0.5)
    positive = ConditioningSetTimestepRange().set_range(positive, start, end)[0]
    return positive, latent, {"resolved_frame_index": 38 if last else 0, "latent": encoded}


def add_guide(positive, latent, kind="image", frame_idx=19, value=3.75, audio_frames=2):
    video = video_marker(value)
    audio = torch.arange(32 * 2 * audio_frames, dtype=torch.float32).reshape(1, 32, 2, audio_frames) / 128 + value
    keyframe = {"resolved_frame_index": frame_idx}
    if kind in ("image", "both"):
        keyframe["latent"] = video
    if kind in ("audio", "both"):
        keyframe["audio_latent"] = audio
    result = MiniMaxH3AddGuide.execute(
        positive, latent, frame_idx,
        vae=FixedEncoder(video), audio_vae=FixedEncoder(audio),
        image=torch.full((1, 32, 32, 3), 0.75) if "latent" in keyframe else None,
        audio={"waveform": torch.ones(1, 2, 1600), "sample_rate": 32000} if "audio_latent" in keyframe else None,
    ).result[0]
    return result, keyframe


def assert_consumed_anchors(conditioning, latent, histories):
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=24, audio_latents_dim=32, text_dim=8,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    video, audio = latent["samples"].unbind()
    noise, shapes = pack_latents((video, audio))
    adapter = MiniMaxH3.__new__(MiniMaxH3)
    nn.Module.__init__(adapter)
    adapter.diffusion_model = model
    adapter.concat_keys = ()
    adapter.manual_cast_dtype = None
    adapter.latent_shapes = shapes
    adapter.model_sampling = ModelSamplingAV(None)
    # Use the real adapter's augmentation controls to inspect exact anchor rows.
    encoded = comfy.samplers.encode_model_conds(
        adapter.extra_conds, convert_cond(conditioning), noise, torch.device("cpu"), "positive",
        latent_shapes=shapes, minimax_visual_cond_noise_aug=1.0, minimax_audio_cond_noise_aug=1.0,
    )
    captured = {}

    def capture(name):
        def hook(module, inputs):
            captured[name] = inputs[0].detach().clone()
        return hook

    handles = [model.video_patch_proj.register_forward_pre_hook(capture("video")),
               model.audio_patch_proj.register_forward_pre_hook(capture("audio"))]
    try:
        for entry, history in zip(encoded, histories):
            payload = entry["model_conds"]["minimax_payload"].cond
            layout = payload["layout"]
            context = entry["model_conds"]["c_crossattn"].cond
            packed = model._embed_and_pack(video, audio, context, layout, payload, {})
            assert torch.isfinite(packed).all()

            video_rows, audio_rows, times = [], [], []
            for anchor in history:
                origin = 2 + anchor["resolved_frame_index"] * 5 / 3
                if "latent" in anchor:
                    video_rows.append(anchor["latent"][0, :, 0].reshape(1, 96))
                    times.append(origin)
                if "audio_latent" in anchor:
                    z = anchor["audio_latent"]
                    audio_rows.append(torch.stack([z[0, :, channel, t] for channel in range(2) for t in range(z.shape[-1])]))
                    times.extend(origin + t for channel in range(2) for t in range(z.shape[-1]))
            expected_video = torch.cat(video_rows + [torch.zeros(video.shape[2], 96)])
            expected_audio = torch.cat(audio_rows + [torch.zeros(2 * audio.shape[-1], 32)])
            torch.testing.assert_close(captured["video"], expected_video, rtol=0, atol=0)
            torch.testing.assert_close(captured["audio"], expected_audio, rtol=0, atol=0)
            condition_times = torch.cat([layout.position_ids[a:b, 0] for a, b, kind in layout.segments if kind in ("cond", "cond_audio")])
            torch.testing.assert_close(condition_times, torch.tensor(times, dtype=torch.float64), rtol=0, atol=1e-12)
    finally:
        for handle in handles:
            handle.remove()
    return adapter, encoded, noise


@pytest.mark.parametrize("kind", ["image", "audio", "both"])
def test_add_guide_preserves_each_scheduled_image_history(monkeypatch, kind):
    monkeypatch.setattr(args, "gpu_only", False)
    branch_a, latent, anchor_a = image_branch("A", 1.25)
    branch_b, _, anchor_b = image_branch("B", 2.5, last=True)
    combined = ConditioningCombine().combine(branch_a, branch_b)[0]
    before = deepcopy(combined)
    latent_before = [value.clone() for value in latent["samples"].unbind()]

    result, common = add_guide(combined, latent, kind)

    adapter, encoded, noise = assert_consumed_anchors(result, latent, [[anchor_a, common], [anchor_b, common]])
    comfy.samplers.calculate_start_end_timesteps(adapter, encoded)
    for progress, expected_branch in ((0.25, "A"), (0.75, "B")):
        sigma = torch.tensor([adapter.model_sampling.percent_to_sigma(progress)])
        active = [entry["branch"] for entry in encoded if comfy.samplers.get_area_and_mult(entry, noise, sigma) is not None]
        assert active == [expected_branch]
    assert is_equal(combined, before)
    for output, source, original in zip(result, combined, before):
        assert output[0] is source[0]
        assert is_equal({key: value for key, value in output[1].items() if key != "minimax_keyframes"},
                        {key: value for key, value in original[1].items() if key != "minimax_keyframes"})
    for actual, original in zip(latent["samples"].unbind(), latent_before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("prior", ["missing", "empty", "none"])
def test_add_guide_keeps_later_history_when_first_entry_has_none(monkeypatch, prior):
    monkeypatch.setattr(args, "gpu_only", False)
    first = FixedTextEncoder().encode_from_tokens_scheduled("A")
    if prior != "missing":
        first[0][1]["minimax_keyframes"] = [] if prior == "empty" else None
    second, latent, previous = image_branch("B", 2.5, last=True)
    combined = ConditioningCombine().combine(first, second)[0]
    before = deepcopy(combined)

    result, common = add_guide(combined, latent, "both")

    assert_consumed_anchors(result, latent, [[common], [previous, common]])
    assert is_equal(combined, before)


@pytest.mark.parametrize("with_previous", [False, True])
def test_add_guide_single_entry_appends_even_at_same_frame(monkeypatch, with_previous):
    monkeypatch.setattr(args, "gpu_only", False)
    if with_previous:
        positive, latent, previous = image_branch("A", 1.25)
        history = [previous]
    else:
        positive = FixedTextEncoder().encode_from_tokens_scheduled("A")
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 39).result[0]
        history = []
    before = deepcopy(positive)

    result, common = add_guide(positive, latent, "both", frame_idx=0)

    assert_consumed_anchors(result, latent, [history + [common]])
    assert len(result[0][1]["minimax_keyframes"]) == len(history) + 1
    assert is_equal(positive, before)


def test_add_guide_commutes_with_combine_and_preserves_audio_branch_order(monkeypatch):
    monkeypatch.setattr(args, "gpu_only", False)
    latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 39).result[0]
    branch_a, anchor_a = add_guide(FixedTextEncoder().encode_from_tokens_scheduled("A"), latent, "audio", 0, 1.25, 1)
    branch_b, anchor_b = add_guide(FixedTextEncoder().encode_from_tokens_scheduled("B"), latent, "audio", 38, 2.5, 1)
    original = deepcopy((branch_a, branch_b))
    each_a, common = add_guide(branch_a, latent, "both")
    each_b, _ = add_guide(branch_b, latent, "both")

    for first, second, first_history, second_history, expected_first, expected_second in (
        (branch_a, branch_b, anchor_a, anchor_b, each_a, each_b),
        (branch_b, branch_a, anchor_b, anchor_a, each_b, each_a),
    ):
        combined = ConditioningCombine().combine(first, second)[0]
        actual, _ = add_guide(combined, latent, "both")
        assert_consumed_anchors(actual, latent, [[first_history, common], [second_history, common]])
        expected = ConditioningCombine().combine(expected_first, expected_second)[0]
        assert is_equal(actual, expected)
    assert is_equal((branch_a, branch_b), original)
