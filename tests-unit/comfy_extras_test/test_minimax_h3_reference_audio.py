from copy import deepcopy

import pytest
import torch
from torch import nn

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    from comfy.conds import is_equal
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_base import MiniMaxH3
    from comfy.model_sampling import ModelSamplingAV
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
finally:
    args.cpu = _original_cpu


class TextEncoder:
    def tokenize(self, prompt, minimax_ref_items):
        self.items = minimax_ref_items
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.arange(16, dtype=torch.float32).reshape(1, 2, 8), {"prompt": tokens}]]


class AudioEncoder:
    audio_sample_rate = 32000

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def encode(self, waveform):
        self.inputs.append(waveform.clone())
        # Keep stereo/time markers while replacing only the neural encoder.
        samples = waveform.unfold(1, 800, 800).mean(-1).movedim(1, -1)
        encoded = samples[:, None] + torch.arange(32).reshape(1, 32, 1, 1) / 32
        self.outputs.append(encoded)
        return encoded


class VideoEncoder:
    def __init__(self):
        self.inputs = []

    def encode(self, frames):
        self.inputs.append(frames.clone())
        t = 2 + 5 * ((frames.shape[0] - 5) // 17)
        return torch.full((1, 24, t, frames.shape[1] // 16, frames.shape[2] // 16), frames.mean())


def soundtrack(value=0.1, frames=3, sample_rate=32000):
    count = frames * sample_rate // 40
    waveform = torch.arange(count, dtype=torch.float32).reshape(1, 1, -1) / count / 10
    waveform = waveform + torch.tensor([value, value + 0.25]).reshape(1, 2, 1)
    return {"waveform": waveform, "sample_rate": sample_rate}


def reference(clip, **kwargs):
    return MiniMaxH3ReferenceToVideo.execute(clip, "reference", 32, 32, 39, **kwargs).result


@pytest.fixture
def packing_model(monkeypatch):
    monkeypatch.setattr(args, "gpu_only", False)
    with torch.random.fork_rng(devices=[]):
        return MiniMaxH3Model(
            hidden_size=8, num_layers=0, token_refiner_num_layers=0,
            num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
            latents_dim=24, audio_latents_dim=32, text_dim=8,
            timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
            dtype=torch.float32, device="cpu", operations=nn,
        )


def consumed_audio(model, conditioning, latent, expected_audio):
    video, audio = latent["samples"].unbind()
    adapter = MiniMaxH3.__new__(MiniMaxH3)
    nn.Module.__init__(adapter)
    adapter.diffusion_model = model
    adapter.concat_keys = ()
    adapter.manual_cast_dtype = None
    adapter.latent_shapes = [video.shape, audio.shape]
    adapter.model_sampling = ModelSamplingAV(None)
    context, metadata = conditioning[0]
    encoded = adapter.extra_conds(
        cross_attn=context, device=torch.device("cpu"), latent_shapes=adapter.latent_shapes,
        minimax_audio_cond_noise_aug=1.0, minimax_visual_cond_noise_aug=1.0, **metadata,
    )
    payload = encoded["minimax_payload"].cond
    layout = payload["layout"]
    captured = []
    handle = model.audio_patch_proj.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        packed = model._embed_and_pack(video, audio, encoded["c_crossattn"].cond, layout, payload, {})
    finally:
        handle.remove()
    assert torch.isfinite(packed).all()
    rows = [torch.stack([z[0, :, channel, t] for channel in range(2) for t in range(z.shape[-1])])
            for z in expected_audio]
    expected = torch.cat(rows + [torch.zeros(2 * audio.shape[-1], 32)])
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    reference_rows = sum(2 * z.shape[-1] for z in expected_audio)
    assert not layout.audio_update[:reference_rows].any()
    assert layout.audio_update[reference_rows:].all()
    return captured[0][:reference_rows], payload


def assert_audio_only_layout(payload, audio_latents):
    layout = payload["layout"]
    assert not payload.get("cond_video_latents", [])
    assert layout.img_update.all()
    assert all("latent" not in block for block in payload["refs"])
    assert all(kind != "ref_img" for _, _, kind in layout.segments)
    times, cursor = [], 2
    for z in audio_latents:
        times.extend(cursor + t for channel in range(2) for t in range(z.shape[-1]))
        cursor += z.shape[-1]
    actual = torch.cat([layout.position_ids[a:b, 0] for a, b, kind in layout.segments if kind == "ref_audio"])
    torch.testing.assert_close(actual, torch.tensor(times, dtype=torch.float64), rtol=0, atol=0)
    for a, _, kind in layout.segments:
        if kind in ("video", "audio"):
            assert layout.position_ids[a, 0] == cursor


@pytest.mark.parametrize("with_video,with_audio,kind", [
    (False, False, None), (True, False, "video"),
    (False, True, "audio"), (True, True, "video_audio"),
])
def test_paired_soundtrack_uses_available_encoders(packing_model, with_video, with_audio, kind):
    clip, video_vae, audio_vae = TextEncoder(), VideoEncoder(), AudioEncoder()
    frames, audio = torch.full((5, 32, 32, 3), 0.4), soundtrack()
    original = deepcopy((frames, audio))
    conditioning, latent = reference(
        clip, vae=video_vae if with_video else None, audio_vae=audio_vae if with_audio else None,
        ref_videos={"ref_video_1": frames}, ref_video_audios={"ref_video_audio_1": audio},
    )
    assert len(video_vae.inputs) == int(with_video)
    assert len(audio_vae.inputs) == int(with_audio)
    assert [item["type"] for item in clip.items] == ["audio", "video"]
    assert clip.items[1]["timestamps"] == [0.0]
    torch.testing.assert_close(clip.items[1]["data"], frames[:1], rtol=0, atol=0)
    refs = conditioning[0][1].get("minimax_refs", [])
    assert [block["kind"] for block in refs] == ([kind] if kind else [])
    if with_video:
        assert set(refs[0]) == {"kind", "latent_t", "latent_h", "latent_w", "latent", "audio_latent", "ref_audio_t"}
        torch.testing.assert_close(video_vae.inputs[0], frames, rtol=0, atol=0)
    if with_audio:
        torch.testing.assert_close(audio_vae.inputs[0], audio["waveform"].movedim(1, -1), rtol=0, atol=0)
        torch.testing.assert_close(refs[0]["audio_latent"], audio_vae.outputs[0], rtol=0, atol=0)
        assert refs[0]["ref_audio_t"] == 3
    _, payload = consumed_audio(packing_model, conditioning, latent, audio_vae.outputs)
    if kind == "audio":
        assert set(refs[0]) == {"kind", "ref_audio_t", "audio_latent"}
        assert_audio_only_layout(payload, audio_vae.outputs)
    elif kind == "video_audio":
        layout = payload["layout"]
        origins = [layout.position_ids[a, 0] for a, _, name in layout.segments if name in ("ref_img", "ref_audio")]
        assert origins == [2, 2]
    assert is_equal((frames, audio), original)


@pytest.mark.parametrize("sample_rate", [32000, 16000])
def test_paired_soundtrack_matches_standalone_audio(packing_model, sample_rate):
    frames, audio = torch.full((22, 32, 32, 3), 0.6), soundtrack(sample_rate=sample_rate)
    before = deepcopy(audio)
    paired_encoder, standalone_encoder = AudioEncoder(), AudioEncoder()
    paired, latent = reference(TextEncoder(), audio_vae=paired_encoder,
                               ref_videos={"ref_video_2": frames}, ref_video_audios={"ref_video_audio_2": audio})
    standalone, standalone_latent = reference(TextEncoder(), audio_vae=standalone_encoder, ref_audios={"ref_audio_1": audio})

    paired_rows, payload = consumed_audio(packing_model, paired, latent, standalone_encoder.outputs)
    standalone_rows, _ = consumed_audio(packing_model, standalone, standalone_latent, standalone_encoder.outputs)
    torch.testing.assert_close(paired_rows, standalone_rows, rtol=0, atol=0)
    assert is_equal(paired_encoder.inputs, standalone_encoder.inputs)
    assert paired_encoder.inputs[0].shape == (1, 2400, 2)
    assert_audio_only_layout(payload, standalone_encoder.outputs)
    assert is_equal(audio, before)


@pytest.mark.parametrize("with_video", [False, True])
def test_soundtracks_follow_video_suffix_and_order(packing_model, with_video):
    first, second, standalone = soundtrack(0.1, 2), soundtrack(0.5, 5), soundtrack(0.8, 3)
    inputs = {
        "ref_videos": {"ref_video_2": torch.full((24, 32, 32, 3), 0.3), "ref_video_0": torch.full((6, 32, 32, 3), 0.7)},
        "ref_video_audios": {"ref_video_audio_0": second, "ref_video_audio_2": first},
        "ref_audios": {"ref_audio_2": standalone},
    }
    before = deepcopy(inputs)
    clip, audio_vae, video_vae = TextEncoder(), AudioEncoder(), VideoEncoder()
    conditioning, latent = reference(clip, vae=video_vae if with_video else None, audio_vae=audio_vae, **inputs)
    oracle = AudioEncoder()
    reference(TextEncoder(), audio_vae=oracle, ref_audios={"ref_audio_0": first, "ref_audio_1": second, "ref_audio_2": standalone})

    _, payload = consumed_audio(packing_model, conditioning, latent, oracle.outputs)
    assert is_equal(audio_vae.inputs, oracle.inputs)
    assert [block["kind"] for block in payload["refs"]] == (["video_audio"] * 2 if with_video else ["audio"] * 2) + ["audio"]
    assert [item["type"] for item in clip.items] == ["audio", "video", "audio", "video", "audio"]
    assert clip.items[1]["timestamps"] == [0.0, 0.5]
    assert clip.items[3]["timestamps"] == [0.0]
    if with_video:
        assert [value.shape[0] for value in video_vae.inputs] == [22, 5]
    else:
        assert_audio_only_layout(payload, oracle.outputs)
    assert is_equal(inputs, before)


@pytest.mark.parametrize("with_video", [False, True])
def test_empty_and_orphan_slots_do_not_create_audio_references(packing_model, with_video):
    clip, audio_vae = TextEncoder(), AudioEncoder()
    inputs = {
        "ref_videos": {"ref_video_0": None, "ref_video_2": torch.zeros(5, 32, 32, 3)},
        "ref_video_audios": {"ref_video_audio_0": soundtrack(0.1), "ref_video_audio_1": soundtrack(0.2), "ref_video_audio_2": None},
        "ref_audios": {"ref_audio_0": None, "ref_audio_2": soundtrack(0.9)},
    }
    before = deepcopy(inputs)
    conditioning, latent = reference(clip, vae=VideoEncoder() if with_video else None, audio_vae=audio_vae, **inputs)
    assert len(audio_vae.inputs) == 1
    torch.testing.assert_close(audio_vae.inputs[0], inputs["ref_audios"]["ref_audio_2"]["waveform"].movedim(1, -1), rtol=0, atol=0)
    assert [item["type"] for item in clip.items] == ["video", "audio"]
    _, payload = consumed_audio(packing_model, conditioning, latent, audio_vae.outputs)
    assert [block["kind"] for block in payload["refs"]] == (["video"] if with_video else []) + ["audio"]
    assert is_equal(inputs, before)


def test_short_reference_video_is_rejected_before_audio_encoding(packing_model):
    audio_vae = AudioEncoder()
    with pytest.raises(ValueError, match="at least 5 frames"):
        reference(TextEncoder(), audio_vae=audio_vae, ref_videos={"ref_video_1": torch.zeros(4, 32, 32, 3)},
                  ref_video_audios={"ref_video_audio_1": soundtrack()})
    assert not audio_vae.inputs
