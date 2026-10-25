import json
import types
import uuid
from unittest.mock import patch

import pytest
import torch

import comfy.nested_tensor
import comfy.ops
import comfy.sd
import comfy.utils
from comfy import memory_management
from comfy import model_detection
from comfy import model_management
from comfy import samplers
from comfy import supported_models
from comfy.ldm.kandinsky6 import piflow
from comfy.ldm.kandinsky6.audio_vae import Kandinsky6AudioVAE, MelConverter
from comfy.ldm.kandinsky6.core_contract import AUDIO_DEFAULTS
from comfy.ldm.kandinsky6.detection import detect_kandinsky6, to_native_state_dict
from comfy.ldm.kandinsky6.model import Kandinsky6NativeAVDiT
from comfy_extras import nodes_kandinsky6 as k6_nodes
from comfy_extras.nodes_kandinsky6 import (
    Kandinsky6AddGuide,
    Kandinsky6AddLatentGuide,
    Kandinsky6EmptyLatent,
    Kandinsky6ImageToVideoAudio,
    Kandinsky6PiFlowGuider,
)
from comfy_extras.nodes_lt import LTXVConcatAVLatent


# Released Pro dimensions. Detection compares checkpoint shapes against these,
# so the synthetic state dict must use the real values even though the tensors
# are meta and never allocated.
_PRO = {
    "model_dim": 4096,
    "model_dim_a": 2048,
    "ff_dim": 16384,
    "ff_dim_a": 7168,
    "time_dim": 1024,
    "time_dim_a": 1024,
    "in_text_dim": 3584,
    "in_text_dim2": 768,
    "head_dim": 128,
    "head_dim_a": 128,
    "in_visual_dim": 16,
    "out_visual_dim": 16,
    "in_audio_dim": 40,
    "num_visual_blocks": 60,
    "num_text_blocks": 4,
}


def _meta(*shape):
    return torch.empty(*shape, device="meta")


def _k6_pro_state_dict():
    d = _PRO
    patch_volume = 4
    sd = {
        "visual_embeddings.in_layer.weight": _meta(d["model_dim"], (2 * d["in_visual_dim"] + 1) * patch_volume),
        "visual_token_type_embeddings.weight": _meta(2, d["model_dim"]),
        "video_time_embeddings.out_layer.weight": _meta(d["time_dim"], d["time_dim"]),
        "audio_time_embeddings.out_layer.weight": _meta(d["time_dim_a"], d["time_dim_a"]),
        "video_text_embeddings.in_layer.weight": _meta(d["model_dim"], d["in_text_dim"]),
        "video_pooled_text_embeddings.in_layer.weight": _meta(d["time_dim"], d["in_text_dim2"]),
        "audio_embeddings.in_layer.weight": _meta(d["model_dim_a"], d["in_audio_dim"]),
        "audio_outLayer.out_layer.weight": _meta(d["in_audio_dim"], d["model_dim_a"]),
        "out_layer.out_layer.weight": _meta(d["out_visual_dim"] * patch_volume, d["model_dim"]),
    }
    for i in range(d["num_visual_blocks"]):
        sd[f"visual_blocks.{i}.videoT.self_attention.to_query.weight"] = _meta(d["model_dim"], d["model_dim"])
    for stem in ("video_text_transformer_blocks", "audio_text_transformer_blocks"):
        hidden = d["model_dim"] if stem.startswith("video") else d["model_dim_a"]
        for i in range(d["num_text_blocks"]):
            sd[f"{stem}.{i}.self_attention.to_query.weight"] = _meta(hidden, hidden)
    sd["visual_blocks.0.videoT.feed_forward.in_layer.weight"] = _meta(d["ff_dim"], d["model_dim"])
    sd["visual_blocks.0.audioT.feed_forward.in_layer.weight"] = _meta(d["ff_dim_a"], d["model_dim_a"])
    sd["visual_blocks.0.videoT.self_attention.query_norm.weight"] = _meta(d["head_dim"])
    sd["visual_blocks.0.audioT.self_attention.query_norm.weight"] = _meta(d["head_dim_a"])
    sd["visual_blocks.0.va_modulation.out_layer.weight"] = _meta(2 * d["model_dim"] + d["model_dim_a"], d["time_dim"])
    sd["visual_blocks.0.va_cross_attention.to_key.weight"] = _meta(d["model_dim_a"], d["model_dim_a"])
    return sd


# diffusers layout -> native layout, same mapping detection applies.
_DIFFUSERS_TO_NATIVE = [
    ("visual_transformer_blocks.", "visual_blocks."),
    ("audio_out_layer.", "audio_outLayer."),
    (".timestep_embedder.linear_1.", ".in_layer."),
    (".timestep_embedder.linear_2.", ".out_layer."),
    (".video_dec_block.", ".videoT."),
    (".audio_dec_block.", ".audioT."),
    (".feed_forward.net.0.proj.", ".feed_forward.in_layer."),
    (".feed_forward.net.2.", ".feed_forward.out_layer."),
    (".attn.", ".self_attention."),
]


def _to_diffusers_key(key):
    for diffusers, native in _DIFFUSERS_TO_NATIVE:
        key = key.replace(native, diffusers)
    return key


def test_detect_kandinsky6_pro():
    cfg = detect_kandinsky6(_k6_pro_state_dict(), "")
    assert cfg is not None
    assert cfg["image_model"] == "kandinsky6"
    assert cfg["in_visual_dim"] == _PRO["in_visual_dim"]
    assert cfg["out_visual_dim"] == _PRO["out_visual_dim"]
    assert cfg["model_dim"] == _PRO["model_dim"]
    assert cfg["model_dim_a"] == _PRO["model_dim_a"]
    assert cfg["num_visual_blocks"] == _PRO["num_visual_blocks"]
    assert cfg["num_text_blocks"] == _PRO["num_text_blocks"]
    assert cfg["visual_token_type_num_embeddings"] == 2
    assert cfg["visual_embed_dim"] == (2 * _PRO["in_visual_dim"] + 1) * 4


def test_detect_kandinsky6_leaves_state_dict_untouched():
    diffusers = {_to_diffusers_key(key): value for key, value in _k6_pro_state_dict().items()}
    original_keys = set(diffusers)

    cfg = detect_kandinsky6(diffusers, "")

    assert cfg is not None
    assert cfg["image_model"] == "kandinsky6"
    assert set(diffusers) == original_keys


def test_to_native_state_dict_converts_diffusers_layout():
    native = _k6_pro_state_dict()
    diffusers = {_to_diffusers_key(key): value for key, value in native.items()}

    converted = to_native_state_dict(diffusers)

    assert set(converted) == set(native)
    assert all(converted[key] is value for key, value in native.items())
    # The input dict is left as the caller provided it.
    assert set(diffusers) == {_to_diffusers_key(key) for key in native}
    # A native layout passes through without a copy.
    assert to_native_state_dict(native) is native


def test_supported_model_converts_diffusers_unet_state_dict():
    native = _k6_pro_state_dict()
    diffusers = {_to_diffusers_key(key): value for key, value in native.items()}
    model_config = supported_models.Kandinsky6(detect_kandinsky6(native, ""))

    converted = model_config.process_unet_state_dict(diffusers)

    assert set(converted) == set(native)


def test_detect_kandinsky6_accepts_distilled_checkpoints():
    sd = _k6_pro_state_dict()
    # A multi-grid DX output head marks a distilled release; both heads expand
    # by the same n_grid factor.
    sd["out_layer.out_layer.weight"] = _meta(2 * _PRO["out_visual_dim"] * 4, _PRO["model_dim"])
    sd["audio_outLayer.out_layer.weight"] = _meta(2 * _PRO["in_audio_dim"], _PRO["model_dim_a"])

    cfg = detect_kandinsky6(sd, "")
    assert cfg["n_grid"] == 2
    assert cfg["out_visual_dim"] == 2 * _PRO["out_visual_dim"]
    assert cfg["out_audio_dim"] == 2 * _PRO["in_audio_dim"]


def test_detect_kandinsky6_rejects_mismatched_audio_dx_grids():
    sd = _k6_pro_state_dict()
    # The video head declares two grids but the audio head is left unexpanded.
    sd["out_layer.out_layer.weight"] = _meta(2 * _PRO["out_visual_dim"] * 4, _PRO["model_dim"])

    with pytest.raises(ValueError, match="audio output head"):
        detect_kandinsky6(sd, "")


def test_detect_kandinsky6_rejects_mixed_key_layouts():
    diffusers = {_to_diffusers_key(key): value for key, value in _k6_pro_state_dict().items()}
    # The native spelling of a key the diffusers layout also renames.
    diffusers["visual_blocks.0.videoT.feed_forward.in_layer.weight"] = _meta(_PRO["ff_dim"], _PRO["model_dim"])

    with pytest.raises(ValueError, match="mixes Diffusers and native keys"):
        detect_kandinsky6(diffusers, "")


def test_detect_kandinsky6_ignores_non_k6_state_dicts():
    # Kandinsky 5 shaped keys: no audio stream, no fused blocks.
    sd = {
        "visual_embeddings.in_layer.weight": _meta(4096, 132),
        "visual_embeddings.in_layer.bias": _meta(4096),
        "visual_transformer_blocks.0.cross_attention.key_norm.weight": _meta(128),
    }
    assert detect_kandinsky6(sd, "") is None


def test_model_config_from_unet_detects_kandinsky6():
    sd = _k6_pro_state_dict()
    model_config = model_detection.model_config_from_unet(sd, "")
    assert model_config.unet_config["image_model"] == "kandinsky6"


_SMALL_BIGVGAN_CONFIG = {
    "resblock": "1",
    "upsample_rates": [4, 4],
    "upsample_kernel_sizes": [8, 8],
    "upsample_initial_channel": 32,
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3]],
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_bias_at_final": False,
    "use_tanh_at_final": False,
    "num_mels": 128,
}


def _seeded_audio_vae(with_encoder=False):
    # torch.empty params stay NaN without a checkpoint load; seed them so forward
    # tests exercise real values.
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG, with_encoder=with_encoder)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return model


def test_audio_vae_decodes_to_comfy_audio_layout():
    model = _seeded_audio_vae()
    latent_frames = 3
    out = model.decode(torch.randn(1, latent_frames, 40))
    # The VAE decoder yields 2 mel frames per latent frame, then BigVGAN
    # upsamples by the product of its upsample rates.
    expected_samples = latent_frames * 2 * 4 * 4
    assert tuple(out.shape) == (1, 1, expected_samples)
    assert out.dtype == torch.float32


def _k6_audio_tiling_wrapper():
    # Minimal stand-in for comfy.sd.VAE carrying the attributes decode_tiled_1d reads.
    # upscale_ratio is the small config's real 2x VAE times 4x4 BigVGAN upsampling.
    return types.SimpleNamespace(
        first_stage_model=_seeded_audio_vae(),
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
        vae_output_dtype=lambda: torch.float32,
        output_device=torch.device("cpu"),
        process_output=lambda audio: audio,
        latent_time_first=True,
        upscale_ratio=32,
        output_channels=1,
    )


def _k6_audio_vae():
    model = _seeded_audio_vae()
    metadata = {"kandinsky6_audio_vae": "1", "bigvgan_config": json.dumps(_SMALL_BIGVGAN_CONFIG)}
    vae = comfy.sd.VAE(sd=model.state_dict(), metadata=metadata, device=torch.device("cpu"))
    # Match the tiler's output sizing to the small config's real upsampling.
    vae.upscale_ratio = 32
    return vae


def test_audio_vae_decode_tiled_k6_tiles_time_axis():
    # T and C differ so a tiler slicing the channel axis cannot produce the right size.
    wrapper = _k6_audio_tiling_wrapper()
    torch.manual_seed(0)
    samples = torch.randn(1, 50, 40)
    full = wrapper.first_stage_model.decode(samples)
    out = comfy.sd.VAE.decode_tiled_1d(wrapper, samples, tile_x=16, overlap=4)
    assert tuple(out.shape) == tuple(full.shape) == (1, 1, 50 * 32)
    assert torch.isfinite(out).all()
    # The decoder's middle attention is non-local, so tiled output approximates the
    # full decode: it must be close, not bit-identical.
    assert (out - full).abs().mean() < 0.05


def test_audio_vae_decode_tiled_k6_single_tile_matches_full():
    wrapper = _k6_audio_tiling_wrapper()
    torch.manual_seed(0)
    samples = torch.randn(1, 10, 40)
    out = comfy.sd.VAE.decode_tiled_1d(wrapper, samples, tile_x=16, overlap=4)
    assert torch.equal(out, wrapper.first_stage_model.decode(samples))


def test_vae_decode_tiled_routes_k6_audio_to_duration_tiling():
    vae = _k6_audio_vae()
    torch.manual_seed(0)
    samples = torch.randn(1, 50, 40)
    out = vae.decode_tiled(samples, tile_x=16, tile_y=16, overlap=4)
    assert tuple(out.shape) == (1, 50 * 32, 1)
    assert torch.isfinite(out).all()


def test_vae_decode_oom_fallback_tiles_k6_audio():
    vae = _k6_audio_vae()
    torch.manual_seed(0)
    samples = torch.randn(1, 50, 40)
    lengths = []
    real_decode = vae.first_stage_model.decode

    def flaky_decode(a, **kwargs):
        lengths.append(a.shape[1])
        if a.shape[1] == samples.shape[1]:
            raise model_management.OOM_EXCEPTION("test")
        return real_decode(a, **kwargs)

    vae.first_stage_model.decode = flaky_decode
    # Pin the budget so the fallback must pick a tile below the full length on any host.
    vae.patcher.get_free_memory = lambda device: 4_000_000_000
    out = vae.decode(samples)
    assert lengths[0] == samples.shape[1]
    assert all(length < samples.shape[1] for length in lengths[1:])
    assert tuple(out.shape) == (1, 50 * 32, 1)
    assert torch.isfinite(out).all()


def test_audio_vae_state_dict_matches_checkpoint_layout():
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
    keys = set(model.state_dict())
    assert keys
    assert all(
        key.startswith(("vae.data_mean", "vae.data_std", "vae.decoder.", "vocoder."))
        for key in keys
    )


def test_audio_vae_uses_contract_scaling_factor():
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
    assert model.scaling_factor == float(AUDIO_DEFAULTS["scaling_factor"])
    assert model.mean_value == 0.0

    torch.manual_seed(0)
    model = _seeded_audio_vae()
    z = torch.randn(1, 2, 40)
    expected = model.vocoder(model.vae.decode((z / model.scaling_factor).transpose(1, 2)))
    assert torch.equal(model.decode(z), expected)


def test_mel_converter_maps_waveform_to_mel_shape():
    torch.manual_seed(0)
    mel = MelConverter()(torch.randn(1, 10000))
    # One 512-hop mel frame per 512 samples, 128 bands, fp32 for the encoder.
    assert tuple(mel.shape) == (1, 128, 10000 // 512)
    assert mel.dtype == torch.float32


def test_audio_vae_full_encodes_waveform_to_latent():
    torch.manual_seed(0)
    model = _seeded_audio_vae(with_encoder=True)
    latent = model.encode(torch.randn(1, 1, 20000))
    # The encoder halves the 512-hop mel frames: 1024 samples per latent frame.
    assert tuple(latent.shape) == (1, 20000 // 1024, 40)


def test_audio_vae_full_state_dict_round_trips():
    torch.manual_seed(0)
    sd = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG, with_encoder=True).state_dict()
    assert "vae.encoder.conv_in.weight" in sd
    assert "mel_converter.mel_basis" in sd
    fresh = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG, with_encoder=True)
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_vae_build_detects_k6_audio_encoder():
    torch.manual_seed(0)
    full = _seeded_audio_vae(with_encoder=True)
    metadata = {"kandinsky6_audio_vae": "1", "bigvgan_config": json.dumps(_SMALL_BIGVGAN_CONFIG)}
    vae = comfy.sd.VAE(sd=full.state_dict(), metadata=metadata, device=torch.device("cpu"))
    assert vae.first_stage_model.mel_converter is not None
    latent = vae.encode(torch.randn(1, 20000, 1))
    assert tuple(latent.shape) == (1, 20000 // 1024, 40)


def test_vae_build_keeps_k6_audio_decoder_only():
    decoder_only = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
    metadata = {"kandinsky6_audio_vae": "1", "bigvgan_config": json.dumps(_SMALL_BIGVGAN_CONFIG)}
    vae = comfy.sd.VAE(sd=decoder_only.state_dict(), metadata=metadata, device=torch.device("cpu"))
    assert not hasattr(vae.first_stage_model, "mel_converter")
    with pytest.raises(RuntimeError, match="cannot encode"):
        vae.memory_used_encode((1, 1, 1000), torch.float32)


def test_reshape_mask_maps_2d_mask_onto_1d_latent_time_axis():
    # The columns of a 2D mask image are the audio time axis; the rows collapse.
    mask = torch.linspace(0.0, 1.0, steps=8).reshape(1, 1, 1, 8)
    out = comfy.utils.reshape_mask(mask, (1, 218, 40))
    assert tuple(out.shape) == (1, 218, 40)
    assert torch.all(out == out[:, :, :1].expand(-1, -1, 40))
    assert out[0, 0, 0] < out[0, -1, 0]
    assert abs(float(out[0, -1, 0]) - 1.0) < 1e-5
    # A full-shape 3D mask passes through unchanged.
    full = torch.rand(1, 218, 40)
    assert torch.equal(comfy.utils.reshape_mask(full, (1, 218, 40)), full)


def test_concat_av_latent_fits_time_first_audio():
    reference = torch.zeros(1, 20, 40)

    audio, mask = LTXVConcatAVLatent.fit_audio(
        reference, torch.randn(1, 17, 40), torch.zeros(1, 1, 4, 17))
    # A short track is zero-padded; the padded tail stays unmasked so the
    # model generates it.
    assert tuple(audio.shape) == (1, 20, 40) and tuple(mask.shape) == (1, 20, 40)
    assert torch.all(mask[:, :17] == 0) and torch.all(mask[:, 17:] == 1)

    audio, mask = LTXVConcatAVLatent.fit_audio(
        reference, torch.randn(1, 25, 40), torch.ones(1, 1, 4, 25))
    assert tuple(audio.shape) == (1, 20, 40) and tuple(mask.shape) == (1, 20, 40)

    # A batch mismatch is never fitted: narrowing the batch would silently drop items.
    with pytest.raises(ValueError):
        LTXVConcatAVLatent.fit_audio(reference, torch.randn(2, 20, 40), torch.zeros(1, 1, 4, 20))


def test_empty_latent_node_builds_joint_latent():
    latent = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == (1, 16, 31, 60, 108)
    assert tuple(audio.shape) == (1, 218, 40)
    assert latent["frame_rate"] == 24.0
    assert latent["sample_rate"] == 44100


def test_empty_latent_node_rounds_unaligned_shapes():
    # Width/height floor to the VAE factor, the frame count to 4*n+1, and
    # batch sizes above 1 build unchanged. 122 frames round down to 121,
    # which the audio stream is sized from as well.
    latent = Kandinsky6EmptyLatent.execute(865, 479, 122, 2)[0]
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == (2, 16, 31, 59, 108)
    assert tuple(audio.shape) == (2, 218, 40)


class _FakeVae:
    """Deterministic stand-in for the K6 video VAE: frames encode to the mean of their pixels.

    Like the real VAE, a 4D batch of frames is one video: the batch axis becomes time.
    """

    def spacial_compression_encode(self):
        return 8

    def encode(self, pixels):
        x = pixels.movedim(-1, 1)
        if x.ndim == 4:
            x = x.movedim(1, 0).unsqueeze(0)  # [1, 3, N, H, W]
        n = 1 + ((x.shape[2] - 1) // 4) * 4
        k = 1 + (n - 1) // 4
        value = x[:, :, :n].mean(dim=(1, 2, 3, 4), keepdim=True)
        return value.expand(-1, 1, k, x.shape[3] // 8, x.shape[4] // 8).repeat(1, 16, 1, 1, 1)


def _cond():
    return [({}, {})]


def test_i2v_node_pins_start_and_end_frames():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    start = torch.full((1, 960, 1280, 3), 0.25)
    end = torch.full((1, 960, 1280, 3), 0.75)

    out_positive, out_negative, out_latent = Kandinsky6ImageToVideoAudio.execute(
        _cond(), _cond(), joint, start_image=start, end_image=end, vae=_FakeVae())

    video, _ = out_latent["samples"].unbind()
    assert torch.allclose(video[:, :, 0], torch.full_like(video[:, :, 0], 0.25))
    assert torch.allclose(video[:, :, -1], torch.full_like(video[:, :, -1], 0.75))
    assert torch.count_nonzero(video[:, :, 1:-1]) == 0

    video_mask, audio_mask = out_latent["noise_mask"].unbind()
    assert video_mask[0, 0, 0].item() == 0.0
    assert video_mask[0, 0, -1].item() == 0.0
    assert video_mask[0, 0, 1:-1].min().item() == 1.0
    assert audio_mask.min().item() == 1.0

    # The start frame doubles as the model's I2VA reference on both conds.
    reference = out_positive[0][1]["k6_reference"]
    assert torch.equal(reference, video[:, :, :1])
    assert torch.equal(out_negative[0][1]["k6_reference"], reference)


def test_i2v_node_builds_latent_from_image():
    # 960x720 image -> 120x90 latents, the default 121 frames -> 31 latent frames.
    start = torch.full((1, 720, 960, 3), 0.3)
    _, _, out_latent = Kandinsky6ImageToVideoAudio.execute(_cond(), _cond(), None, start_image=start, vae=_FakeVae())
    video, audio = out_latent["samples"].unbind()
    assert tuple(video.shape) == (1, 16, 31, 90, 120)
    assert torch.allclose(video[:, :, 0], torch.full_like(video[:, :, 0], 0.3))
    assert out_latent["noise_mask"].unbind()[0][0, 0, 0].item() == 0.0


def test_i2v_node_requires_latent_or_image():
    with pytest.raises(ValueError):
        Kandinsky6ImageToVideoAudio.execute(_cond(), _cond(), None)


def test_i2v_node_requires_vae_for_images():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    with pytest.raises(ValueError):
        Kandinsky6ImageToVideoAudio.execute(_cond(), _cond(), joint, start_image=torch.rand(1, 480, 864, 3))


def test_add_guide_node_pins_mid_frame():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    image = torch.full((1, 480, 864, 3), 0.5)

    out = Kandinsky6AddGuide.execute(joint, 60, vae=_FakeVae(), image=image)[0]

    video, _ = out["samples"].unbind()
    video_mask, _ = out["noise_mask"].unbind()
    # pixel frame 60 sits on latent frame 15
    assert torch.allclose(video[:, :, 15], torch.full_like(video[:, :, 15], 0.5))
    assert video_mask[0, 0, 15].item() == 0.0
    assert video_mask[0, 0, 14].item() == 1.0
    assert video_mask[0, 0, 16].item() == 1.0


def test_add_guide_node_negative_index_counts_from_end():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    image = torch.full((1, 480, 864, 3), 0.5)

    out = Kandinsky6AddGuide.execute(joint, -1, vae=_FakeVae(), image=image)[0]

    video_mask, _ = out["noise_mask"].unbind()
    assert video_mask[0, 0, -1].item() == 0.0
    assert video_mask[0, 0, -2].item() == 1.0


def test_add_guide_node_clip_guide():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    # 5 frames -> 2 latent frames; the clip starts at pixel frame 8 -> latent 2.
    clip = torch.full((5, 480, 864, 3), 0.5)

    out = Kandinsky6AddGuide.execute(joint, 8, vae=_FakeVae(), image=clip)[0]
    video, _ = out["samples"].unbind()
    video_mask, _ = out["noise_mask"].unbind()
    assert torch.allclose(video[:, :, 2:4], torch.full_like(video[:, :, 2:4], 0.5))
    assert video_mask[0, 0, 2:4].max().item() == 0.0
    assert video_mask[0, 0, 1].item() == 1.0
    assert video_mask[0, 0, 4].item() == 1.0

    # The same clip at the very end does not fit: 30 + 2 > 31 latent frames.
    with pytest.raises(ValueError):
        Kandinsky6AddGuide.execute(joint, -1, vae=_FakeVae(), image=clip)


def test_add_latent_guide_node_pins_latent_frames():
    joint = Kandinsky6EmptyLatent.execute(864, 480, 121, 1)[0]
    guide = torch.full((1, 16, 2, 60, 108), 0.9)

    out = Kandinsky6AddLatentGuide.execute(joint, {"samples": guide}, -3)[0]
    video, _ = out["samples"].unbind()
    video_mask, _ = out["noise_mask"].unbind()
    # latent_idx -3 of 31 frames -> 28, pinning latent frames 28 and 29
    assert torch.allclose(video[:, :, 28:30], torch.full_like(video[:, :, 28:30], 0.9))
    assert video_mask[0, 0, 28:30].max().item() == 0.0
    assert video_mask[0, 0, 30].item() == 1.0

    # A joint AV guide contributes its video stream.
    joint_guide = comfy.nested_tensor.NestedTensor((guide, torch.zeros(1, 10, 40)))
    out2 = Kandinsky6AddLatentGuide.execute(joint, {"samples": joint_guide}, 5)[0]
    video2, _ = out2["samples"].unbind()
    assert torch.allclose(video2[:, :, 5:7], torch.full_like(video2[:, :, 5:7], 0.9))

    with pytest.raises(ValueError):
        Kandinsky6AddLatentGuide.execute(joint, {"samples": torch.full((1, 16, 2, 30, 54), 0.9)}, 0)


def _small_dit():
    # disable_weight_init leaves torch.empty garbage behind; bound the weights
    # so the forward stays finite for the value checks below.
    torch.manual_seed(0)
    # aimdo's lazy weight init leaves weights None until a state dict loads;
    # these test models are built without one, so use the plain init path.
    with patch.object(memory_management, "aimdo_enabled", False):
        model = Kandinsky6NativeAVDiT(
            in_visual_dim=16,
            out_visual_dim=16,
            in_audio_dim=40,
            in_text_dim=32,
            in_text_dim2=32,
            time_dim=32,
            model_dim=32,
            ff_dim=64,
            time_dim_a=32,
            model_dim_a=32,
            ff_dim_a=64,
            head_dim_a=16,
            visual_embed_dim=(2 * 16 + 1) * 4,
            patch_size=(1, 2, 2),
            num_text_blocks=1,
            num_visual_blocks=1,
            axes_dims=(8, 4, 4),
            rope_scale_factor=(1.0, 1.0, 1.0),
            cross_gates=True,
            fix_modulation=True,
            ca_rope=True,
            visual_token_type_num_embeddings=2,
            dtype=torch.float32,
            device="cpu",
            operations=comfy.ops.disable_weight_init,
        )
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        else:
            torch.nn.init.zeros_(param)
    return model


def test_dit_appends_i2va_reference_from_conditioning():
    model = _small_dit()
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    timestep = torch.tensor([0.5])
    reference = torch.randn(1, 16, 1, 4, 4)

    out_t2va, _ = model(x=(video, audio), timestep=timestep, context=context, y=pooled)
    out_i2va, _ = model(
        x=(video, audio), timestep=timestep, context=context, y=pooled,
        k6_reference=reference,
    )

    # The sampler only ever sees the generated frames: the reference tail is
    # appended inside the model and cropped from the output.
    assert tuple(out_i2va.shape) == tuple(video.shape)
    assert not torch.equal(out_t2va, out_i2va)


def test_dit_expands_single_reference_to_video_batch():
    model = _small_dit()
    video = torch.randn(2, 16, 2, 4, 4)
    audio = torch.randn(2, 4, 40)
    context = torch.randn(2, 6, 32)
    pooled = torch.randn(2, 32)
    timestep = torch.tensor([0.5, 0.5])
    reference = torch.randn(1, 16, 1, 4, 4)

    out, _ = model(
        x=(video, audio), timestep=timestep, context=context, y=pooled,
        k6_reference=reference,
    )

    assert tuple(out.shape) == tuple(video.shape)


def test_dit_appends_multi_frame_reference_tail():
    model = _small_dit()
    video = torch.randn(1, 16, 3, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    timestep = torch.tensor([0.5])
    reference = torch.randn(1, 16, 2, 4, 4)

    out_t2va, _ = model(x=(video, audio), timestep=timestep, context=context, y=pooled)
    out_chained, _ = model(
        x=(video, audio), timestep=timestep, context=context, y=pooled,
        k6_reference=reference,
    )

    # The sampler only ever sees the generated frames: the multi-frame
    # reference tail is appended inside the model and cropped from the output.
    assert tuple(out_chained.shape) == tuple(video.shape)
    assert not torch.equal(out_t2va, out_chained)


def test_dit_reference_tail_rope_covers_generated_and_reference_frames():
    model = _small_dit()
    calls = []
    original = model.rope_encode_3d

    def recording(t, h, w, **kwargs):
        calls.append(int(t))
        return original(t, h, w, **kwargs)

    model.rope_encode_3d = recording
    video = torch.randn(1, 16, 3, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    timestep = torch.tensor([0.5])
    reference = torch.randn(1, 16, 2, 4, 4)

    model(x=(video, audio), timestep=timestep, context=context, y=pooled,
          k6_reference=reference)

    # Generated frames keep positions 0..T-1; the reference tail gets its own
    # run of positions 0..K-1 so it overlaps the start of the clip.
    assert calls == [3, 2]


def test_dit_single_frame_tail_rope_unchanged():
    # The unified two-call RoPE must reduce to the old single-frame behavior:
    # a one-frame tail sits at position 0, the first frame of the generated
    # run.
    model = _small_dit()
    tail = model.rope_encode_3d(1, 4, 4)
    full = model.rope_encode_3d(3, 4, 4)
    tokens_per_frame = 2 * 2
    assert torch.equal(tail, full[:, :tokens_per_frame])


def _small_distilled_dit(n_grid=2):
    # A distilled head expands both output heads into n_grid DX grids.
    torch.manual_seed(0)
    # aimdo's lazy weight init leaves weights None until a state dict loads;
    # these test models are built without one, so use the plain init path.
    with patch.object(memory_management, "aimdo_enabled", False):
        model = Kandinsky6NativeAVDiT(
            in_visual_dim=16,
            out_visual_dim=16 * n_grid,
            in_audio_dim=40,
            out_audio_dim=40 * n_grid,
            n_grid=n_grid,
            in_text_dim=32,
            in_text_dim2=32,
            time_dim=32,
            model_dim=32,
            ff_dim=64,
            time_dim_a=32,
            model_dim_a=32,
            ff_dim_a=64,
            head_dim_a=16,
            visual_embed_dim=(2 * 16 + 1) * 4,
            patch_size=(1, 2, 2),
            num_text_blocks=1,
            num_visual_blocks=1,
            axes_dims=(8, 4, 4),
            rope_scale_factor=(1.0, 1.0, 1.0),
            cross_gates=True,
            fix_modulation=True,
            ca_rope=True,
            visual_token_type_num_embeddings=2,
            dtype=torch.float32,
            device="cpu",
            operations=comfy.ops.disable_weight_init,
        )
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        else:
            torch.nn.init.zeros_(param)
    return model


def test_distilled_dit_returns_raw_grids_when_not_collapsing():
    model = _small_distilled_dit(n_grid=2)
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    timestep = torch.tensor([0.5])

    v_raw, a_raw = model(
        x=(video, audio), timestep=timestep, context=context, y=pooled,
        collapse_grids=False,
    )
    v_col, a_col = model(x=(video, audio), timestep=timestep, context=context, y=pooled)

    # collapse_grids=False keeps the n_grid channel expansion for the PiFlow
    # guider; the default call averages the grids for a standard sampler.
    assert tuple(v_raw.shape) == (1, 32, 2, 4, 4)
    assert tuple(a_raw.shape) == (1, 4, 80)
    assert tuple(v_col.shape) == (1, 16, 2, 4, 4)
    assert tuple(a_col.shape) == (1, 4, 40)
    assert torch.allclose(v_col, v_raw.view(1, 2, 16, 2, 4, 4).mean(dim=1))
    assert torch.allclose(a_col, a_raw.view(1, 4, 2, 40).mean(dim=2))


def test_piflow_rollout_evaluates_dit_once_per_segment():
    model = _small_distilled_dit(n_grid=2)
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)

    calls = {"n": 0}
    original = model.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.forward = counting
    try:
        v_out, a_out = piflow.rollout(model, video, audio, context, pooled, steps=4, dtype=torch.float32)
    finally:
        model.forward = original

    # One DiT evaluation per segment, and the streams keep their shape.
    assert calls["n"] == 4
    assert tuple(v_out.shape) == tuple(video.shape)
    assert tuple(a_out.shape) == tuple(audio.shape)
    assert torch.isfinite(v_out).all() and torch.isfinite(a_out).all()


def test_piflow_rollout_audio_mask_keeps_clean_region():
    model = _small_distilled_dit(n_grid=2)
    torch.manual_seed(0)
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    audio_clean = torch.randn(1, 4, 40)

    _, a = piflow.rollout(
        model, video, audio, context, pooled, steps=4, dtype=torch.float32,
        audio_mask=torch.zeros(1, 4, 40), audio_clean=audio_clean,
    )
    # A fully masked audio stream comes back exactly as the clean latent.
    assert torch.equal(a, audio_clean)

    mask = torch.zeros(1, 4, 40)
    mask[:, 2:] = 1.0
    _, a2 = piflow.rollout(
        model, video, audio, context, pooled, steps=4, dtype=torch.float32,
        audio_mask=mask, audio_clean=audio_clean,
    )
    # Masked frames stay clean; the rest is generated.
    assert torch.equal(a2[:, :2], audio_clean[:, :2])
    assert not torch.equal(a2[:, 2:], audio_clean[:, 2:])


def test_piflow_rollout_video_mask_keeps_clean_frames():
    model = _small_distilled_dit(n_grid=2)
    torch.manual_seed(0)
    video = torch.randn(1, 16, 4, 4, 4)
    audio = torch.randn(1, 4, 40)
    context = torch.randn(1, 6, 32)
    pooled = torch.randn(1, 32)
    video_clean = torch.randn(1, 16, 4, 4, 4)

    v, _ = piflow.rollout(
        model, video, audio, context, pooled, steps=4, dtype=torch.float32,
        video_mask=torch.zeros(1, 1, 4, 1, 1), video_clean=video_clean,
    )
    # A fully masked video stream comes back exactly as the clean latent.
    assert torch.equal(v, video_clean)

    mask = torch.zeros(1, 1, 4, 1, 1)
    mask[:, :, 2:] = 1.0
    v2, _ = piflow.rollout(
        model, video, audio, context, pooled, steps=4, dtype=torch.float32,
        video_mask=mask, video_clean=video_clean,
    )
    # Masked frames stay clean; the rest is generated.
    assert torch.equal(v2[:, :, :2], video_clean[:, :, :2])
    assert not torch.equal(v2[:, :, 2:], video_clean[:, :, 2:])


def test_piflow_guider_is_a_basic_guider():
    # The PiFlow guider is a CFG guider that keeps a single conditioning, so it
    # slots into the custom sampler like the other guiders.
    assert issubclass(Kandinsky6PiFlowGuider, samplers.CFGGuider)


def test_piflow_guider_drives_cli_progress_bar(monkeypatch):
    # PiFlow runs its own segment loop, so it must create the CLI progress bar
    # the k_diffusion samplers do; the preview callback alone only feeds the UI.
    steps = 4
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    noise, latent_shapes = comfy.utils.pack_latents((video, audio))

    guider = object.__new__(Kandinsky6PiFlowGuider)
    guider.inner_model = types.SimpleNamespace(
        get_dtype_inference=lambda: torch.float32,
        diffusion_model=None,
        process_latent_out=lambda x: x,
    )
    guider.conds = {"positive": [{"cross_attn": torch.randn(1, 4, 32),
                                  "pooled_output": torch.randn(1, 32),
                                  "uuid": uuid.uuid4()}]}
    guider.model_options = {"transformer_options": {}}

    # The rollout hands back to the per-segment callback once per segment.
    def fake_rollout(dit, v, a, context, pooled, *, steps, dtype, **kwargs):
        cb = kwargs.get("callback")
        for i in range(steps):
            if cb is not None:
                cb(i, (v, a), (v, a), steps)
        return v, a
    monkeypatch.setattr(k6_nodes.piflow, "rollout", fake_rollout)

    bars = {}
    class FakeBar:
        def __init__(self, total=None, disable=None):
            bars["total"], bars["disable"], bars["updates"] = total, disable, 0
        def update(self, n=1):
            bars["updates"] += n
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(k6_nodes, "tqdm", FakeBar)

    guider.inner_sample(
        noise, noise, torch.device("cpu"), None,
        torch.linspace(1.0, 0.0, steps + 1), None, None, False, 0,
        latent_shapes=latent_shapes,
    )

    # One bar sized to the segment count, advanced once per segment.
    assert bars["total"] == steps
    assert bars["disable"] is False
    assert bars["updates"] == steps


def test_piflow_guider_passes_stream_masks_to_rollout(monkeypatch):
    # A latent noise mask reaches the rollout per stream: the video mask pins
    # the guide frames, the audio mask the custom track.
    steps = 4
    video = torch.randn(1, 16, 2, 4, 4)
    audio = torch.randn(1, 4, 40)
    noise, latent_shapes = comfy.utils.pack_latents((video, audio))
    video_mask = torch.ones(1, 16, 2, 4, 4)
    video_mask[0, :, 0] = 0.0
    audio_mask = torch.zeros(1, 4, 40)
    denoise_mask, _ = comfy.utils.pack_latents((video_mask, audio_mask))

    guider = object.__new__(Kandinsky6PiFlowGuider)
    guider.inner_model = types.SimpleNamespace(
        get_dtype_inference=lambda: torch.float32,
        diffusion_model=None,
        process_latent_in=lambda x: x,
        process_latent_out=lambda x: x,
        latent_format=types.SimpleNamespace(process_in=lambda x: x * 0.5),
    )
    reference = torch.randn(1, 16, 1, 4, 4)
    guider.conds = {"positive": [{"cross_attn": torch.randn(1, 4, 32),
                                  "pooled_output": torch.randn(1, 32),
                                  "uuid": uuid.uuid4(),
                                  "k6_reference": reference}]}
    guider.model_options = {"transformer_options": {}}

    seen = {}
    def fake_rollout(dit, v, a, context, pooled, *, steps, dtype, **kwargs):
        seen["video_mask"] = kwargs.get("video_mask")
        seen["video_clean"] = kwargs.get("video_clean")
        seen["audio_mask"] = kwargs.get("audio_mask")
        seen["reference"] = kwargs.get("reference")
        return v, a
    monkeypatch.setattr(k6_nodes.piflow, "rollout", fake_rollout)

    guider.inner_sample(
        noise, noise, torch.device("cpu"), None,
        torch.linspace(1.0, 0.0, steps + 1), denoise_mask, None, False, 0,
        latent_shapes=latent_shapes,
    )

    assert torch.equal(seen["video_mask"], video_mask)
    assert torch.equal(seen["video_clean"], video)
    assert torch.equal(seen["audio_mask"], audio_mask)
    # The conditioning carries VAE-space latents; the rollout's reference tail
    # must be in the trajectory's model space.
    assert torch.equal(seen["reference"], reference * 0.5)
