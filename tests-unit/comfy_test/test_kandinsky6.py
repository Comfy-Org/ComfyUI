import json
import types

import pytest
import torch

import comfy.ops
import comfy.sd
from comfy import model_detection
from comfy import model_management
from comfy import samplers
from comfy import supported_models
from comfy.ldm.kandinsky6 import piflow
from comfy.ldm.kandinsky6.audio_vae import Kandinsky6AudioVAE
from comfy.ldm.kandinsky6.core_contract import AUDIO_DEFAULTS
from comfy.ldm.kandinsky6.detection import detect_kandinsky6, to_native_state_dict
from comfy.ldm.kandinsky6.model import Kandinsky6NativeAVDiT
from comfy_extras.nodes_kandinsky6 import (
    Kandinsky6EmptyLatent,
    Kandinsky6PiFlowGuider,
)


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


def test_audio_vae_decodes_to_comfy_audio_layout():
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
    latent_frames = 3
    out = model.decode(torch.randn(1, latent_frames, 40))
    # The VAE decoder yields 2 mel frames per latent frame, then BigVGAN
    # upsamples by the product of its upsample rates.
    expected_samples = latent_frames * 2 * 4 * 4
    assert tuple(out.shape) == (1, 1, expected_samples)
    assert out.dtype == torch.float32


def _k6_audio_tiling_wrapper():
    # Minimal stand-in for comfy.sd.VAE carrying the attributes decode_tiled_1d_k6 reads.
    # upscale_ratio is the small config's real 2x VAE times 4x4 BigVGAN upsampling.
    return types.SimpleNamespace(
        first_stage_model=Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG),
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
        vae_output_dtype=lambda: torch.float32,
        output_device=torch.device("cpu"),
        process_output=lambda audio: audio,
        upscale_ratio=32,
        output_channels=1,
    )


def _k6_audio_vae():
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
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
    out = comfy.sd.VAE.decode_tiled_1d_k6(wrapper, samples, tile_x=16, overlap=4)
    assert tuple(out.shape) == tuple(full.shape) == (1, 1, 50 * 32)
    assert torch.isfinite(out).all()
    # The decoder's middle attention is non-local, so tiled output approximates the
    # full decode: it must be close, not bit-identical.
    assert (out - full).abs().mean() < 0.05


def test_audio_vae_decode_tiled_k6_single_tile_matches_full():
    wrapper = _k6_audio_tiling_wrapper()
    torch.manual_seed(0)
    samples = torch.randn(1, 10, 40)
    out = comfy.sd.VAE.decode_tiled_1d_k6(wrapper, samples, tile_x=16, overlap=4)
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
    model = Kandinsky6AudioVAE(_SMALL_BIGVGAN_CONFIG)
    z = torch.randn(1, 2, 40)
    expected = model.vocoder(model.vae.decode((z / model.scaling_factor).transpose(1, 2)))
    assert torch.equal(model.decode(z), expected)


def test_empty_latent_node_builds_joint_latent():
    latent = Kandinsky6EmptyLatent().build(864, 480, 121, 1)[0]
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == (1, 16, 31, 60, 108)
    assert tuple(audio.shape) == (1, 218, 40)
    assert latent["frame_rate"] == 24.0
    assert latent["sample_rate"] == 44100


def test_empty_latent_node_rounds_unaligned_shapes():
    # Width/height floor to the VAE factor, the frame count to 4*n+1, and
    # batch sizes above 1 build unchanged. 122 frames round down to 121,
    # which the audio stream is sized from as well.
    latent = Kandinsky6EmptyLatent().build(865, 479, 122, 2)[0]
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == (2, 16, 31, 59, 108)
    assert tuple(audio.shape) == (2, 218, 40)


def _small_dit():
    # disable_weight_init leaves torch.empty garbage behind; bound the weights
    # so the forward stays finite for the value checks below.
    torch.manual_seed(0)
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


def _small_distilled_dit(n_grid=2):
    # A distilled head expands both output heads into n_grid DX grids.
    torch.manual_seed(0)
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


def test_piflow_guider_is_a_basic_guider():
    # The PiFlow guider is a CFG guider that keeps a single conditioning, so it
    # slots into the custom sampler like the other guiders.
    assert issubclass(Kandinsky6PiFlowGuider, samplers.CFGGuider)
