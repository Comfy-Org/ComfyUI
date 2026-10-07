import pytest
import torch

import comfy.nested_tensor
from comfy import model_detection
from comfy.ldm.kandinsky6.audio_vae import Kandinsky6AudioVAE
from comfy.ldm.kandinsky6.core_contract import AUDIO_DEFAULTS
from comfy.ldm.kandinsky6.detection import detect_kandinsky6
from comfy_extras.nodes_kandinsky6 import (
    Kandinsky6EmptyLatent,
    Kandinsky6RemoveReferenceLatent,
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
    assert cfg["out_audio_dim"] == _PRO["in_audio_dim"]
    assert cfg["model_dim"] == _PRO["model_dim"]
    assert cfg["model_dim_a"] == _PRO["model_dim_a"]
    assert cfg["n_grid"] == 1
    assert cfg["num_visual_blocks"] == _PRO["num_visual_blocks"]
    assert cfg["num_text_blocks"] == _PRO["num_text_blocks"]
    assert cfg["visual_token_type_num_embeddings"] == 2
    assert cfg["visual_embed_dim"] == (2 * _PRO["in_visual_dim"] + 1) * 4


def test_detect_kandinsky6_renames_diffusers_keys_in_place():
    native = _k6_pro_state_dict()
    diffusers = {_to_diffusers_key(key): value for key, value in native.items()}

    cfg = detect_kandinsky6(diffusers, "")

    assert cfg is not None
    assert cfg["image_model"] == "kandinsky6"
    assert set(diffusers) == set(native)


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


@pytest.mark.parametrize(
    "width, height, length, batch_size",
    [
        (865, 480, 121, 1),  # width not divisible by 16
        (864, 479, 121, 1),  # height not divisible by 16
        (864, 480, 122, 1),  # frame count not of the form 4*n + 1
        (864, 480, 121, 2),  # batch_size > 1
    ],
)
def test_empty_latent_node_rejects_invalid_shapes(width, height, length, batch_size):
    with pytest.raises(ValueError):
        Kandinsky6EmptyLatent().build(width, height, length, batch_size)


def test_remove_reference_latent_strips_tail():
    latent = Kandinsky6EmptyLatent().build(864, 480, 121, 1)[0]
    video, audio = latent["samples"].unbind()
    generated_frames = video.shape[2]
    tail = torch.zeros_like(video[:, :, :1])
    latent["samples"] = comfy.nested_tensor.NestedTensor((torch.cat((video, tail), dim=2), audio))
    latent["k6_reference_tail"] = True
    latent["k6_generated_video_latent_frames"] = generated_frames

    out = Kandinsky6RemoveReferenceLatent().remove(latent)[0]
    out_video, out_audio = out["samples"].unbind()

    assert tuple(out_video.shape) == tuple(video.shape)
    assert torch.equal(out_audio, audio)
    assert "k6_reference_tail" not in out
    assert "k6_generated_video_latent_frames" not in out


def test_remove_reference_latent_requires_tail_metadata():
    latent = Kandinsky6EmptyLatent().build(864, 480, 121, 1)[0]
    with pytest.raises(ValueError):
        Kandinsky6RemoveReferenceLatent().remove(latent)
