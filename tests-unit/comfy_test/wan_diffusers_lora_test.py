"""WAN LoRAs saved in Diffusers/PEFT naming map onto ComfyUI's native WAN weights."""

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.wan.model  # noqa: E402
import comfy.lora  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.supported_models  # noqa: E402

# Diffusers name -> native WAN name, as in diffusers' _convert_non_diffusers_wan_lora_to_diffusers.
BLOCK_NAMES = {
    "attn1.to_q": "self_attn.q", "attn1.to_k": "self_attn.k", "attn1.to_v": "self_attn.v", "attn1.to_out.0": "self_attn.o",
    "attn2.to_q": "cross_attn.q", "attn2.to_k": "cross_attn.k", "attn2.to_v": "cross_attn.v", "attn2.to_out.0": "cross_attn.o",
    "ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2",
}
IMAGE_BLOCK_NAMES = {"attn2.add_k_proj": "cross_attn.k_img", "attn2.add_v_proj": "cross_attn.v_img"}
OTHER_NAMES = {
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "proj_out": "head.head",
}
IMAGE_OTHER_NAMES = {
    "condition_embedder.image_embedder.ff.net.0.proj": "img_emb.proj.1",
    "condition_embedder.image_embedder.ff.net.2": "img_emb.proj.3",
}
NUM_LAYERS = 2
SMALL = {"dim": 16, "ffn_dim": 32, "num_heads": 2, "num_layers": NUM_LAYERS, "text_dim": 8, "freq_dim": 8, "in_dim": 4, "out_dim": 4}
PREFIXES = ["transformer.", ""]


def t2v():
    return comfy.supported_models.WAN21_T2V({"image_model": "wan2.1", "model_type": "t2v", **SMALL}).get_model({})


def i2v():
    return comfy.supported_models.WAN21_T2V({"image_model": "wan2.1", "model_type": "i2v", **SMALL}).get_model({})


def ti2v():  # Wan 2.2 TI2V 5B layout: t2v blocks, 48-channel latent
    return comfy.supported_models.WAN22_T2V({"image_model": "wan2.1", "model_type": "t2v", **SMALL, "in_dim": 48, "out_dim": 48}).get_model({})


def targets(image):
    names = dict(OTHER_NAMES, **(IMAGE_OTHER_NAMES if image else {}))
    blocks = dict(BLOCK_NAMES, **(IMAGE_BLOCK_NAMES if image else {}))
    for i in range(NUM_LAYERS):
        for name, native in blocks.items():
            names["blocks.{}.{}".format(i, name)] = "blocks.{}.{}".format(i, native)
    return names


FIXTURES = [(t2v, comfy.model_base.WAN21, False), (i2v, comfy.model_base.WAN21, True), (ti2v, comfy.model_base.WAN22, False)]


@pytest.mark.parametrize("build,cls,image", FIXTURES)
def test_every_diffusers_name_maps_to_its_native_weight_and_bias(build, cls, image):
    model = build()
    assert type(model) is cls
    sd = model.state_dict()
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    for name, native in targets(image).items():
        for p in PREFIXES:
            assert key_map["{}{}".format(p, name)] == "diffusion_model.{}.weight".format(native)
            assert key_map["{}{}.lora_B.bias".format(p, name)] == "diffusion_model.{}.bias".format(native)
            assert "diffusion_model.{}.bias".format(native) in sd

    absent = {} if image else dict(IMAGE_OTHER_NAMES, **{"blocks.0.{}".format(n): n for n in IMAGE_BLOCK_NAMES})
    for name in absent:
        for p in PREFIXES:
            assert "{}{}".format(p, name) not in key_map
            assert "{}{}.lora_B.bias".format(p, name) not in key_map


def apply(patch, weight, key, strength):
    return comfy.lora.calculate_weight([(strength, patch, 1.0, None, None)], weight.clone(), key)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("alpha", [None, 3.0])
def test_rank2_factors_and_bias_apply_with_alpha_scaling(prefix, alpha):
    model = i2v()
    sd = model.state_dict()
    rank, strength = 2, 0.7
    scale = (alpha / rank if alpha is not None else 1.0) * strength
    generator = torch.Generator().manual_seed(0)
    lora, expected = {}, {}
    for name, native in targets(image=True).items():
        weight = sd["diffusion_model.{}.weight".format(native)]
        out_dim, in_dim = weight.shape
        down = torch.randn(rank, in_dim, generator=generator)
        up = torch.randn(out_dim, rank, generator=generator)
        bias = torch.randn(out_dim, generator=generator)
        lora["{}{}.lora_A.weight".format(prefix, name)] = down
        lora["{}{}.lora_B.weight".format(prefix, name)] = up
        lora["{}{}.lora_B.bias".format(prefix, name)] = bias
        if alpha is not None:
            lora["{}{}.alpha".format(prefix, name)] = torch.tensor(alpha)
        expected["diffusion_model.{}.weight".format(native)] = scale * (up @ down)
        expected["diffusion_model.{}.bias".format(native)] = scale * bias

    patches = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model, {}))
    assert set(patches) == set(expected)

    # square (q/k/v share a shape, so a swapped destination changes the delta) and rectangular targets
    for native in ["blocks.0.self_attn.q", "blocks.0.self_attn.k", "blocks.1.cross_attn.v", "blocks.1.cross_attn.k_img",
                   "blocks.0.ffn.0", "blocks.1.ffn.2", "time_embedding.0", "img_emb.proj.3", "head.head"]:
        for kind in ("weight", "bias"):
            key = "diffusion_model.{}.{}".format(native, kind)
            base = torch.zeros_like(sd[key], dtype=torch.float32)  # model weights are uninitialized
            updated = apply(patches[key], base, key, strength)
            torch.testing.assert_close(updated - base, expected[key])


def test_fp16_bias_scaled_by_alpha_does_not_overflow():
    # 40000 * (alpha 4 / rank 2) = 80000 is past fp16; scaled in fp32 and applied at 0.1 it is 8000.
    model = t2v()
    key = "diffusion_model.head.head.bias"
    out_dim, in_dim = model.state_dict()["diffusion_model.head.head.weight"].shape
    lora = {
        "proj_out.lora_A.weight": torch.zeros(2, in_dim, dtype=torch.float16),
        "proj_out.lora_B.weight": torch.zeros(out_dim, 2, dtype=torch.float16),
        "proj_out.lora_B.bias": torch.full((out_dim,), 40000.0, dtype=torch.float16),
        "proj_out.alpha": torch.tensor(4.0),
    }
    patches = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model, {}))
    base = torch.zeros(out_dim, dtype=torch.float32)
    torch.testing.assert_close(apply(patches[key], base, key, 0.1) - base, torch.full((out_dim,), 8000.0))


def test_diffusers_wan_names_stay_wan_only():
    config = comfy.supported_models.WAN21_T2V({"image_model": "wan2.1", "model_type": "i2v", **SMALL})
    other = comfy.model_base.BaseModel(config, unet_model=comfy.ldm.wan.model.WanModel)
    key_map = comfy.lora.model_lora_keys_unet(other, {})
    for name in targets(image=True):
        for p in PREFIXES:
            assert "{}{}".format(p, name) not in key_map
            assert "{}{}.lora_B.bias".format(p, name) not in key_map
