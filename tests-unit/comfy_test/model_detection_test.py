from collections import defaultdict

import torch

from comfy.model_detection import detect_unet_config, model_config_from_unet, model_config_from_unet_config
import comfy.supported_models


def _freeze(value):
    """Recursively convert a value to a hashable form so configs can be
    compared/used as dict keys or set members."""
    if isinstance(value, dict):
        return frozenset((k, _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


def _make_longcat_comfyui_sd():
    """Minimal ComfyUI-format state dict for pre-converted LongCat-Image weights."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
    C_IN = 16
    C_CTX = 3584

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, C_CTX)
    sd["txt_in.bias"] = torch.empty(H)

    sd["time_in.in_layer.weight"] = torch.empty(H, 256)
    sd["time_in.in_layer.bias"] = torch.empty(H)
    sd["time_in.out_layer.weight"] = torch.empty(H, H)
    sd["time_in.out_layer.bias"] = torch.empty(H)

    sd["final_layer.adaLN_modulation.1.weight"] = torch.empty(2 * H, H)
    sd["final_layer.adaLN_modulation.1.bias"] = torch.empty(2 * H)
    sd["final_layer.linear.weight"] = torch.empty(C_IN * 4, H)
    sd["final_layer.linear.bias"] = torch.empty(C_IN * 4)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
        sd[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.empty(3 * H, H)
        sd[f"double_blocks.{i}.img_mod.lin.weight"] = torch.empty(H, H)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


def _make_flux_schnell_comfyui_sd():
    """Minimal ComfyUI-format state dict for standard Flux Schnell."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
    C_IN = 16

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, 4096)
    sd["txt_in.bias"] = torch.empty(H)

    sd["double_blocks.0.img_attn.norm.key_norm.weight"] = torch.empty(128)
    sd["double_blocks.0.img_attn.qkv.weight"] = torch.empty(3 * H, H)
    sd["double_blocks.0.img_mod.lin.weight"] = torch.empty(H, H)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


def _make_seedvr2_7b_separate_mm_sd():
    return {
        "blocks.35.mlp.vid.proj_out.weight": torch.empty(3072, 1),
        "positive_conditioning": torch.empty(58, 5120),
        "negative_conditioning": torch.empty(64, 5120),
    }


def _make_seedvr2_7b_shared_mm_sd():
    return {
        "blocks.35.mlp.all.proj_in_gate.weight": torch.empty(1, 1),
        "positive_conditioning": torch.empty(58, 5120),
        "negative_conditioning": torch.empty(64, 5120),
    }


def _make_seedvr2_3b_shared_mm_sd():
    return {
        "blocks.31.mlp.all.proj_in_gate.weight": torch.empty(1, 1),
        "positive_conditioning": torch.empty(58, 5120),
        "negative_conditioning": torch.empty(64, 5120),
    }


def _make_pid_v1_5_sd(latent_proj_channels=16):
    sd = {
        "pixel_embedder.proj.weight": torch.empty(16, 3, device="meta"),
        "lq_proj.latent_proj.0.weight": torch.empty(1024, latent_proj_channels, 3, 3, device="meta"),
        "lq_proj.pit_head.weight": torch.empty(1536, 1024, device="meta"),
        "lq_proj.gate_modules.0.content_proj.weight": torch.empty(1, 3072, device="meta"),
        "pixel_blocks.0.attn.q_norm.weight": torch.empty(72, device="meta"),
        "pixel_blocks.0.adaLN_modulation.0.weight": torch.empty(24576, 1536, device="meta"),
        "pixel_blocks.0.adaLN_modulation.0.bias": torch.empty(24576, device="meta"),
    }
    for i in range(7):
        sd[f"lq_proj.gate_modules.{i}.log_alpha"] = torch.empty((), device="meta")
    return sd


def _make_joyimage_edit_plus_sd():
    sd = {
        "img_in.weight": torch.empty(4096, 16, 1, 2, 2, device="meta"),
        "condition_embedder.time_embedder.linear_1.weight": torch.empty(1, device="meta"),
        "double_blocks.0.attn.img_attn_q_norm.weight": torch.empty(128, device="meta"),
    }
    for i in range(40):
        sd[f"double_blocks.{i}.attn.img_attn_qkv.weight"] = torch.empty(1, device="meta")
    return sd


def _add_model_diffusion_prefix(sd):
    return {f"model.diffusion_model.{k}": v for k, v in sd.items()}


class TestModelDetection:
    """Verify that first-match model detection selects the correct model
    based on list ordering and unet_config specificity."""

    def test_longcat_before_schnell_in_models_list(self):
        """LongCatImage must appear before FluxSchnell in the models list."""
        models = comfy.supported_models.models
        longcat_idx = next(i for i, m in enumerate(models) if m.__name__ == "LongCatImage")
        schnell_idx = next(i for i, m in enumerate(models) if m.__name__ == "FluxSchnell")
        assert longcat_idx < schnell_idx, (
            f"LongCatImage (index {longcat_idx}) must come before "
            f"FluxSchnell (index {schnell_idx}) in the models list"
        )

    def test_longcat_comfyui_detected_as_longcat(self):
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 3584
        assert unet_config["vec_in_dim"] is None
        assert unet_config["guidance_embed"] is False
        assert unet_config["txt_ids_dims"] == [1, 2]

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "LongCatImage"

    def test_longcat_comfyui_keys_pass_through_unchanged(self):
        """Pre-converted weights should not be transformed by process_unet_state_dict."""
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        model_config = model_config_from_unet_config(unet_config, sd)

        processed = model_config.process_unet_state_dict(dict(sd))
        assert "img_in.weight" in processed
        assert "txt_in.weight" in processed
        assert "time_in.in_layer.weight" in processed
        assert "final_layer.linear.weight" in processed

    def test_flux_schnell_comfyui_detected_as_flux_schnell(self):
        sd = _make_flux_schnell_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 4096
        assert unet_config["txt_ids_dims"] == []

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "FluxSchnell"

    def test_seedvr2_7b_separate_mm_detection_config(self):
        sd = _make_seedvr2_7b_separate_mm_sd()
        unet_config = detect_unet_config(sd, "")

        assert unet_config is not None
        assert unet_config["image_model"] == "seedvr2"
        assert unet_config["vid_dim"] == 3072
        assert unet_config["heads"] == 24
        assert unet_config["num_layers"] == 36
        assert unet_config["mm_layers"] == 36
        assert unet_config["mlp_type"] == "normal"
        assert unet_config["rope_type"] == "rope3d"
        assert unet_config["rope_dim"] == 64

    def test_seedvr2_7b_shared_mm_detection_config(self):
        sd = _make_seedvr2_7b_shared_mm_sd()
        unet_config = detect_unet_config(sd, "")

        assert unet_config is not None
        assert unet_config["image_model"] == "seedvr2"
        assert unet_config["vid_dim"] == 3072
        assert unet_config["heads"] == 24
        assert unet_config["num_layers"] == 36
        assert unet_config["mm_layers"] == 10
        assert unet_config["mlp_type"] == "swiglu"
        assert unet_config["rope_type"] == "rope3d"
        assert unet_config["rope_dim"] == 64

    def test_seedvr2_3b_shared_mm_detection_config(self):
        sd = _make_seedvr2_3b_shared_mm_sd()
        unet_config = detect_unet_config(sd, "")

        assert unet_config is not None
        assert unet_config["image_model"] == "seedvr2"
        assert unet_config["vid_dim"] == 2560
        assert unet_config["heads"] == 20
        assert unet_config["num_layers"] == 32
        assert unet_config["mlp_type"] == "swiglu"

    def test_seedvr2_model_match_requires_conditioning_tensors(self):
        sd = _make_seedvr2_7b_shared_mm_sd()
        unet_config = detect_unet_config(sd, "")

        assert type(model_config_from_unet_config(unet_config, sd)).__name__ == "SeedVR2"

        del sd["positive_conditioning"]
        assert model_config_from_unet_config(unet_config, sd) is None

    def test_seedvr2_model_match_accepts_full_checkpoint_prefix(self):
        sd = _add_model_diffusion_prefix(_make_seedvr2_7b_shared_mm_sd())

        assert type(model_config_from_unet(sd, "model.diffusion_model.")).__name__ == "SeedVR2"

    def test_pid_v1_5_detection(self):
        sd = _make_pid_v1_5_sd()
        unet_config = detect_unet_config(sd, "")

        assert unet_config == {
            "image_model": "pid",
            "lq_latent_channels": 16,
            "lq_hidden_dim": 1024,
            "latent_spatial_down_factor": 8,
            "lq_interval": 2,
            "lq_latent_unpatchify_factor": 1,
            "lq_conv_padding_mode": "replicate",
            "lq_gate_per_token": True,
            "pit_lq_inject": True,
            "rope_ref_h": 2048,
            "rope_ref_w": 2048,
        }
        assert type(model_config_from_unet_config(unet_config, sd)).__name__ == "PiD"

    def test_pid_v1_5_flux2_detection(self):
        unet_config = detect_unet_config(_make_pid_v1_5_sd(latent_proj_channels=32), "")

        assert unet_config["lq_latent_channels"] == 128
        assert unet_config["latent_spatial_down_factor"] == 16
        assert unet_config["lq_latent_unpatchify_factor"] == 2

    def test_pid_v1_5_pixel_adaln_conversion(self):
        sd = _make_pid_v1_5_sd()
        model_config = model_config_from_unet_config(detect_unet_config(sd, ""), sd)
        processed = model_config.process_unet_state_dict(sd)

        assert processed["pixel_blocks.0.attn.q_norm.weight"].shape == (72,)
        assert processed["pixel_blocks.0.adaLN_modulation_msa.weight"].shape == (12288, 1536)
        assert processed["pixel_blocks.0.adaLN_modulation_mlp.weight"].shape == (12288, 1536)
        assert processed["pixel_blocks.0.adaLN_modulation_msa.bias"].shape == (12288,)
        assert processed["pixel_blocks.0.adaLN_modulation_mlp.bias"].shape == (12288,)

    def test_joyimage_edit_plus_detection(self):
        sd = _make_joyimage_edit_plus_sd()
        unet_config = detect_unet_config(sd, "")

        assert unet_config == {
            "image_model": "joyimage",
            "in_channels": 16,
            "hidden_size": 4096,
            "patch_size": [1, 2, 2],
            "num_layers": 40,
            "num_attention_heads": 32,
            "text_dim": 4096,
        }
        assert type(model_config_from_unet_config(unet_config, sd)).__name__ == "JoyImage"

    def test_incomplete_joyimage_signature_is_not_detected(self):
        sd = _make_joyimage_edit_plus_sd()
        del sd["double_blocks.0.attn.img_attn_q_norm.weight"]
        assert detect_unet_config(sd, "") is None

    def test_pixal3d_projection_detection_ignores_packed_weight_width(self):
        sd = {
            "img2shape.t_embedder.mlp.0.weight": torch.empty(8, 8, device="meta"),
            "img2shape.blocks.0.cross_attn.proj_linear.weight": torch.empty(8, 1024, device="meta"),
            "structure_model.blocks.0.cross_attn.proj_linear.weight": torch.empty(8, 512, device="meta"),
        }

        unet_config = detect_unet_config(sd, "")

        assert unet_config["proj_in_channels_shape"] == 2048
        assert unet_config["proj_in_channels_structure"] == 1024

    def test_unet_config_and_required_keys_combination_is_unique(self):
        """Each model in the registry must have a unique combination of
        ``unet_config`` and ``required_keys``. If two models share the same
        combination, ``BASE.matches`` cannot disambiguate between them and the
        first one in the list will always win."""
        models = comfy.supported_models.models
        groups = defaultdict(list)
        for model in models:
            key = (_freeze(model.unet_config), _freeze(model.required_keys))
            groups[key].append(model.__name__)

        duplicates = {k: names for k, names in groups.items() if len(names) > 1}
        assert not duplicates, (
            "Found models sharing the same (unet_config, required_keys) "
            "combination, which makes detection ambiguous: "
            + "; ".join(", ".join(names) for names in duplicates.values())
        )
