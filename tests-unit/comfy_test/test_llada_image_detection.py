import json
import logging

import pytest
import torch
from comfy_kitchen.tensor import TensorWiseINT8Layout

from comfy.cli_args import args

args.cpu = True

import comfy.model_detection
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.utils
import nodes
from comfy.ldm.llada_image.model import LLaDAImage
from comfy.text_encoders.llada_image import LLaDAImageTEModel


PREFIX = "model.diffusion_model."


@pytest.mark.parametrize("checkpoint_dtype", (torch.bfloat16, torch.float32))
def test_aio_clip_target_preserves_checkpoint_dtype(checkpoint_dtype):
    configs = make_component_configs()
    adapter = comfy.supported_models.LLaDAImage({
        "image_model": "llada_image",
        "variant": "base",
        "llada2_config": configs["text_encoder"],
        "queryformer_config": configs["queryformer"],
        "text_projection_config": configs["text_projection"],
        "sigvq_config": configs["sigvq"],
    })
    target = adapter.clip_target({
        "text_encoders.llada2.model.language_model.word_embeddings.weight": torch.empty(1, 1, device="meta", dtype=checkpoint_dtype),
    })
    model = target.clip(device="meta", dtype=torch.float16, **target.params)
    assert isinstance(model, LLaDAImageTEModel)
    assert model.dtypes == {checkpoint_dtype}
    assert model.llada2.dtype == checkpoint_dtype
    assert model.llada2.model.language_model.word_embeddings.weight.dtype == checkpoint_dtype


def make_state_dict():
    return {
        f"{PREFIX}all_x_embedder.1-1.weight": torch.empty(32, 128),
        f"{PREFIX}all_final_layer.1-1.linear.weight": torch.empty(128, 32),
        f"{PREFIX}noise_refiner.0.attention.to_q.weight": torch.empty(32, 32),
        f"{PREFIX}sigvq_refiner.0.attention.to_q.weight": torch.empty(32, 32),
        f"{PREFIX}context_refiner.0.attention.to_q.weight": torch.empty(32, 32),
        f"{PREFIX}layers.0.attention.to_q.weight": torch.empty(32, 32),
        f"{PREFIX}layers.1.attention.to_q.weight": torch.empty(32, 32),
    }


def make_metadata(variant="turbo", component_configs=None):
    config = {
        "transformer": {
            "all_patch_size": [1],
            "all_f_patch_size": [1],
            "n_heads": 2,
            "cap_feat_dim": 8,
            "semantic_feat_dim": 10,
            "axes_dims": [4, 6, 6],
        },
        "llada_image": {"variant": variant},
    }
    config.update(component_configs or {})
    return {"config": json.dumps(config)}


def make_component_configs():
    return {
        "text_encoder": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "n_group": 2,
            "topk_group": 1,
            "pad_token_id": 0,
            "mask_token_id": 63,
            "end_of_image_token_id": 62,
            "image_token_offset": 32,
        },
        "queryformer": {
            "num_queries": 5,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 24,
        },
        "text_projection": {
            "hidden_size": 16,
            "intermediate_size": 28,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "projection_dim": 8,
        },
        "sigvq": {
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 16,
            "intermediate_size": 28,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "codebook_size": 8,
            "codebook_embed_dim": 4,
            "semantic_embed_dim": 10,
        },
    }


def test_detection_uses_shapes_and_checkpoint_config():
    detected = comfy.model_detection.detect_unet_config(
        make_state_dict(), PREFIX, make_metadata()
    )

    assert detected == {
        "image_model": "llada_image",
        "variant": "turbo",
        "all_patch_size": (1,),
        "all_f_patch_size": (1,),
        "in_channels": 128,
        "dim": 32,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 8,
        "semantic_feat_dim": 10,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": (4, 6, 6),
    }


def test_supported_model_sets_exact_flow_sampling_contract():
    model_config = comfy.model_detection.model_config_from_unet(
        make_state_dict(), PREFIX, metadata=make_metadata("base")
    )

    assert isinstance(model_config, comfy.supported_models.LLaDAImage)
    assert model_config.sampling_settings == {"multiplier": 1.0, "shift": 1.0}
    assert model_config.latent_format.latent_channels == 128


def test_detection_rejects_missing_variant_metadata():
    with pytest.raises(ValueError, match="base or turbo"):
        comfy.model_detection.detect_unet_config(
            make_state_dict(), PREFIX, {"config": json.dumps({"transformer": {}})}
        )


def test_detection_rejects_config_shape_disagreement():
    metadata = make_metadata()
    config = json.loads(metadata["config"])
    config["transformer"]["dim"] = 4096

    with pytest.raises(ValueError, match="state dict implies 32"):
        comfy.model_detection.detect_unet_config(
            make_state_dict(), PREFIX, {"config": json.dumps(config)}
        )


def test_detection_rejects_incompatible_flux2_latent_channels():
    state_dict = make_state_dict()
    state_dict[f"{PREFIX}all_x_embedder.1-1.weight"] = torch.empty(32, 4)
    state_dict[f"{PREFIX}all_final_layer.1-1.linear.weight"] = torch.empty(4, 32)

    with pytest.raises(ValueError, match="requires 128 transformer input channels"):
        comfy.model_detection.detect_unet_config(
            state_dict, PREFIX, make_metadata("base")
        )


@pytest.mark.parametrize(
    ("component", "key", "value"),
    [
        ("queryformer", "hidden_size", 17),
        ("text_projection", "hidden_size", 17),
        ("text_projection", "projection_dim", 9),
        ("sigvq", "semantic_embed_dim", 11),
    ],
)
def test_detection_rejects_cross_component_config_disagreement(
    component, key, value
):
    component_configs = make_component_configs()
    component_configs[component][key] = value

    with pytest.raises(ValueError, match="config mismatch"):
        comfy.model_detection.detect_unet_config(
            make_state_dict(), PREFIX, make_metadata("base", component_configs)
        )


def test_aio_file_loads_model_clip_and_vae_through_checkpoint_path(
    tmp_path, monkeypatch
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_json = torch.tensor(
        list(tokenizer.to_str().encode("utf-8")), dtype=torch.uint8
    )
    expected_word_embeddings = torch.randn(64, 16)
    expected_queries = torch.randn(5, 16)
    expected_projection = torch.randn(8, 16)
    expected_codebook = torch.randn(8, 10)
    state_dict = make_state_dict()
    state_dict.update(
        {
            "text_encoders.llada2.model.language_model.word_embeddings.weight": expected_word_embeddings,
            "text_encoders.queryformer.meta_queries": expected_queries,
            "text_encoders.text_projection.projector.weight": expected_projection,
            "text_encoders.sigvq.prior_token_embedding.weight": expected_codebook,
            "text_encoders.tokenizer_json": tokenizer_json,
            "vae.bn.running_mean": torch.zeros(128),
        }
    )
    component_configs = make_component_configs()

    class TestVAE:
        def __init__(self, sd, metadata=None, device=None):
            self.state_dict = sd
            self.metadata = metadata

    monkeypatch.setattr(comfy.sd, "VAE", TestVAE)
    checkpoint = tmp_path / "llada-image-aio.safetensors"
    metadata = make_metadata("base", component_configs)
    comfy.utils.save_torch_file(state_dict, checkpoint, metadata=metadata)

    monkeypatch.setattr(
        nodes.folder_paths,
        "get_full_path_or_raise",
        lambda category, name: str(checkpoint),
    )
    monkeypatch.setattr(nodes.folder_paths, "get_folder_paths", lambda category: [])
    monkeypatch.setattr(
        comfy.model_management, "unet_dtype", lambda *args, **kwargs: torch.float32
    )
    monkeypatch.setattr(
        comfy.model_management, "text_encoder_dtype", lambda device=None: torch.float32
    )
    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)

    assert model is not None
    assert clip is not None
    assert clip.cond_stage_model.clip_name == "llada2"
    assert clip.cond_stage_model.llada2.config.hidden_size == 16
    assert clip.cond_stage_model.llada2.queryformer.meta_queries.shape == (5, 16)
    assert clip.cond_stage_model.llada2.text_projection.projector.out_features == 8
    assert clip.cond_stage_model.llada2.sigvq.prior_token_embedding.num_embeddings == 8
    assert torch.equal(
        clip.cond_stage_model.llada2.model.language_model.word_embeddings.weight,
        expected_word_embeddings,
    )
    assert torch.equal(
        clip.cond_stage_model.llada2.queryformer.meta_queries, expected_queries
    )
    assert torch.equal(
        clip.cond_stage_model.llada2.text_projection.projector.weight,
        expected_projection,
    )
    assert torch.equal(
        clip.cond_stage_model.llada2.sigvq.prior_token_embedding.weight,
        expected_codebook,
    )
    assert set(vae.state_dict) == {"bn.running_mean"}
    assert vae.metadata == metadata


@pytest.mark.parametrize("variant", ("base", "turbo"))
@pytest.mark.parametrize("quantized", (False, True))
def test_complete_tiny_aio_loads_and_runs_native_generation_and_editing(
    tmp_path, monkeypatch, variant, quantized, caplog
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    component_configs = make_component_configs()
    native_model = LLaDAImage(
        all_patch_size=(1,),
        all_f_patch_size=(1,),
        in_channels=128,
        dim=32,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        cap_feat_dim=8,
        semantic_feat_dim=10,
        axes_dims=(4, 6, 6),
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )
    native_clip = LLaDAImageTEModel(
        dtype=torch.float32,
        llada2_config=component_configs["text_encoder"],
        queryformer_config=component_configs["queryformer"],
        text_projection_config=component_configs["text_projection"],
        sigvq_config=component_configs["sigvq"],
    )
    torch.manual_seed(61)
    for module in (native_model, native_clip):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    with torch.no_grad():
        language_model = native_clip.llada2.model.language_model
        language_model.word_embeddings.weight.zero_()
        language_model.word_embeddings.weight[:, 0] = 100.0
        for name, parameter in language_model.named_parameters():
            if name == "norm.weight" or name.endswith("layernorm.weight"):
                parameter.fill_(1.0)

    expected_model = {
        key: value.detach().clone() for key, value in native_model.state_dict().items()
    }
    expected_clip = {
        key: value.detach().clone() for key, value in native_clip.state_dict().items()
    }
    for name, module in native_clip.named_modules():
        if hasattr(module, "_orig_shape"):
            prefix = f"{name}." if name else ""
            expected_clip[f"{prefix}weight"] = torch.randn(module._orig_shape) * 0.02
            if getattr(module, "bias", None) is not None:
                expected_clip[f"{prefix}bias"] = module.bias.detach().clone()
    expected_clip["llada2.model.lm_head.weight"].zero_()
    expected_clip["llada2.model.lm_head.weight"][32, 0] = 1.0
    if quantized:
        linear_key = "layers.0.attention.to_q.weight"
        expert_key = next(k for k in expected_clip if k.endswith(".experts.gate_proj.weight"))
        for state, key, rotate in ((expected_model, linear_key, True), (expected_clip, expert_key, False)):
            weight = state[key]
            q, params = TensorWiseINT8Layout.quantize(
                weight, per_channel=True, convrot=rotate, convrot_groupsize=16,
            )
            base = key.removesuffix(".weight")
            config = {"format": "int8_tensorwise"}
            if rotate:
                config.update(convrot=True, convrot_groupsize=16)
            else:
                config["num_experts"] = weight.shape[0]
            state[key] = q
            state[base + ".weight_scale"] = params.scale
            state[base + ".comfy_quant"] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    adapter = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": variant}
    )
    checkpoint_state = {
        f"{PREFIX}{key}": value for key, value in expected_model.items()
    }
    checkpoint_state.update(
        adapter.process_clip_state_dict_for_saving(dict(expected_clip))
    )
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    checkpoint_state["text_encoders.tokenizer_json"] = torch.tensor(
        list(tokenizer.to_str().encode("utf-8")), dtype=torch.uint8
    )
    checkpoint_state["vae.bn.running_mean"] = torch.zeros(128)

    class TestVAE:
        def __init__(self, sd, metadata=None, device=None):
            self.state_dict = sd
            self.metadata = metadata

        @staticmethod
        def encode(image):
            return torch.zeros(
                image.shape[0], 128, image.shape[1] // 16, image.shape[2] // 16
            )

    monkeypatch.setattr(comfy.sd, "VAE", TestVAE)
    checkpoint = tmp_path / f"complete-llada-image-{variant}-aio.safetensors"
    metadata = make_metadata(variant, component_configs)
    comfy.utils.save_torch_file(checkpoint_state, checkpoint, metadata=metadata)
    monkeypatch.setattr(
        nodes.folder_paths,
        "get_full_path_or_raise",
        lambda category, name: str(checkpoint),
    )
    monkeypatch.setattr(nodes.folder_paths, "get_folder_paths", lambda category: [])
    monkeypatch.setattr(
        comfy.model_management, "unet_dtype", lambda *args, **kwargs: torch.float32
    )
    monkeypatch.setattr(
        comfy.model_management, "text_encoder_dtype", lambda device=None: torch.float32
    )

    with caplog.at_level(logging.DEBUG):
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
    assert not any("clip unexpected" in record.message or "clip missing" in record.message for record in caplog.records)
    assert torch.equal(clip.get_sd()["tokenizer_json"], checkpoint_state["text_encoders.tokenizer_json"])

    assert model.model.model_sampling.llada_image_variant == variant
    actual_model = model.model.diffusion_model.state_dict()
    actual_clip = clip.cond_stage_model.state_dict()
    assert set(actual_model) == set(expected_model)
    assert set(actual_clip) == set(expected_clip)
    for key, expected in expected_model.items():
        assert torch.equal(actual_model[key], expected), key
    for key, expected in expected_clip.items():
        assert torch.equal(actual_clip[key], expected), key

    conditioning = clip.encode_from_tokens_scheduled(
        clip.tokenize("a tiny blue square"), show_pbar=False
    )
    negative_conditioning = clip.encode_from_tokens_scheduled(
        clip.tokenize(""), show_pbar=False
    )
    context, conditioning_values = conditioning[0]
    attention_mask = conditioning_values["attention_mask"]
    latent = torch.randn(1, 128, 2, 3)
    sigma = torch.tensor([0.5])

    generated = model.model.apply_model(
        latent,
        sigma,
        c_crossattn=context,
        attention_mask=attention_mask,
    )

    from comfy_extras.nodes_llada_image import (
        LLaDAImageEditConditioning,
        LLaDAImageVQConditioning,
    )

    vq_positive, vq_negative = LLaDAImageVQConditioning.execute(
        clip, "a tiny blue square", "", 64, 64
    )
    vq_context, vq_values = vq_positive[0]
    vq_generated = model.model.apply_model(
        latent,
        sigma,
        c_crossattn=vq_context,
        attention_mask=vq_values["attention_mask"],
        semantic_features=vq_values["semantic_features"],
        semantic_mask=vq_values["semantic_mask"],
    )

    edit_positive, edit_negative, edit_target = (
            LLaDAImageEditConditioning.execute(
                clip,
                vae,
                torch.rand(2, 32, 32, 3),
                "make the square red",
                "",
            )
        )
    edit_context, edit_values = edit_positive[0]
    edit_batch_size = edit_target["samples"].shape[0]
    edited = model.model.apply_model(
        edit_target["samples"],
        sigma,
        c_crossattn=edit_context.repeat(edit_batch_size, 1, 1),
        attention_mask=edit_values["attention_mask"].repeat(edit_batch_size, 1),
        semantic_features=edit_values["semantic_features"],
        semantic_mask=edit_values["semantic_mask"],
        source_latents=edit_values["source_latents"],
    )

    from comfy_extras.nodes_custom_sampler import (
        CFGGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_llada_image import (
        LLaDAImageScheduler,
        SamplerLLaDAImageTurbo,
    )

    sampled_modes = {}
    for mode_index, (mode, positive, negative, target) in enumerate(
        (
            (
                "text",
                conditioning,
                negative_conditioning,
                {"samples": torch.zeros_like(latent)},
            ),
            (
                "vq",
                vq_positive,
                vq_negative,
                {"samples": torch.zeros_like(latent)},
            ),
            ("editing", edit_positive, edit_negative, edit_target),
        )
    ):
        noise = RandomNoise.execute(79 + mode_index)[0]
        guider = CFGGuider.execute(
            model,
            positive,
            negative,
            5.0 if variant == "base" else 1.0,
        )[0]
        sampler = (
            KSamplerSelect.execute("euler")[0]
            if variant == "base"
            else SamplerLLaDAImageTurbo.execute()[0]
        )
        sigmas = LLaDAImageScheduler.execute(model, 2)[0]
        sampled_modes[mode] = SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, target
        )[0]

    assert generated.shape == latent.shape
    assert vq_generated.shape == latent.shape
    assert edited.shape == edit_target["samples"].shape
    assert torch.isfinite(generated).all()
    assert torch.isfinite(vq_generated).all()
    assert torch.isfinite(edited).all()
    assert sampled_modes["text"]["samples"].shape == latent.shape
    assert sampled_modes["vq"]["samples"].shape == latent.shape
    assert sampled_modes["editing"]["samples"].shape == edit_target["samples"].shape
    assert all(
        torch.isfinite(result["samples"]).all()
        for result in sampled_modes.values()
    )
    assert vq_positive[0][1]["semantic_features"].shape == (1, 16, 10)
    assert vq_negative[0][1]["semantic_features"].shape == (1, 0, 10)
    assert edit_positive[0][1]["source_latents"].shape == (2, 128, 2, 2)
    assert edit_negative[0][1]["source_latents"].shape == (2, 128, 2, 2)
