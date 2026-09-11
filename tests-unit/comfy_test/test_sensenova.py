from types import SimpleNamespace

import torch

from comfy.cli_args import args

args.cpu = True

from comfy import model_base, model_detection
import comfy.latent_formats
import comfy.sample
import nodes
from comfy.ldm.sensenova import model as sensenova_model
from comfy.ldm.sensenova.conditioning import (
    block_causal_mask,
    condition_input_ids,
    conditioned_input_length,
    preprocess_references,
    thw_indexes,
)
from comfy.ldm.sensenova.model import _match_prefix_batch, _pad_to_merged_patch_size
from comfy.ldm.sensenova.sampling import (
    SenseNovaModelSampling,
    resolution_noise_scale,
    upstream_sigmas,
)
from comfy.text_encoders.sensenova import SenseNovaTokenizer
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from comfy_extras.nodes_sensenova import SenseNovaSamplingOptions


def _minimal_state_dict():
    return {
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": torch.empty(
            1024, 3, 16, 16, device="meta"
        ),
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": torch.empty(
            4096, 4096, device="meta"
        ),
    }


def _generation_input_ids():
    return torch.tensor(
        [
            [
                151644,
                8948,
                198,
                1,
                151645,
                198,
                151644,
                872,
                198,
                2,
                151645,
                198,
                151644,
                77091,
                198,
                151670,
            ]
        ],
        dtype=torch.long,
    )


def _tokenize_generation_prompt(text):
    tokenizer = SenseNovaTokenizer()
    values = tokenizer.tokenize_with_weights(text)["sensenova_u15"][0]
    return torch.tensor([[int(value[0]) for value in values]], dtype=torch.long)


def test_sensenova_top_level_checkpoint_detection():
    state_dict = _minimal_state_dict()

    assert model_detection.unet_prefix_from_state_dict(state_dict) == ""
    assert model_detection.detect_unet_config(state_dict, "") == {
        "image_model": "sensenova_u15"
    }
    assert (
        type(model_detection.model_config_from_unet(state_dict, "")).__name__
        == "SenseNovaU15"
    )


def test_sensenova_detection_rejects_incompatible_dimensions():
    state_dict = _minimal_state_dict()
    state_dict["language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"] = (
        torch.empty(2048, 2048, device="meta")
    )

    assert model_detection.detect_unet_config(state_dict, "") is None


def test_sensenova_model_config_builds_pixel_space_outputs():
    model_config = model_detection.model_config_from_unet(_minimal_state_dict(), "")
    state_dict = {
        "language_model.lm_head.weight": torch.empty(1),
        "kept": torch.empty(1),
    }

    processed = model_config.process_unet_state_dict(state_dict)
    assert set(processed) == {"kept"}
    assert torch.equal(processed["kept"], state_dict["kept"])
    assert "pixel_space_vae" in model_config.process_vae_state_dict({})
    assert "_sensenova_te_sentinel" in model_config.process_clip_state_dict({})


def test_sensenova_sampling_matches_upstream_schedule_and_resolution_scale():
    config = SimpleNamespace(sampling_settings={"shift": 3.0, "noise_scale": 1.0})
    sampling = SenseNovaModelSampling(config)

    expected = upstream_sigmas(50, 3.0)
    actual = sampling.sigma(torch.linspace(0.0, 1000.0, 51))
    assert torch.allclose(actual, expected)
    assert sampling.percent_to_sigma(0.0) == 1.0
    assert sampling.percent_to_sigma(1.0) == 0.0
    assert resolution_noise_scale(2048, 2048) == 8.0
    assert resolution_noise_scale(4096, 4096) == 16.0

    scaled_sampling = SenseNovaModelSampling(
        SimpleNamespace(sampling_settings={"shift": 3.0, "noise_scale": 0.5})
    )
    noise = torch.ones(1, 3, 256, 256)
    latent = torch.zeros_like(noise)
    scaled = scaled_sampling.noise_scaling(torch.ones(1), noise, latent)
    assert torch.allclose(scaled, torch.full_like(noise, 0.5))


def test_shared_reference_images_append_when_chained():
    conditioning = [[torch.empty(1), {}]]
    first_image = torch.empty(1, 8, 8, 3)
    second_image = torch.empty(1, 8, 8, 3)

    first = HiDreamO1ReferenceImages.execute(
        positive=conditioning,
        negative=conditioning,
        images={"image_1": first_image},
    )
    second = HiDreamO1ReferenceImages.execute(
        positive=first[0],
        negative=first[1],
        images={"image_1": second_image},
    )

    references = second[0][0][1]["reference_latents"]
    assert len(references) == 2
    assert references[0] is first_image
    assert references[1] is second_image
    assert second[1][0][1]["reference_latents"] == references
    assert second[1][0][1]["prompt_type"] == "negative"


def test_shared_reference_images_use_numeric_socket_order():
    conditioning = [[torch.empty(1), {}]]
    first_image = torch.empty(1, 8, 8, 3)
    second_image = torch.empty(1, 8, 8, 3)
    extra_image = torch.empty(1, 8, 8, 3)

    output = HiDreamO1ReferenceImages.execute(
        positive=conditioning,
        negative=conditioning,
        images={
            "image_2": second_image,
            "extra_image": extra_image,
            "image_1": first_image,
        },
    )

    references = output[0][0][1]["reference_latents"]
    assert references[0] is first_image
    assert references[1] is second_image
    assert references[2] is extra_image


def test_shared_reference_images_allow_empty_inputs_and_image_batches():
    conditioning = [[torch.empty(1), {}]]
    empty = HiDreamO1ReferenceImages.execute(
        positive=conditioning,
        negative=conditioning,
        images={},
    )
    assert empty[0] is conditioning
    assert empty[1] is conditioning

    image_batch = torch.empty(2, 8, 8, 3)
    attached = HiDreamO1ReferenceImages.execute(
        positive=conditioning,
        negative=conditioning,
        images={"image_1": image_batch},
    )
    assert attached[0][0][1]["reference_latents"] == [image_batch]


def test_hidream_o1_ignores_shared_negative_marker(monkeypatch):
    calls = []

    def build_extra_conds(text_input_ids, noise, ref_images, target_patch_size):
        calls.append((text_input_ids, noise, ref_images, target_patch_size))
        return {
            "input_ids": text_input_ids,
            "ar_len": text_input_ids.shape[1] - 1,
        }

    monkeypatch.setattr(model_base, "build_extra_conds", build_extra_conds)
    model = object.__new__(model_base.HiDreamO1)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    noise = torch.empty(1, 3, 64, 64)
    references = [torch.empty(1, 32, 32, 3)]

    positive = model.extra_conds(
        text_input_ids=input_ids,
        noise=noise,
        reference_latents=references,
    )
    negative = model.extra_conds(
        text_input_ids=input_ids,
        noise=noise,
        reference_latents=references,
        prompt_type="negative",
    )

    assert len(calls) == 2
    for call_input_ids, call_noise, call_references, call_patch_size in calls:
        assert call_input_ids is input_ids
        assert call_noise is noise
        assert call_references is references
        assert call_patch_size == 32
    assert positive.keys() == negative.keys()
    assert torch.equal(positive["input_ids"].cond, negative["input_ids"].cond)
    assert positive["ar_len"].cond == negative["ar_len"].cond


def test_sensenova_reference_preprocessing_preserves_size_and_splits_batches():
    references = preprocess_references(
        [torch.rand(2, 9, 13, 1), torch.empty(1, 0, 0, 0)]
    )

    assert len(references) == 3
    assert all(reference.shape == (1, 3, 9, 13) for reference in references[:2])
    assert references[2].shape == (1, 3, 0, 0)
    assert _pad_to_merged_patch_size(references[0]).shape == (1, 3, 32, 32)
    assert _pad_to_merged_patch_size(references[2]).shape == (1, 3, 32, 32)


def test_sensenova_reference_shape_estimate_uses_padded_image_sizes():
    model = object.__new__(model_base.SenseNovaU15)
    input_ids = _generation_input_ids()

    shapes = model.extra_conds_shapes(
        reference_latents=[torch.empty(2, 33, 65, 3)],
        text_input_ids=input_ids,
    )

    grids = [(2, 3), (2, 3)]
    length = conditioned_input_length(input_ids.shape[1], grids)
    assert shapes["reference_images"] == [1, 3, 12288]
    assert shapes["prefix_mask"] == [1, 1, length, length]
    prefix_shape = [
        1,
        sensenova_model.NUM_KV_HEADS,
        sensenova_model.NUM_LAYERS * length * sensenova_model.HEAD_DIM,
    ]
    assert shapes["prefix_keys"] == prefix_shape
    assert shapes["prefix_values"] == prefix_shape

    negative_shapes = model.extra_conds_shapes(
        reference_latents=[torch.empty(2, 33, 65, 3)],
        text_input_ids=input_ids,
        prompt_type="negative",
    )
    negative_length = conditioned_input_length(
        input_ids.shape[1], grids, image_only=True
    )
    assert negative_shapes["prefix_mask"] == [
        1,
        1,
        negative_length,
        negative_length,
    ]


def test_standard_empty_latent_adapts_to_sensenova_pixel_format():
    latent = nodes.EmptyLatentImage().generate(width=64, height=96, batch_size=2)[0]
    model = SimpleNamespace(
        get_model_object=lambda name: comfy.latent_formats.HiDreamO1Pixel()
    )

    samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent["downscale_ratio_spacial"],
    )

    assert samples.shape == (2, 3, 96, 64)


def test_sensenova_prefix_conditioning_adapts_to_mismatched_batches():
    input_ids = torch.tensor([[1], [2]])
    indexes = torch.arange(6).reshape(2, 3, 1)
    mask = torch.zeros(2, 1, 1, 1)

    input_ids, indexes, mask = _match_prefix_batch(3, input_ids, indexes, mask)

    assert input_ids[:, 0].tolist() == [1, 2, 2]
    assert indexes.shape == (3, 3, 1)
    assert mask.shape == (3, 1, 1, 1)


def test_reference_node_and_sensenova_sampling_do_not_add_quality_limits():
    sampling_inputs = {
        input.id: input for input in SenseNovaSamplingOptions.define_schema().inputs
    }
    assert sampling_inputs["shift"].min is None
    assert sampling_inputs["shift"].max is None

    reference_inputs = {
        input.id: input for input in HiDreamO1ReferenceImages.define_schema().inputs
    }
    images = reference_inputs["images"]
    assert images.optional
    assert images.template.min == 0
    assert len(images.template.names) == 100


def test_sensenova_prefix_preprocessing_runs_each_prefix_layer_once(monkeypatch):
    calls = []
    rope_calls = []

    prepare_mrope = sensenova_model._prepare_mrope

    def tracked_prepare_mrope(indexes, device, dtype):
        rope_calls.append((indexes.shape, device, dtype))
        return prepare_mrope(indexes, device, dtype)

    monkeypatch.setattr(sensenova_model, "_prepare_mrope", tracked_prepare_mrope)

    class Layer:
        def forward_prefix(self, prefix, prefix_rope, prefix_mask, transformer_options):
            calls.append(
                (
                    transformer_options["block_index"],
                    tuple(axis[0].shape for axis in prefix_rope),
                    prefix_mask.dtype,
                )
            )
            key = prefix[..., :1].unsqueeze(1)
            value = key + 1
            return prefix + 1, key, value

    input_ids = torch.tensor([[1, 2, 3]])
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=lambda values: torch.zeros(
                    *values.shape, sensenova_model.HIDDEN_SIZE
                ),
                layers=[Layer(), Layer()],
            )
        )
    )
    model._prepare_prefix = lambda *args: sensenova_model.SenseNovaU15._prepare_prefix(
        model, *args
    )

    prefix_keys, prefix_values, prefix_time = (
        sensenova_model.SenseNovaU15.preprocess_prefix(model, input_ids)
    )

    assert calls == [
        (
            0,
            (
                torch.Size([1, 1, 3, 64]),
                torch.Size([1, 1, 3, 32]),
                torch.Size([1, 1, 3, 32]),
            ),
            torch.float32,
        ),
        (
            1,
            (
                torch.Size([1, 1, 3, 64]),
                torch.Size([1, 1, 3, 32]),
                torch.Size([1, 1, 3, 32]),
            ),
            torch.float32,
        ),
    ]
    assert rope_calls == [(torch.Size([3, 3]), torch.device("cpu"), torch.float32)]
    assert len(prefix_keys) == 2
    assert len(prefix_values) == 2
    assert prefix_keys[0].shape == (1, 1, 3, 1)
    assert prefix_time.tolist() == [3]


def test_sensenova_model_base_preprocesses_prefix_conditioning():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask):
        calls.append((input_ids, references, indexes, prefix_mask))
        return (
            [torch.zeros(1, 1, 3, 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, 3, 1, dtype=torch.bfloat16)],
            torch.tensor([3]),
        )

    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(
        dtype=torch.bfloat16,
        preprocess_prefix=preprocess_prefix,
    )
    input_ids = torch.tensor([[1, 2, 3]])

    conds = model.extra_conds(
        text_input_ids=input_ids,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    assert calls[0][0] is input_ids
    assert calls[0][1:] == (None, None, None)
    assert "text_input_ids" not in conds
    assert conds["prefix_keys"].cond[0].dtype == torch.bfloat16
    assert conds["prefix_values"].cond[0].dtype == torch.bfloat16
    assert conds["prefix_time"].cond.tolist() == [3]


def test_sensenova_uses_prompt_type_for_negative_reference_conditioning():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask):
        calls.append((input_ids, references, indexes, prefix_mask))
        return (
            [torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, 1, 1, dtype=torch.bfloat16)],
            torch.tensor([1]),
        )

    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(
        dtype=torch.bfloat16,
        preprocess_prefix=preprocess_prefix,
    )
    input_ids = _generation_input_ids()
    reference = torch.rand(1, 32, 32, 3)

    model.extra_conds(
        text_input_ids=input_ids,
        reference_latents=[reference],
        prompt_type="negative",
        device=torch.device("cpu"),
    )

    expected_ids = condition_input_ids(input_ids, [(1, 1)], image_only=True)
    assert torch.equal(calls[0][0], expected_ids)
    assert len(calls[0][1]) == 1
    assert calls[0][1][0].shape == (1, 3, 32, 32)
    assert calls[0][2].shape == (1, 3, expected_ids.shape[1])
    assert calls[0][3].shape == (1, 1, expected_ids.shape[1], expected_ids.shape[1])


def test_sensenova_preprocessed_prefix_matches_raw_forward():
    class Layer:
        def forward_prefix(self, prefix, prefix_rope, prefix_mask, transformer_options):
            key = prefix[..., :1].unsqueeze(1)
            return prefix + 1, key, key + 1

        def forward_generation(
            self, image, image_rope, prefix_key, prefix_value, transformer_options
        ):
            offset = (prefix_key + prefix_value).mean(dim=(1, 2, 3))
            return image + offset[:, None, None]

    class VisionModel:
        def __call__(self, image):
            batch, _, height, width = image.shape
            length = (height // sensenova_model.MERGED_PATCH_SIZE) * (
                width // sensenova_model.MERGED_PATCH_SIZE
            )
            return image.new_zeros(batch, length, sensenova_model.HIDDEN_SIZE)

    class TimestepEmbedder:
        def __init__(self):
            self.shapes = []

        def __call__(self, timesteps, dtype):
            self.shapes.append(timesteps.shape)
            return torch.zeros(
                timesteps.shape[0], sensenova_model.HIDDEN_SIZE, dtype=dtype
            )

    class Head:
        def __call__(self, image):
            return (
                image[:, :3]
                .repeat_interleave(sensenova_model.MERGED_PATCH_SIZE, dim=2)
                .repeat_interleave(sensenova_model.MERGED_PATCH_SIZE, dim=3)
            )

    timestep_embedder = TimestepEmbedder()
    noise_scale_embedder = TimestepEmbedder()
    backbone = SimpleNamespace(
        embed_tokens=lambda values: torch.zeros(
            *values.shape, sensenova_model.HIDDEN_SIZE
        ),
        layers=[Layer(), Layer()],
        norm_mot_gen=lambda image: image,
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=backbone),
        fm_modules={
            "vision_model_mot_gen": VisionModel(),
            "timestep_embedder": timestep_embedder,
            "noise_scale_embedder": noise_scale_embedder,
            "fm_head": Head(),
        },
    )
    model._prepare_prefix = lambda *args: sensenova_model.SenseNovaU15._prepare_prefix(
        model, *args
    )
    input_ids = torch.tensor([[1, 2, 3]])
    image = torch.zeros(1, 3, 64, 64)
    timesteps = torch.tensor([0.5])

    raw = sensenova_model.SenseNovaU15._forward(
        model,
        image,
        timesteps,
        text_input_ids=input_ids,
        transformer_options={},
    )
    prefix_keys, prefix_values, prefix_time = (
        sensenova_model.SenseNovaU15.preprocess_prefix(model, input_ids)
    )
    preprocessed = sensenova_model.SenseNovaU15._forward(
        model,
        image,
        timesteps,
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        prefix_time=prefix_time,
        transformer_options={},
    )

    assert torch.equal(raw, preprocessed)
    assert timestep_embedder.shapes == [torch.Size([1]), torch.Size([1])]
    assert noise_scale_embedder.shapes == [torch.Size([1]), torch.Size([1])]


def test_sensenova_reference_tokens_and_indexes():
    input_ids = _generation_input_ids()
    grids = [(2, 3), (1, 2)]

    conditioned = condition_input_ids(input_ids, grids)
    indexes = thw_indexes(conditioned, grids)

    assert conditioned.shape[1] == conditioned_input_length(input_ids.shape[1], grids)
    assert torch.count_nonzero(conditioned == 151669) == 8
    assert indexes.shape == (1, 3, conditioned.shape[1])


def test_sensenova_prefix_mask_matches_attention_dtype(monkeypatch):
    query = torch.empty(1, 32, 3, 128, dtype=torch.bfloat16)
    key = torch.empty(1, 8, 3, 128, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    captured = {}

    def optimized_attention(query, key, value, heads, **kwargs):
        captured.update(query=query, key=key, value=value, heads=heads, kwargs=kwargs)
        return torch.empty(1, 3, 4096, dtype=torch.bfloat16)

    monkeypatch.setattr(sensenova_model, "optimized_attention", optimized_attention)
    attention = SimpleNamespace(
        _project=lambda hidden_states, rope, generation: (query, key, value),
        o_proj=lambda output: output,
    )

    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bfloat16)
    output, _, _ = sensenova_model.Attention.forward_prefix(
        attention,
        torch.empty(1, 3, 4096, dtype=torch.bfloat16),
        torch.empty(3, 1, 3),
        mask,
        {},
    )

    assert output.shape == (1, 3, 4096)
    assert captured["kwargs"]["mask"] is mask


def test_sensenova_prefix_mask_is_created_in_the_model_dtype():
    indexes = torch.tensor([[[0, 1, 1], [0, 0, 0], [0, 0, 0]]])

    mask = block_causal_mask(indexes, dtype=torch.bfloat16)

    assert mask.dtype == torch.bfloat16
    assert torch.equal(
        mask[0, 0],
        torch.tensor(
            [
                [0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.bfloat16,
        ),
    )


def test_sensenova_reference_tokens_allow_more_than_ten_images():
    input_ids = _tokenize_generation_prompt("test")
    grids = [(1, 1)] * 12

    conditioned = condition_input_ids(input_ids, grids)
    indexes = thw_indexes(conditioned, grids)
    expected_text = (
        "".join(
            f"Image-{index}:<img><IMG_CONTEXT></img>\n"
            for index in range(1, 13)
        )
        + "test"
    )

    assert torch.equal(conditioned, _tokenize_generation_prompt(expected_text))
    assert conditioned.shape[1] == conditioned_input_length(input_ids.shape[1], grids)
    assert torch.count_nonzero(conditioned == 151669) == 12
    assert indexes.shape == (1, 3, conditioned.shape[1])


def test_sensenova_reference_tokens_tolerate_nonstandard_prompt_templates():
    conditioned = condition_input_ids(torch.tensor([[1, 2]]), [(1, 1)])

    assert torch.count_nonzero(conditioned == 151669) == 1


def test_sensenova_tokenizer_control_token_ids():
    tokenizer = SenseNovaTokenizer()
    backend = tokenizer.sensenova_u15.tokenizer

    assert len(backend) == 151936
    assert backend.convert_tokens_to_ids(
        ["<IMG_CONTEXT>", "<img>", "</img>", "<FAKE_PAD_253>"]
    ) == [151669, 151670, 151671, 151935]
    assert "<|im_start|>" in backend.all_special_tokens
    assert "<|vision_pad|>" in backend.all_special_tokens
    assert tokenizer.tokenize_with_weights("")["sensenova_u15"][0][-1][0] == 151670
