from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

from comfy import model_base, model_detection
import comfy.latent_formats
import comfy.ops
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
from comfy.ldm.sensenova.interleave import (
    prefix_arguments,
)
from comfy.ldm.sensenova.model import _match_prefix_batch, _pad_to_merged_patch_size
from comfy.ldm.sensenova.sampling import (
    SenseNovaModelSampling,
    resolution_noise_scale,
    upstream_sigmas,
)
from comfy.text_encoders.sensenova import (
    SenseNovaQwenTokenizer,
    SenseNovaTokenizer,
    build_generation_prompt,
)
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from comfy_extras.nodes_sensenova import (
    expand_interleave_metadata,
    ModelSamplingSenseNova,
)
from sensenova_test_utils import (
    PatchEmbedding,
    _generation_input_ids,
    _text_model_patcher,
    _tokenize_generation_prompt,
)


def _minimal_u15_state_dict(has_lm_head=False):
    state_dict = {
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": torch.empty(
            1024, 3, 16, 16, device="meta"
        ),
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": torch.empty(
            4096, 4096, device="meta"
        ),
        "fm_modules.fm_head.conv1.weight": torch.empty(
            1024, 1024, 3, 3, device="meta"
        ),
    }
    if has_lm_head:
        state_dict["language_model.lm_head.weight"] = torch.empty(
            151936, 4096, device="meta"
        )
    return state_dict


@pytest.mark.parametrize("has_lm_head", [False, True])
def test_sensenova_top_level_checkpoint_detection(has_lm_head):
    state_dict = _minimal_u15_state_dict(has_lm_head)

    assert model_detection.unet_prefix_from_state_dict(state_dict) == ""
    assert model_detection.detect_unet_config(state_dict, "") == {
        "image_model": "sensenova_u15",
        "has_lm_head": has_lm_head,
    }
    assert (
        type(model_detection.model_config_from_unet(state_dict, "")).__name__
        == "SenseNovaU15"
    )


def test_sensenova_detection_rejects_incompatible_dimensions():
    state_dict = _minimal_u15_state_dict()
    state_dict["language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"] = (
        torch.empty(2048, 2048, device="meta")
    )

    assert model_detection.detect_unet_config(state_dict, "") is None


def test_sensenova_detection_does_not_treat_u1_mlp_head_as_u15():
    state_dict = _minimal_u15_state_dict()
    state_dict.pop("fm_modules.fm_head.conv1.weight")
    state_dict["fm_modules.fm_head.0.weight"] = torch.empty(
        4096, 1024, device="meta"
    )

    assert model_detection.detect_unet_config(state_dict, "") is None


def test_sensenova_model_config_builds_pixel_space_outputs():
    model_config = model_detection.model_config_from_unet(_minimal_u15_state_dict(), "")
    state_dict = {
        "language_model.lm_head.weight": torch.empty(1),
        "kept": torch.empty(1),
    }

    processed = model_config.process_unet_state_dict(state_dict)
    assert set(processed) == {"language_model.lm_head.weight", "kept"}
    assert torch.equal(
        processed["language_model.lm_head.weight"],
        state_dict["language_model.lm_head.weight"],
    )
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
        [torch.rand(2, 9, 13, 1), torch.empty(1, 9, 13, 0)]
    )

    assert len(references) == 3
    assert all(reference.shape == (1, 3, 9, 13) for reference in references[:2])
    assert references[2].shape == (1, 3, 9, 13)
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


def test_sensenova_sampling_and_reference_schema_limits():
    sampling_inputs = {
        input.id: input for input in ModelSamplingSenseNova.define_schema().inputs
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
        def text_modules(self):
            return ()

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
    model.prepare_mrope = sensenova_model.SenseNovaU15.prepare_mrope
    model._preprocess_prefix_state = lambda *args, **kwargs: sensenova_model.SenseNovaU15._preprocess_prefix_state(model, *args, **kwargs)

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


@pytest.mark.parametrize("prompt_type,hooked", [(None, False), ("positive", False), ("negative", False), ("positive", True)])
def test_sensenova_model_base_preprocesses_prefix_conditioning(prompt_type, hooked):
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, **kwargs):
        calls.append((input_ids, references, indexes, prefix_mask))
        return (
            [torch.zeros(1, 1, 3, 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, 3, 1, dtype=torch.bfloat16)],
            torch.tensor([3]),
        )

    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.current_patcher = None
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(dtype=torch.bfloat16, preprocess_prefix=preprocess_prefix)
    input_ids = torch.tensor([[1, 2, 3]])
    conds = model.extra_conds(
        text_input_ids=input_ids, prompt_type=prompt_type,
        hooks=object() if hooked else None, device=torch.device("cpu"),
    )

    if hooked:
        assert calls == []
        assert conds["text_input_ids"].cond is input_ids
        assert "prefix_keys" not in conds
    else:
        assert len(calls) == 1
        assert calls[0][0] is input_ids
        assert calls[0][1:] == (None, None, None)
        assert "text_input_ids" not in conds
        assert conds["prefix_keys"].cond[0].dtype == torch.bfloat16
        assert conds["prefix_values"].cond[0].dtype == torch.bfloat16
        assert conds["prefix_time"].cond.tolist() == [3]


@pytest.mark.parametrize("height,width", [(0, 32), (32, 0), (0, 0), (0, 65), (65, 0)])
@pytest.mark.parametrize("batched", [False, True])
def test_sensenova_sampler_encodes_zero_sized_references(monkeypatch, height, width, batched):
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 8)
    model = _text_model_patcher([42] * 8).model
    model.concat_keys = ()
    model.diffusion_model.vision_model = sensenova_model.VisionModel(operations=torch.nn)
    reference = torch.empty(height, width, 3)
    if batched:
        reference = reference.unsqueeze(0)

    conds = model.extra_conds(
        text_input_ids=_generation_input_ids(), reference_latents=[reference],
        device=torch.device("cpu"), hooks=object(),
    )
    input_ids = conds["text_input_ids"].cond
    references = conds["reference_images"].cond
    prefix, indexes, mask, _ = model.diffusion_model._prepare_prefix(
        input_ids, references, conds["prefix_indexes"].cond, conds["prefix_mask"].cond,
    )
    encoded = model.diffusion_model.vision_model(_pad_to_merged_patch_size(references[0]))
    selected = input_ids == 151669
    assert selected.sum().item() == max(1, (height + 31) // 32) * max(1, (width + 31) // 32)
    torch.testing.assert_close(prefix[selected], encoded.flatten(0, 1))
    assert prefix.isfinite().all()
    assert indexes.shape[-1] == mask.shape[-1] == prefix.shape[1]


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("input_values", [[1, 2, 3], [1, 2, 3, 151670]])
def test_sensenova_empty_reference_batch_preserves_text_conditioning(negative, expanded, input_values):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = torch.tensor([input_values])

    conds = model.extra_conds(
        text_input_ids=input_ids, reference_latents=[torch.empty(0, 32, 64, 3)],
        sensenova_interleave_expanded=expanded,
        prompt_type="negative" if negative else "positive",
        device=torch.device("cpu"), hooks=object(),
    )

    assert set(conds) == {"text_input_ids"}
    assert conds["text_input_ids"].cond is input_ids


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("references", [None, [], [torch.empty(0, 32, 64, 3)]])
def test_sensenova_empty_reference_batch_rejects_expanded_image_tokens(negative, expanded, references):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = torch.tensor([[151670, 151669, 151669, 151671]])

    with pytest.raises(ValueError, match="2 image context tokens.*require 0"):
        model.extra_conds(
            text_input_ids=input_ids, reference_latents=references,
            sensenova_interleave_expanded=expanded,
            prompt_type="negative" if negative else "positive",
            device=torch.device("cpu"), hooks=object(),
        )


@pytest.mark.parametrize("markers", ["<IMG_CONTEXT>", "<img>", "</img>", "<img></img>", "</img><img>",
                                     "<|im_end|><|im_start|>system", "<think></think>"])
@pytest.mark.parametrize("reference_shapes", [(), ((32, 64),), ((32, 64), (64, 64))])
def test_sensenova_sampler_accepts_user_markers_as_text(markers, reference_shapes):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    text = f"A poster with {markers} in the title"
    input_ids = _tokenize_generation_prompt(text)

    conds = model.extra_conds(
        text_input_ids=input_ids,
        reference_latents=[torch.zeros(1, height, width, 3) for height, width in reference_shapes],
        prompt_type="positive", device=torch.device("cpu"), hooks=object(),
    )
    ids = conds["text_input_ids"].cond[0].tolist()

    assert ids.count(151669) == sum(height // 32 * (width // 32) for height, width in reference_shapes)
    assert ids.count(151670) == len(reference_shapes) + 1
    assert ids.count(151671) == len(reference_shapes)
    assert ids.count(151644) == 3
    assert ids.count(151645) == 2
    assert text in SenseNovaTokenizer().decode(ids, skip_special_tokens=False)


@pytest.mark.parametrize("markers", ["<IMG_CONTEXT>", "<img>", "</img>", "</img><img>"])
def test_sensenova_negative_prefix_rebuild_discards_literal_image_markers(markers):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = _tokenize_generation_prompt(f"A poster with {markers} in the title")

    conds = model.extra_conds(
        text_input_ids=input_ids,
        reference_latents=[torch.zeros(1, 32, 64, 3), torch.zeros(1, 64, 64, 3)],
        prompt_type="negative", device=torch.device("cpu"), hooks=object(),
    )

    assert conds["text_input_ids"].cond.tolist() == [[
        151644, 872, 198,
        151670, 151669, 151669, 151671,
        151670, 151669, 151669, 151669, 151669, 151671,
        151645, 198, 151644, 77091, 198, 151670,
    ]]


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_sensenova_hooked_prefix_preserves_reference_blocks(negative, expanded):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    model.manual_cast_dtype = torch.bfloat16
    image_block = [151670, 151669, 151669, 151671]
    input_ids = torch.tensor([[1, 2, 3, *image_block, 198]] if expanded else [[1, 2, 3]])
    reference = torch.ones(1, 32, 64, 3)

    conds = model.extra_conds(
        text_input_ids=input_ids, reference_latents=[reference],
        sensenova_interleave_expanded=expanded,
        prompt_type="negative" if negative else "positive",
        device=torch.device("cpu"), hooks=object(),
    )

    if negative and not expanded:
        expected = [[151644, 872, 198, *image_block, 151645, 198, 151644, 77091, 198, 151670]]
    else:
        expected = [[1, 2, 3, *image_block, 198]]
    assert conds["text_input_ids"].cond.tolist() == expected
    assert "prefix_keys" not in conds
    assert conds["reference_images"].cond[0].shape == (1, 3, 32, 64)
    assert conds["reference_images"].cond[0].dtype == torch.float32
    torch.testing.assert_close(
        conds["reference_images"].cond[0][0, :, 0, 0],
        torch.tensor([2.2489083, 2.4285714, 2.64]),
    )
    assert conds["prefix_indexes"].cond[0, :, 4:6].tolist() == [[4, 4], [0, 0], [0, 1]]
    mask = conds["prefix_mask"].cond
    assert mask.dtype == torch.bfloat16
    assert mask[0, 0, 4, 5] == 0
    assert torch.isneginf(mask[0, 0, 3, 4])
    assert torch.equal(reference, torch.ones(1, 32, 64, 3))


@pytest.mark.parametrize("context_slots", [3, 5])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("generated_start", [None, 0])
def test_sensenova_sampler_rejects_mismatched_interleave_image_tokens(context_slots, negative, generated_start):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = torch.tensor([[151670, *([151669] * context_slots), 151671]])

    with pytest.raises(ValueError, match=f"{context_slots} image context tokens.*require 4"):
        model.extra_conds(
            text_input_ids=input_ids, reference_latents=[torch.zeros(1, 64, 64, 3)],
            sensenova_interleave_expanded=True,
            sensenova_interleave_start=generated_start,
            prompt_type="negative" if negative else "positive",
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("values,error", [
    ([151670, 151669, 151669, 151669, 151671, 151670, 151669, 151669, 151669, 151671], "image block 1.*require 2"),
    ([151670, 151669, 151669, 151671, 151670, 151669, 151669, 151669, 151669], "image blocks do not match"),
    ([151669, 151669, 151671, 151670, 151669, 151669, 151669, 151669, 151671], "image blocks do not match"),
    ([151669, 151670, 151669, 151671, 151670, 151669, 151669, 151669, 151669, 151671], "image block 1.*require 2"),
], ids=["wrong-block-sizes", "missing-end", "missing-start", "context-outside-block"])
def test_sensenova_sampler_rejects_misaligned_interleave_image_blocks(values, error):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = torch.tensor([values])

    with pytest.raises(ValueError, match=error):
        model.extra_conds(
            text_input_ids=input_ids,
            reference_latents=[torch.zeros(1, 32, 64, 3), torch.zeros(1, 64, 64, 3)],
            sensenova_interleave_expanded=True, prompt_type="positive",
            device=torch.device("cpu"), hooks=object(),
        )


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("pending_image", [False, True])
def test_sensenova_sampler_preserves_multiple_interleave_image_blocks(negative, pending_image):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    values = [151670, 151669, 151669, 151671, 198, 151670, 151669, 151669, 151669, 151669, 151671]
    if pending_image:
        values.append(151670)
    input_ids = torch.tensor([values])

    conds = model.extra_conds(
        text_input_ids=input_ids,
        reference_latents=[torch.zeros(1, 32, 64, 3), torch.zeros(1, 64, 64, 3)],
        sensenova_interleave_expanded=True,
        prompt_type="negative" if negative else "positive",
        device=torch.device("cpu"), hooks=object(),
    )

    assert conds["text_input_ids"].cond is input_ids
    assert len(conds["reference_images"].cond) == 2
    indexes = conds["prefix_indexes"].cond
    assert indexes[0, 1:, 1:3].tolist() == [[0, 0], [0, 1]]
    assert indexes[0, 1:, 6:10].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]


def test_sensenova_model_base_does_not_reinsert_expanded_interleave_images():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, transformer_options=None, text_suffix=None, compute_dtype=None):
        calls.append((input_ids, references, indexes, prefix_mask))
        return (
            [torch.zeros(1, 1, input_ids.shape[1], 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, input_ids.shape[1], 1, dtype=torch.bfloat16)],
            torch.tensor([input_ids.shape[1]]),
        )

    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.current_patcher = None
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(
        dtype=torch.bfloat16,
        preprocess_prefix=preprocess_prefix,
    )
    metadata = expand_interleave_metadata(
        {
            "text_input_ids": torch.tensor([[1, 2, 3]]),
            "reference_latents": [torch.zeros(1, 33, 65, 3)],
        },
        image_only=False,
    )

    model.extra_conds(
        text_input_ids=metadata["text_input_ids"],
        reference_latents=metadata["reference_latents"],
        sensenova_interleave_expanded=True,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    assert calls[0][0].shape == metadata["text_input_ids"].shape
    assert torch.equal(calls[0][0], metadata["text_input_ids"])


def test_sensenova_uses_prompt_type_for_negative_reference_conditioning():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, transformer_options=None, text_suffix=None, compute_dtype=None):
        calls.append((input_ids, references, indexes, prefix_mask))
        return (
            [torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, 1, 1, dtype=torch.bfloat16)],
            torch.tensor([1]),
        )

    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.current_patcher = None
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


@pytest.mark.parametrize("text_suffix", [None, torch.tensor([[42, 151670, 43, 151669, 151671, 151668, 271, 151670]])])
def test_sensenova_preprocessed_prefix_matches_raw_forward(text_suffix):
    class Layer:
        def text_modules(self):
            return ()

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
        embed_tokens=lambda values: values.float().unsqueeze(-1).repeat(
            1, 1, sensenova_model.HIDDEN_SIZE
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
    model.prepare_mrope = sensenova_model.SenseNovaU15.prepare_mrope
    model.predict_velocity = sensenova_model.SenseNovaU15.predict_velocity
    model.embed_timesteps = lambda *args: sensenova_model.SenseNovaU15.embed_timesteps(model, *args)
    model._preprocess_prefix_state = lambda *args, **kwargs: sensenova_model.SenseNovaU15._preprocess_prefix_state(model, *args, **kwargs)
    input_ids = torch.tensor([[1, 2, 3]])
    image = torch.zeros(1, 3, 64, 64)
    timesteps = torch.tensor([0.5])

    raw = sensenova_model.SenseNovaU15._forward(
        model,
        image,
        timesteps,
        text_input_ids=input_ids,
        text_suffix=text_suffix,
        transformer_options={},
    )
    prefix_keys, prefix_values, prefix_time = (
        sensenova_model.SenseNovaU15.preprocess_prefix(model, input_ids, text_suffix=text_suffix)
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
    assert timestep_embedder.shapes == [torch.Size([1])] * 2
    assert noise_scale_embedder.shapes == [torch.Size([1])] * 2
    if text_suffix is not None:
        without_suffix = sensenova_model.SenseNovaU15._forward(
            model, image, timesteps, text_input_ids=input_ids, transformer_options={},
        )
        assert not torch.equal(raw, without_suffix)


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
        comfy_attention=sensenova_model.ComfyAttention(),
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

    expected = SenseNovaQwenTokenizer().tokenize_with_weights(
        build_generation_prompt(expected_text), disable_weights=True,
    )[0]
    assert conditioned.tolist() == [[int(pair[0]) for pair in expected]]
    assert conditioned.shape[1] == conditioned_input_length(input_ids.shape[1], grids)
    assert torch.count_nonzero(conditioned == 151669) == 12
    assert indexes.shape == (1, 3, conditioned.shape[1])


def test_sensenova_reference_tokens_tolerate_nonstandard_prompt_templates():
    conditioned = condition_input_ids(torch.tensor([[1, 2]]), [(1, 1)])

    assert torch.count_nonzero(conditioned == 151669) == 1


def test_sensenova_interleave_negative_reference_waits_for_image_event():
    tokenizer = SenseNovaTokenizer()
    pairs = tokenizer.tokenize_with_weights("", mode="interleave")["sensenova_u15"][0]
    input_ids = torch.tensor([[int(pair[0]) for pair in pairs]])

    conditioned = condition_input_ids(
        input_ids,
        [(1, 2)],
        image_only=True,
        append_image_start=False,
    )

    assert torch.count_nonzero(conditioned == 151669) == 2
    assert conditioned[0, -1].item() == 198
    assert conditioned.shape[1] == 12


def test_sensenova_interleave_replay_matches_continuous_decode(monkeypatch):
    for name, value in (("HIDDEN_SIZE", 32), ("INTERMEDIATE_SIZE", 48), ("NUM_HEADS", 4), ("NUM_KV_HEADS", 2), ("HEAD_DIM", 8)):
        monkeypatch.setattr(sensenova_model, name, value)
    model = _text_model_patcher(list(range(32))).model
    model.concat_keys = ()
    diffusion = model.diffusion_model
    backbone = diffusion.language_model.model
    backbone.layers = torch.nn.ModuleList([
        sensenova_model.DecoderLayer(device="cpu", dtype=torch.float32, operations=comfy.ops.manual_cast)
        for _ in range(2)
    ])
    backbone.norm = comfy.ops.manual_cast.RMSNorm(32, eps=1e-6)

    diffusion.vision_model = PatchEmbedding(3, 32, kernel_size=32, stride=32)
    generator = torch.Generator().manual_seed(23)
    for name, parameter in diffusion.named_parameters():
        if "norm" in name:
            parameter.detach().fill_(1)
        else:
            parameter.detach().normal_(std=0.08, generator=generator)

    # Explicit upstream events, not a second parser of the serialized history.
    ids = [1, 2, 151669, 151671, 3, 151670, 151669, 151669, 151671,
           151669, 151671, 4, 151670, 151669, 151669, 151669, 151669, 151671,
           151669, 151671, 5]
    times = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14, 15, 16]
    heights = [0] * 15 + [1, 1] + [0] * 4
    widths = [0] * 7 + [1] + [0] * 6 + [1, 0, 1] + [0] * 4
    references = [
        torch.linspace(-0.2, 1.2, 32 * 64 * 3).reshape(1, 32, 64, 3),
        torch.linspace(1.1, -0.1, 64 * 64 * 3).reshape(1, 64, 64, 3),
    ]
    image_events = {6: (2, references[0]), 13: (4, references[1])}
    checkpoints = {5: 0, 6: 0, 9: 1, 12: 1, 13: 1, 18: 2, 21: 2}
    position = torch.empty(1, dtype=torch.int64)
    seqlen = torch.zeros(1, dtype=torch.int32)
    caches = [sensenova_model.FixedKV(
        torch.zeros(1, len(ids) + 2, 2, 8), torch.zeros(1, len(ids) + 2, 2, 8),
        0, position, seqlen,
    ) for _ in backbone.layers]

    def decode_text(token, time, state):
        hidden = backbone.embed_tokens(torch.tensor([[token]]))
        rope = sensenova_model._prepare_mrope(torch.tensor([[time], [0], [0]]), hidden.device, hidden.dtype)
        state[0].prepare(1)
        for layer, cache in zip(backbone.layers, state):
            hidden = layer.forward_decode(hidden, rope, cache, {})
            cache.advance(1)
        return hidden

    cursor = 0
    while cursor < len(ids):
        if cursor in image_events:
            count, image = image_events[cursor]
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            hidden = diffusion.vision_model((image.movedim(-1, 1) - mean) / std)
            indexes = torch.tensor([times[cursor:cursor + count], heights[cursor:cursor + count], widths[cursor:cursor + count]])
            rope = sensenova_model._prepare_mrope(indexes, hidden.device, hidden.dtype)
            for layer, cache in zip(backbone.layers, caches):
                attention = layer.self_attn
                query, key, value = attention._project(layer.input_layernorm(hidden), rope, False)
                cache.key[:, cursor:cursor + count].copy_(key.transpose(1, 2))
                cache.value[:, cursor:cursor + count].copy_(value.transpose(1, 2))
                # Image patches attend the entire new image and the prior cache.
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, cache.key[:, :cursor + count].transpose(1, 2),
                    cache.value[:, :cursor + count].transpose(1, 2), enable_gqa=True,
                ).transpose(1, 2).flatten(2)
                hidden = hidden + attention.o_proj(output)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                cache.advance(count)
            seqlen.add_(count)
            cursor += count
        else:
            hidden = decode_text(ids[cursor], times[cursor], caches)
            cursor += 1
        if cursor not in checkpoints:
            continue

        metadata = {
            "text_input_ids": torch.tensor([ids[:cursor]]),
            "reference_latents": references[:checkpoints[cursor]],
            "sensenova_interleave_expanded": True,
            "sensenova_interleave_start": 2,
        }
        args = prefix_arguments(metadata, torch.device("cpu"), torch.float32)
        if args[2] is not None:
            assert args[2].tolist() == [[times[:cursor], heights[:cursor], widths[:cursor]]]
        keys, values, time = diffusion.preprocess_prefix(*args)
        conds = model.extra_conds(**metadata, device=torch.device("cpu"))
        assert time.item() == times[cursor - 1] + 1
        torch.testing.assert_close(conds["prefix_time"].cond, time)
        for index, cache in enumerate(caches):
            # Batched replay and token-wise decode accumulate FP32 rounding differently across CPU backends.
            torch.testing.assert_close(keys[index], cache.key[:, :cursor].transpose(1, 2), atol=1e-5, rtol=2e-5)
            torch.testing.assert_close(values[index], cache.value[:, :cursor].transpose(1, 2), atol=1e-5, rtol=2e-5)
            torch.testing.assert_close(conds["prefix_keys"].cond[index], keys[index])
            torch.testing.assert_close(conds["prefix_values"].cond[index], values[index])

        # Branch the continuous cache to compare two actual greedy predictions.
        first = diffusion.language_model.lm_head(backbone.norm(hidden))[:, -1].argmax().item()
        branch_position, branch_length = position.clone(), seqlen.clone()
        branch = [sensenova_model.FixedKV(cache.key.clone(), cache.value.clone(), cursor, branch_position, branch_length) for cache in caches]
        next_hidden = decode_text(first, times[cursor - 1] + 1, branch)
        second = diffusion.language_model.lm_head(backbone.norm(next_hidden))[:, -1].argmax().item()
        generated, stop = diffusion.generate_text_tokens(*args, max_tokens=2, stop_token_ids=())
        assert (generated, stop) == ([first, second], None)


def test_sensenova_memory_estimate_uses_regular_prefix_length():
    model = object.__new__(model_base.SenseNovaU15)
    input_ids = torch.empty(1, 10, dtype=torch.long)

    shapes = model.extra_conds_shapes(
        text_input_ids=input_ids,
        prompt_type="positive",
    )

    assert shapes["prefix_mask"] == [1, 1, 10, 10]
    assert shapes["prefix_keys"] == [
        1,
        sensenova_model.NUM_KV_HEADS,
        sensenova_model.NUM_LAYERS
        * 10
        * sensenova_model.HEAD_DIM,
    ]
