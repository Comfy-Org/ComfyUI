from types import SimpleNamespace

import pytest
import torch
import comfy_aimdo.host_buffer
import comfy_kitchen

from comfy.cli_args import args

args.cpu = True

from comfy import model_base, model_detection
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_prefetch
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
    build_interleave_prompt,
    build_interleave_unconditional_prompt,
)
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from comfy_execution.graph_utils import ExecutionBlocker
from comfy_extras.nodes_sensenova import (
    append_interleave_image,
    append_interleave_tokens,
    expand_interleave_metadata,
    generate_text_event,
    SenseNovaGenerate,
    SenseNovaDecode,
    SenseNovaSamplingOptions,
    SenseNovaTextEncode,
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
    state_dict = _minimal_u15_state_dict()

    assert model_detection.unet_prefix_from_state_dict(state_dict) == ""
    assert model_detection.detect_unet_config(state_dict, "") == {
        "image_model": "sensenova_u15",
        "has_lm_head": False,
    }
    assert (
        type(model_detection.model_config_from_unet(state_dict, "")).__name__
        == "SenseNovaU15"
    )


def test_sensenova_checkpoint_detection_preserves_thinking_capability():
    config = model_detection.detect_unet_config(
        _minimal_u15_state_dict(has_lm_head=True), ""
    )

    assert config == {
        "image_model": "sensenova_u15",
        "has_lm_head": True,
    }


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
    assert sampling_inputs["shift"].min == 0.01
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

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, transformer_options=None):
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


def test_sensenova_model_base_preprocesses_regular_prefix_only():
    regular_calls = []

    def preprocess_prefix(*args, **kwargs):
        regular_calls.append(args)
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
    model.diffusion_model = SimpleNamespace(
        dtype=torch.bfloat16,
        preprocess_prefix=preprocess_prefix,
    )
    input_ids = torch.tensor([[1, 2, 3]])

    positive = model.extra_conds(
        text_input_ids=input_ids,
        prompt_type="positive",
        device=torch.device("cpu"),
    )
    model.extra_conds(
        text_input_ids=input_ids,
        prompt_type="negative",
        device=torch.device("cpu"),
    )
    hooked = model.extra_conds(
        text_input_ids=input_ids,
        prompt_type="positive",
        hooks=object(),
        device=torch.device("cpu"),
    )

    assert len(regular_calls) == 2
    assert positive["prefix_time"].cond.tolist() == [3]
    assert hooked["text_input_ids"].cond.tolist() == input_ids.tolist()


@pytest.mark.parametrize("negative", [False, True])
def test_sensenova_empty_reference_batch_preserves_text_conditioning(negative):
    model = _text_model_patcher([42, 151668]).model
    model.concat_keys = ()
    input_ids = torch.tensor([[1, 2, 3]])

    conds = model.extra_conds(
        text_input_ids=input_ids, reference_latents=[torch.empty(0, 32, 64, 3)],
        prompt_type="negative" if negative else "positive",
        device=torch.device("cpu"), hooks=object(),
    )

    assert set(conds) == {"text_input_ids"}
    assert conds["text_input_ids"].cond is input_ids


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


def test_sensenova_model_base_does_not_reinsert_expanded_interleave_images():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, transformer_options=None):
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


def test_sensenova_thinking_returns_stop_without_decoding_suffix():
    model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    tokens = iter((42, 151668))
    decoded = []

    model._preprocess_prefix_state = lambda *args: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden: torch.tensor([next(tokens)])

    def decode(token, caches, hidden, rope, transformer_options=None):
        decoded.append(int(token.item()))

    model._decode_text_token = decode
    progress_updates = []
    interrupt_calls = []

    token_ids = model.generate_thinking_tokens(
        torch.tensor([[1, 2, 3]]),
        max_think_tokens=4,
        progress=progress_updates.append,
        interrupt=lambda: interrupt_calls.append(True),
    )

    assert token_ids == [42, 151668]
    assert decoded == [42]
    assert progress_updates == [1, 2]
    assert interrupt_calls == [True, True]


def test_sensenova_thinking_decode_returns_generated_token_ids():
    model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    tokens = iter((42, 151668))

    model._preprocess_prefix_state = lambda *args: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden: torch.tensor([next(tokens)])

    def decode(token, caches, hidden, rope, transformer_options=None):
        pass

    model._decode_text_token = decode

    token_ids = model.generate_thinking_tokens(
        torch.tensor([[1, 2, 3]]), max_think_tokens=4
    )

    assert token_ids == [42, 151668]


@pytest.mark.parametrize(
    ("tokens", "max_think_tokens", "expected_decoded"),
    [
        ((42, 43), 1, [42, 151668]),
        ((151645,), 4, [151668]),
    ],
)
def test_sensenova_thinking_closes_truncated_reasoning(
    tokens, max_think_tokens, expected_decoded
):
    model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    token_iterator = iter(tokens)
    decoded = []

    model._preprocess_prefix_state = lambda *args: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden: torch.tensor([next(token_iterator)])

    def decode(token, caches, hidden, rope, transformer_options=None):
        decoded.append(int(token.item()))

    model._decode_text_token = decode
    token_ids = model.generate_thinking_tokens(
        torch.tensor([[1, 2, 3]]), max_think_tokens=max_think_tokens
    )

    assert token_ids == expected_decoded
    assert decoded == []


def test_sensenova_uses_prompt_type_for_negative_reference_conditioning():
    calls = []

    def preprocess_prefix(input_ids, references, indexes, prefix_mask, transformer_options=None):
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


def test_sensenova_preprocessed_prefix_matches_raw_forward():
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
    assert timestep_embedder.shapes == [torch.Size([1])] * 2
    assert noise_scale_embedder.shapes == [torch.Size([1])] * 2


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

    assert torch.equal(conditioned, _tokenize_generation_prompt(expected_text))
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
    assert conditioned.shape[1] == conditioned_input_length(
        input_ids.shape[1],
        [(1, 2)],
        image_only=True,
        append_image_start=False,
    )


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


def test_sensenova_generation_prompt_selects_thinking_protocol():
    no_thinking = build_generation_prompt("test")
    thinking = build_generation_prompt("test", thinking=True)

    assert no_thinking.endswith("<think>\n\n</think>\n\n<img>")
    assert thinking.endswith("<think>\n")
    assert thinking.rsplit("<|im_start|>assistant\n", 1)[-1] == "<think>\n"


def test_sensenova_interleave_prompt_leaves_image_event_to_the_model():
    no_thinking = build_interleave_prompt("test")
    thinking = build_interleave_prompt("test", thinking=True)

    assert no_thinking.endswith("<think>\n\n</think>\n\n")
    assert not no_thinking.endswith("<img>")
    assert thinking.endswith("<|im_start|>assistant\n")
    assert build_interleave_unconditional_prompt() == (
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    )


def test_sensenova_interleave_text_encode_selects_interleave_protocol():
    calls = []

    class Clip:
        def tokenize(self, text, **kwargs):
            calls.append(("tokenize", text, kwargs))
            return {"tokens": text}

        def encode_from_tokens_scheduled(self, tokens, add_dict):
            calls.append(("encode", tokens, add_dict))
            return [[torch.empty(1), add_dict]]

    output = SenseNovaTextEncode.execute(
        clip=Clip(),
        text="test",
        thinking=False,
        mode="interleave",
    ).args[0]

    assert calls[0] == (
        "tokenize",
        "test",
        {"thinking": False, "mode": "interleave"},
    )
    assert calls[1][2]["sensenova_interleave"] is True
    assert output[0][1]["sensenova_interleave"] is True

    tokenizer = SenseNovaTokenizer()
    values = tokenizer.tokenize_with_weights(
        "test", mode="interleave"
    )["sensenova_u15"][0]
    assert int(values[-1][0]) != 151670


def test_sensenova_nodes_use_family_capability_names():
    assert SenseNovaTextEncode.define_schema().display_name == "SenseNova Text Encode"
    assert SenseNovaGenerate.define_schema().display_name == "SenseNova Generate"


def test_sensenova_model_base_preserves_patch_size_compatibility_alias():
    assert model_base.SenseNovaU15.PATCH_SIZE == sensenova_model.MERGED_PATCH_SIZE


def test_sensenova_interleave_reference_is_part_of_the_initial_prefix():
    tokenizer = SenseNovaTokenizer()
    pairs = tokenizer.tokenize_with_weights(
        "continue this story", mode="interleave"
    )["sensenova_u15"][0]
    input_ids = torch.tensor([[int(pair[0]) for pair in pairs]])

    conditioned, references, indexes, prefix_mask = prefix_arguments(
        {
            "text_input_ids": input_ids,
            "reference_latents": [torch.ones(1, 33, 65, 3)],
        },
        torch.device("cpu"),
        torch.float32,
        image_only=False,
    )

    patch_size = sensenova_model.MERGED_PATCH_SIZE
    expected_image_tokens = ((33 + patch_size - 1) // patch_size) * (
        (65 + patch_size - 1) // patch_size
    )
    assert torch.count_nonzero(conditioned == 151669) == expected_image_tokens
    assert conditioned[0, -1] == input_ids[0, -1]
    assert len(references) == 1
    assert references[0].shape == (1, 3, 33, 65)
    assert indexes.shape == (1, 3, conditioned.shape[1])
    assert prefix_mask.shape == (1, 1, conditioned.shape[1], conditioned.shape[1])


def test_sensenova_text_encode_leaves_image_thinking_to_generate():
    calls = []

    class Clip:
        def tokenize(self, text, **kwargs):
            calls.append(("tokenize", text, kwargs))
            return {"tokens": text}

        def encode_from_tokens_scheduled(self, tokens, add_dict):
            calls.append(("encode", tokens, add_dict))
            return [[torch.empty(1), add_dict]]

    output = SenseNovaTextEncode.execute(
        clip=Clip(),
        text="test",
        thinking=True,
    ).args[0]

    assert calls[0] == ("tokenize", "test", {"thinking": True})
    assert calls[1][2] == {"sensenova_thinking": True}
    assert output[0][1] == {"sensenova_thinking": True}


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("max_tokens,compute_dtype,text_bytes,reference_bytes", [
    (32, torch.float32, 16_171_008, 19_267_584),
    (8192, torch.float32, 2_823_733_248, 2_826_829_824),
    (32, torch.bfloat16, 8_085_504, 9_633_792),
    (8192, torch.bfloat16, 1_411_866_624, 1_413_414_912),
])
def test_sensenova_generate_returns_prefix_conditioning_and_text(
    monkeypatch, with_reference, storage_dtype, max_tokens, compute_dtype, text_bytes, reference_bytes,
):
    calls = []
    memory_budgets = []

    class DiffusionModel:
        dtype = storage_dtype

        def generate_thinking_tokens(
            self,
            input_ids,
            references,
            indexes,
            prefix_mask,
            **kwargs,
        ):
            calls.append((input_ids, references, indexes, prefix_mask, kwargs))
            return [41, 42, 151668]

    class Model:
        load_device = torch.device("cpu")
        model_options = {}
        model = object.__new__(model_base.SenseNovaU15)
        torch.nn.Module.__init__(model)
        model.diffusion_model = DiffusionModel()
        model.manual_cast_dtype = compute_dtype

        def is_dynamic(self):
            return False

        def pre_run(self):
            calls.append("pre_run")

        def cleanup(self):
            calls.append("cleanup")

    class Clip:
        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [41, 42, 151668]
            return "  plan the composition</think>"

    def load_model(models, memory_required=0):
        calls.append("load")
        memory_budgets.append(memory_required)

    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        load_model,
    )
    conditioning = [[
        torch.empty(1),
        {
            "text_input_ids": _generation_input_ids()[:, :-1],
            "sensenova_thinking": True,
            "reference_latents": [torch.ones(1, 33, 65, 3)] if with_reference else [],
        },
    ]]
    output = SenseNovaGenerate.execute(
        model=Model(),
        clip=Clip(),
        conditioning=conditioning,
        max_think_tokens=max_tokens,
    )

    updated, thinking, negative, image_requested = output.args
    # 15 text positions, plus 9 for the 2x3 reference grid, followed by decode capacity.
    assert memory_budgets == [reference_bytes if with_reference else text_bytes]
    assert thinking == "plan the composition"
    assert negative == []
    assert image_requested is True
    metadata = updated[0][1]
    assert metadata["text_input_ids"][0, -5:].tolist() == [41, 42, 151668, 271, 151670]
    assert torch.equal(metadata["text_input_ids"][:, :-5], conditioning[0][1]["text_input_ids"])
    assert not {"prefix_keys", "prefix_values", "prefix_time"} & metadata.keys()
    assert metadata["reference_latents"] is conditioning[0][1]["reference_latents"]
    assert metadata["sensenova_thinking"] is False
    assert conditioning[0][1]["text_input_ids"].shape[1] == 15
    assert conditioning[0][1]["sensenova_thinking"] is True
    prefix_call = next(item for item in calls if isinstance(item, tuple))
    if with_reference:
        assert prefix_call[1][0].shape == (1, 3, 33, 65)
        assert prefix_call[1][0].dtype == compute_dtype
        assert prefix_call[2].shape[0:2] == (1, 3)
        assert prefix_call[3].shape[-1] == prefix_call[2].shape[-1]
        assert prefix_call[3].dtype == compute_dtype
    assert calls[0] == "load"
    assert calls[1] == "pre_run"
    assert calls[-1] == "cleanup"

    sampler_prefixes = []

    def sampler_prefill(input_ids, references, indexes, prefix_mask, transformer_options=None):
        sampler_prefixes.append((input_ids, references))
        return [torch.full((1, 1, 1, 1), 17.0)], [], torch.tensor([input_ids.shape[1]])

    sampler_model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(sampler_model)
    sampler_model.concat_keys = ()
    sampler_model.current_patcher = None
    sampler_model.manual_cast_dtype = None
    sampler_model.diffusion_model = SimpleNamespace(dtype=torch.float32, preprocess_prefix=sampler_prefill)
    sampled = sampler_model.extra_conds(**metadata, device=torch.device("cpu"))

    assert sampled["prefix_keys"].cond[0].item() == 17.0
    assert sampler_prefixes[0][0][0, -5:].tolist() == [41, 42, 151668, 271, 151670]
    assert (sampler_prefixes[0][0] == 151669).sum().item() == (6 if with_reference else 0)
    if with_reference:
        assert sampler_prefixes[0][1][0].shape == (1, 3, 33, 65)


@pytest.mark.parametrize("interleave,token_ids,expected", [
    (False, [42, 151668], "K"),
    (False, [151668], ""),
    (True, [151667, 42, 151668, 43, 151670], "KL"),
    (True, [42, 151668, 43, 151645], "KL"),
    (True, [151645], ""),
])
def test_sensenova_generate_returns_current_text_without_think_wrappers(
    monkeypatch, interleave, token_ids, expected,
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_thinking": True,
        "sensenova_interleave": interleave,
    }]]
    output = SenseNovaGenerate.execute(
        model=_text_model_patcher(token_ids), clip=SenseNovaQwenTokenizer(),
        conditioning=positive,
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]],
        max_think_tokens=8,
    )

    assert output.args[1] == expected
    assert output.ui.as_dict() == {"text": (expected,)}
    history = [token for token in token_ids if token != 151645]
    if not interleave:
        history.extend([271, 151670])
    assert output.args[0][0][1]["text_input_ids"].tolist() == [[1, *history]]
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]
    assert SenseNovaGenerate.define_schema().outputs[1].display_name == "generated_text"


def test_sensenova_generate_passthrough_without_thinking(monkeypatch):
    def fail_load(_models):
        raise AssertionError("thinking=false must not load the model")

    monkeypatch.setattr(comfy.model_management, "load_models_gpu", fail_load)
    conditioning = [[torch.empty(1), {"sensenova_thinking": False}]]

    output = SenseNovaGenerate.execute(
        model=object(),
        clip=object(),
        conditioning=conditioning,
        max_think_tokens=32,
    )

    assert output.args[0] is conditioning
    assert output.args[1] == ""
    assert output.args[2] == []
    assert output.args[3] is True
    assert output.ui.as_dict() == {"text": ("",)}


def test_sensenova_interleave_history_expands_reference_images_once():
    metadata = {
        "text_input_ids": torch.tensor([[1, 2, 3]]),
        "reference_latents": [torch.zeros(1, 33, 65, 3)],
    }

    expanded = expand_interleave_metadata(metadata, image_only=False)

    assert expanded["sensenova_interleave_expanded"] is True
    assert expanded["text_input_ids"].shape[1] == 3 + 6 + 3
    assert torch.count_nonzero(expanded["text_input_ids"] == 151669) == 6


def _text_model_patcher(token_ids):
    """A small deterministic language model using the real token decode path."""
    width = len(token_ids)
    embeddings = torch.zeros(151936, width)
    head = torch.zeros_like(embeddings)
    for index, (previous, following) in enumerate(zip([1, *token_ids], token_ids)):
        embeddings[previous, index] = 1
        head[following, index] = 1
    backbone = torch.nn.Module()
    backbone.embed_tokens = torch.nn.Embedding.from_pretrained(embeddings)
    backbone.layers = torch.nn.ModuleList()
    backbone.norm = torch.nn.Identity()
    language_model = torch.nn.Module()
    language_model.model = backbone
    language_model.lm_head = torch.nn.Linear(width, 151936, bias=False)
    language_model.lm_head.weight = torch.nn.Parameter(head, requires_grad=False)
    diffusion_model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(diffusion_model)
    diffusion_model.has_lm_head = True
    diffusion_model.dtype = torch.float32
    diffusion_model.language_model = language_model
    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.diffusion_model = diffusion_model
    model.manual_cast_dtype = None
    model.current_patcher = None
    return comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )


def test_sensenova_interleave_token_limit_can_resume_to_image(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 43, 151670])
    clip = SenseNovaQwenTokenizer()
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]
    first = SenseNovaGenerate.execute(
        model=model, clip=clip, conditioning=positive, negative=negative, max_think_tokens=1,
    )
    assert first.args[3] is False
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[1, 42]]

    second = SenseNovaGenerate.execute(
        model=model, clip=clip, conditioning=first.args[0], negative=first.args[2], max_think_tokens=8,
    )
    assert second.args[3] is True
    assert second.args[0][0][1]["text_input_ids"].tolist() == [[1, 42, 43, 151670]]
    assert second.args[2][0][1]["text_input_ids"].tolist() == [[9, 151670]]
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[1, 42]]


def test_sensenova_interleave_reuses_image_negative_start_token(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    negative_ids = _tokenize_generation_prompt("")
    negative = [[torch.empty(1), {"text_input_ids": negative_ids}]]
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]

    output = SenseNovaGenerate.execute(
        model=_text_model_patcher([151670]), clip=SenseNovaQwenTokenizer(),
        conditioning=positive, negative=negative, max_think_tokens=8,
    )

    assert output.args[3] is True
    assert output.args[2][0][1]["text_input_ids"].tolist() == [
        [151644, 872, 198, 151645, 198, 151644, 77091, 198, 151670]
    ]
    assert torch.equal(negative[0][1]["text_input_ids"], negative_ids)
    assert "sensenova_interleave_pending_image" not in negative[0][1]


@pytest.mark.parametrize("negative_mode", ["image", "interleave"])
@pytest.mark.parametrize("negative_text", ["", "blurry"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_interleave_negative_modes_preserve_generation_and_decode(
    monkeypatch, negative_mode, negative_text, with_reference
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    values = SenseNovaTokenizer().tokenize_with_weights(
        negative_text, mode=negative_mode
    )["sensenova_u15"][0]
    input_ids = [int(value[0]) for value in values]
    metadata = {"text_input_ids": torch.tensor([input_ids])}
    if negative_mode == "interleave":
        metadata["sensenova_interleave"] = True
    if with_reference:
        metadata["reference_latents"] = [torch.ones(1, 33, 65, 3)]
    negative = [[torch.empty(1), metadata]]
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]
    model = _text_model_patcher([42, 151670])
    clip = SenseNovaQwenTokenizer()

    first = SenseNovaGenerate.execute(
        model=model, clip=clip, conditioning=positive, negative=negative,
        max_think_tokens=1,
    )
    assert first.args[3] is False
    second = SenseNovaGenerate.execute(
        model=model, clip=clip, conditioning=first.args[0], negative=first.args[2],
        max_think_tokens=8,
    )

    if with_reference:
        expected = [
            151644, 872, 198, 151670,
            151669, 151669, 151669, 151669, 151669, 151669,
            151671, 151645, 198, 151644, 77091, 198, 151670,
        ]
    elif negative_mode == "image":
        expected = input_ids
    else:
        expected = [*input_ids, 151670]
    assert second.args[3] is True
    assert second.args[2][0][1]["text_input_ids"].tolist() == [expected]
    assert metadata["text_input_ids"].tolist() == [input_ids]
    assert "sensenova_interleave_pending_image" not in metadata

    image = torch.full((1, 32, 32, 3), 0.5)
    text, images, thinking = SenseNovaDecode.execute(
        clip=clip, conditioning=second.args[0], samples={"samples": torch.zeros(1, 3, 32, 32)},
    ).args
    assert text == "K<image>"
    assert thinking == ""
    assert len(images) == 1
    assert torch.equal(images[0], image)


@pytest.mark.parametrize("reference_branch", ["positive", "negative", "both"])
def test_sensenova_generate_interleave_ignores_empty_reference_batches(monkeypatch, reference_branch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]
    for branch, conditioning in (("positive", positive), ("negative", negative)):
        if reference_branch in (branch, "both"):
            conditioning[0][1]["reference_latents"] = [torch.empty(0, 33, 65, 3)]

    output = SenseNovaGenerate.execute(
        model=_text_model_patcher([42, 151670]), clip=SenseNovaQwenTokenizer(),
        conditioning=positive, negative=negative, max_think_tokens=8,
    )

    assert output.args[3] is True
    assert output.args[1] == "K"
    assert output.args[0][0][1]["text_input_ids"].tolist() == [[1, 42, 151670]]
    assert output.args[2][0][1]["text_input_ids"].tolist() == [[9, 151670]]
    for conditioning in (output.args[0], output.args[2]):
        assert conditioning[0][1]["sensenova_interleave_input_images"] == 0
        assert conditioning[0][1]["reference_latents"] == []
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]
    assert negative[0][1]["text_input_ids"].tolist() == [[9]]


@pytest.mark.parametrize("interleave", [False, True])
def test_sensenova_generate_initialization_failure_cleans_up_and_can_run_again(monkeypatch, interleave):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 151645 if interleave else 151668])
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_thinking": True,
        "sensenova_interleave": interleave,
    }]]
    kwargs = dict(
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive,
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]],
        max_think_tokens=8,
    )

    def fail_initialization(patcher):
        assert patcher.model.current_patcher is patcher
        raise RuntimeError("initialization failed")

    model.add_callback_with_key("on_pre_run", "test_failure", fail_initialization)
    with pytest.raises(RuntimeError, match="initialization failed"):
        SenseNovaGenerate.execute(**kwargs)

    assert model.model.current_patcher is None
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]
    model.remove_callbacks_with_key("on_pre_run", "test_failure")
    assert SenseNovaGenerate.execute(**kwargs).args[1] == "K"
    assert model.model.current_patcher is None


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_sensenova_generate_cancel_cleans_up_and_can_run_again(monkeypatch, interleave, dynamic):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 43, 151645 if interleave else 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_thinking": True,
        "sensenova_interleave": interleave,
    }]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]]
    kwargs = dict(
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive,
        negative=negative, max_think_tokens=8,
    )

    def cancel_on_progress(value, total, preview, node_id=None):
        assert model.model.current_patcher is model
        assert bool(comfy.model_prefetch.PREFETCH_QUEUES) == dynamic
        comfy.model_management.interrupt_current_processing()

    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", cancel_on_progress)
    try:
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            SenseNovaGenerate.execute(**kwargs)
    finally:
        comfy.model_management.interrupt_current_processing(False)
    assert comfy.model_prefetch.PREFETCH_QUEUES == []
    assert model.model.current_patcher is None
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]

    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    completed = SenseNovaGenerate.execute(**kwargs)
    expected = [[1, 42, 43]] if interleave else [[1, 42, 43, 151668, 271, 151670]]
    assert completed.args[0][0][1]["text_input_ids"].tolist() == expected
    assert model.model.current_patcher is None
    assert comfy.model_prefetch.PREFETCH_QUEUES == []


@pytest.mark.parametrize("dynamic", [False, True])
def test_sensenova_generate_uses_native_prefetch_for_dynamic_models(monkeypatch, dynamic):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 43, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    options = {"prefetch_dynamic_vbars": not dynamic}
    model.model_options["transformer_options"] = options
    active_prefetch = []
    monkeypatch.setattr(
        comfy.utils, "PROGRESS_BAR_HOOK",
        lambda *args, **kwargs: active_prefetch.append(bool(comfy.model_prefetch.PREFETCH_QUEUES)),
    )
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]), "sensenova_thinking": True,
    }]]
    try:
        output = SenseNovaGenerate.execute(
            model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
            max_think_tokens=8,
        )
        assert output.args[0][0][1]["text_input_ids"].tolist() == [[1, 42, 43, 151668, 271, 151670]]
        assert output.args[1] == "KL"
        assert active_prefetch and all(active == dynamic for active in active_prefetch)
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model_options["transformer_options"] == {"prefetch_dynamic_vbars": not dynamic}
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()


@pytest.mark.parametrize("dynamic", [False, True])
def test_sensenova_sampler_prefix_uses_current_patcher_prefetch(monkeypatch, dynamic):
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    model.model.concat_keys = ()
    model.pre_run()
    try:
        conds = model.model.extra_conds(
            text_input_ids=torch.tensor([[1]]), device=model.load_device,
        )
        assert conds["prefix_time"].cond.tolist() == [1]
        assert len(conds["prefix_keys"].cond) == 2
        for key, value in zip(conds["prefix_keys"].cond, conds["prefix_values"].cond):
            assert key.shape == value.shape == (1, 1, 1, 8)
            assert torch.count_nonzero(key) == torch.count_nonzero(value) == 0
        assert bool(comfy.model_prefetch.PREFETCH_QUEUES) == dynamic
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
        model.cleanup()


def _add_identity_decoder_layers(monkeypatch, model):
    backbone = model.model.diffusion_model.language_model.model
    for name, value in {
        "HIDDEN_SIZE": backbone.embed_tokens.embedding_dim,
        "INTERMEDIATE_SIZE": 4, "NUM_HEADS": 1, "NUM_KV_HEADS": 1, "HEAD_DIM": 8,
    }.items():
        monkeypatch.setattr(sensenova_model, name, value)
    backbone.layers = torch.nn.ModuleList([
        sensenova_model.DecoderLayer(operations=torch.nn) for _ in range(2)
    ])
    for parameter in backbone.layers.parameters():
        torch.nn.init.zeros_(parameter)


def test_sensenova_generate_layer_failure_releases_prefetch_and_can_retry(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    model = comfy.model_patcher.ModelPatcherDynamic(
        model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
    )
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]), "sensenova_thinking": True,
    }]]
    kwargs = dict(model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning, max_think_tokens=8)

    def fail_layer(module, inputs):
        assert comfy.model_prefetch.PREFETCH_QUEUES
        raise RuntimeError("layer failed")

    handle = model.model.diffusion_model.language_model.model.layers[0].mlp.register_forward_pre_hook(fail_layer)
    try:
        with pytest.raises(RuntimeError, match="layer failed"):
            SenseNovaGenerate.execute(**kwargs)
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model.current_patcher is None
        assert conditioning[0][1]["text_input_ids"].tolist() == [[1]]
    finally:
        handle.remove()
        comfy.model_prefetch.cleanup_prefetch_queues()
    assert SenseNovaGenerate.execute(**kwargs).args[1] == "K"
    assert comfy.model_prefetch.PREFETCH_QUEUES == []


def test_sensenova_generate_releases_native_prefetch_on_initialization_failure(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    model = _text_model_patcher([42, 151668])
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]), "sensenova_thinking": True,
    }]]

    def fail_initialization(patcher):
        queue = comfy.model_prefetch.make_prefetch_queue(
            [], patcher.load_device, {"prefetch_dynamic_vbars": True},
        )
        assert queue is not None
        raise RuntimeError("initialization failed after prefetch")

    model.add_callback_with_key("on_pre_run", "prefetch_failure", fail_initialization)
    try:
        with pytest.raises(RuntimeError, match="initialization failed after prefetch"):
            SenseNovaGenerate.execute(
                model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
                max_think_tokens=8,
            )
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model.current_patcher is None
        model.remove_callbacks_with_key("on_pre_run", "prefetch_failure")
        assert SenseNovaGenerate.execute(
            model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
            max_think_tokens=8,
        ).args[1] == "K"
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()


@pytest.mark.parametrize(
    "storage_dtype,compute_dtype",
    [(torch.float32, torch.float32), (torch.float8_e4m3fn, torch.bfloat16)],
)
@pytest.mark.parametrize("configuration", ["module", "global"])
def test_sensenova_decode_attention_uses_configured_backend(monkeypatch, storage_dtype, compute_dtype, configuration):
    monkeypatch.setattr(comfy_kitchen, "flash_attention_decode_is_available", lambda device: True)
    monkeypatch.setattr(comfy_kitchen.registry, "is_available", lambda backend: True)
    attention = sensenova_model.Attention(
        device=torch.device("cpu"), dtype=storage_dtype, operations=comfy.ops.manual_cast,
    )
    for parameter in attention.parameters():
        parameter.detach().zero_()
    attention.o_proj.weight.detach().copy_(torch.eye(4096))
    visible_lengths = iter((3, 4))

    def configured_backend(query, key, value, heads, **kwargs):
        assert isinstance(query, sensenova_model.AttentionTensorContainer)
        assert isinstance(key, sensenova_model.AttentionTensorContainer)
        assert isinstance(value, sensenova_model.AttentionTensorContainer)
        assert query.take().shape == (1, 32, 1, 128)
        length = next(visible_lengths)
        assert key.take().shape == (1, 8, 5, 128)
        values = value.take()
        assert values.dtype == compute_dtype
        assert values.shape == (1, 8, 5, 128)
        assert torch.isfinite(values).all()
        assert kwargs["mask"][0, 0, 0].tolist() == [0.0] * length + [float("-inf")] * (5 - length)
        assert heads == 32
        assert kwargs["enable_gqa"] is True
        return values.new_ones((1, 1, 4096))

    if configuration == "module":
        attention.comfy_attention.function = configured_backend
    else:
        monkeypatch.setattr(sensenova_model, "optimized_attention", configured_backend)
    hidden = torch.zeros(1, 1, 4096, dtype=compute_dtype)
    rope = tuple(
        (torch.ones(1, 1, 1, dim, dtype=compute_dtype), torch.zeros(1, 1, 1, dim, dtype=compute_dtype))
        for dim in (64, 32, 32)
    )
    prefix = torch.ones(1, 8, 2, 128, dtype=compute_dtype)
    cache = sensenova_model.FixedKV(
        torch.zeros(1, 5, 8, 128, dtype=compute_dtype), torch.zeros(1, 5, 8, 128, dtype=compute_dtype), 2,
        torch.empty(1, dtype=torch.int64), torch.tensor([2], dtype=torch.int32),
    )
    cache.key[:, :2].copy_(prefix.transpose(1, 2))
    cache.value[:, :2].copy_(prefix.transpose(1, 2))
    for _ in (2, 3):
        cache.prepare(1)
        output = attention.forward_decode(hidden, rope, cache, {})
        cache.advance(1)

        assert output.dtype == compute_dtype
        assert torch.equal(output, torch.ones_like(hidden))
    assert cache.key.shape == cache.value.shape == (1, 5, 8, 128)
    assert cache.key.dtype == cache.value.dtype == compute_dtype
    assert torch.equal(cache.key[:, :2], prefix.transpose(1, 2))
    assert torch.equal(cache.value[:, :2], prefix.transpose(1, 2))
    assert torch.count_nonzero(cache.key[:, 2:]) == 0
    assert torch.count_nonzero(cache.value[:, 2:]) == 0


def test_sensenova_interleave_text_event_stops_before_image_sampling():
    class Model:
        has_lm_head = True
        generate_text_tokens = sensenova_model.SenseNovaU15.generate_text_tokens

        def __init__(self):
            self.tokens = iter((101, 151670))

        def _preprocess_prefix_state(self, *args):
            return torch.zeros(1, 1, 1), [torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1, 1, 1)], torch.tensor([3])

        def _next_text_token(self, hidden):
            return torch.tensor([next(self.tokens)])

        def _decode_text_token(
            self, token, caches, hidden, rope, transformer_options=None
        ):
            pass

    event = generate_text_event(
        Model(),
        (torch.tensor([[1]]), None, None, None),
        max_text_tokens=8,
    )

    assert event.token_ids == [101, 151670]
    assert event.image_requested is True
    assert event.stop_reason == "image"


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float8_e4m3fn])
def test_sensenova_interleave_generate_can_resume_after_ksampler(monkeypatch, storage_dtype):
    class DiffusionModel:
        dtype = storage_dtype
        has_lm_head = True
        generate_text_tokens = sensenova_model.SenseNovaU15.generate_text_tokens

        def __init__(self):
            self.stages = [iter((101, 151670)), iter((102, 151645))]
            self.tokens = None

        def _preprocess_prefix_state(self, *args):
            if args[1]:
                assert args[1][0].dtype == torch.float32
                assert args[3].dtype == torch.float32
            self.tokens = self.stages.pop(0)
            return torch.zeros(1, 1, 1), [torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1, 1, 1)], torch.tensor([3])

        def _next_text_token(self, hidden):
            return torch.tensor([next(self.tokens)])

        def _decode_text_token(
            self, token, caches, hidden, rope, transformer_options=None
        ):
            pass

    diffusion_model = DiffusionModel()
    calls = []
    memory_budgets = []

    class Model:
        load_device = torch.device("cpu")
        model_options = {}
        model = object.__new__(model_base.SenseNovaU15)
        torch.nn.Module.__init__(model)
        model.diffusion_model = diffusion_model
        model.manual_cast_dtype = torch.float32

        def is_dynamic(self):
            return False

        def pre_run(self):
            calls.append("pre_run")

        def cleanup(self):
            calls.append("cleanup")

    class Clip:
        def decode(self, token_ids, skip_special_tokens=True):
            return {101: "before", 102: "after"}[token_ids[0]]

    def load_model(models, memory_required=0):
        calls.append("load")
        memory_budgets.append(memory_required)

    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        load_model,
    )
    monkeypatch.setattr(
        comfy.model_management,
        "throw_exception_if_processing_interrupted",
        lambda: None,
    )
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[7, 8]]), "sensenova_interleave": True}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]

    first = SenseNovaGenerate.execute(
        model=Model(),
        clip=Clip(),
        conditioning=positive,
        negative=negative,
        max_think_tokens=8,
    )
    first_positive, first_text, first_negative, requested = first.args

    assert first_text == "before"
    assert requested is True
    assert first_positive[0][1]["text_input_ids"].tolist() == [[7, 8, 101, 151670]]
    assert first_negative[0][1]["text_input_ids"].tolist() == [[9, 151670]]
    assert first_positive[0][1]["sensenova_interleave_pending_image"] is True

    second = SenseNovaGenerate.execute(
        model=Model(),
        clip=Clip(),
        conditioning=first_positive,
        negative=first_negative,
        samples={"samples": torch.tensor([-2.0, 2.0]).repeat(1, 3, 32, 32)},
        max_think_tokens=8,
    )
    second_positive, second_text, second_negative, requested = second.args
    positive_ids = second_positive[0][1]["text_input_ids"]
    negative_ids = second_negative[0][1]["text_input_ids"]

    assert second_text == "after"
    assert requested is False
    assert positive_ids[0, -4:].tolist() == [151669, 151669, 151671, 102]
    assert negative_ids[0, -3:].tolist() == [151669, 151669, 151671]
    assert len(second_positive[0][1]["reference_latents"]) == 1
    assert second_positive[0][1]["reference_latents"][0].shape == (1, 32, 64, 3)
    assert second_positive[0][1]["reference_latents"][0][0, 0, :2, 0].tolist() == [-0.5, 1.5]
    assert second_positive[0][1]["sensenova_interleave_pending_image"] is False
    # The second prefix already contains the sampled image's two patch tokens.
    assert memory_budgets == [3_440_640, 5_160_960]
    assert calls == ["load", "pre_run", "cleanup", "load", "pre_run", "cleanup"]

    for _ in range(3):
        finished = SenseNovaGenerate.execute(
            model=object(),
            clip=object(),
            conditioning=second_positive,
            negative=second_negative,
            max_think_tokens=8,
        )
        assert finished.args[0] is second_positive
        assert finished.args[1] == ""
        assert finished.args[2] is second_negative
        assert finished.args[3] is False


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


def test_sensenova_decode_excludes_prompt_and_input_images():
    tokenizer = SenseNovaQwenTokenizer()
    metadata = expand_interleave_metadata(
        {
            "text_input_ids": _generation_input_ids()[:, :-1],
            "reference_latents": [torch.ones(1, 32, 64, 3)],
            "sensenova_interleave": True,
        },
        image_only=False,
    )
    before = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        "<think>plan<image1><img>", disable_weights=True
    )[0]]
    after = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        "inspect</think>answer <image1>", disable_weights=True
    )[0]]
    metadata = append_interleave_tokens(metadata, before, True)
    metadata = append_interleave_image(metadata, torch.zeros(1, 3, 32, 64))
    metadata = append_interleave_tokens(metadata, after, False)
    conditioning = [[torch.empty(1), metadata]]

    text, images, thinking = SenseNovaDecode.execute(
        clip=tokenizer, conditioning=conditioning,
    ).args

    assert text == "answer <image1>"
    assert thinking == "plan<image1>\n\ninspect"
    assert len(images) == 1
    assert torch.equal(images[0], torch.full((1, 32, 64, 3), 0.5))
    assert len(metadata["reference_latents"]) == 2


def test_sensenova_decode_includes_pending_image_once_without_mutating_history():
    tokenizer = SenseNovaQwenTokenizer()
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]},
        image_only=False,
    )
    before = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        "before<img>", disable_weights=True
    )[0]]
    between = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        "between<img>", disable_weights=True
    )[0]]
    metadata = append_interleave_tokens(metadata, before, True)
    metadata = append_interleave_image(metadata, torch.zeros(1, 3, 32, 64))
    metadata = append_interleave_tokens(metadata, between, True)
    conditioning = [[torch.empty(1), metadata]]
    final_image = torch.ones(1, 64, 32, 3)

    for _ in range(2):
        text, images, thinking = SenseNovaDecode.execute(
            clip=tokenizer, conditioning=conditioning, samples={"samples": torch.ones(1, 3, 64, 32)},
        ).args
        assert text == "before<image>between<image>"
        assert thinking == ""
        assert len(images) == 2
        assert torch.equal(images[0], torch.full((1, 32, 64, 3), 0.5))
        assert torch.equal(images[1], final_image)

    assert len(metadata["reference_latents"]) == 1
    assert metadata["sensenova_interleave_pending_image"] is True


def test_sensenova_decode_accepts_pending_latent_without_mutating_history():
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]}, image_only=False,
    )
    metadata = append_interleave_tokens(metadata, [151670], True)
    metadata = append_interleave_image(metadata, torch.zeros(1, 3, 32, 64))
    metadata = append_interleave_tokens(metadata, [151670], True)
    conditioning = [[torch.empty(1), metadata]]
    samples = {"samples": torch.tensor([[[[-2.0, 0.0, 2.0]]]]).expand(1, 3, 1, 3).clone()}

    for _ in range(2):
        text, images, thinking = SenseNovaDecode.execute(
            clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
            samples=samples,
        ).args
        assert text == "<image><image>"
        assert thinking == ""
        assert len(images) == 2
        assert torch.equal(images[0], torch.full((1, 32, 64, 3), 0.5))
        assert images[1].shape == (1, 1, 3, 3)
        assert images[1][0, 0, :, 0].tolist() == [0.0, 0.5, 1.0]

    assert samples["samples"][0, 0, 0].tolist() == [-2.0, 0.0, 2.0]
    assert len(metadata["reference_latents"]) == 1
    assert metadata["sensenova_interleave_pending_image"] is True


@pytest.mark.parametrize(
    ("generated", "expected_text", "expected_thinking"),
    [
        ("answer", "answer", ""),
        ("<think>plan</think>answer", "answer", "plan"),
        ("<think>unfinished", "", "unfinished"),
        ("plan</think>answer", "answer", "plan"),
        ("<think>plan<img>inspect</think>answer", "answer", "plan\n\ninspect"),
        ("<think>plan</think>before<img>after <image1>", "before\n\n<image>\n\nafter <image1>", "plan"),
        ("", "", ""),
    ],
)
def test_sensenova_decode_text_only_and_ignores_stale_image(generated, expected_text, expected_thinking):
    tokenizer = SenseNovaQwenTokenizer()
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]},
        image_only=False,
    )
    token_ids = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        generated, disable_weights=True
    )[0]]
    metadata = append_interleave_tokens(metadata, token_ids, False)

    text, images, thinking = SenseNovaDecode.execute(
        clip=tokenizer, conditioning=[[torch.empty(1), metadata]],
        samples={"samples": torch.ones(1, 3, 32, 32)},
    ).args

    assert text == expected_text
    assert thinking == expected_thinking
    assert len(images) == 1
    assert isinstance(images[0], ExecutionBlocker)
    assert images[0].message is None


def test_sensenova_decode_has_typed_inputs_and_image_list_output():
    schema = SenseNovaDecode.GET_SCHEMA()
    assert [value.id for value in schema.inputs] == ["clip", "conditioning", "samples"]
    assert [value.display_name for value in schema.outputs] == ["text", "images", "thinking"]
    assert schema.outputs[1].is_output_list is True
    assert not schema.is_input_list


def test_sensenova_decode_clamps_images_without_changing_reference_history():
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]},
        image_only=False,
    )
    metadata = append_interleave_tokens(metadata, [151670], True)
    metadata = append_interleave_image(metadata, torch.tensor([[[[-2.0, 2.0]]]]).expand(1, 3, 1, 2))

    _, images, _ = SenseNovaDecode.execute(
        clip=SenseNovaQwenTokenizer(), conditioning=[[torch.empty(1), metadata]],
    ).args

    assert images[0][0, 0, :, 0].tolist() == [0.0, 1.0]
    assert metadata["reference_latents"][0][0, 0, :, 0].tolist() == [-0.5, 1.5]
