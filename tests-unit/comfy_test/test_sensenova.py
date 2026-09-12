from pathlib import Path
from types import SimpleNamespace

import pytest
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
from comfy.ldm.sensenova.interleave import (
    InterleaveResult,
    SenseNovaInterleaveSession,
    build_interleave_result,
    interleave_result_to_markdown,
    prefix_arguments,
)
from comfy.ldm.sensenova.model import _match_prefix_batch, _pad_to_merged_patch_size
from comfy.ldm.sensenova.sampling import (
    SenseNovaModelSampling,
    resolution_noise_scale,
    upstream_sigmas,
)
from comfy.text_encoders.sensenova import (
    SenseNovaTokenizer,
    build_generation_prompt,
    build_interleave_prompt,
    build_interleave_unconditional_prompt,
)
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
import comfy_extras.nodes_sensenova as sensenova_nodes
from comfy_extras.nodes_sensenova import (
    SenseNovaInterleave,
    SenseNovaInterleavePreview,
    SenseNovaSamplingOptions,
    SenseNovaTextEncode,
    SenseNovaThinkingPreview,
    interleave_output_samples,
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


def test_sensenova_model_base_accepts_live_interleave_prefix():
    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(dtype=torch.bfloat16)
    keys = [torch.zeros(1, 1, 3, 1, dtype=torch.bfloat16)]
    values = [torch.ones(1, 1, 3, 1, dtype=torch.bfloat16)]
    time = torch.tensor([3])

    conds = model.extra_conds(
        prefix_keys=keys,
        prefix_values=values,
        prefix_time=time,
        device=torch.device("cpu"),
    )

    assert conds["prefix_keys"].cond is keys
    assert conds["prefix_values"].cond is values
    assert conds["prefix_time"].cond is time


def test_sensenova_model_base_preprocesses_thinking_only_for_positive():
    thinking_calls = []
    regular_calls = []

    def preprocess_thinking_prefix_with_tokens(
        *args, max_think_tokens, progress=None, interrupt=None
    ):
        thinking_calls.append((args, max_think_tokens, progress, interrupt))
        return (
            [torch.zeros(1, 1, 5, 1, dtype=torch.bfloat16)],
            [torch.ones(1, 1, 5, 1, dtype=torch.bfloat16)],
            torch.tensor([5]),
            [41, 42],
        )

    def preprocess_prefix(*args):
        regular_calls.append(args)
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
        preprocess_thinking_prefix_with_tokens=preprocess_thinking_prefix_with_tokens,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    thinking_result = {"enabled": True, "token_ids": None}

    positive = model.extra_conds(
        text_input_ids=input_ids,
        sensenova_thinking=True,
        sensenova_max_think_tokens=17,
        sensenova_thinking_result=thinking_result,
        prompt_type="positive",
        device=torch.device("cpu"),
    )
    model.extra_conds(
        text_input_ids=input_ids,
        sensenova_thinking=True,
        prompt_type="negative",
        device=torch.device("cpu"),
    )
    hooked = model.extra_conds(
        text_input_ids=input_ids,
        sensenova_thinking=True,
        sensenova_max_think_tokens=23,
        sensenova_thinking_result=thinking_result,
        prompt_type="positive",
        hooks=object(),
        device=torch.device("cpu"),
    )

    assert len(thinking_calls) == 1
    assert thinking_calls[0][1] == 17
    assert callable(thinking_calls[0][2])
    assert callable(thinking_calls[0][3])
    assert thinking_result["token_ids"] == [41, 42]
    assert len(regular_calls) == 1
    assert positive["prefix_time"].cond.tolist() == [5]
    assert hooked["text_input_ids"].cond.tolist() == input_ids.tolist()
    assert hooked["sensenova_thinking"].cond is True
    assert hooked["sensenova_max_think_tokens"].cond == 23
    assert (
        hooked["sensenova_thinking_interrupt"].cond
        is comfy.model_management.throw_exception_if_processing_interrupted
    )
    assert hooked["sensenova_thinking_result"].cond is thinking_result


def test_sensenova_thinking_decode_appends_stop_and_image_suffix():
    model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    tokens = iter((42, sensenova_model.THINK_END_TOKEN_ID))
    decoded = []

    model._preprocess_prefix_state = lambda *args: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden: torch.tensor([next(tokens)])

    def decode(token, keys, values, prefix_time, transformer_options=None):
        decoded.append(int(token.item()))
        return torch.zeros(1, 1, 1), keys, values, prefix_time + 1

    model._decode_text_token = decode
    progress_updates = []
    interrupt_calls = []

    _, _, prefix_time = model.preprocess_thinking_prefix(
        torch.tensor([[1, 2, 3]]),
        max_think_tokens=4,
        progress=progress_updates.append,
        interrupt=lambda: interrupt_calls.append(True),
    )

    assert decoded == [
        42,
        sensenova_model.THINK_END_TOKEN_ID,
        *sensenova_model.THINK_SUFFIX_TOKEN_IDS,
    ]
    assert prefix_time.tolist() == [7]
    assert progress_updates == [1, 2]
    assert interrupt_calls == [True, True]


def test_sensenova_thinking_decode_returns_generated_token_ids():
    model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    tokens = iter((42, sensenova_model.THINK_END_TOKEN_ID))

    model._preprocess_prefix_state = lambda *args: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden: torch.tensor([next(tokens)])

    def decode(token, keys, values, prefix_time, transformer_options=None):
        return torch.zeros(1, 1, 1), keys, values, prefix_time + 1

    model._decode_text_token = decode

    _, _, _, token_ids = model.preprocess_thinking_prefix_with_tokens(
        torch.tensor([[1, 2, 3]]), max_think_tokens=4
    )

    assert token_ids == [42, sensenova_model.THINK_END_TOKEN_ID]


@pytest.mark.parametrize(
    ("tokens", "max_think_tokens", "expected_decoded"),
    [
        ((42, 43), 1, [42, sensenova_model.THINK_END_TOKEN_ID]),
        ((sensenova_model.EOS_TOKEN_ID,), 4, [sensenova_model.THINK_END_TOKEN_ID]),
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

    def decode(token, keys, values, prefix_time, transformer_options=None):
        decoded.append(int(token.item()))
        return torch.zeros(1, 1, 1), keys, values, prefix_time + 1

    model._decode_text_token = decode
    _, _, prefix_time = model.preprocess_thinking_prefix(
        torch.tensor([[1, 2, 3]]), max_think_tokens=max_think_tokens
    )

    assert decoded == [
        *expected_decoded,
        *sensenova_model.THINK_SUFFIX_TOKEN_IDS,
    ]
    assert prefix_time.tolist() == [
        3 + len(expected_decoded) + len(sensenova_model.THINK_SUFFIX_TOKEN_IDS)
    ]


def test_sensenova_interleave_generates_text_image_and_resumed_text():
    image_start = 151670
    eos = 151645

    class Model:
        def _preprocess_prefix_state(self, input_ids, *args):
            branch = int(input_ids.item())
            first_token = 101 if branch == 1 else 0
            return first_token, [[]], [[]], torch.tensor([3])

        def _next_text_token(self, hidden):
            return torch.tensor([hidden])

        def _decode_text_token(
            self, token, keys, values, prefix_time, transformer_options=None
        ):
            token_id = int(token.item())
            keys = [keys[0] + [token_id]]
            next_token = {101: image_start, image_start: 0, 102: eos}[token_id]
            return next_token, keys, values, prefix_time + 1

        def append_interleave_image(
            self, image, keys, values, prefix_time, transformer_options=None
        ):
            assert torch.equal(image, torch.full((1, 3, 32, 32), 0.25))
            keys = [keys[0] + ["image"]]
            return 102, keys, values, prefix_time + 2

    sampled_prefixes = []

    def sample_image(positive, negative):
        sampled_prefixes.append(
            ([list(values) for values in positive.keys], [list(values) for values in negative.keys])
        )
        return torch.full((1, 3, 32, 32), 0.25)

    session = SenseNovaInterleaveSession(
        Model(),
        positive_prefix=(torch.tensor([[1]]),),
        negative_prefix=(torch.tensor([[2]]),),
        decode_tokens=lambda token_ids: {101: "before ", 102: "after"}[
            token_ids[0]
        ],
    )

    result = session.generate(sample_image, max_text_tokens=8, max_images=1)

    assert result.text == "before <image>after"
    assert result.token_ids == [101, image_start, 102, eos]
    assert result.stop_reason == "eos"
    assert len(result.images) == 1
    assert torch.equal(result.images[0], torch.full((1, 3, 32, 32), 0.25))
    assert sampled_prefixes[0][0] == [[101, image_start]]
    assert sampled_prefixes[0][1] == [[image_start]]


def test_sensenova_interleave_generates_multiple_images_in_one_session():
    image_start = 151670
    eos = 151645

    class Model:
        def __init__(self):
            self.append_calls = 0

        def _preprocess_prefix_state(self, input_ids, *args):
            branch = int(input_ids.item())
            first_token = 101 if branch == 1 else 0
            return first_token, [[]], [[]], torch.tensor([3])

        def _next_text_token(self, hidden):
            return torch.tensor([hidden])

        def _decode_text_token(
            self, token, keys, values, prefix_time, transformer_options=None
        ):
            token_id = int(token.item())
            next_token = {
                101: image_start,
                image_start: 0,
                102: image_start,
                103: eos,
            }[token_id]
            return next_token, keys, values, prefix_time + 1

        def append_interleave_image(
            self, image, keys, values, prefix_time, transformer_options=None
        ):
            self.append_calls += 1
            next_token = 102 if self.append_calls <= 2 else 103
            return next_token, keys, values, prefix_time + 2

    sampled_images = []

    def sample_image(positive, negative):
        image = torch.full((1, 3, 32, 32), len(sampled_images) + 1.0)
        sampled_images.append(image)
        return image

    session = SenseNovaInterleaveSession(
        Model(),
        positive_prefix=(torch.tensor([[1]]),),
        negative_prefix=(torch.tensor([[2]]),),
        decode_tokens=lambda token_ids: {101: "first", 102: "second", 103: "end"}[
            token_ids[0]
        ],
    )

    result = session.generate(sample_image, max_text_tokens=8, max_images=2)

    assert result.text == "first<image>second<image>end"
    assert result.token_ids == [101, image_start, 102, image_start, 103, eos]
    assert result.stop_reason == "eos"
    assert len(result.images) == 2
    assert torch.equal(result.images[0], torch.ones(1, 3, 32, 32))
    assert torch.equal(result.images[1], torch.full((1, 3, 32, 32), 2.0))


def test_sensenova_interleave_appends_generated_image_to_prefix(monkeypatch):
    captured = {}

    class Layer:
        def forward_decode(
            self,
            hidden_states,
            rope,
            prefix_key,
            prefix_value,
            transformer_options,
            attention_mask=None,
        ):
            captured["hidden_states"] = hidden_states
            captured["attention_mask"] = attention_mask
            next_key = torch.cat(
                (prefix_key, torch.full((1, 1, hidden_states.shape[1], 1), 7.0)),
                dim=2,
            )
            next_value = torch.cat(
                (prefix_value, torch.full((1, 1, hidden_states.shape[1], 1), 8.0)),
                dim=2,
            )
            return hidden_states + 1, next_key, next_value

    class VisionModel:
        def __call__(self, image):
            captured["vision_input"] = image
            return torch.zeros(1, 2, sensenova_model.HIDDEN_SIZE)

    def embed_tokens(token_ids):
        assert token_ids.tolist() == [[151671]]
        return torch.ones(1, 1, sensenova_model.HIDDEN_SIZE)

    def prepare_mrope(indexes, device, dtype):
        captured["indexes"] = indexes
        return (None, None, None)

    monkeypatch.setattr(sensenova_model, "_prepare_mrope", prepare_mrope)
    model = SimpleNamespace(
        vision_model=VisionModel(),
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=embed_tokens, layers=[Layer()])
        ),
    )
    image = torch.zeros(1, 3, 32, 64)
    keys = [torch.zeros(1, 1, 3, 1)]
    values = [torch.zeros(1, 1, 3, 1)]

    hidden, next_keys, next_values, next_time = (
        sensenova_model.SenseNovaU15.append_interleave_image(
            model,
            image,
            keys,
            values,
            torch.tensor([3]),
            transformer_options={},
        )
    )

    expected = torch.tensor(
        [0.06550218, 0.19642857, 0.41777778]
    ).view(1, 3, 1, 1)
    assert torch.allclose(captured["vision_input"], expected.expand(1, 3, 32, 64))
    assert captured["hidden_states"].shape == (1, 3, sensenova_model.HIDDEN_SIZE)
    assert torch.equal(
        captured["indexes"],
        torch.tensor([[[3, 3, 4]], [[0, 0, 0]], [[0, 1, 0]]]),
    )
    assert captured["attention_mask"].shape == (1, 1, 3, 6)
    assert torch.isneginf(captured["attention_mask"][0, 0, :2, 5]).all()
    assert captured["attention_mask"][0, 0, 2, 5] == 0
    assert hidden.shape == (1, 3, sensenova_model.HIDDEN_SIZE)
    assert next_keys[0].shape == (1, 1, 6, 1)
    assert next_values[0].shape == (1, 1, 6, 1)
    assert next_time.tolist() == [5]


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
    thinking_calls = []

    def preprocess_thinking_prefix(*args, **kwargs):
        thinking_calls.append((args, kwargs))
        return prefix_keys, prefix_values, prefix_time

    model.preprocess_thinking_prefix = preprocess_thinking_prefix
    thinking_options = {}
    thinking_interrupt = object()
    thinking = sensenova_model.SenseNovaU15._forward(
        model,
        image,
        timesteps,
        text_input_ids=input_ids,
        sensenova_thinking=True,
        sensenova_max_think_tokens=7,
        sensenova_thinking_interrupt=thinking_interrupt,
        transformer_options=thinking_options,
    )
    thinking_result = {"enabled": True, "token_ids": None}
    thinking_token_calls = []

    def preprocess_thinking_prefix_with_tokens(*args, **kwargs):
        thinking_token_calls.append((args, kwargs))
        return prefix_keys, prefix_values, prefix_time, [41, 42]

    model.preprocess_thinking_prefix_with_tokens = (
        preprocess_thinking_prefix_with_tokens
    )
    thinking_with_result = sensenova_model.SenseNovaU15._forward(
        model,
        image,
        timesteps,
        text_input_ids=input_ids,
        sensenova_thinking=True,
        sensenova_max_think_tokens=7,
        sensenova_thinking_result=thinking_result,
        sensenova_thinking_interrupt=thinking_interrupt,
        transformer_options=thinking_options,
    )

    assert torch.equal(raw, preprocessed)
    assert torch.equal(thinking, preprocessed)
    assert torch.equal(thinking_with_result, preprocessed)
    assert thinking_calls[0][0][:4] == (input_ids, None, None, None)
    assert thinking_calls[0][1]["max_think_tokens"] == 7
    assert thinking_calls[0][1]["interrupt"] is thinking_interrupt
    assert thinking_calls[0][1]["transformer_options"] is thinking_options
    assert thinking_token_calls[0][1]["max_think_tokens"] == 7
    assert thinking_token_calls[0][1]["interrupt"] is thinking_interrupt
    assert thinking_result["token_ids"] == [41, 42]
    assert timestep_embedder.shapes == [torch.Size([1])] * 4
    assert noise_scale_embedder.shapes == [torch.Size([1])] * 4


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
        max_think_tokens=64,
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


def test_sensenova_interleave_node_uses_standard_sampling_inputs():
    schema = SenseNovaInterleave.define_schema()
    inputs = {value.id for value in schema.inputs}

    assert inputs == {
        "model",
        "clip",
        "positive",
        "negative",
        "noise_seed",
        "cfg",
        "sampler",
        "sigmas",
        "latent_image",
        "max_text_tokens",
        "max_images",
    }
    assert [output.display_name for output in schema.outputs] == [
        "samples",
        "text",
        "interleave_result",
    ]


def test_sensenova_nodes_use_family_capability_names():
    assert SenseNovaTextEncode.define_schema().display_name == "SenseNova Text Encode"
    assert SenseNovaInterleave.define_schema().display_name == "SenseNova Interleave"


def test_sensenova_model_base_preserves_patch_size_compatibility_alias():
    assert model_base.SenseNovaU15.PATCH_SIZE == sensenova_model.MERGED_PATCH_SIZE


def test_sensenova_interleave_result_preserves_article_order_and_thinking():
    result = InterleaveResult(
        text="<think>plan</think>Hello<image>After",
        images=[torch.empty(1, 8, 8, 3)],
        token_ids=[1, 2, 3],
        stop_reason="eos",
    )

    payload = build_interleave_result(result)

    assert payload["parts"] == [
        {"type": "think", "text": "plan"},
        {"type": "text", "text": "Hello"},
        {"type": "image", "index": 0},
        {"type": "text", "text": "After"},
    ]
    assert payload["think_text"] == "plan"
    assert payload["token_ids"] == [1, 2, 3]
    assert interleave_result_to_markdown(payload, include_think=False) == (
        "Hello\n\n[image:0]\n\nAfter"
    )


def test_sensenova_interleave_result_preserves_thinking_across_images():
    result = InterleaveResult(
        text="<think>plan<image>inspect</think>answer",
        images=[torch.empty(1, 8, 8, 3)],
        token_ids=[],
        stop_reason="eos",
    )

    payload = build_interleave_result(result)

    assert payload["parts"] == [
        {"type": "think", "text": "plan"},
        {"type": "image", "index": 0},
        {"type": "think", "text": "inspect"},
        {"type": "text", "text": "answer"},
    ]
    assert payload["think_text"] == "plan\n\ninspect"
    assert interleave_result_to_markdown(payload, include_think=False) == (
        "[image:0]\n\nanswer"
    )


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


def test_sensenova_interleave_result_marks_missing_images():
    result = InterleaveResult(
        text="Before<image>Middle<image>After",
        images=[torch.empty(1, 8, 8, 3)],
        token_ids=[],
        stop_reason="eos",
    )

    payload = build_interleave_result(result)

    assert payload["parts"][3] == {"type": "image", "index": 1, "missing": True}


def test_sensenova_interleave_without_images_keeps_a_decodable_latent():
    latent_samples = torch.randn(1, 3, 8, 8)
    result = InterleaveResult("text only", [], [1], "eos")

    output = interleave_output_samples(result, latent_samples)

    assert output is latent_samples


def test_sensenova_interleave_preview_builds_inline_ui_parts(monkeypatch):
    monkeypatch.setattr(
        "comfy_extras.nodes_sensenova._save_preview_images",
        lambda images: [
            {"filename": "preview.png", "subfolder": "", "type": "temp"}
        ],
    )
    result = {
        "parts": [
            {"type": "think", "text": "plan"},
            {"type": "text", "text": "Hello"},
            {"type": "image", "index": 0},
        ]
    }

    output = SenseNovaInterleavePreview.execute(
        interleave_result=result,
        include_think=False,
        images=torch.empty(1, 8, 8, 3),
    )

    assert output.args == ("Hello\n\n[image:0]",)
    assert output.ui["parts"] == [
        {"type": "text", "text": "Hello"},
        {
            "type": "image",
            "index": 0,
            "filename": "preview.png",
            "subfolder": "",
            "image_type": "temp",
        },
    ]


def _mock_interleave_preview_images(monkeypatch, count=1):
    monkeypatch.setattr(
        "comfy_extras.nodes_sensenova._save_preview_images",
        lambda images: [
            {"filename": f"preview_{index}.png", "subfolder": "", "type": "temp"}
            for index in range(count)
        ],
    )


def test_sensenova_interleave_preview_places_thinking_images_at_final_references(
    monkeypatch,
):
    _mock_interleave_preview_images(monkeypatch, count=3)
    result = {
        "parts": [
            {"type": "think", "text": "plan first image"},
            {"type": "image", "index": 0},
            {"type": "think", "text": "plan second image"},
            {"type": "image", "index": 1},
            {"type": "think", "text": "plan third image"},
            {"type": "image", "index": 2},
            {
                "type": "text",
                "text": (
                    "First description\n<image1>\n"
                    "Second description\n<image2>\n"
                    "Third description\n<image3>"
                ),
            },
        ]
    }

    output = SenseNovaInterleavePreview.execute(
        interleave_result=result,
        include_think=False,
        images=torch.empty(3, 8, 8, 3),
    )

    assert output.args == (
        "First description\n\n[image:0]\n\n"
        "Second description\n\n[image:1]\n\n"
        "Third description\n\n[image:2]",
    )
    assert [part["type"] for part in output.ui["parts"]] == [
        "text",
        "image",
        "text",
        "image",
        "text",
        "image",
    ]
    assert [
        part["index"] for part in output.ui["parts"] if part["type"] == "image"
    ] == [0, 1, 2]


def test_sensenova_interleave_preview_removes_unresolved_numbered_references(
    monkeypatch,
):
    _mock_interleave_preview_images(monkeypatch)
    result = {
        "parts": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Before<image99>After"},
        ]
    }

    output = SenseNovaInterleavePreview.execute(
        interleave_result=result,
        include_think=False,
        images=torch.empty(1, 8, 8, 3),
    )

    assert output.args == ("[image:0]\n\nBefore\n\nAfter",)
    assert [part["type"] for part in output.ui["parts"]] == [
        "image",
        "text",
        "text",
    ]


def test_sensenova_interleave_preview_hides_references_when_showing_thinking(
    monkeypatch,
):
    _mock_interleave_preview_images(monkeypatch)
    result = {
        "parts": [
            {"type": "think", "text": "plan"},
            {"type": "image", "index": 0},
            {"type": "text", "text": "Answer<image1>Done"},
        ]
    }

    output = SenseNovaInterleavePreview.execute(
        interleave_result=result,
        include_think=True,
        images=torch.empty(1, 8, 8, 3),
    )

    assert "<image1>" not in output.args[0]
    assert [part["type"] for part in output.ui["parts"]] == [
        "think",
        "image",
        "text",
        "text",
    ]
    assert [
        part["index"] for part in output.ui["parts"] if part["type"] == "image"
    ] == [0]


def test_sensenova_interleave_preview_removes_non_positive_image_references(
    monkeypatch,
):
    _mock_interleave_preview_images(monkeypatch)
    result = {
        "parts": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Before<image0>Middle<image00>After"},
        ]
    }

    output = SenseNovaInterleavePreview.execute(
        interleave_result=result,
        include_think=False,
        images=torch.empty(1, 8, 8, 3),
    )

    assert output.args == ("[image:0]\n\nBefore\n\nMiddle\n\nAfter",)
    assert [
        part["index"] for part in output.ui["parts"] if part["type"] == "image"
    ] == [0]


def test_sensenova_frontend_extension_is_packaged_with_preview_nodes():
    web_directory = Path(sensenova_nodes.__file__).parent / sensenova_nodes.WEB_DIRECTORY
    script = web_directory / "sensenova_interleave_preview.js"

    assert script.is_file()
    script_text = script.read_text(encoding="utf-8")
    assert "SenseNovaThinkingPreview" in script_text
    assert "SenseNovaInterleavePreview" in script_text
    assert "previewText" in script_text


def test_sensenova_text_encode_adds_reasoning_policy():
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
        max_think_tokens=64,
    ).args[0]

    assert calls[0] == ("tokenize", "test", {"thinking": True})
    assert calls[1][2] == {
        "sensenova_thinking": True,
        "sensenova_max_think_tokens": 64,
        "sensenova_thinking_result": {
            "enabled": True,
            "token_ids": None,
        },
    }
    assert output[0][1]["sensenova_thinking"] is True


def test_sensenova_thinking_preview_decodes_tokens_after_sampling():
    decode_calls = []

    class Clip:
        def decode(self, token_ids, skip_special_tokens=True):
            decode_calls.append((token_ids, skip_special_tokens))
            return "  inspect the layout  "

    conditioning = [
        [
            torch.empty(1),
            {
                "sensenova_thinking_result": {
                    "enabled": True,
                    "token_ids": [41, 42],
                }
            },
        ]
    ]
    output = SenseNovaThinkingPreview.execute(
        clip=Clip(),
        conditioning=conditioning,
        samples={"samples": torch.empty(1, 3, 8, 8)},
    )

    assert output.args == ("inspect the layout",)
    assert output.ui.as_dict() == {"text": ("inspect the layout",)}
    assert decode_calls == [([41, 42], True)]


@pytest.mark.parametrize(
    ("conditioning", "expected"),
    [
        (
            [[torch.empty(1), {}]],
            "SenseNova thinking is disabled for this conditioning.",
        ),
        (
            [
                [
                    torch.empty(1),
                    {
                        "sensenova_thinking_result": {
                            "enabled": True,
                            "token_ids": None,
                        }
                    },
                ]
            ],
            "SenseNova thinking has not run. Connect samples from the KSampler that uses this conditioning.",
        ),
    ],
)
def test_sensenova_thinking_preview_explains_unavailable_results(
    conditioning, expected
):
    output = SenseNovaThinkingPreview.execute(
        clip=SimpleNamespace(decode=lambda *args, **kwargs: "unused"),
        conditioning=conditioning,
        samples={"samples": torch.empty(1, 3, 8, 8)},
    )

    assert output.args == (expected,)


def test_sensenova_thinking_memory_estimate_includes_decode_limit():
    model = object.__new__(model_base.SenseNovaU15)
    input_ids = torch.empty(1, 10, dtype=torch.long)

    shapes = model.extra_conds_shapes(
        text_input_ids=input_ids,
        sensenova_thinking=True,
        sensenova_max_think_tokens=5,
        prompt_type="positive",
    )

    expected_length = 10 + 5 + 1 + len(sensenova_model.THINK_SUFFIX_TOKEN_IDS)
    assert shapes["prefix_mask"] == [1, 1, 10, 10]
    assert shapes["prefix_keys"] == [
        1,
        sensenova_model.NUM_KV_HEADS,
        sensenova_model.NUM_LAYERS
        * expected_length
        * sensenova_model.HEAD_DIM,
    ]
