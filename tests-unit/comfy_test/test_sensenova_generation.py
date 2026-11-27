
import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.hooks
import comfy.model_management
import comfy.ops
import comfy.sd
import nodes
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from comfy.ldm.sensenova import model as sensenova_model
from comfy.ldm.sensenova.conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    IMAGE_START_ID,
    IM_END_ID,
    IM_START_ID,
    NEWLINE_ID,
    multi_turn_suffix_ids,
    prepare_incremental_prefix_inputs,
    prepare_interleave_image,
    prepare_prefix_inputs,
)
from comfy.text_encoders.sensenova import (
    INTERLEAVE_SYSTEM_MESSAGE,
    SenseNovaQwenTokenizer,
    SenseNovaTokenizer,
    build_generation_prompt,
    build_interleave_unconditional_prompt,
    build_unconditional_prompt,
)
from comfy_execution.graph_utils import ExecutionBlocker
from comfy_extras.nodes_sensenova import (
    append_interleave_image,
    append_interleave_tokens,
    expand_interleave_metadata,
    expand_multi_turn_metadata,
    SenseNovaConditioning,
    SenseNovaInterleaveDecode,
)
from sensenova_test_utils import (
    prefix_arguments,
    prepared_prefix,
    PatchEmbedding,
    _generation_input_ids,
    _run_prepared_generation,
    _text_clip,
    _tokenize_generation_prompt,
)


def test_sensenova_text_returns_stop_without_decoding_suffix():
    model = object.__new__(sensenova_model.SenseNovaTextModel)
    torch.nn.Module.__init__(model)
    model.has_lm_head = True
    tokens = iter((42, 151668))
    decoded = []

    model._preprocess_prefix_state = lambda *args, **kwargs: (
        torch.zeros(1, 1, 1),
        [torch.zeros(1, 1, 1, 1)],
        [torch.zeros(1, 1, 1, 1)],
        torch.tensor([3]),
    )
    model._next_text_token = lambda hidden, **sampling: torch.tensor([next(tokens)])

    def decode(token, caches, hidden, rope, attention_mask, transformer_options=None):
        decoded.append(int(token.item()))

    model._decode_text_token = decode
    progress_updates = []
    interrupt_calls = []

    token_ids, stop_token = model.generate_text_tokens(
        torch.tensor([[1, 2, 3]]),
        max_tokens=4,
        stop_token_ids=(151668,),
        progress=progress_updates.append,
        interrupt=lambda: interrupt_calls.append(True),
    )

    assert token_ids == [42, 151668]
    assert stop_token == 151668
    assert decoded == [42]
    assert progress_updates == [1, 2]
    assert interrupt_calls == [True, True]


def test_sensenova_decode_shares_mask_and_updates_only_temporal_rope(monkeypatch):
    model = _text_clip([42, 43, 44, 151668]).cond_stage_model
    prefix = torch.zeros(1, 1, 3, 4)
    monkeypatch.setattr(model, "_preprocess_prefix_state", lambda *args, **kwargs: (
        torch.zeros(1, 3, 4), [prefix, prefix], [prefix, prefix], torch.tensor([7]),
    ))
    predictions = iter((42, 43, 44, 151668))
    monkeypatch.setattr(model, "_next_text_token", lambda *args, **kwargs: torch.tensor([next(predictions)]))
    masks = []

    class Layer(torch.nn.Module):
        def forward_decode(self, hidden, rope, cache, attention_mask, transformer_options):
            step = len(masks) // 2
            assert cache.position.item() == cache.index == 3 + step
            assert cache.seqlen.item() == 4 + step
            assert len(rope) == 1
            expected = sensenova_model._prepare_mrope(torch.tensor([[7 + step], [0], [0]]), hidden.device, hidden.dtype)
            torch.testing.assert_close(rope[0], expected[0])
            assert attention_mask.flatten().tolist() == [0.] * (4 + step) + [float("-inf")] * (3 - step)
            masks.append(attention_mask)
            return hidden

    model.language_model.model.layers = torch.nn.ModuleList([Layer(), Layer()])
    generated, stop = model.generate_text_tokens(torch.tensor([[1, 2, 3]]), max_tokens=5, stop_token_ids=(151668,))
    assert (generated, stop) == ([42, 43, 44, 151668], 151668)
    assert len(masks) == 6 and all(mask is masks[0] for mask in masks)


@pytest.mark.parametrize(
    ("tokens", "max_think_tokens", "expected_decoded"),
    [
        ((42, 43), 1, [42]),
        ((151645,), 4, []),
        ((151668,), 4, [151668]),
    ],
)
def test_sensenova_thinking_preserves_reference_stop_protocol(
    monkeypatch, tokens, max_think_tokens, expected_decoded
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_clip(tokens)
    monkeypatch.setattr(model.cond_stage_model, "_decode_text_token", lambda *args: pytest.fail("Decoding past the stop token"))
    result = SenseNovaConditioning._execute_image(
        clip=model,
        conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([[1]])}]],
        max_tokens=max_think_tokens,
    )

    assert result.args[0][0][1]["text_input_ids"].tolist() == [[1, *expected_decoded, 271, 151670]]
    assert result.args[1] == ("K" if expected_decoded == [42] else "")


def test_sensenova_interleave_references_do_not_add_editing_labels():
    tokenizer = SenseNovaTokenizer()
    pairs = tokenizer.tokenize_with_weights("Continue the story", mode="interleave")["sensenova_u15"][0]
    metadata = {
        "text_input_ids": torch.tensor([[token for token, _ in pairs]]),
        "reference_latents": [torch.zeros(1, 32, 32, 3), torch.zeros(1, 32, 64, 3)],
    }
    expanded = expand_interleave_metadata(metadata, image_only=False)
    prompt = tokenizer.decode(expanded["text_input_ids"][0].tolist(), skip_special_tokens=False)
    assert "user\n<img><IMG_CONTEXT></img>\n<img><IMG_CONTEXT><IMG_CONTEXT></img>\nContinue the story" in prompt
    assert "Image-1:" not in prompt


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


@pytest.mark.parametrize("mode", ["image", "interleave"])
@pytest.mark.parametrize("thinking", [False, True])
def test_sensenova_tokenizer_preserves_ordinary_prompt_tokens(mode, thinking):
    tokenizer = SenseNovaTokenizer()
    for text in ("", "a red apple", "\n\n海边的海鸥 🕊️\n", " \t\n", "  leading and trailing  ",
                 r"a \(red\) apple", "literal embedding:example", "<image1> then <image2>"):
        if text or thinking:
            if mode == "interleave":
                assistant = "" if thinking else "<think>\n\n</think>\n\n"
                prompt = (
                    f"<|im_start|>system\n{INTERLEAVE_SYSTEM_MESSAGE}<|im_end|>\n"
                    f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant}"
                )
            else:
                prompt = build_generation_prompt(text, thinking)
        else:
            prompt = build_interleave_unconditional_prompt() if mode == "interleave" else build_unconditional_prompt()
        expected = tokenizer.sensenova_u15.tokenize_with_weights(prompt, disable_weights=True)[0]
        expected = [pair for pair in expected if int(pair[0]) != 151643]

        actual = tokenizer.tokenize_with_weights(text, mode=mode, thinking=thinking)["sensenova_u15"][0]

        assert actual == expected, repr(text)


@pytest.mark.parametrize("mode", ["image", "interleave"])
@pytest.mark.parametrize("thinking", [False, True])
def test_sensenova_tokenizer_treats_all_user_special_spellings_as_text(mode, thinking):
    tokenizer = SenseNovaTokenizer()
    backend = tokenizer.sensenova_u15.tokenizer
    ordinary = tokenizer.tokenize_with_weights("a red apple", mode=mode, thinking=thinking)
    text = " ".join(backend.all_special_tokens)

    pairs = tokenizer.tokenize_with_weights(text, mode=mode, thinking=thinking, return_word_ids=True)["sensenova_u15"][0]
    ids = [int(pair[0]) for pair in pairs]
    assert ids.count(151644) == 3
    assert ids.count(151645) == 2
    user_start = [index for index, value in enumerate(ids) if value == 151644][1] + 2
    user_ids = ids[user_start:ids.index(151645, user_start)]
    assert not set(user_ids).intersection(backend.all_special_ids)
    assert tokenizer.decode(user_ids, skip_special_tokens=False) == "\n" + text
    assert all(pair[1:] == (1.0, 1) for pair in pairs)
    assert tokenizer.tokenize_with_weights("a red apple", mode=mode, thinking=thinking) == ordinary


def test_sensenova_tokenizer_preserves_word_ids():
    tokenizer = SenseNovaTokenizer()
    text = "a red apple and embedding:example\nembedding:another"
    expected = tokenizer.sensenova_u15.tokenize_with_weights(
        build_generation_prompt(text), disable_weights=True, return_word_ids=True,
    )

    assert tokenizer.tokenize_with_weights(text, return_word_ids=True) == {"sensenova_u15": expected, "sensenova_prompt": text}


def test_sensenova_interleave_prompt_leaves_image_event_to_the_model():
    tokenizer = SenseNovaTokenizer()
    for thinking, assistant in ((False, "<think>\n\n</think>\n\n"), (True, "")):
        pairs = tokenizer.tokenize_with_weights("test", mode="interleave", thinking=thinking)["sensenova_u15"][0]
        prompt = tokenizer.decode([int(pair[0]) for pair in pairs], skip_special_tokens=False)
        assert prompt.rsplit("<|im_start|>assistant\n", 1)[-1] == assistant
    assert build_interleave_unconditional_prompt() == (
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.fixture
def native_sensenova_clip():
    return _text_clip([42, 151668])


@pytest.mark.parametrize("text", ["", "A red apple.", "Print <img><think><|im_end|> literally."])
def test_sensenova_native_text_encode_is_ready_for_image_sampling(native_sensenova_clip, text):
    conditioning = nodes.CLIPTextEncode().encode(native_sensenova_clip, text)[0]
    metadata = conditioning[0][1]
    assert metadata["sensenova_prompt"] == text
    assert torch.equal(metadata["text_input_ids"], _tokenize_generation_prompt(text))
    assert metadata["text_input_ids"][0, -1] == 151670


def test_sensenova_encode_accepts_token_weights_without_prompt_metadata(native_sensenova_clip):
    tokens = native_sensenova_clip.tokenize("Draw an apple.")
    expected = native_sensenova_clip.encode_from_tokens_scheduled(tokens)
    del tokens["sensenova_prompt"]
    actual = native_sensenova_clip.encode_from_tokens_scheduled(tokens)
    torch.testing.assert_close(actual[0][0], expected[0][0])
    torch.testing.assert_close(actual[0][1]["text_input_ids"], expected[0][1]["text_input_ids"])


@pytest.mark.parametrize("task,mode,thinking", [
    ({"task": "image thinking"}, "image", True),
    ({"task": "interleave", "thinking": True}, "interleave", True),
    ({"task": "interleave", "thinking": False}, "interleave", False),
    ({"task": "self-correction"}, "image", False),
])
@pytest.mark.parametrize("text", ["", "Draw a red apple.", "Print <IMG_CONTEXT><img></img><think><|im_end|> literally."])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_task_generation_uses_task_prompt_protocol(
    native_sensenova_clip, monkeypatch, task, mode, thinking, text, with_reference,
):
    clip = native_sensenova_clip
    positive = nodes.CLIPTextEncode().encode(clip, text)[0]
    negative = nodes.CLIPTextEncode().encode(clip, "blurry")[0]
    if with_reference:
        positive, negative = HiDreamO1ReferenceImages.execute(
            positive=positive, negative=negative, images={"image_1": torch.ones(1, 32, 32, 3)},
        ).args
    positive[0][1]["strength"] = 0.7
    original_ids = positive[0][1]["text_input_ids"].clone()
    generated = [42, 151668 if task["task"] == "image thinking" else 151670]
    model = _text_clip(generated)
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", len(generated))
    model.cond_stage_model.vision_model = PatchEmbedding(3, len(generated), kernel_size=32, stride=32)
    embeddings = model.cond_stage_model.language_model.model.embed_tokens.weight.detach()
    for token in (198, 271):
        embeddings[token].copy_(embeddings[1])
    result = SenseNovaConditioning.execute(
        clip=model, conditioning=positive,
        task=task, negative=negative, max_tokens=8,
    ).args
    protocol_tokens = clip.tokenize(text, mode=mode, thinking=thinking)
    protocol_conditioning = clip.encode_from_tokens_scheduled(protocol_tokens)
    protocol_conditioning[0][1].update({key: value for key, value in positive[0][1].items() if key != "text_input_ids"})
    expected = protocol_conditioning[0][1]
    if task["task"] == "image thinking":
        expected_ids = torch.cat((expected["text_input_ids"], torch.tensor([[*generated, 271, 151670]])), dim=1)
    else:
        expected = expand_interleave_metadata(expected, image_only=False)
        expected_ids = torch.cat((expected["text_input_ids"], torch.tensor([generated])), dim=1)
    assert result[3] is True
    assert torch.equal(result[0][0][1]["text_input_ids"], expected_ids)
    assert result[0][0][1]["strength"] == 0.7
    assert torch.equal(positive[0][1]["text_input_ids"], original_ids)
    assert "sensenova_interleave_expanded" not in positive[0][1]
    if task["task"] != "image thinking":
        continuation = _text_clip([151671, 43, 151645])
        monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 3)
        continuation.cond_stage_model.vision_model = PatchEmbedding(3, 3, kernel_size=32, stride=32)
        def unexpected_tokenize(*args, **kwargs):
            raise AssertionError("An expanded history must not be tokenized again")
        monkeypatch.setattr(clip, "tokenize", unexpected_tokenize)
        resumed = SenseNovaConditioning.execute(
            clip=continuation, conditioning=result[0], negative=result[2],
            task={**task, "samples": {"samples": torch.zeros(1, 3, 32, 32)}},
            max_tokens=8,
        ).args
        assert resumed[3] is False
        assert resumed[0][0][1]["sensenova_interleave_finished"] is True
        assert resumed[0][0][1]["text_input_ids"].tolist() == [expected_ids[0].tolist() + [151669, 151671, 43]]
        assert len(resumed[0][0][1]["reference_latents"]) == int(with_reference) + 1
        assert resumed[0][0][1]["reference_latents"][-1] is resumed[2][0][1]["reference_latents"][-1]


def test_sensenova_generate_requires_head_only_for_token_generation(native_sensenova_clip):
    clip = native_sensenova_clip
    conditioning = nodes.CLIPTextEncode().encode(clip, "Draw an apple.")[0]
    model = _text_clip([42, 151668])
    model.cond_stage_model.has_lm_head = False
    del model.cond_stage_model.language_model.lm_head
    with pytest.raises(RuntimeError, match="required for text generation"):
        SenseNovaConditioning.execute(clip=model, conditioning=conditioning,
            task={"task": "image thinking"}, max_tokens=8)


def test_sensenova_self_correction_prompt_matches_checkpoint_protocol():
    tokenizer = SenseNovaTokenizer()
    pairs = tokenizer.tokenize_with_weights("test", mode="image", thinking=False)["sensenova_u15"][0]
    original_ids = torch.tensor([[int(pair[0]) for pair in pairs]])
    expanded = expand_interleave_metadata({"text_input_ids": original_ids}, image_only=False)
    prompt = tokenizer.decode(expanded["text_input_ids"][0].tolist(), skip_special_tokens=False)
    expected = build_generation_prompt("test", thinking=False).removesuffix("<img>")
    assert prompt == expected
    assert "<img>" not in prompt
    assert tokenizer.decode(original_ids[0].tolist(), skip_special_tokens=False) == build_generation_prompt("test", thinking=False)

    negative = tokenizer.tokenize_with_weights("", mode="image")["sensenova_u15"][0]
    expanded_negative = expand_interleave_metadata({
        "text_input_ids": torch.tensor([[int(pair[0]) for pair in negative]]),
    }, image_only=True)
    assert tokenizer.decode(expanded_negative["text_input_ids"][0].tolist(), skip_special_tokens=False) == build_interleave_unconditional_prompt()

    requested = append_interleave_tokens(expanded, [151670], True)
    resumed = expand_interleave_metadata(requested, image_only=False)
    assert resumed["text_input_ids"] is requested["text_input_ids"]
    assert resumed["text_input_ids"][0, -1] == 151670
    assert resumed["sensenova_interleave_pending_image"] is True


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


@pytest.mark.parametrize("prompt_type", [None, "negative"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_generate_returns_prefix_conditioning_and_text(monkeypatch, prompt_type, with_reference):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 3)
    model = _text_clip([41, 42, 151668])
    diffusion = model.cond_stage_model
    diffusion.vision_model = PatchEmbedding(3, 3, kernel_size=32, stride=32)
    embeddings = diffusion.language_model.model.embed_tokens.weight.detach()
    embeddings[198].copy_(embeddings[1])
    input_ids = _generation_input_ids()[:, :-1]
    references = [torch.ones(1, 33, 65, 3)] if with_reference else []
    conditioning = [[torch.empty(1), {
        "text_input_ids": input_ids, "reference_latents": references, "prompt_type": prompt_type,
    }]]

    updated, text, negative, requested = SenseNovaConditioning._execute_image(
        clip=model, conditioning=conditioning, max_tokens=8,
    ).args

    metadata = updated[0][1]
    assert text == "JK"
    assert negative == []
    assert requested is True
    assert metadata["prompt_type"] == prompt_type
    assert metadata["text_input_ids"].tolist() == [input_ids[0].tolist() + [41, 42, 151668, 271, 151670]]
    assert metadata["sensenova_thinking_start"] == input_ids.shape[1]
    assert metadata["reference_latents"] is references
    assert {"prefix_keys", "prefix_values", "prefix_time"} <= metadata.keys()
    assert conditioning[0][1]["text_input_ids"] is input_ids
    assert "sensenova_thinking_start" not in conditioning[0][1]

    conds = prepared_prefix(**{**metadata, "prompt_type": None})
    sampler_ids = conds["text_input_ids"]
    assert (sampler_ids == 151669).sum().item() == (6 if with_reference else 0)
    assert conds["text_suffix"].tolist() == [[41, 42, 151668, 271, 151670]]
    if with_reference:
        assert conds["reference_images"][0].shape == (1, 3, 33, 65)


@pytest.mark.parametrize("interleave,token_ids,expected", [
    (False, [42, 151668], "K"),
    (False, [220, 42, 198, 151668], " K\n"),
    (False, [151668], ""),
    (False, [42, 151667, 43, 151668], "K<think>L"),
    (True, [220, 42, 198, 151645], " K\n"),
    (True, [197, 198, 151645], "\t\n"),
    (True, [151667, 42, 151668, 43, 151670], "KL"),
    (True, [151667, 151668, 271, 197, 42, 220, 151645], "\n\n\tK "),
    (True, [151667, 42, 151668, 271, 197, 43, 220, 151645], "K\n\n\tL "),
    (True, [42, 151668, 43, 151645], "K</think>L"),
    (True, [42, 151667, 43, 151668, 44, 151645], "K<think>L</think>M"),
    (True, [198, 151667, 42, 151668, 43, 151645], "\nKL"),
    (True, [151645], ""),
])
def test_sensenova_generate_returns_current_text_without_think_wrappers(
    monkeypatch, interleave, token_ids, expected,
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_interleave": interleave,
    }]]
    output = _run_prepared_generation(
        generation_mode="interleave" if interleave else "image",
        clip=_text_clip(token_ids),
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


@pytest.mark.parametrize("first_tokens,stage_text,final_text,thinking", [
    ([151667, 42, 151670], "LM<think>N", "M<think>N", "K\n\nL"),
    ([151667, 40, 151668, 42, 151670], "L</think>M<think>N", "K<image>L</think>M<think>N", "I"),
    ([151667, 40, 151668, 271, 197, 42, 220, 151670], "L</think>M<think>N", "\n\n\tK <image>L</think>M<think>N", "I"),
])
def test_sensenova_interleave_display_uses_thinking_history(
    monkeypatch, first_tokens, stage_text, final_text, thinking,
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    clip = SenseNovaQwenTokenizer()
    # Tags in the input prompt must not determine the generated thinking span.
    prompt_ids = [151667, 151668, 1]
    first = _run_prepared_generation(
        generation_mode="interleave",
        clip=_text_clip(first_tokens),
        conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([prompt_ids]), "sensenova_interleave": True}]],
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]],
        max_think_tokens=8,
    )
    assert first.args[3] is True

    # After image feedback the prefix ends with </img>; continue from that token.
    continuation = [151671, 43, 151668, 44, 151667, 45, 151645]
    model = _text_clip(continuation)
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", len(continuation))
    model.cond_stage_model.vision_model = PatchEmbedding(3, len(continuation), kernel_size=32, stride=32)
    second = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=first.args[0], negative=first.args[2],
        samples={"samples": torch.zeros(1, 3, 32, 32)}, max_think_tokens=8,
    )
    assert second.args[1] == stage_text
    assert second.ui.as_dict() == {"text": (stage_text,)}
    assert second.args[3] is False
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[*prompt_ids, *first_tokens]]
    expected_ids = [[*prompt_ids, *first_tokens, 151669, 151671, 43, 151668, 44, 151667, 45]]
    assert second.args[0][0][1]["text_input_ids"].tolist() == expected_ids

    text, images, actual_thinking = SenseNovaInterleaveDecode.execute(clip, second.args[0]).args
    assert (text, actual_thinking) == (final_text, thinking)
    assert len(images) == 1
    assert torch.equal(images[0], torch.full((1, 32, 32, 3), 0.5))
    assert second.args[0][0][1]["text_input_ids"].tolist() == expected_ids


@pytest.mark.parametrize("marker", [151670, 151671, 151669])
@pytest.mark.parametrize("reference_shapes", [(), ((32, 32),), ((32, 64),), ((32, 64), (32, 32))])
def test_sensenova_generated_image_markers_remain_text_for_sampler(monkeypatch, marker, reference_shapes):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    token_ids = [42, marker, 43, 151668]
    model = _text_clip(token_ids)
    width = len(token_ids)
    for name, value in (("HIDDEN_SIZE", width), ("INTERMEDIATE_SIZE", 16), ("NUM_HEADS", 1), ("NUM_KV_HEADS", 1)):
        monkeypatch.setattr(sensenova_model, name, value)

    diffusion = model.cond_stage_model
    diffusion.vision_model = PatchEmbedding(3, width, kernel_size=32, stride=32)
    # The initial reference prefix ends in a newline, not the last prompt token.
    embedding = model.cond_stage_model.language_model.model.embed_tokens.weight.detach()
    embedding[198].copy_(embedding[1])
    positive = [[torch.empty(1), {
        "text_input_ids": _generation_input_ids()[:, :-1],
        "reference_latents": [torch.ones(1, height, width, 3) for height, width in reference_shapes],
    }]]

    result = _run_prepared_generation(
        generation_mode="image",
        clip=model, conditioning=positive, max_think_tokens=8,
    )
    metadata = result.args[0][0][1]
    assert metadata["text_input_ids"][0, 15:].tolist() == [42, marker, 43, 151668, 271, 151670]

    diffusion.language_model.model.layers = torch.nn.ModuleList([
        sensenova_model.DecoderLayer(device="cpu", dtype=torch.float32, operations=comfy.ops.manual_cast)
        for _ in range(2)
    ])
    generator = torch.Generator().manual_seed(17)
    for layer in diffusion.language_model.model.layers:
        for name, parameter in layer.named_parameters():
            if "norm" in name:
                parameter.detach().fill_(1)
            else:
                parameter.detach().normal_(std=0.15, generator=generator)
    conds = model.encode_from_tokens_scheduled(metadata)[0][1]
    assert conds["prefix_time"].tolist() == [{0: 21, 1: 25, 2: 37}[len(reference_shapes)]]

    # An ordinary token with the same embedding is an independent text-only oracle.
    embedding[100].copy_(embedding[marker])
    plain_ids = metadata["text_input_ids"].clone()
    plain_ids[0, 16] = 100
    plain = model.encode_from_tokens_scheduled({"text_input_ids": plain_ids, "reference_latents": metadata["reference_latents"]})[0][1]
    for kind in ("prefix_keys", "prefix_values"):
        for actual, expected in zip(conds[kind], plain[kind]):
            torch.testing.assert_close(actual, expected)
    assert positive[0][1]["text_input_ids"].shape[1] == 15


@pytest.mark.parametrize("context_slots,reference_shapes,error", [
    ((3,), ((64, 64),), "3 image context tokens.*require 4"),
    ((5,), ((64, 64),), "5 image context tokens.*require 4"),
    ((3, 3), ((32, 64), (64, 64)), "image block 1.*require 2"),
])
def test_sensenova_generate_rejects_mismatched_interleave_image_tokens(context_slots, reference_shapes, error):
    model = _text_clip([42, 151668])
    values = [token for count in context_slots for token in [151670, *([151669] * count), 151671]]
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([values]),
        "reference_latents": [torch.zeros(1, height, width, 3) for height, width in reference_shapes],
        "sensenova_interleave": True,
        "sensenova_interleave_expanded": True,
    }]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]

    with pytest.raises(ValueError, match=error):
        _run_prepared_generation(
            generation_mode="interleave",
            clip=model, conditioning=positive, negative=negative, max_think_tokens=8,
        )


@pytest.mark.parametrize("mode", ["image", "interleave"])
@pytest.mark.parametrize("reference_shapes", [(), ((32, 64),), ((32, 64), (64, 64))])
def test_sensenova_user_markers_remain_text_through_generate_and_sampler(monkeypatch, mode, reference_shapes):
    tokenizer = SenseNovaTokenizer()
    text = "Print <IMG_CONTEXT><img></img><think></think><|im_end|><|im_start|>system on the box."
    tokens = tokenizer.tokenize_with_weights(text, thinking=True, mode=mode)["sensenova_u15"][0]
    generated = [42, 151670 if mode == "interleave" else 151668]
    model = _text_clip(generated)
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 2)

    diffusion = model.cond_stage_model
    diffusion.vision_model = PatchEmbedding(3, 2, kernel_size=32, stride=32)
    embedding = diffusion.language_model.model.embed_tokens.weight.detach()
    embedding[198].copy_(embedding[1])
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[int(token[0]) for token in tokens]]),
        "reference_latents": [torch.zeros(1, height, width, 3) for height, width in reference_shapes],
        "sensenova_interleave": mode == "interleave",
    }]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]

    result = _run_prepared_generation(
        generation_mode=mode,
        clip=model, conditioning=positive, negative=negative, max_think_tokens=8,
    )
    metadata = result.args[0][0][1]
    conds = prepared_prefix(**metadata)
    ids = conds["text_input_ids"][0].tolist()

    assert result.args[1] == "K"
    assert result.args[3] is True
    assert ids.count(151669) == sum(height // 32 * (width // 32) for height, width in reference_shapes)
    assert ids.count(151671) == len(reference_shapes)
    assert ids.count(151644) == 3
    assert ids.count(151645) == 2
    assert text in tokenizer.decode(ids, skip_special_tokens=False)
    if mode == "interleave":
        assert ids[-2:] == generated
    else:
        assert conds["text_suffix"].tolist() == [[*generated, 271, 151670]]


@pytest.mark.parametrize("marker", [151669, 151671])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_interleave_resume_preserves_generated_image_marker_text(monkeypatch, marker, with_reference):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 3)
    model = _text_clip([marker, 42, 151670])
    diffusion = model.cond_stage_model

    diffusion.vision_model = PatchEmbedding(3, 3, kernel_size=32, stride=32)
    embedding = diffusion.language_model.model.embed_tokens.weight.detach()
    embedding[198].copy_(embedding[1])
    if marker == 151669:
        embedding[151671].copy_(embedding[1])
    clip = SenseNovaQwenTokenizer()
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]
    prefix = [1]
    if with_reference:
        positive[0][1]["reference_latents"] = [torch.ones(1, 32, 64, 3)]
        prefix.extend([151670, 151669, 151669, 151671, 198])
    first = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=positive, negative=negative, max_think_tokens=1,
    )
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[*prefix, marker]]
    second = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=first.args[0], negative=first.args[2], max_think_tokens=8,
    )
    metadata = second.args[0][0][1]
    assert second.args[3] is True
    assert metadata["text_input_ids"].tolist() == [[*prefix, marker, 42, 151670]]
    conds = prepared_prefix(**metadata)
    assert conds["text_input_ids"].tolist() == [[*prefix, marker, 42, 151670]]

    samples = {"samples": torch.ones(1, 3, 32, 32)}
    third = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=second.args[0], negative=second.args[2],
        samples=samples, max_think_tokens=8,
    )
    next_text = [marker, 42, 151670] if marker == 151669 else [42, 151670]
    assert third.args[0][0][1]["text_input_ids"].tolist() == [
        [*prefix, marker, 42, 151670, 151669, 151671, *next_text]
    ]
    assert third.args[2][0][1]["text_input_ids"].tolist() == [[9, 151670, 151669, 151671, 151670]]
    assert third.args[3] is True
    text, images, thinking = SenseNovaInterleaveDecode.execute(clip, third.args[0], samples=samples).args
    assert (text, thinking) == ("K<image>K<image>", "")
    assert len(images) == 2
    assert all(torch.equal(image, torch.ones(1, 32, 32, 3)) for image in images)


def test_sensenova_interleave_token_limit_can_resume_to_image(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_clip([42, 43, 151670])
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "sensenova_interleave": True}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "sensenova_interleave": True}]]
    first = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=positive, negative=negative, max_think_tokens=1,
    )
    assert first.args[3] is False
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[1, 42]]

    second = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=first.args[0], negative=first.args[2], max_think_tokens=8,
    )
    assert second.args[3] is True
    assert second.args[0][0][1]["text_input_ids"].tolist() == [[1, 42, 43, 151670]]
    assert second.args[2][0][1]["text_input_ids"].tolist() == [[9, 151670]]
    assert first.args[0][0][1]["text_input_ids"].tolist() == [[1, 42]]


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
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 2)
    model = _text_clip([42, 151670])
    model.cond_stage_model.vision_model = PatchEmbedding(3, 2, kernel_size=32, stride=32)
    clip = SenseNovaQwenTokenizer()

    first = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=positive, negative=negative,
        max_think_tokens=1,
    )
    assert first.args[3] is False
    second = _run_prepared_generation(
        generation_mode="interleave",
        clip=model, conditioning=first.args[0], negative=first.args[2],
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
    text, images, thinking = SenseNovaInterleaveDecode.execute(
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

    output = _run_prepared_generation(
        generation_mode="interleave",
        clip=_text_clip([42, 151670]),
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
    metadata = append_interleave_image(metadata, *prepare_interleave_image(torch.zeros(1, 3, 32, 64)))
    metadata = append_interleave_tokens(metadata, after, False)
    conditioning = [[torch.empty(1), metadata]]

    text, images, thinking = SenseNovaInterleaveDecode.execute(
        clip=tokenizer, conditioning=conditioning,
    ).args

    assert text == "answer <image1>"
    assert thinking == "plan<image1>\n\ninspect"
    assert len(images) == 1
    assert torch.equal(images[0], torch.full((1, 32, 64, 3), 0.5))
    assert len(metadata["reference_latents"]) == 2


@pytest.mark.parametrize("previous_images", [0, 1])
def test_sensenova_decode_requires_samples_for_pending_image(previous_images):
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]}, image_only=False,
    )
    for _ in range(previous_images):
        metadata = append_interleave_tokens(metadata, [151670], True)
        metadata = append_interleave_image(metadata, *prepare_interleave_image(torch.zeros(1, 3, 32, 64)))
    metadata = append_interleave_tokens(metadata, [151670], True)

    with pytest.raises(ValueError, match="SenseNova Interleave Decode.*samples.*KSampler"):
        SenseNovaInterleaveDecode.execute(
            clip=SenseNovaQwenTokenizer(), conditioning=[[torch.empty(1), metadata]],
        )

    assert len(metadata["reference_latents"]) == previous_images
    assert metadata["sensenova_interleave_pending_image"] is True


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
    metadata = append_interleave_image(metadata, *prepare_interleave_image(torch.zeros(1, 3, 32, 64)))
    metadata = append_interleave_tokens(metadata, between, True)
    conditioning = [[torch.empty(1), metadata]]
    samples = {"samples": torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 1, 1, 3).expand(1, 3, 1, 3).clone()}

    for _ in range(2):
        text, images, thinking = SenseNovaInterleaveDecode.execute(
            clip=tokenizer, conditioning=conditioning, samples=samples,
        ).args
        assert text == "before<image>between<image>"
        assert thinking == ""
        assert len(images) == 2
        assert torch.equal(images[0], torch.full((1, 32, 64, 3), 0.5))
        assert images[1].shape == (1, 1, 3, 3)
        assert images[1][0, 0, :, 0].tolist() == [0.0, 0.5, 1.0]

    assert len(metadata["reference_latents"]) == 1
    assert metadata["sensenova_interleave_pending_image"] is True

    assert samples["samples"][0, 0, 0].tolist() == [-2.0, 0.0, 2.0]


@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize("consumer", [SenseNovaConditioning, SenseNovaInterleaveDecode])
def test_sensenova_interleave_rejects_non_rgb_pixel_latents(channels, consumer):
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1], "sensenova_interleave": True},
        image_only=False,
    )
    metadata = append_interleave_tokens(metadata, [151670], True)
    conditioning = [[torch.empty(1), metadata]]
    kwargs = dict(
        clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
        samples={"samples": torch.zeros(1, channels, 32, 32)},
    )
    if consumer is SenseNovaConditioning:
        kwargs.update(clip=object(), negative=conditioning, max_think_tokens=8, generation_mode="interleave")

    with pytest.raises(ValueError, match="three-channel RGB pixel latent"):
        if consumer is SenseNovaConditioning:
            _run_prepared_generation(**kwargs)
        else:
            consumer.execute(**kwargs)
    assert metadata["reference_latents"] == []
    assert metadata["sensenova_interleave_pending_image"] is True


@pytest.mark.parametrize("with_stale_samples", [False, True])
@pytest.mark.parametrize(
    ("generated", "expected_text", "expected_thinking"),
    [
        ("answer", "answer", ""),
        (" K\n", " K\n", ""),
        ("\t\n", "\t\n", ""),
        ("<think>plan</think>answer", "answer", "plan"),
        ("<think></think>\n\n\tK ", "\n\n\tK ", ""),
        ("<think>K</think>\n\n\tL ", "\n\n\tL ", "K"),
        ("<think>plan</think>\n\n    answer  \n", "\n\n    answer  \n", "plan"),
        ("<think>plan</think>\t\n", "\t\n", "plan"),
        ("<think>unfinished", "", "unfinished"),
        ("plan</think>answer", "plan</think>answer", ""),
        ("<think>plan<img>inspect</think>answer", "answer", "plan\n\ninspect"),
        ("<think>plan</think>before<img>after <image1>", "before<image>after <image1>", "plan"),
        ("<think>plan</think>\n\n    before  <img>\n\tafter  \n", "\n\n    before  <image>\n\tafter  \n", "plan"),
        ("Use <think> tags to reason.", "Use <think> tags to reason.", ""),
        ("<img><think>quoted</think>", "<image><think>quoted</think>", ""),
        ("Use </think> to close the block.", "Use </think> to close the block.", ""),
        ("<think>plan</think>Use <think> and </think> tags.", "Use <think> and </think> tags.", "plan"),
        ("<think>Quote <think> and <image> literally.</think>answer", "answer", "Quote <think> and <image> literally."),
        ("<think>Use <image> as text.<img>inspect</think>answer", "answer", "Use <image> as text.\n\ninspect"),
        ("\n <think>plan</think>answer", "answer", "plan"),
        ("", "", ""),
    ],
)
def test_sensenova_decode_text_only_and_ignores_stale_image(
    generated, expected_text, expected_thinking, with_stale_samples,
):
    tokenizer = SenseNovaQwenTokenizer()
    metadata = expand_interleave_metadata(
        {"text_input_ids": _generation_input_ids()[:, :-1]},
        image_only=False,
    )
    token_ids = [int(value[0]) for value in tokenizer.tokenize_with_weights(
        generated, disable_weights=True
    )[0]]
    metadata = append_interleave_tokens(metadata, token_ids, False)

    text, images, thinking = SenseNovaInterleaveDecode.execute(
        clip=tokenizer, conditioning=[[torch.empty(1), metadata]],
        samples={"samples": torch.ones(1, 3, 32, 32)} if with_stale_samples else None,
    ).args

    assert text == expected_text
    assert thinking == expected_thinking
    assert len(images) == 1
    assert isinstance(images[0], ExecutionBlocker)
    assert images[0].message is None


def test_sensenova_decode_has_typed_inputs_and_image_list_output():
    schema = SenseNovaInterleaveDecode.GET_SCHEMA()
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
    samples = torch.tensor([[[[-2.0, 2.0]]]]).expand(1, 3, 1, 2)
    metadata = append_interleave_image(metadata, *prepare_interleave_image(samples))

    _, images, _ = SenseNovaInterleaveDecode.execute(
        clip=SenseNovaQwenTokenizer(), conditioning=[[torch.empty(1), metadata]],
    ).args

    assert images[0][0, 0, :, 0].tolist() == [0.0, 1.0]
    assert metadata["reference_latents"][0][0, 0, :, 0].tolist() == [-0.5, 1.5]


@pytest.mark.parametrize("task", [{"task": "self-correction"}, {"task": "interleave", "thinking": False}])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_image_thinking_can_start_interleave(native_sensenova_clip, monkeypatch, task, with_reference):
    clip = native_sensenova_clip
    positive = nodes.CLIPTextEncode().encode(clip, "Draw a red apple.")[0]
    negative = nodes.CLIPTextEncode().encode(clip, "")[0]
    positive[0][1]["strength"] = 0.7
    if with_reference:
        reference = torch.ones(1, 32, 32, 3)
        positive[0][1]["reference_latents"] = [reference]
        negative[0][1]["reference_latents"] = [reference]
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 2)
    model = _text_clip([42, 151668])
    diffusion = model.cond_stage_model
    diffusion.vision_model = PatchEmbedding(3, 2, kernel_size=32, stride=32)
    embeddings = diffusion.language_model.model.embed_tokens.weight.detach()
    for token in (198, 271, 151671):
        embeddings[token].copy_(embeddings[1])
    thought = SenseNovaConditioning.execute(
        clip=model, conditioning=positive,
        task={"task": "image thinking"}, max_tokens=8,
    ).args[0]
    original_ids = thought[0][1]["text_input_ids"].clone()
    boundary = thought[0][1]["sensenova_thinking_start"]
    head = diffusion.language_model.lm_head.weight.detach()
    head[151670].copy_(head[151668])
    head[151668].zero_()

    first = SenseNovaConditioning.execute(
        clip=model, conditioning=thought,
        task=task, negative=negative, max_tokens=8,
    ).args
    second = SenseNovaConditioning.execute(
        clip=model, conditioning=first[0], negative=first[2],
        task={**task, "samples": {"samples": torch.zeros(1, 3, 32, 32)}},
        max_tokens=8,
    ).args

    assert first[3] is second[3] is True
    for conditioning in (second[0], second[2]):
        metadata = conditioning[0][1]
        conds = prepared_prefix(**metadata)
        assert len(conds["reference_images"]) == int(with_reference) + 1
        assert "text_suffix" not in conds
        assert "sensenova_thinking_start" not in metadata
    assert second[0][0][1]["strength"] == 0.7
    assert thought[0][1]["sensenova_thinking_start"] == boundary
    assert torch.equal(thought[0][1]["text_input_ids"], original_ids)


@pytest.mark.parametrize("task", ["image thinking", "interleave", "self-correction"])
@pytest.mark.parametrize("finished", [False, True], ids=["empty", "finished"])
def test_sensenova_generate_noop_needs_no_model_or_clip(task, finished):
    conditioning = [[torch.empty(1), {"sensenova_interleave_finished": True}]] if finished else []
    negative = [[torch.empty(1), {}]]

    result = SenseNovaConditioning.execute(
        conditioning=conditioning, negative=negative, task={"task": task}, max_tokens=8,
    )

    assert result.args[0] is conditioning
    assert result.args[1] == ""
    assert result.args[2] is negative
    assert result.args[3] is False
    assert result.ui.as_dict() == {"text": ("",)}


def test_sensenova_generate_uses_clip_without_model():
    inputs = {value.id: value for value in SenseNovaConditioning.GET_SCHEMA().inputs}
    assert "model" not in inputs
    assert inputs["clip"].optional


def test_sensenova_generate_requires_clip_for_text():
    with pytest.raises(ValueError, match="Connect CLIP"):
        SenseNovaConditioning.execute(
            conditioning=[[torch.empty(1), {}]],
            task={"task": "image thinking"}, max_tokens=8,
        )


def test_sensenova_multi_turn_continues_scheduled_clip(native_sensenova_clip, monkeypatch):
    clip = native_sensenova_clip
    hook = comfy.hooks.WeightHook()
    hook.hook_keyframe.add(comfy.hooks.HookKeyframe(0.25))
    hook.hook_keyframe.add(comfy.hooks.HookKeyframe(0.75, start_percent=0.5))
    clip.patcher.forced_hooks = comfy.hooks.HookGroup()
    clip.patcher.forced_hooks.add(hook)
    clip.use_clip_schedule = True
    calls = []

    def encode(tokens):
        assert "past_prefix_keys" not in tokens
        ids = clip.cond_stage_model._input_ids(tokens)
        kv = torch.full((1, 1, ids.shape[1], 1), hook.strength)
        calls.append(hook.strength)
        return ids.unsqueeze(-1).float(), None, {"text_input_ids": ids, "prefix_keys": [kv], "prefix_values": [kv]}

    monkeypatch.setattr(clip.cond_stage_model, "encode_token_weights", encode)
    positive = nodes.CLIPTextEncode().encode(clip, "first")[0]
    negative = nodes.CLIPTextEncode().encode(clip, "")[0]
    for entries in (positive, negative):
        entries[0][1]["strength"] = 0.3
        entries[1][1]["strength"] = 0.9
    calls.clear()
    initial, _, neg_history, _ = SenseNovaConditioning.execute(
        clip=clip, conditioning=positive, negative=negative, task={"task": "multi-turn edit"}, max_tokens=1,
    ).args
    assert calls == [0.25, 0.75] * 2
    assert [m["strength"] for _, m in initial] == [0.3, 0.9]

    # The active schedule changes between turns. Only overlapping ranges contribute.
    hook.hook_keyframe.keyframes[1].start_percent = 0.75
    current = nodes.CLIPTextEncode().encode(clip, "second")[0]
    sampler_hooks = comfy.hooks.HookGroup()
    sampler_hooks.add(comfy.hooks.WeightHook())
    current[0][1].update(start_percent=0.1, end_percent=0.95, hooks=sampler_hooks)
    clip.apply_hooks_to_conds = clip.patcher.forced_hooks
    calls.clear()
    continued, _, neg, _ = SenseNovaConditioning.execute(
        clip=clip, conditioning=current, negative=neg_history,
        task={"task": "multi-turn edit", "previous": initial, "samples": {"samples": torch.zeros(1, 3, 32, 32)}},
        max_tokens=1,
    ).args
    assert calls == [0.25, 0.25, 0.75] * 2
    for entries, history in ((continued, initial), (neg, neg_history)):
        assert [(m["clip_start_percent"], m["clip_end_percent"]) for _, m in entries] == [
            (0.0, 0.5), (0.5, 0.75), (0.75, 1.0),
        ]
        for (_, meta), weight in zip(entries, (0.25, 0.25, 0.75), strict=True):
            assert meta["text_input_ids"].shape[1] > history[0][1]["text_input_ids"].shape[1]
            assert torch.all(meta["prefix_keys"][0] == weight)
            assert torch.all(meta["prefix_values"][0] == weight)
    assert [m["strength"] for _, m in neg] == [0.3, 0.9, 0.9]
    assert [m["hooks"] for _, m in continued] == [sampler_hooks, sampler_hooks, clip.apply_hooks_to_conds]
    assert all(m["hooks"] is clip.apply_hooks_to_conds for _, m in neg)
    assert [(m.get("start_percent"), m.get("end_percent")) for _, m in continued] == [
        (0.1, 0.95), (0.1, 0.95), (None, None),
    ]


def test_sensenova_multi_turn_empty_prompt_places_reference_in_user_message():
    metadata = {"text_input_ids": _tokenize_generation_prompt(""), "reference_latents": [torch.zeros(1, 32, 32, 3)]}
    expanded = expand_multi_turn_metadata(metadata, image_only=False)
    ids = expanded["text_input_ids"][0].tolist()
    decoded = SenseNovaQwenTokenizer().tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded == "<|im_start|>user\n<img><IMG_CONTEXT></img><|im_end|>\n<|im_start|>assistant\n<img>"


def test_sensenova_multi_turn_edit_builds_protocol_and_uses_incremental_cache():
    tokenizer = SenseNovaQwenTokenizer()

    class RecordingClip:
        def __init__(self):
            self.calls = []

        def encode_from_tokens_scheduled(self, tokens):
            self.calls.append(dict(tokens))
            ids = tokens["text_input_ids"]
            refs = tokens.get("reference_latents") or []
            seq_len = int(ids.shape[1])
            return [
                [
                    ids.unsqueeze(-1).float(),
                    {
                        "text_input_ids": ids,
                        "reference_latents": refs,
                        "prefix_keys": [torch.zeros((1, 8, seq_len, 128))],
                        "prefix_values": [torch.zeros((1, 8, seq_len, 128))],
                        "prefix_time": torch.tensor([seq_len]),
                        "sensenova_reference_count": len(refs),
                    },
                ]
            ]

    def encode_prompt(prompt):
        ids = torch.tensor(
            [[token for token, _ in tokenizer.tokenize_with_weights(
                f"<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>",
                disable_weights=True,
            )[0]]],
            dtype=torch.long,
        )
        return [[ids.unsqueeze(-1).float(), {"text_input_ids": ids, "sensenova_prompt": prompt}]]

    clip = RecordingClip()
    seed = torch.full((1, 32, 32, 3), 0.5)
    turn0_in = encode_prompt("turn one")
    turn0_in[0][1]["reference_latents"] = [seed]

    cond0, text0, neg0, req0 = SenseNovaConditioning.execute(
        clip=clip,
        conditioning=turn0_in,
        task={"task": "multi-turn edit", "previous": None, "samples": None},
        max_tokens=16,
    ).args
    assert text0 == "" and neg0 == [] and req0 is True
    assert len(clip.calls) == 1
    assert "past_prefix_keys" not in clip.calls[0]
    decoded0 = tokenizer.tokenizer.decode(cond0[0][1]["text_input_ids"][0].tolist(), skip_special_tokens=False)
    assert "<img><IMG_CONTEXT></img>turn one<|im_end|>" in decoded0

    # Inject a stale thinking key to verify narrow metadata boundary strips it
    cond0[0][1]["sensenova_thinking_start"] = 3
    samples0 = {"samples": torch.tensor([[[[-2.0, 2.0]]]]).repeat(1, 3, 32, 16)}
    cond1, text1, neg1, req1 = SenseNovaConditioning.execute(
        clip=clip,
        conditioning=encode_prompt("turn two"),
        task={"task": "multi-turn edit", "previous": cond0, "samples": samples0},
        max_tokens=16,
    ).args
    assert text1 == "" and neg1 == [] and req1 is True
    assert len(clip.calls) == 2
    assert "past_prefix_keys" in clip.calls[1]
    assert "sensenova_thinking_start" not in cond1[0][1]
    assert cond1[0][1]["reference_latents"][1].min().item() == 0.0
    assert cond1[0][1]["reference_latents"][1].max().item() == 1.0
    decoded1 = tokenizer.tokenizer.decode(cond1[0][1]["text_input_ids"][0].tolist(), skip_special_tokens=False)
    assert decoded1.endswith(
        "<img><IMG_CONTEXT></img><|im_end|>\n<|im_start|>user\nturn two<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
    )


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_sensenova_incremental_prefix_matches_full_prefill(monkeypatch, device):
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 32)
    monkeypatch.setattr(sensenova_model, "INTERMEDIATE_SIZE", 64)
    monkeypatch.setattr(sensenova_model, "NUM_LAYERS", 2)
    monkeypatch.setattr(sensenova_model, "NUM_HEADS", 2)
    monkeypatch.setattr(sensenova_model, "NUM_KV_HEADS", 1)
    monkeypatch.setattr(sensenova_model, "HEAD_DIM", 16)

    torch.manual_seed(0)
    text_model = sensenova_model.SenseNovaTextModel(
        has_lm_head=False, device=device, dtype=torch.float32, operations=comfy.ops.manual_cast
    )
    text_model.vision_model = PatchEmbedding(3, 32, kernel_size=32, stride=32, device=device)
    for name, param in text_model.named_parameters():
        if 'norm' in name:
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.normal_(param, std=0.05)
    text_model.eval()

    img0 = torch.rand(1, 32, 32, 3, device=device)
    img1 = torch.rand(1, 32, 32, 3, device=device)
    turn0_ids = torch.tensor([[IM_START_ID, 872, NEWLINE_ID, 151670, IMAGE_CONTEXT_ID, IMAGE_END_ID, 100, IM_END_ID, NEWLINE_ID, IM_START_ID, 77091, NEWLINE_ID, 151670]], dtype=torch.long, device=device)
    suffix_ids = torch.tensor([[IMAGE_CONTEXT_ID, IMAGE_END_ID, IM_END_ID, NEWLINE_ID, IM_START_ID, 872, NEWLINE_ID, 200, IM_END_ID, NEWLINE_ID, IM_START_ID, 77091, NEWLINE_ID, 151670]], dtype=torch.long, device=device)
    full_ids = torch.cat((turn0_ids, suffix_ids), dim=1)

    ids0, refs0, idx0, mask0 = prepare_prefix_inputs(turn0_ids, [img0], 32, torch.float32, expand_images=False)
    k0, v0, t0 = text_model.preprocess_prefix(ids0, refs0, idx0, mask0, compute_dtype=torch.float32, output_device="cpu")
    assert all(t.device.type == "cpu" for t in [*k0, *v0])

    s_ids, s_refs, s_idx, s_mask = prepare_incremental_prefix_inputs(
        suffix_ids, [img1], 32, torch.float32, past_length=int(k0[0].shape[2]), start_time=int(t0[0].item())
    )
    k_inc, v_inc, t_inc = text_model.preprocess_prefix(
        s_ids, s_refs, s_idx, s_mask, compute_dtype=torch.float32, past_keys=k0, past_values=v0
    )

    f_ids, f_refs, f_idx, f_mask = prepare_prefix_inputs(full_ids, [img0, img1], 32, torch.float32, expand_images=False)
    k_full, v_full, t_full = text_model.preprocess_prefix(f_ids, f_refs, f_idx, f_mask, compute_dtype=torch.float32)

    assert torch.equal(t_inc, t_full)
    for layer_idx in range(len(k_full)):
        torch.testing.assert_close(k_inc[layer_idx], k_full[layer_idx], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_inc[layer_idx], v_full[layer_idx], atol=1e-5, rtol=1e-5)

    # Also verify incremental vs full prefill when the new turn adds an extra reference image
    img2 = torch.rand(1, 32, 32, 3, device=device)
    suffix_with_ref = torch.tensor([[IMAGE_CONTEXT_ID, IMAGE_END_ID, IM_END_ID, NEWLINE_ID, IM_START_ID, 872, NEWLINE_ID, IMAGE_START_ID, IMAGE_CONTEXT_ID, IMAGE_END_ID, 200, IM_END_ID, NEWLINE_ID, IM_START_ID, 77091, NEWLINE_ID, IMAGE_START_ID]], dtype=torch.long, device=device)
    full_with_ref = torch.cat((turn0_ids, suffix_with_ref), dim=1)
    s2_ids, s2_refs, s2_idx, s2_mask = prepare_incremental_prefix_inputs(
        suffix_with_ref, [img1, img2], 32, torch.float32, past_length=int(k0[0].shape[2]), start_time=int(t0[0].item())
    )
    k_inc2, v_inc2, t_inc2 = text_model.preprocess_prefix(
        s2_ids, s2_refs, s2_idx, s2_mask, compute_dtype=torch.float32, past_keys=k0, past_values=v0
    )
    f2_ids, f2_refs, f2_idx, f2_mask = prepare_prefix_inputs(full_with_ref, [img0, img1, img2], 32, torch.float32, expand_images=False)
    k_full2, v_full2, t_full2 = text_model.preprocess_prefix(f2_ids, f2_refs, f2_idx, f2_mask, compute_dtype=torch.float32)
    assert torch.equal(t_inc2, t_full2)
    for layer_idx in range(len(k_full2)):
        torch.testing.assert_close(k_inc2[layer_idx], k_full2[layer_idx], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_inc2[layer_idx], v_full2[layer_idx], atol=1e-5, rtol=1e-5)

    # Verify streaming output_device="cpu" matches default incremental output
    k_stream, v_stream, t_stream = text_model.preprocess_prefix(
        s2_ids, s2_refs, s2_idx, s2_mask, compute_dtype=torch.float32,
        past_keys=k0, past_values=v0, output_device=torch.device("cpu"),
    )
    assert torch.equal(t_stream, t_full2)
    for layer_idx in range(len(k_full2)):
        assert k_stream[layer_idx].device == torch.device("cpu")
        assert v_stream[layer_idx].device.type == "cpu"
        torch.testing.assert_close(k_stream[layer_idx], k_full2[layer_idx].cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_stream[layer_idx], v_full2[layer_idx].cpu(), atol=1e-5, rtol=1e-5)


def test_sensenova_multi_turn_edit_handles_empty_prompt_new_refs_and_clip_mismatch():
    tokenizer = SenseNovaQwenTokenizer()

    class PatcherStub:
        def __init__(self, patches_uuid):
            self.model = object()
            self.patches_uuid = patches_uuid
            self.forced_hooks = None
            self.injections = {}

    class RecordingClip:
        def __init__(self, cond_stage_model=None, patches_uuid=None):
            self.calls = []
            if patches_uuid is not None:
                self.cond_stage_model = cond_stage_model
                self.patcher = PatcherStub(patches_uuid)
                self.use_clip_schedule = False
                self.layer_idx = None

        def encode_from_tokens_scheduled(self, tokens):
            self.calls.append(dict(tokens))
            ids = tokens["text_input_ids"]
            refs = tokens.get("reference_latents") or []
            seq_len = int(ids.shape[1])
            return [
                [
                    ids.unsqueeze(-1).float(),
                    {
                        "text_input_ids": ids,
                        "reference_latents": refs,
                        "prefix_keys": [torch.zeros((1, 8, seq_len, 128))],
                        "prefix_values": [torch.zeros((1, 8, seq_len, 128))],
                        "prefix_time": torch.tensor([seq_len]),
                        "sensenova_reference_count": len(refs),
                    },
                ]
            ]

    def encode_prompt(prompt, include_system=True):
        text = (
            f"<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
            if include_system
            else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
        )
        ids = torch.tensor(
            [[token for token, _ in tokenizer.tokenize_with_weights(text, disable_weights=True)[0]]],
            dtype=torch.long,
        )
        return [[ids.unsqueeze(-1).float(), {"text_input_ids": ids, "sensenova_prompt": prompt}]]

    # 1. Empty prompt without system message keeps the user turn in multi_turn_suffix_ids
    empty_turn_ids = encode_prompt("", include_system=False)[0][1]["text_input_ids"]
    suffix_empty = multi_turn_suffix_ids(empty_turn_ids, [IMAGE_CONTEXT_ID, IMAGE_END_ID])
    decoded_empty = tokenizer.tokenizer.decode(suffix_empty[0].tolist(), skip_special_tokens=False)
    assert decoded_empty == "<IMG_CONTEXT></img><|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"

    shared_model = object()
    clip_a = RecordingClip(cond_stage_model=shared_model, patches_uuid="uuid-1")
    seed = torch.full((1, 32, 32, 3), 0.5)
    turn0_in = encode_prompt("turn one")
    turn0_in[0][1]["reference_latents"] = [seed]
    neg0_in = encode_prompt("", include_system=False)
    neg0_in[0][1]["reference_latents"] = [seed]
    mask = torch.ones(1, 4, 4)
    turn0_in[0][1].update(mask=mask, area=(4, 4, 0, 0), strength=0.25)
    neg0_in[0][1]["strength"] = 0.5

    cond0, _, neg0, _ = SenseNovaConditioning.execute(
        clip=clip_a,
        conditioning=turn0_in,
        negative=neg0_in,
        task={"task": "multi-turn edit", "previous": None, "samples": None},
        max_tokens=16,
    ).args

    assert cond0[0][1]["mask"] is mask and cond0[0][1]["area"] == (4, 4, 0, 0)
    assert cond0[0][1]["strength"] == 0.25 and neg0[0][1]["strength"] == 0.5
    assert "mask" not in clip_a.calls[0] and "strength" not in clip_a.calls[1]

    # 2. Current turn new reference image is expanded and placed after previous sampler output;
    #    negative branch does not duplicate previous reference images.
    samples0 = {"samples": torch.full((1, 3, 32, 32), 0.25)}
    new_ref = torch.full((1, 32, 32, 3), 0.75)
    turn1_in = encode_prompt("add this hat")
    turn1_in[0][1].update(reference_latents=[new_ref], area=(2, 2, 1, 1), strength=0.75)
    cond1, _, neg1, _ = SenseNovaConditioning.execute(
        clip=clip_a,
        conditioning=turn1_in,
        negative=neg0,
        task={"task": "multi-turn edit", "previous": cond0, "samples": samples0},
        max_tokens=16,
    ).args
    assert cond1[0][1]["area"] == (2, 2, 1, 1) and cond1[0][1]["strength"] == 0.75
    assert "mask" not in cond1[0][1] and neg1[0][1]["strength"] == 0.5
    assert "area" not in clip_a.calls[2] and "strength" not in clip_a.calls[3]
    assert "past_prefix_keys" in clip_a.calls[2]
    assert len(cond1[0][1]["reference_latents"]) == 3
    assert torch.equal(cond1[0][1]["reference_latents"][2], new_ref)
    assert len(clip_a.calls[2]["suffix_reference_latents"]) == 2
    assert torch.equal(clip_a.calls[2]["suffix_reference_latents"][1], new_ref)
    assert len(neg1[0][1]["reference_latents"]) == 2
    decoded1 = tokenizer.tokenizer.decode(cond1[0][1]["text_input_ids"][0].tolist(), skip_special_tokens=False)
    assert decoded1.endswith(
        "<img><IMG_CONTEXT></img><|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>add this hat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
    )

    # 3. Same model with different patches_uuid (e.g. LoRA) or different CLIP falls back to full prefill
    clip_patched = RecordingClip(cond_stage_model=shared_model, patches_uuid="uuid-2")
    SenseNovaConditioning.execute(
        clip=clip_patched,
        conditioning=encode_prompt("turn two"),
        task={"task": "multi-turn edit", "previous": cond0, "samples": samples0},
        max_tokens=16,
    )
    assert "past_prefix_keys" not in clip_patched.calls[0]

    # Same model + same patches_uuid (unmodified clone) reuses cache
    clip_clone = RecordingClip(cond_stage_model=shared_model, patches_uuid="uuid-1")
    SenseNovaConditioning.execute(
        clip=clip_clone,
        conditioning=encode_prompt("turn two"),
        task={"task": "multi-turn edit", "previous": cond0, "samples": samples0},
        max_tokens=16,
    )
    assert "past_prefix_keys" in clip_clone.calls[0]

    def resume(clip, previous=cond0, negative=None):
        return SenseNovaConditioning.execute(
            clip=clip, conditioning=turn1_in, negative=negative,
            task={"task": "multi-turn edit", "previous": previous, "samples": samples0}, max_tokens=16,
        ).args

    # A text-ended history is invalid both with the original CLIP and on a cache miss.
    prev_ids = cond0[0][1]["text_input_ids"].clone()
    prev_ids[0, -1] = IM_END_ID
    text_history = [[cond0[0][0], {**cond0[0][1], "text_input_ids": prev_ids}]]
    for clip in (clip_a, clip_patched):
        with pytest.raises(ValueError, match="ends with an image request"):
            resume(clip, previous=text_history)

    for attr, owner, active, inactive in (("forced_hooks", clip_a.patcher, object(), None),
                                          ("injections", clip_a.patcher, {"bypass_lora": [object()]}, {}),
                                          ("use_clip_schedule", clip_a, True, False)):
        setattr(owner, attr, active)
        hooked, *_ = resume(clip_a)
        assert "past_prefix_keys" not in clip_a.calls[-1]
        setattr(owner, attr, inactive)
        resume(clip_a, previous=hooked)
        assert "past_prefix_keys" not in clip_a.calls[-1]

    # Attaching a batch to encoded negative conditioning adds it to the next user message.
    extra = torch.rand(2, 32, 32, 3)
    _, neg_with_refs = HiDreamO1ReferenceImages.execute(positive=cond0, negative=neg0, images={"image_1": extra}).args
    _, _, edited_negative, _ = resume(clip_a, negative=neg_with_refs)
    neg_meta = edited_negative[0][1]
    assert "past_prefix_keys" not in clip_a.calls[-1]
    assert neg_meta["sensenova_reference_count"] == 4
    torch.testing.assert_close(torch.cat(neg_meta["reference_latents"][2:]), extra)
    neg_ids = neg_meta["text_input_ids"]
    decoded = tokenizer.tokenizer.decode(neg_ids[0].tolist(), skip_special_tokens=False)
    assert decoded.endswith("<|im_start|>user\n<img><IMG_CONTEXT></img><img><IMG_CONTEXT></img><|im_end|>\n<|im_start|>assistant\n<img>")
    prepare_prefix_inputs(neg_ids, neg_meta["reference_latents"], 32, torch.float32, expand_images=False)
    resume(clip_a, negative=edited_negative)
    assert "past_prefix_keys" in clip_a.calls[-1]

    replaced = [[neg0[0][0], {**neg0[0][1], "reference_latents": [new_ref]}]]
    resume(clip_a, negative=replaced)
    assert "past_prefix_keys" not in clip_a.calls[-1]
