from types import SimpleNamespace

import pytest
import torch

from comfy import ops
from comfy.text_encoders import qwen_vl
from comfy.text_encoders.llama import Qwen25_7BVLI, Qwen25_7BVLI_Config
from comfy.text_encoders.qwen_image import Qwen25_7BVLIModel, QwenImageTokenizer


def test_qwen25vl_generation_config_has_stop_tokens_and_untied_head():
    config = Qwen25_7BVLI_Config()

    assert config.stop_tokens == [151643, 151645]
    assert config.lm_head is True


def test_qwen25vl_untied_head_loads_normalized_checkpoint_key(monkeypatch):
    config = Qwen25_7BVLI_Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    monkeypatch.setattr(
        qwen_vl, "Qwen2VLVisionTransformer", lambda **kwargs: torch.nn.Identity()
    )
    model = Qwen25_7BVLI(
        config.__dict__, device="cpu", dtype=torch.float32, operations=torch.nn
    )
    checkpoint_head = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loaded = model.load_state_dict(
        {"model.lm_head.weight": checkpoint_head}, strict=False
    )

    assert not loaded.unexpected_keys
    assert torch.equal(model.model.lm_head.weight, checkpoint_head)


def test_qwen_image_tokenizer_preserves_textgenerate_image():
    image = torch.zeros((1, 4, 4, 3))
    seen = {}

    def tokenize_with_weights(text, return_word_ids=False, **kwargs):
        seen["text"] = text
        return [[(151655, 1.0, 0)]]

    tokenizer = object.__new__(QwenImageTokenizer)
    tokenizer.clip_name = "qwen25_7b"
    tokenizer.clip = "clip_qwen25_7b"
    tokenizer.clip_qwen25_7b = SimpleNamespace(
        tokenize_with_weights=tokenize_with_weights
    )
    tokenizer.llama_template = "diffusion text: {}"
    tokenizer.llama_template_images = "diffusion image: {}"

    tokens = tokenizer.tokenize_with_weights(
        "describe this", image=image, thinking=False
    )

    assert "<|vision_start|><|image_pad|><|vision_end|>" in seen["text"]
    assert "describe this" in seen["text"]
    image_embed = tokens["qwen25_7b"][0][0][0]
    assert image_embed["type"] == "image"
    assert torch.equal(image_embed["data"], image)


def test_qwen_image_tokenizer_honors_explicit_template_with_thinking_flag():
    seen = {}

    def tokenize_with_weights(text, return_word_ids=False, **kwargs):
        seen["text"] = text
        return [[(1, 1.0, 0)]]

    tokenizer = object.__new__(QwenImageTokenizer)
    tokenizer.clip_name = "qwen25_7b"
    tokenizer.clip = "clip_qwen25_7b"
    tokenizer.clip_qwen25_7b = SimpleNamespace(
        tokenize_with_weights=tokenize_with_weights
    )
    tokenizer.llama_template = "diffusion text: {}"
    tokenizer.llama_template_images = "diffusion image: {}"

    tokenizer.tokenize_with_weights(
        "prompt", llama_template="custom: {}", thinking=False
    )

    assert seen["text"] == "custom: prompt"


def test_qwen_image_tokenizer_inserts_image_into_custom_user_turn():
    image = torch.zeros((1, 4, 4, 3))
    seen = {}

    def tokenize_with_weights(text, return_word_ids=False, **kwargs):
        seen["text"] = text
        return [[(151655, 1.0, 0)]]

    tokenizer = object.__new__(QwenImageTokenizer)
    tokenizer.clip_name = "qwen25_7b"
    tokenizer.clip = "clip_qwen25_7b"
    tokenizer.clip_qwen25_7b = SimpleNamespace(
        tokenize_with_weights=tokenize_with_weights
    )
    tokenizer.llama_template = "diffusion text: {}"
    tokenizer.llama_template_images = "diffusion image: {}"

    tokenizer.tokenize_with_weights(
        "<|im_start|>user\nlook closely", image=image, skip_template=True
    )

    assert seen["text"] == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>look closely"
    )


def test_qwen_image_tokenizer_adds_missing_custom_template_image_placeholders():
    images = torch.zeros((2, 4, 4, 3))
    seen = {}

    def tokenize_with_weights(text, return_word_ids=False, **kwargs):
        seen["text"] = text
        return [[(151655, 1.0, 0), (151655, 1.0, 0)]]

    tokenizer = object.__new__(QwenImageTokenizer)
    tokenizer.clip_name = "qwen25_7b"
    tokenizer.clip = "clip_qwen25_7b"
    tokenizer.clip_qwen25_7b = SimpleNamespace(
        tokenize_with_weights=tokenize_with_weights
    )

    tokens = tokenizer.tokenize_with_weights(
        "inspect both", llama_template="custom: {}", image=images
    )
    embedded_images = [
        item[0]
        for item in tokens["qwen25_7b"][0]
        if isinstance(item[0], dict) and item[0].get("type") == "image"
    ]

    assert seen["text"].count("<|image_pad|>") == 2
    assert len(embedded_images) == 2
    assert torch.equal(embedded_images[0]["data"], images[0:1])
    assert torch.equal(embedded_images[1]["data"], images[1:2])


def test_qwen_image_tokenizer_rejects_unmatched_image_placeholder():
    tokenizer = object.__new__(QwenImageTokenizer)

    with pytest.raises(
        ValueError, match="more image placeholders than supplied images"
    ):
        tokenizer.tokenize_with_weights(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>describe",
            skip_template=True,
        )


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_qwen_image_tokenizer_rejects_unsupported_modalities(modality):
    tokenizer = object.__new__(QwenImageTokenizer)

    with pytest.raises(NotImplementedError, match="does not support video or audio"):
        tokenizer.tokenize_with_weights("prompt", **{modality: object()})


def test_qwen_image_tokenizer_encodes_textgenerate_image_with_local_tokenizer():
    image = torch.zeros((1, 4, 4, 3))
    tokenizer = QwenImageTokenizer()

    tokens = tokenizer.tokenize_with_weights(
        "describe this", image=image, thinking=False
    )
    image_embeds = [
        item[0] for item in tokens["qwen25_7b"][0] if isinstance(item[0], dict)
    ]

    assert len(image_embeds) == 1
    assert image_embeds[0]["type"] == "image"
    assert torch.equal(image_embeds[0]["data"], image)


def test_qwen_image_tokenizer_keeps_diffusion_template():
    seen = {}

    def tokenize_with_weights(text, return_word_ids=False, **kwargs):
        seen["text"] = text
        return [[(151655, 1.0, 0)]]

    tokenizer = object.__new__(QwenImageTokenizer)
    tokenizer.clip_name = "qwen25_7b"
    tokenizer.clip = "clip_qwen25_7b"
    tokenizer.clip_qwen25_7b = SimpleNamespace(
        tokenize_with_weights=tokenize_with_weights
    )
    tokenizer.llama_template = "diffusion text: {}"
    tokenizer.llama_template_images = "diffusion image: {}"

    tokens = tokenizer.tokenize_with_weights(
        "prompt", images=[torch.zeros((1, 4, 4, 3))]
    )

    assert seen["text"] == (
        "<|vision_start|><|image_pad|><|vision_end|>diffusion image: prompt"
    )
    assert tokens["qwen25_7b"][0][0][0]["type"] == "image"


def test_qwen25vl_exposes_mrope_position_builder_for_generation():
    model = object.__new__(Qwen25_7BVLI)

    position_ids = model.build_position_ids(
        torch.zeros((1, 9, 8)),
        [{"type": "image", "index": 1, "size": 6, "extra": torch.tensor([[1, 4, 6]])}],
    )

    assert position_ids.shape == (3, 9)
    assert position_ids[:, 1:7].tolist() == [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 2],
        [1, 2, 3, 1, 2, 3],
    ]
    assert position_ids[:, 7:].tolist() == [[4, 5], [4, 5], [4, 5]]


def test_qwen25vl_generation_uses_mrope_then_stops_on_eos(monkeypatch):
    monkeypatch.setattr(Qwen25_7BVLI_Config, "head_dim", 8)
    monkeypatch.setattr(Qwen25_7BVLI_Config, "rope_dims", [1, 1, 2])
    grid = torch.tensor([[1, 2, 2]])
    monkeypatch.setattr(
        qwen_vl,
        "process_qwen2vl_images",
        lambda image: (torch.ones((1, 8)), grid),
    )

    class VisionStub(torch.nn.Module):
        def forward(self, image, image_grid):
            return image

    monkeypatch.setattr(
        qwen_vl, "Qwen2VLVisionTransformer", lambda **kwargs: VisionStub()
    )
    config = {
        "vocab_size": 151646,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
    }
    transformer = Qwen25_7BVLI(
        config, device="cpu", dtype=torch.float32, operations=ops.disable_weight_init
    )
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad_(False)
        parameter.fill_(1.0 if name.endswith("norm.weight") else 0.0)
    clip_model = object.__new__(Qwen25_7BVLIModel)
    torch.nn.Module.__init__(clip_model)
    clip_model.transformer = transformer
    clip_model.execution_device = torch.device("cpu")
    clip_model.special_tokens = {"pad": 151643}
    positions = []
    original_compute_freqs = transformer.model.compute_freqs_cis

    def record_positions(position_ids, device):
        positions.append(position_ids.clone())
        return original_compute_freqs(position_ids, device)

    transformer.model.compute_freqs_cis = record_positions
    generated_ids = iter((1, 151643))

    def logits(_hidden):
        token_id = next(generated_ids)
        result = torch.full((1, 1, config["vocab_size"]), -1.0)
        result[..., token_id] = 1.0
        return result

    transformer.logits = logits
    image = {"type": "image", "data": torch.zeros((1, 2, 2, 3))}
    tokens = {"qwen25_7b": [[(image, 1.0, 0)]]}

    result = clip_model.generate(tokens, do_sample=False, max_length=4)

    assert result == [1, 151643]
    assert [position.tolist() for position in positions] == [[[0], [0], [0]], [[1]]]

    generated_ids = iter((1, 151643))
    result = clip_model.generate(tokens, do_sample=True, top_k=1, max_length=4)

    assert result == [1, 151643]


def test_qwen25vl_generation_rejects_masked_prompt_embeddings():
    clip_model = object.__new__(Qwen25_7BVLIModel)
    torch.nn.Module.__init__(clip_model)
    clip_model.execution_device = torch.device("cpu")
    clip_model.process_tokens = lambda tokens, device: (
        torch.zeros((1, 2, 4)),
        torch.tensor([[1, 0]]),
        [1],
        [],
    )

    with pytest.raises(ValueError, match="does not support padded prompt embeddings"):
        clip_model.generate({"qwen25_7b": [[(1, 1.0, 0)]]})


def test_qwen25vl_forward_keeps_generation_position_ids():
    model = object.__new__(Qwen25_7BVLI)
    torch.nn.Module.__init__(model)
    model.model = lambda input_ids, *args, **kwargs: kwargs
    position_ids = torch.tensor([[0], [0], [0]])

    result = model.forward(
        None, embeds=torch.zeros((1, 1, 4)), position_ids=position_ids
    )

    assert result["position_ids"] is position_ids
