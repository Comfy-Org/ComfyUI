import torch
import pytest

import comfy.sd
import comfy.text_encoders.flux as flux
import comfy.text_encoders.qwen3vl as qwen3vl


@pytest.mark.parametrize(
    ("model_type", "hidden_size", "vision_hidden_size"),
    (("qwen3vl_4b", 2560, 1024), ("qwen3vl_8b", 4096, 1152)),
)
def test_klein_qwen3vl_keeps_klein_language_config(monkeypatch, model_type, hidden_size, vision_hidden_size):
    captured = {}

    class DummyLanguage(torch.nn.Module):
        def __init__(self, config, device, dtype, ops):
            super().__init__()
            self.config = config
            captured["language_config"] = config

    class DummyVisual(torch.nn.Module):
        def __init__(self, config, device, dtype, ops):
            super().__init__()
            captured["vision_config"] = config

    monkeypatch.setattr(flux.comfy.text_encoders.llama, "Llama2_", DummyLanguage)
    monkeypatch.setattr(qwen3vl, "Qwen3VLVisionModel", DummyVisual)

    model = flux._make_klein_qwen3vl_model(model_type)({}, torch.float32, "cpu", object())

    assert isinstance(model.model, DummyLanguage)
    assert isinstance(model.visual, DummyVisual)
    assert captured["language_config"].hidden_size == hidden_size
    assert captured["language_config"].rope_dims is None
    assert not getattr(captured["language_config"], "interleaved_mrope", False)
    assert captured["language_config"].rope_theta == 1000000.0
    assert captured["vision_config"]["hidden_size"] == vision_hidden_size
    assert captured["vision_config"]["out_hidden_size"] == hidden_size


def test_klein_qwen3vl_uses_merged_visual_embeddings_without_metadata(monkeypatch):
    merged = torch.empty((1, 4, 2560))
    deepstack = [torch.empty((4, 2560))]

    class DummyVisual(torch.nn.Module):
        def forward(self, image, grid):
            return merged, deepstack

    monkeypatch.setattr(
        flux.comfy.text_encoders.qwen_vl,
        "process_qwen2vl_images",
        lambda image, **kwargs: (torch.empty((1, 3, 16, 16)), [(1, 1, 1)]),
    )

    model_class = flux._make_klein_qwen3vl_model("qwen3vl_4b")
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.visual = DummyVisual()

    result, extra = model.preprocess_embed(
        {"type": "image", "data": torch.empty((1, 32, 32, 3))},
        "cpu",
    )

    assert result is merged
    assert extra is None


@pytest.mark.parametrize(
    ("old_tokenizer", "new_tokenizer"),
    ((flux.KleinTokenizer, flux.KleinVLTokenizer), (flux.KleinTokenizer8B, flux.KleinVLTokenizer8B)),
)
def test_klein_vl_text_template_matches_existing_klein(old_tokenizer, new_tokenizer):
    prompt = "a small red cube"
    old_tokens = next(iter(old_tokenizer().tokenize_with_weights(prompt).values()))
    new_tokens = next(iter(new_tokenizer().tokenize_with_weights(prompt).values()))
    assert old_tokens == new_tokens


@pytest.mark.parametrize(
    ("old_tokenizer", "new_tokenizer"),
    ((flux.KleinTokenizer, flux.KleinVLTokenizer), (flux.KleinTokenizer8B, flux.KleinVLTokenizer8B)),
)
def test_klein_vl_text_template_override_matches_existing_klein(old_tokenizer, new_tokenizer):
    prompt = "a small red cube"
    template = "<|im_start|>user\nDescribe this: {}<|im_end|>\n<|im_start|>assistant\n"
    old_tokens = next(iter(old_tokenizer().tokenize_with_weights(prompt, False, template).values()))
    new_tokens = next(iter(new_tokenizer().tokenize_with_weights(prompt, False, template).values()))
    assert old_tokens == new_tokens


def test_klein_vl_custom_template_owns_image_block_placement():
    image = torch.zeros((1, 32, 32, 3))
    prompt = f"Picture 1: {flux.KLEIN_VL_IMAGE_BLOCK}describe"
    tokens = flux.KleinVLTokenizer().tokenize_with_weights(prompt, False, "{}", images=[image])
    rows = next(iter(tokens.values()))
    image_entries = [token for row in rows for token, _ in row if isinstance(token, dict)]
    image_pad_tokens = [token for row in rows for token, _ in row if token == 151655]
    assert len(image_entries) == 1
    assert image_pad_tokens == []


def test_klein_vl_tokenizer_exposes_native_image_entry():
    tokens = flux.KleinVLTokenizer().tokenize_with_weights("describe", images=[torch.zeros((1, 32, 32, 3))])
    assert list(tokens) == ["qwen3_4b"]
    rows = next(iter(tokens.values()))
    image_entries = [token for row in rows for token, _ in row if isinstance(token, dict)]
    assert len(image_entries) == 1
    assert image_entries[0]["type"] == "image"


def test_klein_qwen3vl_pads_after_visual_expansion(monkeypatch):
    clip_model = flux.KleinQwen3VLClipModel.__new__(flux.KleinQwen3VLClipModel)
    torch.nn.Module.__init__(clip_model)
    clip_model.special_tokens = {"pad": 151643}

    class DummyEmbeddings:
        def __call__(self, tokens, out_dtype):
            return torch.zeros((*tokens.shape, 2560), device=tokens.device, dtype=out_dtype)

    class DummyTransformer:
        def get_input_embeddings(self):
            return DummyEmbeddings()

    clip_model.transformer = DummyTransformer()
    monkeypatch.setattr(
        flux.sd1_clip.SDClipModel,
        "process_tokens",
        lambda self, tokens, device: (
            torch.ones((1, 19, 2560)),
            torch.ones((1, 19), dtype=torch.long),
            [19],
            [{"type": "image", "index": 1, "size": 4, "extra": None}],
        ),
    )

    embeds, attention_mask, num_tokens, embeds_info = clip_model.process_tokens([], "cpu")

    assert embeds.shape == (1, 512, 2560)
    assert attention_mask.shape == (1, 512)
    assert attention_mask[:, :19].all()
    assert not attention_mask[:, 19:].any()
    assert num_tokens == [19]
    assert embeds_info == [{"type": "image", "index": 1, "size": 4, "extra": None}]


@pytest.mark.parametrize(
    ("width", "model_type", "tokenizer"),
    ((2560, "qwen3vl_4b", flux.KleinVLTokenizer), (4096, "qwen3vl_8b", flux.KleinVLTokenizer8B)),
)
def test_flux2_qwen3vl_loader_keeps_visual_weights(monkeypatch, width, model_type, tokenizer):
    captured = {}
    sentinel_clip = object()

    def fake_klein_vl_te(**kwargs):
        captured["factory_kwargs"] = kwargs
        return sentinel_clip

    class FakeCLIP:
        def __init__(self, target, **kwargs):
            captured["target"] = target
            captured["state_dict"] = kwargs["state_dict"][0]

    monkeypatch.setattr(flux, "klein_vl_te", fake_klein_vl_te)
    monkeypatch.setattr(comfy.sd, "CLIP", FakeCLIP)
    monkeypatch.setattr(
        comfy.sd.comfy.text_encoders.long_clipl,
        "model_options_long_clip",
        lambda state_dict, tokenizer_data, model_options: (tokenizer_data, model_options),
    )

    state_dict = {
        "model.language_model.norm.weight": torch.empty(width),
        "model.visual.deepstack_merger_list.0.norm.weight": torch.empty(1),
        "model.visual.merger.linear_fc2.weight": torch.empty((width, 1)),
    }
    comfy.sd.load_text_encoder_state_dicts([state_dict], clip_type=comfy.sd.CLIPType.FLUX2)

    assert captured["target"].clip is sentinel_clip
    assert captured["target"].tokenizer is tokenizer
    assert captured["factory_kwargs"]["model_type"] == model_type
    assert "model.norm.weight" in captured["state_dict"]
    assert "visual.deepstack_merger_list.0.norm.weight" in captured["state_dict"]
    assert not any(key.startswith("model.visual.") for key in captured["state_dict"])
