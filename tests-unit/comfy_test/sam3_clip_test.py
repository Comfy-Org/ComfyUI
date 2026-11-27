from comfy import sd1_clip
from comfy.text_encoders.sam3_clip import SAM3TokenizerWrapper


def test_sam3_single_prompt_strips_default_detection_limit(monkeypatch):
    calls = []

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        calls.append(text)
        return {"tokens": text}

    monkeypatch.setattr(sd1_clip.SD1Tokenizer, "tokenize_with_weights", tokenize_with_weights)

    tokenizer = object.__new__(SAM3TokenizerWrapper)
    result = tokenizer.tokenize_with_weights("person:1")

    assert calls == ["person"]
    assert result == {"tokens": "person"}
