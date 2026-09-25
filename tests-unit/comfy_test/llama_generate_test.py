import types

import pytest

from comfy.text_encoders.llama import BaseGenerate


def test_generate_raises_clear_error_when_config_cannot_generate():
    """A truncated conditioning-only checkpoint (e.g. the MiniMax H3 Qwen3-VL-32B
    encoder, which has neither an lm_head nor tied word embeddings) must raise
    instead of silently sampling garbage from an unrelated embedding table."""
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(can_generate=False)))

    with pytest.raises(RuntimeError, match="does not support text generation"):
        BaseGenerate.generate(fake_self, embeds=None)


def test_generate_proceeds_when_config_allows_generation():
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(can_generate=True)))

    with pytest.raises(AttributeError):
        # can_generate check passes; fails later touching embeds.device, proving
        # the guard doesn't block checkpoints that support generation.
        BaseGenerate.generate(fake_self, embeds=None)
