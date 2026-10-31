import types

import pytest

from comfy.text_encoders.llama import BaseGenerate, Qwen3_8BConfig, Qwen3VL_32BConfig


def test_generate_raises_clear_error_when_config_cannot_generate():
    """The MiniMax H3 Qwen3-VL-32B encoder (neither an lm_head nor tied word
    embeddings) must raise instead of silently sampling garbage from an
    unrelated embedding table. Uses the real checkpoint config so this also
    catches `can_generate` being removed from `Qwen3VL_32BConfig`."""
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(config=Qwen3VL_32BConfig()))

    with pytest.raises(RuntimeError, match="does not support text generation"):
        BaseGenerate.generate(fake_self, embeds=None)


def test_generate_proceeds_when_config_allows_generation():
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(config=types.SimpleNamespace(can_generate=True)))

    with pytest.raises(AttributeError):
        # can_generate check passes; fails later touching embeds.device, proving
        # the guard doesn't block checkpoints that support generation.
        BaseGenerate.generate(fake_self, embeds=None)


def test_generate_proceeds_when_config_omits_can_generate():
    """Qwen3_8BConfig doesn't set can_generate; the getattr(..., True) fallback
    in generate() must still let it through unaffected."""
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(config=Qwen3_8BConfig()))

    with pytest.raises(AttributeError):
        BaseGenerate.generate(fake_self, embeds=None)
