from types import SimpleNamespace

import comfy.model_management
from comfy.text_encoders.ace15 import ACE15TEModel


def test_memory_estimation_uses_max_tokens(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "should_use_bf16", lambda _: False)
    model = SimpleNamespace(constant=0.4375)
    tokens = {
        "lm_prompt": [[None] * 10],
        "lm_metadata": {"min_tokens": 20, "max_tokens": 80},
    }

    actual = ACE15TEModel.memory_estimation_function(model, tokens)

    assert actual == (10 + 80) * model.constant * 1024 * 1024
