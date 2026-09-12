import types

import pytest

from comfy.sd1_clip import SDClipModel


def test_generate_raises_clear_error_for_non_generation_model():
    """A CLIP text encoder (no .generate on the underlying transformer) must
    fail with an actionable RuntimeError instead of a bare AttributeError
    from deep inside the model call."""
    fake_self = types.SimpleNamespace(transformer=object())

    with pytest.raises(RuntimeError, match="does not support text generation"):
        SDClipModel.generate(
            fake_self,
            tokens=[[(0, 1.0)]],
            do_sample=False,
            max_length=8,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            seed=0,
        )


def test_generate_raises_clear_error_for_non_callable_generate_attribute():
    """A transformer with a non-callable `generate` attribute (e.g. None)
    must also fail with the actionable RuntimeError, not an incidental
    TypeError from trying to call it."""
    fake_self = types.SimpleNamespace(transformer=types.SimpleNamespace(generate=None))

    with pytest.raises(RuntimeError, match="does not support text generation"):
        SDClipModel.generate(
            fake_self,
            tokens=[[(0, 1.0)]],
            do_sample=False,
            max_length=8,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            seed=0,
        )
