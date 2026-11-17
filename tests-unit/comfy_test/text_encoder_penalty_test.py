import torch

from comfy.text_encoders.llama import apply_penalty, penalty_active


def test_zero_repetition_penalty_is_off():
    torch.manual_seed(0)
    logits = torch.randn(2, 128)

    assert torch.equal(apply_penalty(logits, 0.0, 0.0), logits)
    assert torch.isfinite(torch.softmax(apply_penalty(logits, 0.0, 0.0), dim=-1)).all()


def test_negative_repetition_penalty_is_off():
    torch.manual_seed(0)
    logits = torch.randn(2, 128)

    assert torch.equal(apply_penalty(logits, -1.0, 0.0), logits)


def test_penalty_applies_above_zero():
    logits = torch.tensor([[-2.0, 3.0]])

    assert torch.allclose(apply_penalty(logits, 2.0, 0.0), torch.tensor([[-4.0, 1.5]]))
    assert torch.allclose(apply_penalty(logits, 0.5, 0.0), torch.tensor([[-1.0, 6.0]]))


def test_penalty_active_matches_apply_penalty():
    assert not penalty_active(0.0, 0.0)
    assert not penalty_active(1.0, 0.0)
    assert not penalty_active(1.0, None)
    assert penalty_active(1.5, 0.0)
    assert penalty_active(1.0, 0.5)
