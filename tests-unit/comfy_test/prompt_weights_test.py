import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.sd1_clip import escape_important, token_weights


def weights(text):
    return token_weights(escape_important(text), 1.0)


def test_weighted_group():
    assert weights("a (b:1.5) c") == [("a ", 1.0), ("b", 1.5), (" c", 1.0)]


def test_unbalanced_closing_parenthesis_is_literal_text():
    assert weights("smile :) (red hair:1.3)") == [("smile :) ", 1.0), ("red hair", 1.3)]


def test_closing_parenthesis_after_group_does_not_cancel_later_weights():
    assert weights("(a:1.2) b) (c:1.4)") == [("a", 1.2), (" b) ", 1.0), ("c", 1.4)]
