import torch
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.weight_adapter.boft import BOFTAdapter


def _apply(alpha):
    torch.manual_seed(0)
    blocks = torch.randn(1, 2, 2, 2) * 0.1
    weight = torch.eye(4)
    adapter = BOFTAdapter("w", (blocks, None, alpha, None))
    out = adapter.calculate_weight(
        weight.clone(), "w", 1.0, 1.0, 0, lambda x: x,
        intermediate_dtype=torch.float32, original_weight=None,
    )
    return out, weight


class TestBOFTAdapter:
    def test_applies_when_alpha_missing(self):
        # a BOFT LoRA without an ".alpha" key arrives with alpha=None; it must still
        # apply the rotation (None means "no constraint"), like the OFT adapter.
        out, weight = _apply(None)
        assert not torch.equal(out, weight)

    def test_missing_alpha_matches_zero_alpha(self):
        out_none, _ = _apply(None)
        out_zero, _ = _apply(0)
        assert torch.allclose(out_none, out_zero)
