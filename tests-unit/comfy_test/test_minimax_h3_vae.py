import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.quant_ops
from comfy.ldm.minimax.vae import Attention


class _OffloadedScale:
    """Stands in for a buffer that comfy's dynamic-VRAM offload left on another
    device: it only tracks its device label and can be moved with .to(), like
    the real qk_norm_scale tensor the bug report is about."""

    def __init__(self, device):
        self.device = torch.device(device)

    def to(self, device):
        return _OffloadedScale(device)


class _FakeCK:
    def rms_rope_split_half(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    def rms_rope_split_half_(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    @staticmethod
    def _check(query, key, scale):
        if scale.device != query.device:
            # mirrors comfy_kitchen's eager backend, which does no device
            # alignment and refuses to run the fused kernel on mixed devices
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got weight is on "
                f"{scale.device}, different from other tensors on {query.device}"
            )
        return query, key


def test_attention_moves_offloaded_qk_norm_scale_to_input_device(monkeypatch):
    monkeypatch.setattr(comfy.quant_ops, "ck", _FakeCK(), raising=False)

    heads, dim_head = 2, 4
    attn = Attention(heads=heads, dim_head=dim_head)
    # simulate a partially-offloaded VAE: qk_norm_scale sits on a different
    # device than the tensors flowing through the fused rope kernel
    attn._buffers["qk_norm_scale"] = _OffloadedScale("meta")

    batch_size, seq_len = 1, 3
    x = torch.randn(batch_size, seq_len, heads * dim_head)
    rotary_pos_emb = torch.randn(batch_size, 1, seq_len, dim_head // 2, 2, 2)

    class _Norm:
        weight = None
        eps = 1e-5

    with torch.no_grad():
        out = attn.forward(
            x, rotary_pos_emb, pre_norm=_Norm(), residual=None, residual_scale=None
        )

    assert out.device == x.device
