import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.qwen_image21.model import PoseBranchCache, QwenImage21Transformer2DModel  # noqa: E402


class _RecordingPause:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append("pause_enter")

    def __exit__(self, *exc_info):
        self.calls.append("pause_exit")


class _FakePatcher:
    def get_free_memory(self, device):
        return 10 ** 12


def _make_model():
    torch.manual_seed(0)
    model = QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=8, mlp_ratio=2,
        axes_dims_rope=(2, 2, 4), dtype=torch.float32, device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )
    model.reset_prefix_cache(True)
    model.current_patcher = _FakePatcher()
    return model


def _run_forward_and_record(monkeypatch, model, x, timestep, context, ref_latents):
    calls = []
    monkeypatch.setattr(comfy.model_prefetch, "pause_malloc_graph", lambda sync=False: _RecordingPause(calls))

    orig_take = PoseBranchCache.take
    orig_prefetch = PoseBranchCache.prefetch

    def take(self, *args, **kwargs):
        calls.append("take")
        return orig_take(self, *args, **kwargs)

    def prefetch(self, *args, **kwargs):
        calls.append("prefetch")
        return orig_prefetch(self, *args, **kwargs)

    monkeypatch.setattr(PoseBranchCache, "take", take)
    monkeypatch.setattr(PoseBranchCache, "prefetch", prefetch)

    model._forward(x, timestep, context, ref_latents=ref_latents, transformer_options={})
    return calls


def _assert_cache_reads_are_paused(calls):
    # cache.take/prefetch do their own CUDA staging-buffer allocation and stream
    # syncs, which the block-weight dynamic-VRAM malloc graph must not observe
    # (issue #16443: an unpaused allocation there aborts the process).
    assert "take" in calls
    depth = 0
    for call in calls:
        if call == "pause_enter":
            depth += 1
        elif call == "pause_exit":
            depth -= 1
        elif call in ("take", "prefetch"):
            assert depth > 0, f"PoseBranchCache.{call} ran outside pause_malloc_graph()"


def test_prefix_cache_fill_pass_is_paused(monkeypatch):
    model = _make_model()
    x = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 3, 8)
    ref_latents = [torch.randn(1, 4, 4, 4)]
    timestep = torch.tensor([0.5])

    calls = _run_forward_and_record(monkeypatch, model, x, timestep, context, ref_latents)

    _assert_cache_reads_are_paused(calls)


def test_prefix_cache_replay_pass_is_paused(monkeypatch):
    model = _make_model()
    x = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 3, 8)
    ref_latents = [torch.randn(1, 4, 4, 4)]
    timestep = torch.tensor([0.5])

    # first pass fills the cache; the pause contract is exercised again on replay
    model._forward(x, timestep, context, ref_latents=ref_latents, transformer_options={})
    calls = _run_forward_and_record(monkeypatch, model, x, timestep, context, ref_latents)

    _assert_cache_reads_are_paused(calls)
