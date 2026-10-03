import torch

from comfy.cli_args import args as cli_args

_original_cli_args_cpu = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_base  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.supported_models  # noqa: E402

cli_args.cpu = _original_cli_args_cpu


def test_minimax_h3_memory_usage_factor_covers_measured_shortfall(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: False)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: False)

    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    model.memory_usage_factor_conds = ()
    model.memory_usage_shape_process = {}

    input_shape = (1, 32, 49, 92, 144)

    model.memory_usage_factor = 0.114
    old_estimate = model.memory_required(input_shape)

    model.memory_usage_factor = comfy.supported_models.MiniMaxH3.memory_usage_factor
    new_estimate = model.memory_required(input_shape)

    # Issue #15781: factor 0.114 underestimated the real sampling working set
    # by ~1.45x on a 24GB card, so load_models_gpu() pinned too much of the
    # DiT weights and sampling OOM'd.
    assert new_estimate >= old_estimate * 1.45
