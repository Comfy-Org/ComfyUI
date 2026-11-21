import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.trellis2.model as trellis2_model
import comfy.ops


def test_trellis2_fp8_structure_model_initializes_on_cpu():
    model = trellis2_model.SparseStructureFlowModel(
        resolution=2,
        in_channels=2,
        out_channels=2,
        model_channels=12,
        cond_channels=12,
        num_blocks=0,
        num_heads=1,
        dtype=torch.float8_e4m3fn,
        device="cpu",
        operations=comfy.ops.manual_cast,
    )

    assert model.rope_phases.shape[0] == 8
    assert model.rope_phases.dtype == torch.float32
