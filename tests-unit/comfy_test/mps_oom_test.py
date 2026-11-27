"""MPS reports a plain RuntimeError; VAE recovery must recognize that boundary."""
import pytest
import torch

from comfy.cli_args import args as cli_args

_original_cpu = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as mm  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402
from comfy.sd import VAE  # noqa: E402

cli_args.cpu = _original_cpu


MPS_OOM = (
    'MPS backend out of memory (MPS allocated: 11.12 GiB, other allocations: '
    '26.64 MiB, max allowed: 10.67 GiB). Tried to allocate 6.75 MiB on shared pool.'
)


@pytest.fixture(autouse=True)
def cpu_models(monkeypatch):
    # These tests exercise backend error handling without allocating on a GPU.
    monkeypatch.setattr(mm, 'cpu_state', mm.CPUState.CPU)
    monkeypatch.setattr(mm, 'current_loaded_models', [])
    yield
    mm.free_memory(1e30, torch.device('cpu'))


def test_plain_mps_runtime_error_is_oom():
    error = RuntimeError(MPS_OOM)
    assert type(error) is RuntimeError
    assert mm.is_oom(error)
    mm.raise_non_oom(error)


@pytest.mark.parametrize('error', [
    torch.OutOfMemoryError('fixture'),
    torch.cuda.OutOfMemoryError('fixture'),
] + ([torch.AcceleratorError('CUDA error: out of memory')] if hasattr(torch, 'AcceleratorError') else []))
def test_existing_typed_oom_classification(error):
    assert mm.is_oom(error)


class DecoderBoundary(torch.nn.Module):
    """Stand in for the backend operation that fails, not VAE recovery or tiling."""
    def __init__(self, errors=(), oom_above=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.errors = errors
        self.oom_above = oom_above
        self.sides = []

    @property
    def calls(self):
        return len(self.sides)

    def decode(self, value):
        self.sides.append(max(value.shape[-2:]))
        if self.calls <= len(self.errors):
            raise self.errors[self.calls - 1]
        if self.oom_above is not None and self.sides[-1] > self.oom_above:
            raise RuntimeError(MPS_OOM)
        return value * self.weight


def cpu_vae(errors=(), oom_above=None):
    # Configure the real VAE execution path without loading a model checkpoint.
    vae = VAE.__new__(VAE)
    vae.first_stage_model = DecoderBoundary(errors, oom_above)
    vae.device = vae.output_device = torch.device('cpu')
    vae.vae_dtype = torch.float32
    vae.latent_dim = 3
    vae.extra_1d_channel = None
    vae.handles_tiling = False
    vae.disable_offload = False
    vae.output_channels = 3
    vae.upscale_ratio = (1, 1, 1)
    vae.upscale_index_formula = (1, 1, 1)
    vae.process_output = lambda value: value
    vae.memory_used_decode = lambda shape, dtype: 4096
    vae.patcher = ModelPatcher(vae.first_stage_model, vae.device, vae.device)
    return vae


def samples(side=8):
    count = 3 * 3 * side * side
    return torch.arange(count, dtype=torch.float32).reshape(1, 3, 3, side, side) / count


def test_mps_oom_recovers_through_smaller_3d_tiles():
    # Whole-input decode fails like the MPS allocator; the 256-wide fallback tiles fit.
    vae = cpu_vae(oom_above=256)
    latent = samples(320)

    output = vae.decode(latent)

    sides = vae.first_stage_model.sides
    assert sides[0] == 320 and len(sides) > 2 and max(sides[1:]) <= 256
    assert list(output.shape) == [1, 3, 320, 320, 3]
    torch.testing.assert_close(output, latent.movedim(1, -1), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('error', [
    RuntimeError('shape mismatch'),
    RuntimeError('application out of memory bookkeeping'),
    RuntimeError('application says: ' + MPS_OOM),
    ValueError(MPS_OOM),
])
def test_unrelated_errors_propagate_without_tiled_retry(error):
    vae = cpu_vae([error])

    with pytest.raises(type(error)) as caught:
        vae.decode(samples())

    assert caught.value is error
    assert not mm.is_oom(error)
    assert vae.first_stage_model.calls == 1


def test_tiled_retry_failure_propagates_without_retry_loop():
    retry_error = RuntimeError(MPS_OOM)
    vae = cpu_vae([RuntimeError(MPS_OOM), retry_error])

    with pytest.raises(RuntimeError) as caught:
        vae.decode(samples())

    assert caught.value is retry_error
    assert vae.first_stage_model.calls == 2
