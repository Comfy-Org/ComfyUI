from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.minimax import audio_vae_kernels
from comfy.ldm.minimax.audio_vae import LowPassFilter1d, SnakeBeta, UpSample1d


class _FakeCudaTensor:
    def __init__(self, shape=(2, 3, 17), device="cuda:1", dtype=torch.float32, requires_grad=False):
        self.is_cuda = True
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.device = torch.device(device)
        self.requires_grad = requires_grad

    def numel(self):
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def stride(self):
        strides = []
        size = 1
        for dim in reversed(self.shape):
            strides.append(size)
            size *= dim
        return tuple(reversed(strides))

    def contiguous(self):
        return self


class _FakeKernel:
    def __init__(self, active_device):
        self.active_device = active_device
        self.grids = []
        self.launch_devices = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.grids.append(grid)
            self.launch_devices.append(self.active_device())
        return launch


def _upsample_reference(x, filter):
    x = F.pad(x, (5, 5), mode="replicate")
    filter = filter.to(x).expand(x.shape[1], -1, -1)
    x = F.conv_transpose1d(x, filter, stride=2, groups=x.shape[1]).mul_(2)
    return x[..., 15:-15]


def _downsample_reference(x, filter):
    x = F.pad(x, (5, 6), mode="replicate")
    filter = filter.to(x).expand(x.shape[1], -1, -1)
    return F.conv1d(x, filter, stride=2, groups=x.shape[1])


@pytest.mark.parametrize("length", [1, 2, 17, 257])
@pytest.mark.parametrize("upsample", [True, False])
def test_minimax_h3_fir_kernel_matches_torch(length, upsample):
    if not torch.cuda.is_available():
        pytest.skip("requires a supported HIP GPU")

    device = torch.device("cuda")
    layer = UpSample1d() if upsample else LowPassFilter1d(stride=2)
    layer = layer.to(device)
    # Striding exercises the kernel's explicit input strides; lengths cover
    # replicate padding at the boundary and masked final blocks.
    x = torch.randn(2, 3, length * 2, device=device)[:, :, ::2]
    if not audio_vae_kernels.can_use(x, layer.filter):
        pytest.skip("requires a supported HIP GPU")

    actual = layer(x)
    reference = _upsample_reference(x, layer.filter) if upsample else _downsample_reference(x, layer.filter)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


def test_minimax_h3_snake_beta_module_dispatch_matches_torch(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a supported HIP GPU")

    x = torch.randn(2, 3, 513, device="cuda")
    activation = SnakeBeta(3)
    activation.alpha = torch.nn.Parameter(torch.randn(3, device=x.device), requires_grad=False)
    activation.beta = torch.nn.Parameter(torch.randn(3, device=x.device), requires_grad=False)
    if not audio_vae_kernels.can_use(x, activation.alpha, activation.beta):
        pytest.skip("requires a supported HIP GPU")

    kernel = audio_vae_kernels.snake_beta
    calls = []

    def track_dispatch(*args):
        calls.append(args)
        return kernel(*args)

    monkeypatch.setattr(audio_vae_kernels, "snake_beta", track_dispatch)
    actual = activation(x)
    assert len(calls) == 1
    alpha = activation.alpha.exp().view(1, -1, 1)
    beta = activation.beta.exp().view(1, -1, 1)
    reference = x + torch.sin(alpha * x).square() / (beta + 1e-9)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_minimax_h3_audio_kernels_reject_cpu_inputs(dtype):
    x = torch.randn(2, 3, 17, dtype=dtype)
    assert not audio_vae_kernels.can_use(torch.randn(2, 3, 17))
    assert not audio_vae_kernels.can_use(x)


@pytest.mark.parametrize(("arch", "expected"), [
    ("gfx1010", True),  # RDNA1
    ("gfx1010:sramecc+:xnack-", True),
    ("gfx1036", True),  # RDNA2
    ("gfx908", True),  # MI100
    ("gfx1151", True),
    ("gfx906", False),  # MI50
    ("gfx9999", False),
])
def test_minimax_h3_audio_kernel_arch_dispatch(monkeypatch, arch, expected):
    monkeypatch.setattr(audio_vae_kernels, "triton", object())
    monkeypatch.setattr(torch.version, "hip", "7.16")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName=arch))
    assert audio_vae_kernels.can_use(_FakeCudaTensor()) is expected


def test_minimax_h3_audio_kernel_arch_dispatch_requires_hip(monkeypatch):
    monkeypatch.setattr(audio_vae_kernels, "triton", object())
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: pytest.fail("unexpected device query"))
    assert not audio_vae_kernels.can_use(_FakeCudaTensor())


def test_minimax_h3_audio_kernel_launch_uses_input_device(monkeypatch):
    selected_devices = []
    active_device = [torch.device("cuda:0")]

    @contextmanager
    def use_device(device):
        previous_device = active_device[0]
        selected_devices.append(device)
        active_device[0] = device
        try:
            yield
        finally:
            active_device[0] = previous_device

    monkeypatch.setattr(torch.cuda, "device", use_device)
    monkeypatch.setattr(audio_vae_kernels, "triton", SimpleNamespace(cdiv=lambda size, block: (size + block - 1) // block))
    monkeypatch.setattr(torch, "empty", lambda shape, dtype, device: _FakeCudaTensor(shape, device, dtype))

    monkeypatch.delattr(audio_vae_kernels, "_fir2x", raising=False)
    monkeypatch.delattr(audio_vae_kernels, "_snake_beta_f32", raising=False)
    fir_kernel = _FakeKernel(lambda: active_device[0])
    snake_kernel = _FakeKernel(lambda: active_device[0])
    monkeypatch.setattr(audio_vae_kernels, "_fir2x", fir_kernel, raising=False)
    monkeypatch.setattr(audio_vae_kernels, "_snake_beta_f32", snake_kernel, raising=False)

    x = _FakeCudaTensor(device="cuda:1")
    audio_vae_kernels.fir2x(x, _FakeCudaTensor((1, 1, 12), x.device), up=True)
    audio_vae_kernels.snake_beta(x, _FakeCudaTensor((3,), x.device), _FakeCudaTensor((3,), x.device))

    assert selected_devices == [x.device, x.device]
    assert len(fir_kernel.grids) == len(snake_kernel.grids) == 1
    assert fir_kernel.launch_devices == snake_kernel.launch_devices == [x.device]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_minimax_h3_audio_kernels_reject_unsupported_gpu_dtypes(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a supported HIP GPU")

    x = torch.randn(2, 3, 17, device="cuda")
    if not audio_vae_kernels.can_use(x):
        pytest.skip("requires a supported HIP GPU")
    assert not audio_vae_kernels.can_use(x.to(dtype))


def test_minimax_h3_audio_kernels_preserve_autograd():
    if not torch.cuda.is_available():
        pytest.skip("requires a supported HIP GPU")

    x = torch.randn(2, 3, 17, device="cuda")
    if not audio_vae_kernels.can_use(x):
        pytest.skip("requires a supported HIP GPU")
    assert audio_vae_kernels.can_use(x.detach())
    assert not audio_vae_kernels.can_use(x.requires_grad_())
    parameter = torch.randn(3, device=x.device, requires_grad=True)
    assert not audio_vae_kernels.can_use(x.detach(), parameter)


@pytest.mark.parametrize("length", [1, 17, 257])
@pytest.mark.parametrize("upsample", [True, False])
def test_minimax_h3_fir_cpu_fallback_matches_torch(length, upsample):
    layer = UpSample1d() if upsample else LowPassFilter1d(stride=2)
    x = torch.randn(2, 3, length * 2)[:, :, ::2]
    assert not audio_vae_kernels.can_use(x, layer.filter)

    actual = layer(x)
    reference = _upsample_reference(x, layer.filter) if upsample else _downsample_reference(x, layer.filter)
    torch.testing.assert_close(actual, reference)


def test_minimax_h3_snake_beta_cpu_fallback_matches_torch():
    activation = SnakeBeta(3)
    activation.alpha = torch.nn.Parameter(torch.randn(3))
    activation.beta = torch.nn.Parameter(torch.randn(3))
    x = torch.randn(2, 3, 257)

    actual = activation(x)
    alpha = activation.alpha.exp().view(1, -1, 1)
    beta = activation.beta.exp().view(1, -1, 1)
    reference = x + torch.sin(alpha * x).square() / (beta + 1e-9)
    torch.testing.assert_close(actual, reference)
