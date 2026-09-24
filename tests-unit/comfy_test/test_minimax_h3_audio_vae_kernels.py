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
    is_cuda = True
    dtype = torch.float32
    ndim = 3
    device = torch.device("cuda:0")
    requires_grad = False

    def numel(self):
        return 1


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
        pytest.skip("requires an AMD GPU")

    device = torch.device("cuda")
    layer = UpSample1d() if upsample else LowPassFilter1d(stride=2)
    layer = layer.to(device)
    # Striding exercises the kernel's explicit input strides; lengths cover
    # replicate padding at the boundary and masked final blocks.
    x = torch.randn(2, 3, length * 2, device=device)[:, :, ::2]
    if not audio_vae_kernels.can_use(x, layer.filter):
        pytest.skip("requires HIP gfx1151")

    actual = layer(x)
    reference = _upsample_reference(x, layer.filter) if upsample else _downsample_reference(x, layer.filter)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


def test_minimax_h3_snake_beta_kernel_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("requires an AMD GPU")

    x = torch.randn(2, 3, 513, device="cuda")
    alpha = torch.randn(3, device=x.device)
    beta = torch.randn(3, device=x.device)
    if not audio_vae_kernels.can_use(x, alpha, beta):
        pytest.skip("requires HIP gfx1151")

    actual = audio_vae_kernels.snake_beta(x, alpha, beta)
    alpha = alpha.exp().view(1, -1, 1)
    beta = beta.exp().view(1, -1, 1)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_minimax_h3_audio_kernels_reject_unsupported_gpu_dtypes(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires HIP gfx1151")

    x = torch.randn(2, 3, 17, device="cuda")
    if not audio_vae_kernels.can_use(x):
        pytest.skip("requires HIP gfx1151")
    assert not audio_vae_kernels.can_use(x.to(dtype))


def test_minimax_h3_audio_kernels_preserve_autograd():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")

    x = torch.randn(2, 3, 17, device="cuda", requires_grad=True)
    parameter = torch.randn(3, device=x.device, requires_grad=True)
    assert not audio_vae_kernels.can_use(x)
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
