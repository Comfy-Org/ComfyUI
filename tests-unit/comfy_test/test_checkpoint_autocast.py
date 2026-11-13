import pytest
import torch

from comfy.ldm.modules.diffusionmodules.util import checkpoint


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("parameter_only", [False, True])
def test_checkpoint_preserves_autocast(device_type, dtype, cache_enabled, parameter_only):
    if device_type != "cpu" and not getattr(torch, device_type).is_available():
        pytest.skip(f"{device_type} is unavailable")
    if device_type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is unavailable")

    def run(use_checkpoint):
        x = torch.linspace(-1, 1, 32, device=device_type).reshape(4, 8).requires_grad_()
        weight = torch.linspace(-0.5, 0.5, 32, device=device_type).reshape(8, 4).requires_grad_()
        seen = []

        def block(*inputs):
            hidden = (x if parameter_only else inputs[0]) @ weight
            seen.append((torch.is_autocast_enabled(device_type), hidden.dtype, torch.is_autocast_cache_enabled()))
            return torch.nn.functional.silu(hidden)

        inputs = () if parameter_only else (x,)
        params = (x, weight) if parameter_only else (weight,)
        with torch.autocast(device_type, dtype=dtype or torch.bfloat16, enabled=dtype is not None, cache_enabled=cache_enabled):
            output = checkpoint(block, inputs, params, use_checkpoint)
        output.backward(torch.ones_like(output))
        assert not torch.is_autocast_enabled(device_type)
        assert torch.is_autocast_cache_enabled()
        return output, x.grad, weight.grad, seen

    expected = run(False)
    actual = run(True)
    for result, reference in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert actual[3] == expected[3] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_preserves_cpu_and_cuda_autocast():
    def run(use_checkpoint):
        x = torch.linspace(-1, 1, 16).reshape(4, 4).requires_grad_()
        weight = x.detach().cuda().requires_grad_()

        def block(input):
            return input @ input, weight @ weight

        with torch.autocast("cpu", dtype=torch.bfloat16), torch.autocast("cuda", dtype=torch.float16):
            outputs = checkpoint(block, (x,), (weight,), use_checkpoint)
        torch.autograd.backward(outputs, tuple(torch.ones_like(output) for output in outputs))
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")
        return *outputs, x.grad, weight.grad

    for result, reference in zip(run(True), run(False)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
