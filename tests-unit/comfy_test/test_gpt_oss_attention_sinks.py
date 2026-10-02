import pytest
import torch

import comfy.text_encoders.gpt_oss as gpt_oss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real second device to reproduce the lowvram offload bug")
def test_attention_with_sinks_matches_when_sinks_offloaded_to_cpu():
    device = torch.device("cuda")
    num_heads = 2
    q = torch.randn(1, num_heads, 3, 4, dtype=torch.float32, device=device)
    k = torch.randn(1, num_heads, 3, 4, dtype=torch.float32, device=device)
    v = torch.randn(1, num_heads, 3, 4, dtype=torch.float32, device=device)
    sinks_on_device = torch.randn(num_heads, dtype=torch.float32, device=device)

    # Simulates lowvram offload: the sinks parameter ends up on CPU while
    # q/k/v (produced by ops.Linear layers, which self-cast) stay on the GPU.
    sinks_on_cpu = sinks_on_device.cpu()

    expected = gpt_oss._attention_with_sinks(q, k, v, sinks_on_device, None, num_heads, 1)
    actual = gpt_oss._attention_with_sinks(q, k, v, sinks_on_cpu, None, num_heads, 1)

    assert actual.device == q.device
    torch.testing.assert_close(actual, expected)
