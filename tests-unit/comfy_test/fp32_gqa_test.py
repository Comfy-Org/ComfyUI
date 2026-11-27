import types

import pytest
import torch
import torch.nn.functional as functional

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.modules import attention


def _inputs(device, length=17, query_length=1, heads=32, kv_heads=8, head_dim=128, dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(2608)
    query = torch.randn((2, heads, query_length, head_dim), generator=generator, device=device, dtype=dtype)
    # Prefix slices retain the strides of a larger allocated tuple cache.
    key = torch.randn((2, kv_heads, length + 3, head_dim), generator=generator, device=device, dtype=dtype)[:, :, :length]
    value = torch.randn((2, kv_heads, length + 3, head_dim), generator=generator, device=device, dtype=dtype)[:, :, :length]
    return query, key, value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decode optimization")
@pytest.mark.parametrize("skip_reshape", [False, True])
@pytest.mark.parametrize("skip_output_reshape", [False, True])
@pytest.mark.parametrize("length", [776, 4278])
def test_fp32_gqa_decode_matches_sdpa(monkeypatch, skip_reshape, skip_output_reshape, length):
    query, key, value = _inputs("cuda", length=length)
    expected = functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    if not skip_output_reshape:
        expected = expected.transpose(1, 2).reshape(2, 1, 4096)

    def unexpected_sdpa(*args, **kwargs):
        pytest.fail("eligible decode expanded KV through SDPA")

    monkeypatch.setattr(attention.comfy.ops, "scaled_dot_product_attention", unexpected_sdpa)
    if not skip_reshape:
        query, key, value = [tensor.transpose(1, 2).flatten(2) for tensor in (query, key, value)]
    output = attention.attention_pytorch(
        query, key, value, 32, enable_gqa=True,
        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
    assert output.shape == expected.shape
    assert output.dtype == torch.float32 and output.device == expected.device
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("case", ["cpu", "mask", "scale", "prefill", "fp16", "head_dim", "heads", "no_gqa", "batch_limit"])
def test_other_attention_cases_keep_sdpa(monkeypatch, case):
    device = "cuda" if torch.cuda.is_available() and case != "cpu" else "cpu"
    query, key, value = _inputs(
        device, query_length=2 if case == "prefill" else 1,
        heads=16 if case == "heads" else 32,
        kv_heads=32 if case == "no_gqa" else 8,
        head_dim=64 if case == "head_dim" else 128,
        dtype=torch.float16 if case == "fp16" else torch.float32)
    mask = torch.zeros((1, 1, 1, key.shape[2]), device=device) if case == "mask" else None
    if mask is not None:
        mask[..., -1] = float("-inf")
    kwargs = {"enable_gqa": case != "no_gqa"}
    if case == "scale":
        kwargs["scale"] = 0.25
    if case == "batch_limit":
        monkeypatch.setattr(attention, "SDP_BATCH_LIMIT", 1)
    expected = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, **kwargs)
    expected = expected.transpose(1, 2).reshape(2, query.shape[2], -1)
    sdpa_calls = []

    def sdpa(*args, **kwargs):
        sdpa_calls.append(kwargs)
        return functional.scaled_dot_product_attention(*args, **kwargs)

    def unexpected_grouped(*args, **kwargs):
        pytest.fail("ineligible attention used grouped decode")

    monkeypatch.setattr(attention.comfy.ops, "scaled_dot_product_attention", sdpa)
    monkeypatch.setattr(attention.comfy_kitchen, "flash_attention_decode_gqa", unexpected_grouped)
    output = attention.attention_pytorch(query, key, value, query.shape[1], mask=mask, skip_reshape=True, **kwargs)
    assert len(sdpa_calls) == (2 if case == "batch_limit" else 1)
    assert all(call["attn_mask"] is mask for call in sdpa_calls)
    assert all(call["enable_gqa"] == kwargs["enable_gqa"] for call in sdpa_calls)
    if case == "scale":
        assert sdpa_calls[0]["scale"] == 0.25
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decode optimization")
@pytest.mark.parametrize("dispatch", ["preferred", "override"])
def test_custom_attention_still_owns_dispatch(monkeypatch, dispatch):
    query, key, value = _inputs("cuda")
    expected = query.new_full((2, 1, 4096), 7.0)

    def preferred(*args, **kwargs):
        return expected

    def unexpected_grouped(*args, **kwargs):
        pytest.fail("preferred attention was bypassed")

    monkeypatch.setattr(attention.comfy_kitchen, "flash_attention_decode_gqa", unexpected_grouped)
    kwargs = {"preferred_attention": types.SimpleNamespace(function=preferred)}
    if dispatch == "override":
        kwargs = {"transformer_options": {"optimized_attention_override": preferred}}
    output = attention.attention_pytorch(
        query, key, value, 32, enable_gqa=True, skip_reshape=True,
        **kwargs)
    assert output is expected
