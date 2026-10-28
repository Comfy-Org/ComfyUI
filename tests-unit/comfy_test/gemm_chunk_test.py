import torch

import comfy.gemm_chunk as gemm_chunk


def _force_split(monkeypatch, size):
    """Pretend the GPU miscomputes large GEMMs, so the split path runs on any machine."""
    monkeypatch.setattr(gemm_chunk, "_split_needed", lambda tensor: True)
    monkeypatch.setattr(gemm_chunk, "_OVERRIDE", size)


def _count_calls(monkeypatch, module, name):
    calls = []
    original = getattr(module, name)

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_linear_split_matches_unsplit(monkeypatch):
    _force_split(monkeypatch, 4)
    x = torch.randn(10, 6)
    w = torch.randn(5, 6)
    b = torch.randn(5)

    out = gemm_chunk.linear(x, w, b)

    assert out.shape == (10, 5)
    assert torch.allclose(out, torch.nn.functional.linear(x, w, b), atol=1e-6)


def test_linear_split_keeps_the_input_shape(monkeypatch):
    _force_split(monkeypatch, 4)
    x = torch.randn(2, 5, 6)
    w = torch.randn(3, 6)

    out = gemm_chunk.linear(x, w)

    assert out.shape == (2, 5, 3)
    assert torch.allclose(out, torch.nn.functional.linear(x, w), atol=1e-6)


def test_linear_split_runs_once_per_chunk(monkeypatch):
    _force_split(monkeypatch, 4)
    calls = _count_calls(monkeypatch, gemm_chunk.F, "linear")

    gemm_chunk.linear(torch.randn(10, 6), torch.randn(5, 6))

    assert len(calls) == 3  # 4 + 4 + 2 rows


def test_linear_without_a_split_is_the_plain_op(monkeypatch):
    monkeypatch.setattr(gemm_chunk, "_OVERRIDE", 0)
    calls = _count_calls(monkeypatch, gemm_chunk.F, "linear")
    x = torch.randn(10, 6)
    w = torch.randn(5, 6)

    out = gemm_chunk.linear(x, w)

    assert len(calls) == 1
    assert torch.equal(out, torch.nn.functional.linear(x, w))


def test_bmm_split_matches_unsplit(monkeypatch):
    _force_split(monkeypatch, 4)
    a = torch.randn(6, 9, 3)
    b = torch.randn(6, 3, 4)

    out = gemm_chunk.bmm(a, b)

    assert out.shape == (6, 9, 4)
    assert torch.allclose(out, torch.bmm(a, b), atol=1e-6)


def test_einsum_split_matches_unsplit(monkeypatch):
    _force_split(monkeypatch, 4)
    p = torch.randn(10, 4)

    out = gemm_chunk.einsum("fi,fj->fij", p, p)

    assert out.shape == (10, 4, 4)
    assert torch.allclose(out, torch.einsum("fi,fj->fij", p, p), atol=1e-6)


def test_einsum_split_handles_three_operands(monkeypatch):
    _force_split(monkeypatch, 4)
    v = torch.randn(10, 3)
    q = torch.randn(10, 3, 3)

    out = gemm_chunk.einsum("ei,eij,ej->e", v, q, v)

    assert out.shape == (10,)
    assert torch.allclose(out, torch.einsum("ei,eij,ej->e", v, q, v), atol=1e-6)


def test_measure_ignores_a_cpu_device(monkeypatch):
    monkeypatch.setattr(gemm_chunk, "_measured", False)
    monkeypatch.setattr(gemm_chunk, "_row_split", 0)
    monkeypatch.setattr(gemm_chunk, "_batch_split", 0)

    gemm_chunk.measure(torch.device("cpu"))

    assert gemm_chunk.row_split() == 0
    assert gemm_chunk.batch_split() == 0
