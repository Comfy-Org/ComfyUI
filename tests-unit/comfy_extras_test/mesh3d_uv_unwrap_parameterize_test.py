"""Exercise GPU solve selection and the ROCm workspace fallback on CPU tensors."""

import numpy as np
import pytest
import torch

from comfy.cli_args import args as cli_args

# Model management probes the device on import, before pytest fixtures run.
_original_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        cli_args.cpu = True
    import comfy.model_management
    from comfy_extras.mesh3d.uv_unwrap import parameterize
finally:
    cli_args.cpu = _original_cpu


def _one_triangle_chart():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    uv_pins = verts[:, :2].copy()
    faces_gl = np.array([[0, 1, 2]], dtype=np.int64)
    face_pos = np.array([0], dtype=np.int64)
    chart_of_face = np.array([0], dtype=np.int64)
    chart_of_vert = np.array([0, 0, 0], dtype=np.int64)
    vert_offsets = np.array([0, 3], dtype=np.int64)
    chart_ids = np.array([0], dtype=np.int64)
    return verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert, vert_offsets, chart_ids


def _run_chart(device):
    return parameterize.lscm_charts_batch(*_one_triangle_chart(), n_charts=1, device=device)[0]


@pytest.fixture
def cpu_backed_gpu(monkeypatch):
    """Run the GPU branch's tensor algebra without claiming hardware validation."""
    original_to = torch.Tensor.to

    def to_cpu(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            args = (torch.device("cpu"), *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    return torch.device("cuda")


@pytest.mark.parametrize("is_amd", [True, False])
def test_working_gpu_solver_is_used(monkeypatch, cpu_backed_gpu, is_amd):
    expected = _run_chart(torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: is_amd)
    solve = torch.linalg.solve
    calls = []

    def record(a, b):
        calls.append(a.shape)
        return solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", record)
    actual = _run_chart(cpu_backed_gpu)
    assert len(calls) == 1
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rocm_workspace_failure_falls_back_for_one_batch(monkeypatch, cpu_backed_gpu):
    expected = _run_chart(torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: True)
    solve = torch.linalg.solve
    calls = []

    def fail_once(a, b):
        calls.append(a.shape)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasDgetrfBatched")
        return solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", fail_once)
    np.testing.assert_allclose(_run_chart(cpu_backed_gpu), expected, atol=1e-6)
    np.testing.assert_allclose(_run_chart(cpu_backed_gpu), expected, atol=1e-6)
    assert len(calls) == 2


@pytest.mark.parametrize("is_amd,message", [
    (True, "The solver failed because the input matrix is singular"),
    (True, "CUDA error: an illegal memory access was encountered"),
    (False, "HIPBLAS_STATUS_ALLOC_FAILED"),
])
def test_unrelated_solver_errors_propagate(monkeypatch, cpu_backed_gpu, is_amd, message):
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: is_amd)

    def fail(a, b):
        raise RuntimeError(message)

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(RuntimeError, match=message):
        _run_chart(cpu_backed_gpu)


def test_cpu_path_does_not_call_gpu_solver(monkeypatch):
    def unexpected(a, b):
        pytest.fail("CPU path called torch.linalg.solve")

    monkeypatch.setattr(torch.linalg, "solve", unexpected)
    result = _run_chart(torch.device("cpu"))
    assert result.shape == (3, 2)
    assert np.isfinite(result).all()
    assert np.ptp(result) > 0
