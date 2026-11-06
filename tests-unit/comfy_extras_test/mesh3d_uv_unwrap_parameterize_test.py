"""lscm_charts_batch must not run the dense solve on the GPU on AMD/ROCm (issue #16124).

ROCm's hipBLAS backing call for torch.linalg.solve (hipblasDgetrfBatched) raises
HIPBLAS_STATUS_ALLOC_FAILED on these chart-solve batches even with ample free VRAM.
PyTorch reports a ROCm device's .type as "cuda", so the GPU-solve decision must also
check comfy.model_management.is_amd() and fall back to the CPU (numpy/LAPACK) path.
"""

import numpy as np
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
from comfy_extras.mesh3d.uv_unwrap import parameterize


def test_solve_on_gpu_false_for_amd_cuda_type_device(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: True)

    assert parameterize._lscm_solve_on_gpu(torch.device("cuda")) is False


def test_solve_on_gpu_true_for_non_amd_cuda_type_device(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: False)

    assert parameterize._lscm_solve_on_gpu(torch.device("cuda")) is True


def test_solve_on_gpu_false_for_cpu_device(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: False)

    assert parameterize._lscm_solve_on_gpu(torch.device("cpu")) is False


def _one_triangle_chart():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    uv_pins = np.zeros((3, 2), dtype=np.float64)
    faces_gl = np.array([[0, 1, 2]], dtype=np.int64)
    face_pos = np.array([0], dtype=np.int64)
    chart_of_face = np.array([0], dtype=np.int64)
    chart_of_vert = np.array([0, 0, 0], dtype=np.int64)
    vert_offsets = np.array([0, 3], dtype=np.int64)
    chart_ids = np.array([0], dtype=np.int64)
    return verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert, vert_offsets, chart_ids


def test_lscm_charts_batch_completes_on_amd_with_cuda_type_device(monkeypatch):
    """End-to-end: on a machine reporting an AMD/ROCm "cuda"-type device, the batch
    solve must not attempt any GPU tensor transfer (which would otherwise be the
    first operation to fail/hang on real ROCm hardware per issue #16124)."""
    monkeypatch.setattr(comfy.model_management, "is_amd", lambda: True)

    (verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert,
     vert_offsets, chart_ids) = _one_triangle_chart()

    out = parameterize.lscm_charts_batch(
        verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert,
        vert_offsets, chart_ids, n_charts=1, device=torch.device("cuda"))

    assert 0 in out
    assert out[0].shape == (3, 2)
