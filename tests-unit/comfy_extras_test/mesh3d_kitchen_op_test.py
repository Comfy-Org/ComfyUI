"""Backend selection in comfy_extras.mesh3d.postprocess.qem_decimate._kitchen_op."""

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

# Model management probes the device on import, before pytest fixtures run.
_original_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        cli_args.cpu = True
    import comfy.quant_ops
    from comfy_extras.mesh3d.postprocess import qem_decimate
finally:
    cli_args.cpu = _original_cpu


def _op(*args, **kwargs):
    return "result"


def _fake_kitchen(backends, floors=None):
    """comfy-kitchen stand-in: `backends` maps name -> (available, disabled, capabilities)."""
    floors = floors or {}
    info = {
        name: {"available": available, "disabled": disabled, "capabilities": list(caps)}
        for name, (available, disabled, caps) in backends.items()
    }
    registry = SimpleNamespace(
        get_constraints=lambda backend, name: SimpleNamespace(min_compute_capability=floors.get(backend))
    )
    return SimpleNamespace(list_backends=lambda: info, registry=registry, mesh_bvh=_op)


GPU_TENSOR = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))


@pytest.fixture
def capability(monkeypatch):
    def set_capability(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
    set_capability((8, 9))
    return set_capability


def test_hip_backend_is_used_on_amd(monkeypatch, capability):
    ck = _fake_kitchen({"hip": (True, False, ["mesh_bvh"]), "cuda": (True, True, ["mesh_bvh"])})
    monkeypatch.setattr(comfy.quant_ops, "ck", ck, raising=False)
    assert qem_decimate._kitchen_op("mesh_bvh", GPU_TENSOR) is _op


def test_cuda_backend_is_used_on_nvidia(monkeypatch, capability):
    ck = _fake_kitchen({"hip": (False, False, []), "cuda": (True, False, ["mesh_bvh"])}, floors={"cuda": (7, 5)})
    monkeypatch.setattr(comfy.quant_ops, "ck", ck, raising=False)
    assert qem_decimate._kitchen_op("mesh_bvh", GPU_TENSOR) is _op


def test_cuda_compute_capability_floor_is_respected(monkeypatch, capability):
    capability((7, 0))
    ck = _fake_kitchen({"cuda": (True, False, ["mesh_bvh"])}, floors={"cuda": (7, 5)})
    monkeypatch.setattr(comfy.quant_ops, "ck", ck, raising=False)
    assert qem_decimate._kitchen_op("mesh_bvh", GPU_TENSOR) is None


@pytest.mark.parametrize("backends", [
    {"hip": (True, False, [])},                     # HIP backend without the op
    {"hip": (True, True, ["mesh_bvh"])},            # disabled
    {"hip": (False, False, ["mesh_bvh"])},          # unavailable (e.g. unsupported gfx target)
    {"eager": (True, False, ["mesh_bvh"])},         # CPU reference implementations don't count
])
def test_falls_back_to_torch_without_a_usable_gpu_backend(monkeypatch, capability, backends):
    monkeypatch.setattr(comfy.quant_ops, "ck", _fake_kitchen(backends), raising=False)
    assert qem_decimate._kitchen_op("mesh_bvh", GPU_TENSOR) is None


def test_cpu_tensors_use_torch(monkeypatch, capability):
    ck = _fake_kitchen({"hip": (True, False, ["mesh_bvh"])})
    monkeypatch.setattr(comfy.quant_ops, "ck", ck, raising=False)
    assert qem_decimate._kitchen_op("mesh_bvh", SimpleNamespace(is_cuda=False, device=torch.device("cpu"))) is None
