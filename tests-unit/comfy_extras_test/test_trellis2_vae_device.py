import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

spec = importlib.util.spec_from_file_location(
    "trellis2_vae_device_test_module", Path(__file__).parents[2] / "comfy_extras/nodes_trellis2.py",
)
trellis_nodes = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {
    "comfy_extras.nodes_mesh_postprocess": types.SimpleNamespace(pack_variable_mesh_batch=lambda *args: None),
}):
    spec.loader.exec_module(trellis_nodes)


def test_structure_decode_uses_vae_device(monkeypatch):
    monkeypatch.setattr(trellis_nodes.comfy.model_management, "get_torch_device", lambda: torch.device("meta"))

    class StructureModel:
        def decode_structure(self, samples):
            assert samples.device == torch.device("cpu")
            return torch.ones(samples.shape[0], 1, 2, 2, 2)

    vae = types.SimpleNamespace(
        device=torch.device("cpu"), vae_dtype=torch.float32,
        first_stage_model=StructureModel(), prepare_decode=lambda shape: 1,
    )
    samples = {"samples": torch.ones(2, 8, 1, 1, 1)}

    voxel = trellis_nodes.VaeDecodeStructureTrellis2.execute(samples, vae, "2").args[0]
    assert voxel.data.device == torch.device("cpu")
    assert voxel.data.shape == (2, 2, 2, 2)
    assert torch.all(voxel.data == 1)


@pytest.mark.parametrize("node", ["shape", "texture", "upsample"])
def test_sparse_vae_nodes_use_vae_device(monkeypatch, node):
    monkeypatch.setattr(trellis_nodes.comfy.model_management, "get_torch_device", lambda: torch.device("meta"))

    class ReachedDecode(Exception):
        pass

    class Sparse:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

        def to(self, dtype):
            return self

    def decode(sparse, *args, **kwargs):
        assert sparse.feats.device == torch.device("cpu")
        assert sparse.coords.device == torch.device("cpu")
        raise ReachedDecode

    monkeypatch.setattr(trellis_nodes, "shape_norm", Sparse)
    monkeypatch.setattr(trellis_nodes, "SparseTensor", Sparse)
    monkeypatch.setattr(trellis_nodes, "tex_slat_format", types.SimpleNamespace(process_out=lambda x: x))
    monkeypatch.setattr(trellis_nodes, "_move_sparse_tensor_uncached", lambda sparse, device: Sparse(sparse.feats.to(device), sparse.coords.to(device)))

    model = types.SimpleNamespace(
        resolution=torch.tensor(1024), decode_shape_slat=decode,
        decode_tex_slat=decode, upsample_shape=decode,
    )
    vae = types.SimpleNamespace(
        device=torch.device("cpu"), vae_dtype=torch.float32,
        first_stage_model=model, prepare_decode=lambda *args, **kwargs: 1,
    )
    samples = {"samples": torch.ones(1, 32, 1, 1), "coords": torch.zeros(1, 4, dtype=torch.int32)}

    with pytest.raises(ReachedDecode):
        if node == "shape":
            trellis_nodes.VaeDecodeShapeTrellis.execute(samples, vae)
        elif node == "texture":
            trellis_nodes.VaeDecodeTextureTrellis.execute(samples, vae, [Sparse(torch.ones(1, 32), samples["coords"])])
        else:
            trellis_nodes.Trellis2UpsampleStage.execute([], [], samples, vae, 1024)
