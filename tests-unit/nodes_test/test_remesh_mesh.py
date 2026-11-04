import importlib
import sys
import types

import pytest
import torch

from comfy_api.latest import Types


@pytest.fixture
def mesh_module(monkeypatch):
    server = types.ModuleType("server")
    server.PromptServer = type("PromptServer", (), {"instance": None})
    monkeypatch.setitem(sys.modules, "server", server)

    module = importlib.import_module("comfy_extras.nodes_mesh_postprocess")
    monkeypatch.setattr(module.RemeshMesh, "hidden", types.SimpleNamespace(unique_id=None))
    monkeypatch.setattr(module.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "_prepare_gpu_mesh_processing", lambda *args: None)
    return module


def _cube_mesh(extent, center=(0.0, 0.0, 0.0)):
    vertices = torch.tensor(
        [[
            [-extent, -extent, -extent], [extent, -extent, -extent],
            [extent, extent, -extent], [-extent, extent, -extent],
            [-extent, -extent, extent], [extent, -extent, extent],
            [extent, extent, extent], [-extent, extent, extent],
        ]]
    ) + torch.tensor(center)
    faces = torch.tensor(
        [[
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ]],
        dtype=torch.int32,
    )
    return Types.MESH(vertices=vertices, faces=faces)


def _remesh(module, mesh, sign_mode="sdf", band=1.0):
    return module.RemeshMesh.execute(
        mesh,
        resolution=32,
        sign_mode={"sign_mode": sign_mode, "qef": False},
        band=band,
        project_back=0.0,
        fix_poles=False,
        smooth_iters=0,
        drop_small_components=0.0,
        precluster_max_verts=0,
    ).args[0]


def test_remesh_mesh_fits_domain_to_input_bounds(mesh_module):
    result = _remesh(mesh_module, _cube_mesh(1.0))

    assert result.vertices.shape[1] > 0
    torch.testing.assert_close(result.vertices.amin(dim=(0, 1)), torch.full((3,), -1.0))
    torch.testing.assert_close(result.vertices.amax(dim=(0, 1)), torch.full((3,), 1.0))

    result = _remesh(mesh_module, _cube_mesh(1.0, center=(2.0, 0.0, 0.0)))

    torch.testing.assert_close(result.vertices.amin(dim=(0, 1)), torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(result.vertices.amax(dim=(0, 1)), torch.tensor([3.0, 1.0, 1.0]))


def test_remesh_mesh_pads_fitted_domain_for_udf_band(mesh_module, monkeypatch):
    captured = {}

    def fake_remesh(vertices, faces, **kwargs):
        captured.update(kwargs)
        return vertices, faces, kwargs["colors"]

    monkeypatch.setattr(mesh_module, "remesh_narrow_band_dc", fake_remesh)
    _remesh(mesh_module, _cube_mesh(1.0), sign_mode="udf", band=4.0)

    assert captured["scale"] == pytest.approx(2.0 * 44.0 / 32.0)
    torch.testing.assert_close(captured["center"], torch.zeros(3))


def test_remesh_mesh_preserves_unit_cube_domain(mesh_module, monkeypatch):
    captured = {}

    def fake_remesh(vertices, faces, **kwargs):
        captured.update(kwargs)
        return vertices, faces, kwargs["colors"]

    monkeypatch.setattr(mesh_module, "remesh_narrow_band_dc", fake_remesh)
    _remesh(mesh_module, _cube_mesh(0.5))

    assert captured["scale"] == pytest.approx(35.0 / 32.0)
    torch.testing.assert_close(captured["center"], torch.zeros(3))
