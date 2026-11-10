"""util, the pure helpers for pack code, and the execution-provider seam."""
import io
import pytest
import torch
from PIL import Image
from comfy_api.latest import _runtime
from comfy_api.v4 import util


@pytest.mark.parametrize("method", ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"])
@pytest.mark.parametrize("crop", ["disabled", "center"])
@pytest.mark.parametrize("shape", [(2, 4, 13, 17), (1, 3, 2, 9, 11)])
def test_common_upscale_matches_comfy(method, crop, shape):
    import comfy.utils

    torch.manual_seed(0)
    samples = torch.rand(shape)
    if method == "lanczos" and len(shape) == 5:
        pytest.skip("ComfyUI's lanczos path is for images")
    a = util.common_upscale(samples.clone(), 23, 19, method, crop)
    b = comfy.utils.common_upscale(samples.clone(), 23, 19, method, crop)
    assert torch.equal(a, b)


@pytest.mark.parametrize("inds", [None, [0, 2, 2]])
def test_prepare_noise_matches_comfy(inds):
    import comfy.sample

    latent = torch.zeros(3, 4, 8, 8, dtype=torch.float16)
    assert torch.equal(util.prepare_noise(latent, 42, inds), comfy.sample.prepare_noise(latent, 42, inds))


def test_conditioning_helpers_match_comfy():
    import node_helpers

    cond = [[torch.ones(1, 2, 3), {"a": 1, "strength": 0.5}], [torch.zeros(1, 2, 3), {}]]
    for append in (False, True):
        assert util.conditioning_set_values(cond, {"strength": 0.25}, append) == \
            node_helpers.conditioning_set_values(cond, {"strength": 0.25}, append)
    a = util.conditioning_set_values_with_timestep_range(cond, {"x": 2}, 0.2, 0.6)
    b = node_helpers.conditioning_set_values_with_timestep_range(cond, {"x": 2}, 0.2, 0.6)
    assert [(t[0].tolist(), t[1]) for t in a] == [(t[0].tolist(), t[1]) for t in b]


def test_pillow_matches_comfy():
    import node_helpers

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, "PNG")
    data = buffer.getvalue()
    assert util.pillow(lambda d: Image.open(io.BytesIO(d)).convert("RGB").tobytes(), data) == \
        node_helpers.pillow(lambda d: Image.open(io.BytesIO(d)).convert("RGB").tobytes(), data)


def test_constants_match_comfy():
    import comfy.samplers
    import nodes

    assert util.SAMPLER_NAMES == comfy.samplers.KSampler.SAMPLERS
    assert util.SCHEDULER_NAMES == comfy.samplers.KSampler.SCHEDULERS
    assert util.MAX_RESOLUTION == nodes.MAX_RESOLUTION


def test_util_imports_nothing_from_the_engine():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(util))
    folders = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Folders")
    in_folders = {id(n) for n in ast.walk(folders)}
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in imports if isinstance(n, ast.Import) and id(n) not in in_folders for a in n.names} | \
        {n.module for n in imports if isinstance(n, ast.ImportFrom) and n.module}
    assert not {n for n in names if n.split(".")[0] in ("comfy", "nodes", "node_helpers", "folder_paths", "server")}
    # util.folders reads ComfyUI's folder layout (folder_paths), and nothing else of ComfyUI's
    in_folders_names = {a.name for n in imports if isinstance(n, ast.Import) and id(n) in in_folders for a in n.names}
    assert in_folders_names == {"folder_paths"}


@pytest.mark.parametrize("shape,tile,overlap,scale", [((1, 3, 40, 50), (16, 24), 4, 2), ((2, 4, 9, 9), (16, 16), 2, 4),
                                                      ((1, 3, 33, 17), (8, 8), 3, 1)])
def test_tiled_scale_matches_comfy(shape, tile, overlap, scale):
    import comfy.utils
    samples = torch.randn(*shape, generator=torch.Generator().manual_seed(3))

    def function(x):
        return torch.nn.functional.interpolate(x, scale_factor=scale, mode="nearest")[:, :3]
    assert util.get_tiled_scale_steps(shape[3], shape[2], tile[1], tile[0], overlap) == \
        comfy.utils.get_tiled_scale_steps(shape[3], shape[2], tile[1], tile[0], overlap)
    ours = util.tiled_scale(samples, function, tile_x=tile[1], tile_y=tile[0], overlap=overlap, upscale_amount=scale)
    theirs = comfy.utils.tiled_scale(samples, function, tile_x=tile[1], tile_y=tile[0], overlap=overlap, upscale_amount=scale)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("suffix", [".safetensors", ".pt", ".ckpt"])
def test_load_torch_file_matches_comfy(tmp_path, suffix):
    import comfy.utils
    sd = {"a.weight": torch.randn(3, 4), "b.bias": torch.arange(5.0)}
    path = str(tmp_path / f"w{suffix}")
    if suffix == ".safetensors":
        util.save_torch_file(sd, path, metadata={"k": "v"})
    else:
        torch.save({"state_dict": sd} if suffix == ".ckpt" else sd, path)
    ours, theirs = util.load_torch_file(path), comfy.utils.load_torch_file(path)
    assert sorted(ours) == sorted(theirs) and all(torch.equal(ours[k], theirs[k]) for k in ours)
    if suffix == ".safetensors":
        assert util.load_torch_file(path, return_metadata=True)[1] == comfy.utils.load_torch_file(path, return_metadata=True)[1]


def test_folders_are_comfyuis():
    import folder_paths
    assert util.folders.models_dir == folder_paths.models_dir
    assert util.folders.get_output_directory() == folder_paths.get_output_directory()
    assert util.folders.get_temp_directory() == folder_paths.get_temp_directory()
    assert util.folders.get_input_directory() == folder_paths.get_input_directory()
    assert util.folders.get_folder_paths("checkpoints") == folder_paths.get_folder_paths("checkpoints")
    assert util.folders.get_filename_list("checkpoints") == folder_paths.get_filename_list("checkpoints")
    assert util.folders.get_full_path("checkpoints", "nope.safetensors") is None


def test_folders_has_folder_paths_names(monkeypatch):
    """The documented names, folder_paths' other public names, and assignment, all on ComfyUI's module."""
    import folder_paths
    assert util.folders.base_path == folder_paths.base_path
    assert util.folders.folder_names_and_paths is folder_paths.folder_names_and_paths
    assert util.folders.supported_pt_extensions == folder_paths.supported_pt_extensions
    assert util.folders.output_directory == folder_paths.get_output_directory()
    assert util.folders.filter_files_extensions is folder_paths.filter_files_extensions  # not documented, still there
    assert util.folders.__file__ == folder_paths.__file__
    monkeypatch.setattr(folder_paths, "models_dir", folder_paths.models_dir)  # restored after
    util.folders.models_dir = "/elsewhere/models"
    assert folder_paths.models_dir == "/elsewhere/models"
    with pytest.raises(AttributeError):
        util.folders._private


def test_execution_blocker_is_comfyuis():
    from comfy_api.v4 import io
    from comfy_execution.graph_utils import ExecutionBlocker
    assert io.ExecutionBlocker is ExecutionBlocker


def test_no_provider_configured_is_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.delenv("COMFY_PACK_ISOLATION", raising=False)
    monkeypatch.delenv("OOP_CUSTOM_NODES", raising=False)
    assert not _runtime.load_execution_provider()
    assert _runtime.providers.load_custom_nodes


def test_provider_without_entrypoint_fails(tmp_path) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text("value = 1\n")
    with pytest.raises(RuntimeError, match="register"):
        _runtime.load_execution_provider(str(provider))


def test_provider_can_replace_dispatch(monkeypatch, tmp_path) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text("def register(providers):\n    providers.load_custom_nodes = False\n")
    registry = _runtime.Providers()
    monkeypatch.setattr(_runtime, "providers", registry)
    assert _runtime.load_execution_provider(str(provider))
    assert not registry.load_custom_nodes


def test_frontend_config_is_empty_by_default() -> None:
    assert _runtime.Providers().frontend_runtime_config == {}


def test_a_provider_can_supply_a_row_itself(monkeypatch):
    """providers.service_calls: a sandbox's own store or interaction broker stands in for the core row."""
    import asyncio

    from comfy_api.latest import _engine, _runtime

    async def own_store_get(key, scope="pack", default=None):
        return f"{scope}:{key}"

    monkeypatch.setitem(_runtime.providers.service_calls, "store.get", own_store_get)
    with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.EngineCalls(), pack="p"):
        assert asyncio.run(_engine.dispatch(None, "store.get", ("presets",), {})) == "pack:presets"


def test_the_admission_policy_is_one_function(monkeypatch, tmp_path):
    """custom_node_allowed: what native loading applies, for a provider's pack discovery too."""
    import nodes
    from comfy.cli_args import args

    pack = tmp_path / "custom_nodes" / "some_pack"
    pack.mkdir(parents=True)
    monkeypatch.setattr(args, "enable_manager", False)
    monkeypatch.setattr(args, "disable_all_custom_nodes", True)
    monkeypatch.setattr(args, "whitelist_custom_nodes", [])
    assert nodes.custom_node_allowed(str(pack)) is False
    monkeypatch.setattr(args, "whitelist_custom_nodes", ["some_pack"])
    assert nodes.custom_node_allowed(str(pack)) is True
