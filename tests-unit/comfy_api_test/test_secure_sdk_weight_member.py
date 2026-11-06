import asyncio
from collections import OrderedDict
import pickle

import pytest
import torch
from safetensors.torch import save_file

import comfy.utils
from comfy_api.latest import _sdk


class UnapprovedCheckpointObject:
    pass


@pytest.mark.parametrize("container", ["model-only", "model-and-iter", "state-dict-and-model"])
@pytest.mark.parametrize("return_metadata", [False, True])
def test_member_selects_original_key_before_legacy_unwrapping(tmp_path, container, return_metadata):
    model = OrderedDict(weight=torch.arange(6, dtype=torch.float64).reshape(2, 3), counter=torch.tensor(9))
    model._metadata = {"": {"version": 2}}
    root = {"model": model}
    if container == "model-and-iter":
        root["iter"] = 17
    elif container == "state-dict-and-model":
        root["state_dict"] = {"other": torch.zeros(1)}
    path = tmp_path / "original.pt"
    torch.save(root, path)
    result = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=return_metadata, member="model")
    selected = result[0] if return_metadata else result
    if return_metadata:
        assert result[1] is None
    assert list(selected) == list(model)
    assert selected._metadata == model._metadata
    for key, tensor in model.items():
        assert selected[key].dtype == tensor.dtype
        assert torch.equal(selected[key], tensor)


def test_defaults_preserve_flat_and_original_nested_results_and_safetensor_metadata(tmp_path):
    flat = {"weight": torch.ones(2)}
    nested = tmp_path / "nested.pt"
    torch.save({"model": flat, "iter": 17}, nested)
    original = comfy.utils.load_torch_file(str(nested), safe_load=True)
    assert set(original) == {"model", "iter"} and original["iter"] == 17
    for wrapper in (flat, {"state_dict": flat}, {"model": flat}):
        path = tmp_path / "flat.pt"
        torch.save(wrapper, path)
        state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
        assert metadata is None and torch.equal(state["weight"], flat["weight"])
    safe = tmp_path / "flat.safetensors"
    save_file(flat, safe, metadata={"author": "original"})
    state, metadata = comfy.utils.load_torch_file(str(safe), return_metadata=True)
    assert metadata == {"author": "original"} and torch.equal(state["weight"], flat["weight"])
    with pytest.raises(ValueError, match="requires a Torch checkpoint"):
        comfy.utils.load_torch_file(str(safe), member="model")


@pytest.mark.parametrize("bad", [False, 1, "", [], {}, lambda: "model"])
def test_invalid_member_refuses_before_file_read(monkeypatch, bad):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid member reached torch.load")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(TypeError, match="non-empty string"):
        comfy.utils.load_torch_file("/must-not-read.pt", member=bad)


@pytest.mark.parametrize("selected", [17, [], {"weight": 17}, {1: torch.ones(1)}, {"nested": {"weight": torch.ones(1)}}])
def test_selected_member_must_contain_named_tensors(tmp_path, selected):
    path = tmp_path / "invalid.pt"
    torch.save({"model": selected, "iter": 17}, path)
    with pytest.raises(ValueError, match="tensor state dict"):
        comfy.utils.load_torch_file(str(path), member="model")


def test_missing_key_and_non_mapping_root_refuse(tmp_path):
    path = tmp_path / "missing.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, path)
    with pytest.raises(KeyError, match="missing"):
        comfy.utils.load_torch_file(str(path), member="missing")
    torch.save([torch.ones(1)], path)
    with pytest.raises(ValueError, match="member dictionary"):
        comfy.utils.load_torch_file(str(path), member="model")


def test_restricted_reader_does_not_retry_unsafe_pickle(tmp_path, monkeypatch):
    path = tmp_path / "unsafe.pt"
    torch.save({"model": {"weight": torch.ones(1)}, "extra": UnapprovedCheckpointObject()}, path)
    real_load = torch.load
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", observed)
    with pytest.raises(pickle.UnpicklingError):
        comfy.utils.load_torch_file(str(path), safe_load=True, member="model")
    assert len(calls) == 1 and calls[0]["weights_only"] is True


def test_sdk_member_signature_ownership_and_return_metadata(tmp_path):
    path = tmp_path / "nested.pt"
    torch.save({"model": {"weight": torch.ones(2)}, "iter": 17}, path)

    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan("member", "7", "Read"))
        asset = _sdk.AssetRef._wrap(await refs.create("ASSET", str(path)))
        other = _sdk.InProcessRefResolver()
        foreign = _sdk.AssetRef._wrap(await other.create("ASSET", str(path)))
        wrong = _sdk.ImageRef._wrap(await refs.create("IMAGE", str(path)))
        forged = _sdk.AssetRef._wrap(wrong)
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            state, metadata = await ctx.assets.load_state_dict(asset, True, member="model")
            assert metadata is None and torch.equal(state["weight"], torch.ones(2))
            assert set(await ctx.assets.load_state_dict(asset)) == {"model", "iter"}
            with pytest.raises(TypeError, match="ASSET ref"):
                await ctx.assets.load_state_dict(wrong, member="model")
            with pytest.raises(TypeError, match="relabel"):
                await ctx.assets.load_state_dict(forged, member="model")
            with pytest.raises(KeyError, match="unknown ref"):
                await ctx.assets.load_state_dict(foreign, member="model")
            await asset.release()
            with pytest.raises(KeyError, match="unknown ref"):
                await ctx.assets.load_state_dict(asset, member="model")
        with pytest.raises(RuntimeError):
            await ctx.assets.load_state_dict(asset, member="model")

    asyncio.run(run())
