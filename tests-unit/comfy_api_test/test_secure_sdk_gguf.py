"""Closed GGUF model-loading primitives exposed to secure packs."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import folder_paths
from comfy_api.latest import _sdk


def _context():
    return _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan(
        prompt_id="gguf-test",
        node_id="1",
        node_type="gguf-test",
    ))


def test_load_gguf_text_encoders_mixes_closed_catalogue_weights(monkeypatch):
    import comfy.model_management as model_management
    import comfy.sd
    import comfy.utils

    names = ("encoder.gguf", "encoder.safetensors")
    paths = {
        ("clip_gguf", names[0]): "/models/clip/encoder.gguf",
        ("text_encoders", names[1]): "/models/clip/encoder.safetensors",
    }
    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda folder: list(names) if folder == "text_encoders" else [],
    )
    monkeypatch.setattr(
        folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: paths[(folder, name)],
    )
    monkeypatch.setattr(
        folder_paths,
        "get_folder_paths",
        lambda folder: [f"/models/{folder}"],
    )
    monkeypatch.setattr(
        comfy.utils,
        "load_torch_file",
        lambda path, safe_load=True: {"safe": path},
    )
    monkeypatch.setattr(
        model_management,
        "text_encoder_offload_device",
        lambda: "offload-device",
    )

    original_patcher = object()
    loaded = SimpleNamespace(patcher=original_patcher)
    load_call = {}

    def load_text_encoder_state_dicts(**kwargs):
        load_call.update(kwargs)
        return loaded

    monkeypatch.setattr(
        comfy.sd,
        "load_text_encoder_state_dicts",
        load_text_encoder_state_dicts,
    )
    gguf_ops = object()
    gguf_module = SimpleNamespace(
        GGMLOps=gguf_ops,
        gguf_clip_loader=lambda path: {"gguf": path},
        GGUFModelPatcher=SimpleNamespace(
            clone=lambda patcher: ("gguf-patcher", patcher)),
    )
    monkeypatch.setattr(_sdk, "_fixed_gguf_node_module", lambda: gguf_module)

    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            clip_ref = await context.models.load_gguf_text_encoders(
                names, "stable_diffusion")
            return clip_ref, await refs.resolve(clip_ref)

    clip_ref, value = asyncio.run(run())
    assert isinstance(clip_ref, _sdk.ClipRef)
    assert value is loaded
    assert load_call["state_dicts"] == [
        {"gguf": paths[("clip_gguf", names[0])]},
        {"safe": paths[("text_encoders", names[1])]},
    ]
    assert load_call["clip_type"] is comfy.sd.CLIPType.STABLE_DIFFUSION
    assert load_call["model_options"] == {
        "custom_operations": gguf_ops,
        "initial_device": "offload-device",
    }
    assert load_call["embedding_directory"] == ["/models/embeddings"]
    assert loaded.patcher == ("gguf-patcher", original_patcher)


def test_load_gguf_text_encoders_rejects_unsafe_or_ambiguous_inputs(
    monkeypatch,
):
    import comfy.sd
    import comfy.utils

    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda folder: ["scaled.safetensors"]
        if folder == "text_encoders" else [],
    )
    monkeypatch.setattr(
        folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: f"/models/{folder}/{name}",
    )
    monkeypatch.setattr(
        comfy.utils,
        "load_torch_file",
        lambda path, safe_load=True: {"scaled_fp8": object()},
    )

    async def invoke(names, clip_type):
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            return await context.models.load_gguf_text_encoders(
                names, clip_type)

    with pytest.raises(TypeError, match="sequence"):
        asyncio.run(invoke("scaled.safetensors", "stable_diffusion"))
    with pytest.raises(ValueError, match="1 to 4"):
        asyncio.run(invoke([], "stable_diffusion"))
    with pytest.raises(ValueError, match="1 to 4"):
        asyncio.run(invoke(["scaled.safetensors"] * 5, "stable_diffusion"))
    with pytest.raises(ValueError, match="unknown CLIP type"):
        asyncio.run(invoke(["scaled.safetensors"], "not-a-family"))
    with pytest.raises(ValueError, match="unknown text encoder"):
        asyncio.run(invoke(["missing.safetensors"], "stable_diffusion"))
    with pytest.raises(ValueError, match="scaled FP8"):
        asyncio.run(invoke(["scaled.safetensors"], "stable_diffusion"))


def test_load_gguf_text_encoders_requires_compatible_gguf_extension(
    monkeypatch,
):
    import comfy.sd

    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda folder: ["encoder.gguf"] if folder == "text_encoders" else [],
    )
    monkeypatch.setattr(
        folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: f"/models/{folder}/{name}",
    )
    monkeypatch.setattr(
        _sdk,
        "_fixed_gguf_node_module",
        lambda: SimpleNamespace(
            GGMLOps=object(),
            GGUFModelPatcher=SimpleNamespace(clone=lambda value: value),
        ),
    )

    async def run():
        refs = _sdk.InProcessRefResolver()
        context = _context()
        with _sdk.bind_runtime(refs, context, _sdk.InProcessOps()):
            return await context.models.load_gguf_text_encoders(
                ["encoder.gguf"], "stable_diffusion")

    with pytest.raises(RuntimeError, match="missing gguf_clip_loader"):
        asyncio.run(run())
