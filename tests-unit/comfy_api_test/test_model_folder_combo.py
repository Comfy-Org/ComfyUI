"""io.ModelFolder: combo options listed from a model folder when ComfyUI reads the schema."""
import asyncio

import pytest

from comfy_api.index import io


class LoraPick(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="TestLoraPick", inputs=[
            io.Combo.Input("lora_name", options=io.ModelFolder("loras")),
            io.Combo.Input("only_st", options=io.ModelFolder("loras", extensions=["safetensors"]))],
            outputs=[io.String.Output()])

    @classmethod
    async def execute(cls, lora_name, only_st):
        return io.NodeOutput(lora_name)


@pytest.fixture
def loras(tmp_path, monkeypatch):
    import folder_paths

    (tmp_path / "a.safetensors").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"x")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors", ".pt"}))
    folder_paths.filename_list_cache.clear()
    return tmp_path


def _options(name):
    kind, spec = LoraPick.GET_NODE_INFO_V1()["input"]["required"][name][:2]
    assert kind == "COMBO"
    return spec["options"]


def test_options_follow_the_folder(loras):
    import folder_paths

    assert _options("lora_name") == ["a.safetensors", "b.pt"]
    assert _options("only_st") == ["a.safetensors"]
    (loras / "c.safetensors").write_bytes(b"x")
    folder_paths.filename_list_cache.clear()
    assert _options("lora_name") == ["a.safetensors", "b.pt", "c.safetensors"]


def test_validation_accepts_a_file_added_later(loras):
    import execution
    import folder_paths
    import nodes

    nodes.NODE_CLASS_MAPPINGS["TestLoraPick"] = LoraPick
    try:
        (loras / "new.safetensors").write_bytes(b"x")
        folder_paths.filename_list_cache.clear()
        prompt = {"1": {"class_type": "TestLoraPick", "inputs": {"lora_name": "new.safetensors", "only_st": "a.safetensors"}}}
        valid = asyncio.run(execution.validate_inputs("p", prompt, "1", {}))
        assert valid[0] is True, valid
        prompt["1"]["inputs"]["lora_name"] = "missing.safetensors"
        assert asyncio.run(execution.validate_inputs("p", prompt, "1", {}))[0] is False
    finally:
        del nodes.NODE_CLASS_MAPPINGS["TestLoraPick"]


def test_no_folders_means_no_options(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_folder_paths(name, *args, **kwargs):
        if name == "folder_paths":
            raise ImportError("not here")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_folder_paths)
    assert io.ModelFolder("loras").resolve() == []
