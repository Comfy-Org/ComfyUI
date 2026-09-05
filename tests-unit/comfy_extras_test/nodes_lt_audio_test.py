import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


class FakeComfyNode:
    pass


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args


class FakeInput:
    def __init__(self, id, options=None, **kwargs):
        self.id = id
        self.options = options
        self.kwargs = kwargs


class FakeCombo:
    Input = staticmethod(lambda id, **kwargs: FakeInput(id, **kwargs))


class FakeSchema:
    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.kwargs = kwargs


def _fake_io_module():
    io_mock = MagicMock()
    io_mock.ComfyNode = FakeComfyNode
    io_mock.NodeOutput = FakeNodeOutput
    io_mock.Schema = FakeSchema
    io_mock.Combo = FakeCombo
    return io_mock


def _fake_folder_paths(checkpoints=None, vae=None, text_encoders=None):
    checkpoints = checkpoints or []
    vae = vae or []
    text_encoders = text_encoders if text_encoders is not None else ["gemma4-12b-with-proj.safetensors"]
    lists_by_folder = {"checkpoints": checkpoints, "vae": vae, "text_encoders": text_encoders}

    def get_filename_list(folder_name):
        return list(lists_by_folder.get(folder_name, []))

    def get_full_path_or_raise(folder_name, filename):
        if filename not in lists_by_folder.get(folder_name, []):
            raise FileNotFoundError(f"no file '{filename}' registered under '{folder_name}'")
        return f"/models/{folder_name}/{filename}"

    fp = MagicMock()
    fp.get_filename_list.side_effect = get_filename_list
    fp.get_full_path_or_raise.side_effect = get_full_path_or_raise
    fp.get_folder_paths.return_value = ["/models/embeddings"]
    return fp


@contextmanager
def _nodes_lt_audio(folder_paths_mock):
    comfy_api_latest_mock = MagicMock()
    comfy_api_latest_mock.ComfyExtension = object
    comfy_api_latest_mock.io = _fake_io_module()

    nodes_audio_mock = MagicMock()
    nodes_audio_mock.VAEEncodeAudio = type("VAEEncodeAudio", (), {})

    sd_mock = MagicMock()
    utils_mock = MagicMock()
    utils_mock.load_torch_file.return_value = ({}, {})
    model_management_mock = MagicMock()

    modules = {
        "torch": MagicMock(),
        "folder_paths": folder_paths_mock,
        "comfy.utils": utils_mock,
        "comfy.model_management": model_management_mock,
        "comfy.sd": sd_mock,
        "comfy_api.latest": comfy_api_latest_mock,
        "comfy_extras.nodes_audio": nodes_audio_mock,
    }
    comfy_submodule_attrs = {"utils": utils_mock, "model_management": model_management_mock, "sd": sd_mock}

    sentinel = object()
    import comfy as comfy_pkg
    prior_attrs = {name: getattr(comfy_pkg, name, sentinel) for name in comfy_submodule_attrs}
    prior_module = sys.modules.get("comfy_extras.nodes_lt_audio", sentinel)
    sys.modules.pop("comfy_extras.nodes_lt_audio", None)

    try:
        with patch.dict(sys.modules, modules):
            for name, mock in comfy_submodule_attrs.items():
                setattr(comfy_pkg, name, mock)
            module = __import__("comfy_extras.nodes_lt_audio", fromlist=["dummy"])
            yield module, sd_mock
    finally:
        for name, prior in prior_attrs.items():
            if prior is sentinel:
                if hasattr(comfy_pkg, name):
                    delattr(comfy_pkg, name)
            else:
                setattr(comfy_pkg, name, prior)
        sys.modules.pop("comfy_extras.nodes_lt_audio", None)
        if prior_module is not sentinel:
            sys.modules["comfy_extras.nodes_lt_audio"] = prior_module


class TestLTXVAudioVAELoader:
    def test_reads_from_vae_folder_not_checkpoints(self):
        fp = _fake_folder_paths(
            checkpoints=["some_checkpoint.safetensors"],
            vae=["ltx-2.5-audio-vae-bf16.safetensors"],
        )
        with _nodes_lt_audio(fp) as (module, _sd_mock):
            schema = module.LTXVAudioVAELoader.define_schema()
            ckpt_input = next(i for i in schema.inputs if i.id == "ckpt_name")
            assert ckpt_input.options == ["ltx-2.5-audio-vae-bf16.safetensors"]

            module.LTXVAudioVAELoader.execute("ltx-2.5-audio-vae-bf16.safetensors")
            fp.get_full_path_or_raise.assert_called_with("vae", "ltx-2.5-audio-vae-bf16.safetensors")


class TestLTXAVTextEncoderLoader:
    def test_none_ckpt_loads_standalone_text_encoder(self):
        fp = _fake_folder_paths(checkpoints=["some_checkpoint.safetensors"])
        with _nodes_lt_audio(fp) as (module, sd_mock):
            module.LTXAVTextEncoderLoader.execute("gemma4-12b-with-proj.safetensors", "none")

            sd_mock.load_clip.assert_called_once()
            ckpt_paths = sd_mock.load_clip.call_args.kwargs["ckpt_paths"]
            assert ckpt_paths == ["/models/text_encoders/gemma4-12b-with-proj.safetensors"]

    def test_with_ckpt_still_loads_both_files(self):
        fp = _fake_folder_paths(checkpoints=["some_checkpoint.safetensors"])
        with _nodes_lt_audio(fp) as (module, sd_mock):
            module.LTXAVTextEncoderLoader.execute("gemma4-12b-with-proj.safetensors", "some_checkpoint.safetensors")

            ckpt_paths = sd_mock.load_clip.call_args.kwargs["ckpt_paths"]
            assert ckpt_paths == [
                "/models/text_encoders/gemma4-12b-with-proj.safetensors",
                "/models/checkpoints/some_checkpoint.safetensors",
            ]

    def test_none_is_a_valid_schema_option(self):
        fp = _fake_folder_paths(checkpoints=["some_checkpoint.safetensors"])
        with _nodes_lt_audio(fp) as (module, _sd_mock):
            schema = module.LTXAVTextEncoderLoader.define_schema()
            ckpt_input = next(i for i in schema.inputs if i.id == "ckpt_name")
            assert ckpt_input.options[0] == "none"
