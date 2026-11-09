import asyncio
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management
from comfy.model_state import ModelState
from comfy_api.latest import io
from comfy_execution.graph import DynamicPrompt
from comfy_extras import nodes_replacements
from comfy_extras.nodes_textgen import TextGenerate, TextGenerateV2, TextGenerateLTX2Prompt


class Clip:
    tokenizer = SimpleNamespace(clip_name="gemma4")

    def tokenize(self, prompt, **kwargs):
        self.tokenized = kwargs
        return {"text": [[(1, 1.0)]]}

    def generate(self, tokens, state, **kwargs):
        self.generated = kwargs
        state.metadata["turn"] = state.metadata.get("turn", 0) + 1
        return [7]

    def decode(self, ids):
        return "answer"


def node_for(cls, consumed):
    graph = {"preview": {"inputs": {"source": ["generate", 0]}}}
    if consumed:
        graph["save"] = {"inputs": {"state": ["generate", 2]}}
    return cls.PREPARE_CLASS_CLONE({"hidden_inputs": {
        io.Hidden.unique_id: "generate", io.Hidden.dynprompt: DynamicPrompt(graph),
    }})


@pytest.mark.parametrize("cls", [TextGenerate, TextGenerateV2])
def test_text_generation_schemas_preserve_state_and_hidden_inputs(cls):
    schema = cls.GET_SCHEMA()
    assert next(i for i in schema.inputs if i.id == "state").optional
    assert [o.io_type for o in schema.outputs] == ["STRING", "STRING", "MODEL_STATE"]
    assert schema.hidden == [io.Hidden.unique_id, io.Hidden.dynprompt]
    media_ids = {i.id for i in schema.inputs}
    assert ({"image", "video", "audio"} if cls is TextGenerate else {"images", "videos", "audios"}) <= media_ids
    sampling = next(i for i in schema.inputs if i.id == "sampling_mode").options[0]
    assert next(i for i in sampling.inputs if i.id == "presence_penalty").optional == (cls is TextGenerate)


@pytest.mark.parametrize("cls", [TextGenerate, TextGenerateV2])
@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("consumed", [False, True])
@pytest.mark.parametrize("template", [False, True])
def test_text_generation_copies_input_state_and_preserves_output_lifetime(monkeypatch, cls, continuing, consumed, template):
    monkeypatch.setattr(comfy.model_management, "EPHEMERAL_MODELS", [])
    source = ModelState(metadata={"turn": 1}) if continuing else None
    clip = Clip()
    output = node_for(cls, consumed).execute(
        clip, "Continue", 8, {"sampling_mode": "on", "seed": 71},
        state=source, system_prompt="Initial instructions", use_default_template=template,
    )
    state = output.args[2]
    assert output.args[:2] == ("answer", "")
    assert state is not source and state.metadata == {"turn": 2 if continuing else 1}
    if continuing:
        assert source.metadata == {"turn": 1}
    assert clip.tokenized["system_prompt"] == ("Initial instructions" if template and not continuing else "")
    assert clip.generated["seed"] == 71 and clip.generated["do_sample"]
    assert comfy.model_management.EPHEMERAL_MODELS == ([] if consumed else [state])


@pytest.mark.parametrize("cls", [TextGenerate, TextGenerateV2])
def test_text_generation_keeps_media_inputs_with_state(cls):
    clip = Clip()
    images = torch.arange(96).reshape(2, 4, 4, 3)
    video = torch.zeros(3, 4, 4, 3)
    audio = {"waveform": torch.zeros(1, 1, 16), "sample_rate": 16000}
    media = ({"image": images, "video": video, "audio": audio} if cls is TextGenerate else {
        "images": {"image1": images, "image2": None},
        "videos": {"video1": video, "video2": None},
        "audios": {"audio1": audio, "audio2": None},
    })
    node_for(cls, True).execute(clip, "Describe", 8, {"sampling_mode": "off"}, state=ModelState(), **media)
    if cls is TextGenerate:
        assert clip.tokenized["image"] is images
        assert clip.tokenized["video"] is video and clip.tokenized["audio"] is audio
    else:
        assert len(clip.tokenized["images"]) == 2
        assert all(torch.equal(actual, expected) for actual, expected in zip(clip.tokenized["images"], images.split(1)))
        assert clip.tokenized["videos"][0] is video and clip.tokenized["audios"][0] is audio


@pytest.mark.parametrize("cls", [TextGenerate, TextGenerateV2])
def test_unconsumed_state_is_registered_before_tokenization_failure(monkeypatch, cls):
    monkeypatch.setattr(comfy.model_management, "EPHEMERAL_MODELS", [])
    def fail(*args, **kwargs):
        raise RuntimeError("tokenization failed")
    clip = Clip()
    monkeypatch.setattr(clip, "tokenize", fail)
    with pytest.raises(RuntimeError, match="tokenization failed"):
        node_for(cls, False).execute(clip, "Hello", 8, {"sampling_mode": "off"})
    assert len(comfy.model_management.EPHEMERAL_MODELS) == 1


def test_ltx_prompt_keeps_its_two_output_contract(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "EPHEMERAL_MODELS", [])
    schema = TextGenerateLTX2Prompt.GET_SCHEMA()
    assert all(i.id != "state" for i in schema.inputs)
    assert len(schema.outputs) == 2
    assert schema.hidden == [io.Hidden.unique_id, io.Hidden.dynprompt]
    output = node_for(TextGenerateLTX2Prompt, False).execute(Clip(), "A lighthouse", 8, {"sampling_mode": "off"})
    assert output.args == ("answer", "")
    assert len(comfy.model_management.EPHEMERAL_MODELS) == 1


def test_v1_replacement_preserves_state_and_media_connections(monkeypatch):
    replacements = []
    async def register(replacement):
        replacements.append(replacement)
    monkeypatch.setattr(nodes_replacements, "api", SimpleNamespace(node_replacement=SimpleNamespace(register=register)))
    asyncio.run(nodes_replacements.register_replacements_textgenerate())
    replacement, = replacements
    old_inputs = {"clip": ["clip", 0], "state": ["open", 0], "image": ["image", 0],
                  "video": ["video", 0], "audio": ["audio", 0], "prompt": "Continue"}
    migrated = {mapping["new_id"]: old_inputs[mapping["old_id"]] for mapping in replacement.input_mapping
                if mapping.get("old_id") in old_inputs}
    assert replacement.old_node_id == "TextGenerate" and replacement.new_node_id == "TextGenerateV2"
    assert migrated == {"clip": ["clip", 0], "state": ["open", 0], "images.image1": ["image", 0],
                        "videos.video1": ["video", 0], "audios.audio1": ["audio", 0], "prompt": "Continue"}
    assert replacement.output_mapping is None
