import asyncio
import base64
import io
import struct
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from comfy.text_encoders import qwen_vl
from comfy.text_encoders import qwen3vl, qwen_image
from comfy.text_encoders.llama import BaseGenerate
from comfy_api.latest import _llama_cpp, _sdk
from comfy_api.latest._sdk import (
    ClipRef,
    ExecutionPlan,
    ImageRef,
    InProcessCtxProvider,
    InProcessOps,
    InProcessRefResolver,
    LlamaCppModelRef,
    bind_runtime,
)


def _plan():
    return ExecutionPlan(
        prompt_id="qwen",
        node_id="1",
        node_type="qwen-test",
        prompt={"1": {"class_type": "qwen-test"}},
        extra_pnginfo={},
    )


def test_qwen_media_preprocessing_keeps_selected_frames_and_exact_mrope():
    frames = torch.arange(3 * 64 * 96 * 3, dtype=torch.float32).reshape(
        3, 64, 96, 3)
    frames = frames / frames.max()

    qwen25_patches, qwen25_grid, qwen25_mrope = (
        qwen_vl.process_qwen_vl_media(
            frames, family="qwen2_5_vl_7b"))
    qwen3_patches, qwen3_grid, qwen3_mrope = (
        qwen_vl.process_qwen_vl_media(
            frames, family="qwen3vl_4b"))

    assert qwen25_grid.tolist() == [[2, 4, 6]]
    assert qwen25_patches.shape == (48, 3 * 2 * 14 * 14)
    assert qwen25_mrope.shape == (3, 12)
    assert qwen25_mrope[0].tolist() == [0] * 6 + [2] * 6
    assert qwen3_grid.tolist() == [[2, 4, 6]]
    assert qwen3_patches.shape == (48, 3 * 2 * 16 * 16)
    assert qwen3_mrope[0].tolist() == [0] * 12

    # Qwen3 budgets spatial resize using the temporally padded length while
    # retaining the original frame count in the official beta calculation.
    assert qwen_vl._qwen_smart_resize(
        2080,
        2080,
        factor=32,
        min_pixels=4096,
        max_pixels=25_165_824,
        frames=5,
        padded_frames=6,
    ) == (2240, 2240)


def test_qwen_bounded_beam_generation_is_deterministic():
    class FakeModel:
        config = SimpleNamespace(stop_tokens=[3])

        @staticmethod
        def embed_tokens(tokens):
            return tokens.to(dtype=torch.float32).unsqueeze(-1)

        @staticmethod
        def forward(
            _unused, *, embeds, attention_mask, past_key_values,
            input_ids, position_ids, **kwargs,
        ):
            return embeds, None, past_key_values

    class FakeGenerator(BaseGenerate):
        model = FakeModel()

        @staticmethod
        def init_kv_cache(batch, max_cache_len, device, execution_dtype):
            return []

        @staticmethod
        def logits(value):
            marker = int(value[0, -1, 0].item())
            if marker == 9:  # prefill: token 1 narrowly beats token 2
                return torch.tensor([[[0.0, 4.0, 3.8, -10.0]]])
            if marker == 1:  # the best complete branch
                return torch.tensor([[[0.0, -2.0, -2.0, 5.0]]])
            return torch.tensor([[[0.0, -2.0, 4.0, 1.0]]])

    generated = FakeGenerator().generate(
        embeds=torch.tensor([[[9.0]]]),
        do_sample=False,
        max_length=3,
        num_beams=2,
    )
    assert generated == [1, 3]


def test_qwen_static_templates_keep_distinct_still_and_video_media():
    image = torch.zeros((1, 8, 8, 3))
    video = torch.zeros((3, 8, 8, 3))

    qwen3 = qwen3vl.generation_tokenizer(
        "qwen3vl_4b")(embedding_directory=[])
    qwen3_tokens = qwen3.tokenize_with_weights(
        "hello", image=image, video=video)
    qwen3_descriptors = [
        item[0]
        for row in qwen3_tokens[qwen3.clip_name]
        for item in row
        if isinstance(item[0], dict)
    ]
    assert [(item["type"], item.get("segment")) for item in qwen3_descriptors] == [
        ("image", None),
        ("video_segment", 0),
        ("video_segment", 1),
    ]
    plain = qwen3.tokenize_with_weights("hello", thinking=False)
    thinking = qwen3.tokenize_with_weights("hello", thinking=True)

    def decode(tokenizer, rows):
        ids = [item[0] for row in rows[tokenizer.clip_name] for item in row]
        return getattr(tokenizer, tokenizer.clip).decode(
            ids, skip_special_tokens=False)

    assert decode(qwen3, plain).endswith("<|im_start|>assistant\n")
    assert decode(qwen3, thinking).endswith(
        "<|im_start|>assistant\n<think>\n")

    qwen25 = qwen_image.vl_tokenizer(
        "qwen2_5_vl_7b")(embedding_directory=[])
    qwen25_tokens = qwen25.tokenize_with_weights(
        "hello", image=image, video=video)
    qwen25_descriptors = [
        item[0]
        for row in qwen25_tokens[qwen25.clip_name]
        for item in row
        if isinstance(item[0], dict)
    ]
    assert [item["type"] for item in qwen25_descriptors] == [
        "image", "video",
    ]


def test_clip_generation_accepts_still_and_video_with_family_defaults():
    class FakeClip:
        _secure_language_family = "qwen3_vl_4b"

        def __init__(self):
            self.calls = []

        def tokenize(self, prompt, **kwargs):
            self.calls.append(("tokenize", prompt, kwargs))
            return {"qwen": [[(1, 1.0)]]}

        def generate(self, tokens, **kwargs):
            self.calls.append(("generate", tokens, kwargs))
            return [7, 8]

        @staticmethod
        def decode(tokens):
            return " generated "

    async def run():
        refs = InProcessRefResolver()
        value = FakeClip()
        clip = ClipRef._wrap(await refs.create("CLIP", value))
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 8, 8, 3))))
        video = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((3, 8, 8, 3))))
        with bind_runtime(refs, None, InProcessOps()):
            result = await clip.generate_text(
                "describe",
                image=image,
                video=video,
                top_k=None,
                num_beams=2,
            )
            with pytest.raises(ValueError, match="cannot also enable sampling"):
                await clip.generate_text(
                    "x", do_sample=True, num_beams=2)
        return result, value.calls

    result, calls = asyncio.run(run())
    assert result == "generated"
    assert calls[0][2]["image"].shape == (1, 8, 8, 3)
    assert calls[0][2]["video"].shape == (3, 8, 8, 3)
    assert calls[1][2]["top_k"] == 20
    assert calls[1][2]["num_beams"] == 2


def test_qwen_shards_merge_remap_and_dequantize_once(monkeypatch, tmp_path):
    import comfy.sd
    import comfy.text_encoders.hunyuan_video
    import comfy.utils
    import folder_paths

    shard_a = tmp_path / "a.safetensors"
    shard_b = tmp_path / "b.safetensors"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    states = {
        str(shard_a): {
            "model.language_model.embed_tokens.weight": torch.ones((2, 3)),
            "model.language_model.layers.0.mlp.weight": torch.full((2, 3), 2.0),
            "model.language_model.layers.0.mlp.weight_scale_inv": torch.full((1, 1), 3.0),
        },
        str(shard_b): {
            "model.visual.patch_embed.weight": torch.full((1,), 4.0),
            "lm_head.weight": torch.full((2, 3), 5.0),
        },
    }
    captured = {}

    monkeypatch.setattr(
        comfy.utils,
        "load_torch_file",
        lambda path, **kwargs: (dict(states[path]), {}),
    )
    monkeypatch.setattr(
        comfy.utils, "convert_old_quants", lambda state, **kwargs: (state, {}))
    monkeypatch.setattr(
        comfy.utils,
        "calculate_parameters",
        lambda state: sum(value.numel() for value in state.values()),
    )
    monkeypatch.setattr(
        comfy.text_encoders.hunyuan_video, "llama_detect", lambda state: {})
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [])

    class FakeClip:
        def __init__(self, target, **kwargs):
            captured["target"] = target
            captured.update(kwargs)

        @staticmethod
        def generate(*args, **kwargs):
            return []

    monkeypatch.setattr(comfy.sd, "CLIP", FakeClip)

    entry = _sdk._load_qwen_language_model(
        (str(shard_a), str(shard_b)), "qwen3_vl_4b", "cpu")
    state = captured["state_dict"][0]
    assert isinstance(entry.clip, FakeClip)
    assert set(state) == {
        "model.embed_tokens.weight",
        "model.layers.0.mlp.weight",
        "visual.patch_embed.weight",
        "model.lm_head.weight",
    }
    assert torch.equal(
        state["model.layers.0.mlp.weight"],
        torch.full((2, 3), 6.0, dtype=torch.bfloat16),
    )
    assert captured["model_options"]["qwen3vl_4b_model_config"] == {
        "lm_head": True,
    }
    # The public family uses the stable SDK spelling; the native Comfy class
    # keeps its existing internal model key.
    captured["target"].tokenizer(embedding_directory=[])


def test_llama_cpp_vendor_ref_hides_paths_and_encodes_media(
    monkeypatch, tmp_path,
):
    calls = []

    class FakeHandler:
        def __init__(self, **kwargs):
            calls.append(("handler", kwargs))

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(("load", kwargs))

        def create_chat_completion(self, **kwargs):
            calls.append(("generate", kwargs))
            return {"choices": [{"message": {"content": " described "}}]}

    formats = SimpleNamespace(
        Qwen3VLChatHandler=FakeHandler,
        Qwen25VLChatHandler=FakeHandler,
    )
    monkeypatch.setattr(
        _llama_cpp, "_classes", lambda: (FakeLlama, formats))
    _llama_cpp._CACHE.clear()

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    gguf_header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 0)
    model_path.write_bytes(gguf_header)
    mmproj_path.write_bytes(gguf_header)

    import folder_paths

    def resolve(folder, logical):
        assert folder == "text_encoders"
        return str(model_path if logical == "model.gguf" else mmproj_path)

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", resolve)

    async def run():
        refs = InProcessRefResolver()
        context = InProcessCtxProvider().build(_plan())
        with bind_runtime(refs, context, InProcessOps()):
            model = await context.integrations.call("llama_cpp", "load_chat_model", model_weight="model.gguf", mmproj_weight="mmproj.gguf", family="qwen3_vl", device="cpu")
            assert isinstance(model, LlamaCppModelRef)
            image = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((1, 4, 5, 3))))
            video = ImageRef._wrap(await refs.create(
                "IMAGE", torch.zeros((2, 6, 7, 3))))
            result = await model.generate(
                "system", "prompt", image=image, video=video,
                max_tokens=32, seed=9)
            with pytest.raises(ValueError, match="require an mmproj"):
                await context.integrations.call("llama_cpp", "load_chat_model", model_weight="model.gguf", family="qwen3_vl")
        return result

    assert asyncio.run(run()) == "described"
    load = next(value for kind, value in calls if kind == "load")
    assert load["model_path"] == str(model_path)
    assert load["n_gpu_layers"] == 0
    generated = next(value for kind, value in calls if kind == "generate")
    content = generated["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "prompt"}
    assert len(content) == 4
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,")
        for item in content[1:]
    )
    sizes = []
    for item in content[1:]:
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as media:
            sizes.append(media.size)
    assert sizes == [(5, 4), (7, 6), (7, 6)]
    assert "model_path" not in generated
    _llama_cpp._CACHE.clear()
