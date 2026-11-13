import pytest
import safetensors.torch
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.sd
from comfy import sd1_clip
from comfy.text_encoders import qwen35
from comfy_extras.nodes_model_merging import CLIPSave


def tokenizer_json(first="alpha", second="beta"):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, first: 3, second: 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.post_processor = processors.TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    return tokenizer.to_str()


def test_embedded_tokenizers_are_scoped_and_invalid_json_is_not_ignored():
    metadata = {"clip_l_tokenizer_json": tokenizer_json(), "clip_g_tokenizer_json": tokenizer_json("beta", "alpha")}
    left = sd1_clip.SDTokenizer(tokenizer_path="/nonexistent", tokenizer_data=metadata)
    right = sd1_clip.SDTokenizer(tokenizer_path="/nonexistent", embedding_key="clip_g", tokenizer_data=metadata)
    assert [t[0] for t in left.tokenize_with_weights("alpha beta")[0]][:4] == [1, 3, 4, 2]
    assert [t[0] for t in right.tokenize_with_weights("alpha beta")[0]][:4] == [1, 4, 3, 2]
    assert left.decode([1, 3, 4, 2]) == "alpha beta"
    with pytest.raises(Exception, match="expected"):
        sd1_clip.SDTokenizer(tokenizer_data={"clip_l_tokenizer_json": "not json"})


def test_clip_metadata_survives_load_clone_save_and_reload(tmp_path, monkeypatch):
    config = {"hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2, "num_hidden_layers": 2, "vocab_size": 5}
    options = {"clip_l_model_config": config, "model_config": config, "load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "initial_device": torch.device("cpu"), "dtype": torch.float32}
    model = sd1_clip.SD1ClipModel(device="cpu", dtype=torch.float32, model_options=options)
    weights = model.clip_l.transformer.state_dict()
    weights = {k: torch.zeros_like(v) for k, v in weights.items()}
    path = tmp_path / "input.safetensors"
    metadata = {"clip_l_tokenizer_json": tokenizer_json(), "prompt": "not tokenizer state"}
    safetensors.torch.save_file(weights, str(path), metadata=metadata)
    clip = comfy.sd.load_clip([str(path)], model_options=options, disable_dynamic=True).clone()
    assert [t[0] for t in clip.tokenize("alpha beta")["l"][0]][:4] == [1, 3, 4, 2]

    # The tokenizer is required model data even when optional prompt metadata is disabled.
    monkeypatch.setattr(args, "disable_metadata", True)
    saver = CLIPSave()
    saver.output_dir = str(tmp_path)
    saver.save(clip, "saved")
    saved = next(tmp_path.glob("saved*.safetensors"))
    with safetensors.safe_open(saved, framework="pt") as f:
        assert f.metadata() == {"clip_l_tokenizer_json": metadata["clip_l_tokenizer_json"]}
    loaded = comfy.sd.load_clip([str(saved)], model_options=options, disable_dynamic=True)
    assert loaded.tokenize("alpha beta") == clip.tokenize("alpha beta")


def test_qwen_uses_embedded_special_ids_and_reduced_geometry():
    vocab = {"[UNK]": 0, "hello": 1, "<|endoftext|>": 2, "<|im_end|>": 3, "<|image_pad|>": 4, "<|im_start|>": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in vocab if t.startswith("<|")])
    data = tokenizer.to_str()
    wrapped = qwen35.Qwen35ImageTokenizer(tokenizer_data={"qwen35_2b_tokenizer_json": data})
    assert wrapped.qwen35_2b.tokenize_with_weights("") == [[(2, 1.0)]]
    image = torch.zeros(1, 1, 3)
    tokens = wrapped.tokenize_with_weights("<|im_start|>hello<|image_pad|><|im_end|>", images=[image])["qwen35_2b"][0]
    assert [t[0] for t in tokens[:2]] == [5, 1]
    assert tokens[2][0]["data"] is image
    assert tokens[3][0] == 3

    # Meta construction exercises real model configuration without allocating weights.
    model = qwen35.te(model_type="qwen35_27b", vocab_size=9, tokenizer_json=data)(device="meta", dtype=torch.float32)
    inner = model.qwen35_27b
    assert inner.transformer.model.embed_tokens.weight.shape == (9, 5120)
    assert inner.transformer.model.lm_head.weight.shape == (9, 5120)
    assert inner.special_tokens == {"pad": 2}
    assert inner.transformer.model.config.stop_tokens == [2, 3]


def test_bundled_qwen_tokenizer_stays_unchanged():
    tokenizer = qwen35.Qwen35Tokenizer()
    assert tokenizer.tokenize_with_weights("Hello world") == [[(9419, 1.0), (1814, 1.0)]]
    assert tokenizer.tokenize_with_weights("") == [[(248044, 1.0)]]
    assert tokenizer.decode([9419, 1814]) == "Hello world"
