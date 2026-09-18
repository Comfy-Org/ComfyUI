import torch

from comfy.lora import load_lora


def make_lora_dict(prefix="test_layer", rank=4, in_dim=8):
    return {
        "{}.lora_down.weight".format(prefix): torch.zeros((rank, in_dim)),
        "{}.lora_up.weight".format(prefix): torch.zeros((in_dim, rank)),
    }


def get_adapter(load_result, target="model.test_layer"):
    return load_result[target]


def test_metadata_alpha_used_when_tensor_missing():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}
    # SimpleTuner-style file: alpha/rank only in safetensors metadata,
    # namespaced ("transformer.r", "transformer.lora_alpha").
    metadata = {"transformer.r": "16", "transformer.lora_alpha": "64"}

    loaded = load_lora(lora, key_map, metadata=metadata)

    adapter = get_adapter(loaded)
    assert adapter.weights[2] == 64.0 / 16.0


def test_metadata_plain_keys_also_work():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}
    metadata = {"r": "8", "lora_alpha": "24"}

    loaded = load_lora(lora, key_map, metadata=metadata)

    assert get_adapter(loaded).weights[2] == 3.0


def test_explicit_alpha_tensor_wins_over_metadata():
    lora = make_lora_dict()
    lora["test_layer.alpha"] = torch.tensor(2.0)
    key_map = {"test_layer": "model.test_layer"}
    metadata = {"r": "16", "lora_alpha": "64"}

    loaded = load_lora(lora, key_map, metadata=metadata)

    assert get_adapter(loaded).weights[2] == 2.0


def test_no_metadata_keeps_old_behavior():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}

    loaded = load_lora(lora, key_map)

    assert get_adapter(loaded).weights[2] is None


def test_malformed_metadata_ignored():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}
    metadata = {"transformer.r": "sixteen", "transformer.lora_alpha": "x"}

    loaded = load_lora(lora, key_map, metadata=metadata)

    assert get_adapter(loaded).weights[2] is None


def test_ambiguous_namespaces_skip_fallback():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}
    metadata = {
        "text_encoder.lora_alpha": "8",
        "text_encoder.r": "8",
        "transformer.lora_alpha": "64",
        "transformer.r": "16",
    }

    loaded = load_lora(lora, key_map, metadata=metadata)

    assert get_adapter(loaded).weights[2] is None


def test_non_finite_metadata_ignored():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}

    loaded = load_lora(lora, key_map, metadata={"transformer.r": "16", "transformer.lora_alpha": "nan"})
    assert get_adapter(loaded).weights[2] is None

    loaded = load_lora(lora, key_map, metadata={"transformer.r": "16", "transformer.lora_alpha": "inf"})
    assert get_adapter(loaded).weights[2] is None

    loaded = load_lora(lora, key_map, metadata={"transformer.r": "0", "transformer.lora_alpha": "64"})
    assert get_adapter(loaded).weights[2] is None


def test_integer_metadata_values_accepted():
    lora = make_lora_dict()
    key_map = {"test_layer": "model.test_layer"}
    metadata = {"r": 16, "lora_alpha": 64}

    loaded = load_lora(lora, key_map, metadata=metadata)

    assert get_adapter(loaded).weights[2] == 4.0
