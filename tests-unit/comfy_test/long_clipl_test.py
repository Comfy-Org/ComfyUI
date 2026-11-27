import pytest
import torch

from comfy.text_encoders.long_clipl import model_options_long_clip


def test_prefixed_clip_g_state_dict_configures_clip_g():
    state_dict = {
        "clip_g.text_model.embeddings.position_embedding.weight": torch.zeros(248, 1280)
    }

    tokenizer_data, model_options = model_options_long_clip(state_dict, {}, {})

    assert tokenizer_data["clip_g_max_length"] == 248
    assert model_options["clip_g_model_config"]["max_position_embeddings"] == 248
    assert "clip_l_max_length" not in tokenizer_data


def test_prefixed_clip_l_state_dict_configures_clip_l():
    state_dict = {
        "clip_l.text_model.embeddings.position_embedding.weight": torch.zeros(248, 768),
        "clip_g.text_model.embeddings.position_embedding.weight": torch.zeros(248, 1280),
    }

    tokenizer_data, model_options = model_options_long_clip(state_dict, {}, {})

    assert tokenizer_data["clip_l_max_length"] == 248
    assert model_options["clip_l_model_config"]["max_position_embeddings"] == 248


def test_bare_embedding_without_layer_marker_fails_clearly():
    state_dict = {
        "text_model.embeddings.position_embedding.weight": torch.zeros(248, 768)
    }

    with pytest.raises(ValueError, match="Cannot determine CLIP-L or CLIP-G"):
        model_options_long_clip(state_dict, {}, {})
