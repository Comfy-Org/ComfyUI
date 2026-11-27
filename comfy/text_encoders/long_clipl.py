

def model_options_long_clip(sd, tokenizer_data, model_options):
    w = sd.get("clip_l.text_model.embeddings.position_embedding.weight", None)
    if w is None:
        w = sd.get("clip_g.text_model.embeddings.position_embedding.weight", None)
        model_name = "clip_g" if w is not None else None
    else:
        model_name = "clip_l"

    if w is None:
        w = sd.get("text_model.embeddings.position_embedding.weight", None)
        if w is not None:
            if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
                model_name = "clip_g"
            elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
                model_name = "clip_l"

    if w is not None:
        tokenizer_data = tokenizer_data.copy()
        model_options = model_options.copy()
        model_config = model_options.get("model_config", {})
        model_config["max_position_embeddings"] = w.shape[0]
        model_options["{}_model_config".format(model_name)] = model_config
        tokenizer_data["{}_max_length".format(model_name)] = w.shape[0]
    return tokenizer_data, model_options
