"""Checkpoint detection for the Kandinsky 6 joint video/audio DiT."""
import math

from .core_contract import DIT_CONFIG, GENERATION_DEFAULTS

# The generated contract describes Pro. Lite is the same architecture at a
# smaller size, so only its dimensions are listed here (k6_video
# k6_lite_121_480_864_mOffload_nocomp.yaml and the Lite I2VA release config);
# every other setting — flags, sampling, audio — is shared with Pro.
_LITE_DIT_CONFIG = {
    **DIT_CONFIG,
    "time_dim": 512,
    "model_dim": 1792,
    "ff_dim": 7168,
    "num_text_blocks": 2,
    "num_visual_blocks": 32,
    "axes_dims": (16, 24, 24),
    "model_dim_a": 896,
    "time_dim_a": 512,
    "ff_dim_a": 3584,
    "axes_dims_a": (16, 24, 24),
}
# Keyed by the checkpoint's model dimension. The T2VA Lite ships without the
# I2VA token types, so a released size may carry either count.
_RELEASE_DIT_CONFIGS = {
    int(DIT_CONFIG["model_dim"]): DIT_CONFIG,
    int(_LITE_DIT_CONFIG["model_dim"]): _LITE_DIT_CONFIG,
}
_VISUAL_TOKEN_TYPE_COUNTS = (0, int(DIT_CONFIG["visual_token_type_num_embeddings"]))

_K6_REQUIRED_KEYS = (
    "audio_embeddings.in_layer.weight",
    "visual_embeddings.in_layer.weight",
    "out_layer.out_layer.weight",
    "video_time_embeddings.out_layer.weight",
    "audio_time_embeddings.out_layer.weight",
    "video_text_embeddings.in_layer.weight",
    "video_pooled_text_embeddings.in_layer.weight",
    "visual_blocks.0.videoT.feed_forward.in_layer.weight",
    "visual_blocks.0.audioT.feed_forward.in_layer.weight",
    "visual_blocks.0.videoT.self_attention.query_norm.weight",
    "visual_blocks.0.audioT.self_attention.query_norm.weight",
    "visual_blocks.0.va_modulation.out_layer.weight",
    "visual_blocks.0.va_cross_attention.to_key.weight",
)

# Both the original export and the current HF Diffusers layout use the same
# tensors as the native DiT. Normalize module names without copying weights.
_DIFFUSERS_PREFIXES = (
    ("visual_transformer_blocks.", "visual_blocks."),
    ("audio_out_layer.", "audio_outLayer."),
)
_DIFFUSERS_INNER_NAMES = (
    (".timestep_embedder.linear_1.", ".in_layer."),
    (".timestep_embedder.linear_2.", ".out_layer."),
    (".video_dec_block.", ".videoT."),
    (".audio_dec_block.", ".audioT."),
    (".feed_forward.net.0.proj.", ".feed_forward.in_layer."),
    (".feed_forward.net.2.", ".feed_forward.out_layer."),
    (".attn.", ".self_attention."),
)


def _native_key(key):
    for source, target in _DIFFUSERS_PREFIXES:
        if key.startswith(source):
            key = target + key[len(source):]
            break
    for source, target in _DIFFUSERS_INNER_NAMES:
        key = key.replace(source, target)
    return key


def _diffusers_renames(state_dict, key_prefix):
    renames = {}
    for key in state_dict:
        if not key.startswith(key_prefix):
            continue
        target = key_prefix + _native_key(key[len(key_prefix):])
        if target != key:
            if target in state_dict or target in renames.values():
                raise ValueError(
                    "Kandinsky 6 checkpoint mixes Diffusers and native keys: "
                    f"{key!r} conflicts with {target!r}."
                )
            renames[key] = target
    return renames


def to_native_state_dict(state_dict, key_prefix=""):
    """Return the state dict with Diffusers-layout keys converted to native names.

    Used by the supported-model class at weight-load time. The returned
    dictionary holds references to the existing tensors; no weights are copied
    or cast here.
    """
    renames = _diffusers_renames(state_dict, key_prefix)
    if not renames:
        return state_dict
    return {renames.get(key, key): value for key, value in state_dict.items()}


def _detect_k6(state_dict, key_prefix):
    kp = key_prefix

    def get_tensor(name):
        return state_dict[f"{kp}{name}"]

    def count_blocks(stem):
        i = 0
        while '{}{}.{}.videoT.self_attention.to_query.weight'.format(kp, stem, i) in state_dict \
                or '{}{}.{}.self_attention.to_query.weight'.format(kp, stem, i) in state_dict:
            i += 1
        return i

    def count_text_blocks(stem):
        i = 0
        while '{}{}.{}.self_attention.to_query.weight'.format(kp, stem, i) in state_dict:
            i += 1
        return i

    model_dim = get_tensor("visual_embeddings.in_layer.weight").shape[0]
    dit_config = _RELEASE_DIT_CONFIGS.get(model_dim)
    if dit_config is None:
        raise ValueError(
            f"Kandinsky 6 checkpoint has an unreleased model dimension {model_dim}; "
            f"released sizes: {sorted(_RELEASE_DIT_CONFIGS)}."
        )

    patch_size = tuple(int(value) for value in dit_config["patch_size"])
    patch_volume = math.prod(patch_size)
    # Linear input widths come from the matched release contract, not the
    # checkpoint: quantized exports may carry a reduced second dimension.
    in_visual_dim = int(dit_config["in_visual_dim"])

    cfg = {"image_model": "kandinsky6"}
    cfg["in_visual_dim"] = in_visual_dim
    cfg["model_dim"] = model_dim
    cfg["visual_embed_dim"] = (2 * in_visual_dim + 1) * patch_volume
    visual_output_dim = get_tensor("out_layer.out_layer.weight").shape[0]
    if visual_output_dim % patch_volume:
        raise ValueError(
            "Kandinsky 6 visual output projection is not divisible by the "
            f"generated DiT patch volume {patch_volume}."
        )
    cfg["out_visual_dim"] = visual_output_dim // patch_volume
    base_visual_output = int(dit_config["out_visual_dim"])
    if cfg["out_visual_dim"] % base_visual_output:
        raise ValueError("Kandinsky 6 output head does not contain complete DX grids.")
    # A distilled release expands both output heads into n_grid DX grids. The
    # adapter collapses the grids back to a single velocity so a standard
    # flow-matching sampler can consume the checkpoint (PiFlow is not required).
    cfg["n_grid"] = cfg["out_visual_dim"] // base_visual_output
    in_audio_dim = int(dit_config["in_audio_dim"])
    cfg["out_audio_dim"] = in_audio_dim * cfg["n_grid"]
    audio_output_key = f"{kp}audio_outLayer.out_layer.weight"
    if audio_output_key in state_dict and state_dict[audio_output_key].shape[0] != cfg["out_audio_dim"]:
        raise ValueError(
            "Kandinsky 6 audio output head does not match the released "
            f"architecture: expected {cfg['out_audio_dim']} channels, got "
            f"{state_dict[audio_output_key].shape[0]}."
        )
    cfg["model_dim_a"] = get_tensor("audio_embeddings.in_layer.weight").shape[0]
    cfg["in_audio_dim"] = in_audio_dim
    cfg["time_dim"] = get_tensor("video_time_embeddings.out_layer.weight").shape[0]
    cfg["time_dim_a"] = get_tensor("audio_time_embeddings.out_layer.weight").shape[0]
    cfg["in_text_dim"] = int(dit_config["in_text_dim"])
    cfg["in_text_dim2"] = int(dit_config["in_text_dim2"])
    cfg["ff_dim"] = get_tensor("visual_blocks.0.videoT.feed_forward.in_layer.weight").shape[0]
    cfg["ff_dim_a"] = get_tensor("visual_blocks.0.audioT.feed_forward.in_layer.weight").shape[0]

    video_head_dim = get_tensor("visual_blocks.0.videoT.self_attention.query_norm.weight").shape[0]
    cfg["axes_dims"] = tuple(int(value) for value in dit_config["axes_dims"])
    cfg["head_dim_a"] = get_tensor("visual_blocks.0.audioT.self_attention.query_norm.weight").shape[0]
    cfg["num_visual_blocks"] = count_blocks("visual_blocks")
    cfg["num_text_blocks"] = count_text_blocks("video_text_transformer_blocks")
    audio_text_blocks = count_text_blocks("audio_text_transformer_blocks")

    token_type_key = "visual_token_type_embeddings.weight"
    if '{}{}'.format(kp, token_type_key) in state_dict:
        cfg["visual_token_type_num_embeddings"] = get_tensor(token_type_key).shape[0]
    else:
        cfg["visual_token_type_num_embeddings"] = 0

    cfg["patch_size"] = patch_size
    cfg["rope_scale_factor"] = tuple(
        float(value) for value in GENERATION_DEFAULTS["scale_factor"]
    )
    cfg["freqs_scaling"] = float(dit_config["audio_freqs_scaling"])
    cfg["cross_gates"] = bool(dit_config["cross_gates"])
    cfg["fix_modulation"] = bool(dit_config["fix_modulation"])
    cfg["ca_rope"] = bool(dit_config["ca_rope"])

    expected = {
        "out_visual_dim": int(dit_config["out_visual_dim"]) * cfg["n_grid"],
        "model_dim": int(dit_config["model_dim"]),
        "model_dim_a": int(dit_config["model_dim_a"]),
        "time_dim": int(dit_config["time_dim"]),
        "time_dim_a": int(dit_config["time_dim_a"]),
        "ff_dim": int(dit_config["ff_dim"]),
        "ff_dim_a": int(dit_config["ff_dim_a"]),
        "num_visual_blocks": int(dit_config["num_visual_blocks"]),
        "num_text_blocks": int(dit_config["num_text_blocks"]),
    }
    mismatches = [
        f"{name}: checkpoint={cfg[name]!r}, core={value!r}"
        for name, value in expected.items()
        if cfg[name] != value
    ]
    # T2VA checkpoints have no I2VA reference-tail embeddings.
    if cfg["visual_token_type_num_embeddings"] not in _VISUAL_TOKEN_TYPE_COUNTS:
        mismatches.append(
            "visual_token_type_num_embeddings: "
            f"checkpoint={cfg['visual_token_type_num_embeddings']!r}, "
            f"core={_VISUAL_TOKEN_TYPE_COUNTS!r}"
        )
    if video_head_dim != sum(cfg["axes_dims"]):
        mismatches.append(
            f"video_head_dim: checkpoint={video_head_dim!r}, "
            f"core={sum(cfg['axes_dims'])!r}"
        )
    expected_audio_head_dim = sum(int(value) for value in dit_config["axes_dims_a"])
    if cfg["head_dim_a"] != expected_audio_head_dim:
        mismatches.append(
            f"audio_head_dim: checkpoint={cfg['head_dim_a']!r}, "
            f"core={expected_audio_head_dim!r}"
        )
    if audio_text_blocks != expected["num_text_blocks"]:
        mismatches.append(
            f"audio_text_blocks: checkpoint={audio_text_blocks!r}, "
            f"core={expected['num_text_blocks']!r}"
        )

    expected_va_modulation = (
        2 * cfg["model_dim"] + cfg["model_dim_a"]
        if cfg["cross_gates"]
        else 3 * cfg["model_dim"]
    )
    actual_va_modulation = get_tensor("visual_blocks.0.va_modulation.out_layer.weight").shape[0]
    if actual_va_modulation != expected_va_modulation:
        mismatches.append(
            f"va_modulation: checkpoint={actual_va_modulation!r}, "
            f"core={expected_va_modulation!r}"
        )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Kandinsky 6 checkpoint does not match the released architecture "
            f"({details})."
        )
    return cfg


def detect_kandinsky6(state_dict, key_prefix):
    normalized_keys = {
        _native_key(key[len(key_prefix):]) for key in state_dict if key.startswith(key_prefix)
    }
    if not all(name in normalized_keys for name in _K6_REQUIRED_KEYS):
        return None
    renames = _diffusers_renames(state_dict, key_prefix)
    if renames:
        state_dict = {renames.get(key, key): value for key, value in state_dict.items()}
    return _detect_k6(state_dict, key_prefix)
