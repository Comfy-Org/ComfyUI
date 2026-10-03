# Detection + loading for the Hunyuan3D 2.1 paint (hunyuan3d-paintpbr-v2-1) UNet.
# Detection is pure state_dict-shape inspection (used by comfy.model_detection to
# route the checkpoint into the Hunyuan3DPaint model config); loading goes through
# core's load_diffusion_model_state_dict so ModelPatcher/model_management own dtype
# selection, device placement and offload, with paint-specific error diagnosis
# layered on top.

# Fixed SD-2.1 head dim of the released paint checkpoint (config.json:
# attention_head_dim [5, 10, 20, 20] over block_out_channels [320, 640, 1280, 1280]).
_HEAD_DIM = 64


def detect_paint_config(state_dict, prefix=""):
    """Return a UNet2p5DConditionModel config dict if ``state_dict`` looks like the
    Hunyuan3D 2.1 paint UNet, else ``None``. Everything derivable is read from tensor
    shapes; only the (weight-invisible) attention head dim and group count are assumed.

    Raises ValueError (never KeyError) when the checkpoint matches the paint family
    but is missing keys the config derivation needs (truncated / renamed weights).
    """
    ck = lambda k: f"{prefix}{k}"  # noqa: E731
    conv_in = ck("unet.conv_in.weight")
    if conv_in not in state_dict:
        return None
    if ck("unet_dual.conv_in.weight") not in state_dict:
        return None
    if ck("unet.learned_text_clip_albedo") not in state_dict:
        return None

    def need(key):
        full = ck(key)
        if full not in state_dict:
            raise ValueError(
                f"checkpoint looks like a Hunyuan3D 2.1 paint UNet but is missing "
                f"'{full}'; the file appears truncated or its keys were renamed "
                f"(expected the hunyuan3d-paintpbr-v2-1 layout)")
        return state_dict[full]

    in_channels = state_dict[conv_in].shape[1]
    ref_in_channels = state_dict[ck("unet_dual.conv_in.weight")].shape[1]
    out_channels = need("unet.conv_out.weight").shape[0]

    # block_out_channels from each down block's resnet output width
    block_out_channels = []
    i = 0
    while ck(f"unet.down_blocks.{i}.resnets.0.conv2.weight") in state_dict:
        block_out_channels.append(state_dict[ck(f"unet.down_blocks.{i}.resnets.0.conv2.weight")].shape[0])
        i += 1

    # layers_per_block = number of resnets in the first down block
    layers_per_block = 0
    while ck(f"unet.down_blocks.0.resnets.{layers_per_block}.conv2.weight") in state_dict:
        layers_per_block += 1

    # transformer_layers_per_block
    transformer_layers = 0
    while ck(f"unet.down_blocks.0.attentions.0.transformer_blocks.{transformer_layers}.transformer.norm1.weight") in state_dict:
        transformer_layers += 1

    if not block_out_channels or layers_per_block == 0 or transformer_layers == 0:
        raise ValueError(
            "checkpoint looks like a Hunyuan3D 2.1 paint UNet but its down_blocks "
            "layout is incomplete; the file appears truncated (expected the "
            "hunyuan3d-paintpbr-v2-1 layout)")

    cross_attention_dim = need(
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn2.to_k.weight").shape[1]

    num_attention_heads = [max(1, c // _HEAD_DIM) for c in block_out_channels]

    # PBR settings from learned_text_clip_{token}
    pbr_setting = ["albedo"]
    if ck("unet.learned_text_clip_mr") in state_dict:
        pbr_setting.append("mr")
    pbr_token_channels = state_dict[ck("unet.learned_text_clip_albedo")].shape[0]

    use_dino = ck("unet.image_proj_model_dino.proj.weight") in state_dict
    dino_embeddings_dim = 1536
    if use_dino:
        dino_embeddings_dim = state_dict[ck("unet.image_proj_model_dino.proj.weight")].shape[1]

    return {
        "in_channels": in_channels,
        "ref_in_channels": ref_in_channels,
        "out_channels": out_channels,
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": layers_per_block,
        "cross_attention_dim": cross_attention_dim,
        "num_attention_heads": tuple(num_attention_heads),
        "transformer_layers_per_block": transformer_layers,
        "norm_num_groups": 32,
        "pbr_setting": tuple(pbr_setting),
        "pbr_token_channels": pbr_token_channels,
        "dino_embeddings_dim": dino_embeddings_dim,
        "use_dino": use_dino,
    }


def _describe_checkpoint_family(state_dict):
    """Best-effort guess at what an unrecognised checkpoint actually is, so loader
    errors say "this looks like X" instead of a bare key dump."""
    keys = list(state_dict.keys())

    def has(prefix):
        return any(k.startswith(prefix) for k in keys)

    if has("unet.conv_in") and not has("unet_dual."):
        return ("a single-stream 'unet.*' checkpoint without the paint model's "
                "dual-stream reference UNet ('unet_dual.*')")
    if has("model.diffusion_model."):
        return ("a ComfyUI/LDM diffusion checkpoint ('model.diffusion_model.*') - "
                "load it with the standard checkpoint/diffusion-model loader")
    if has("double_blocks.") or has("single_blocks.") or has("joint_blocks."):
        return "a DiT-style diffusion model (double/single/joint blocks)"
    if has("down_blocks.") and has("conv_in."):
        return ("a plain diffusers UNet (unprefixed 'down_blocks.*') without the "
                "paint model's 'unet.*'/'unet_dual.*' dual-stream layout")
    if has("decoder.") and has("encoder."):
        return "a VAE (encoder/decoder) checkpoint"
    sample = ", ".join(sorted(keys)[:3]) if keys else "no keys at all"
    return f"an unrecognised checkpoint (first keys: {sample})"


def load_paint_unet(state_dict, model_options={}):
    """Load a hunyuan3d-paintpbr-v2-1 state_dict through core's diffusion-model
    path (detection -> Hunyuan3DPaint model config -> BaseModel in a ModelPatcher)
    and return the ModelPatcher.

    Raises ValueError with a family diagnosis when the checkpoint is not a
    hunyuan3d-paintpbr-v2-1 paint UNet, or when it matches the family but is
    missing weights (truncated file)."""
    import comfy.sd  # deferred: this module is imported by comfy.model_detection

    config = detect_paint_config(state_dict)
    if config is None:
        raise ValueError(
            f"this looks like {_describe_checkpoint_family(state_dict)}; expected a "
            f"Hunyuan3D 2.1 paint UNet (hunyuan3d-paintpbr-v2-1) with "
            f"'unet.*'/'unet_dual.*' keys and learned_text_clip_* embeddings")

    provided_keys = set(state_dict.keys())
    # core's load path pops weights out of the dict it is given; keep the caller's
    patcher = comfy.sd.load_diffusion_model_state_dict(dict(state_dict), model_options=model_options)
    if patcher is None:
        raise ValueError(
            "checkpoint detected as a Hunyuan3D 2.1 paint UNet but core model "
            "detection could not build it; the file appears corrupted")

    # core loads strict=False and only warns; a paint checkpoint that passes
    # detection but lacks weights would silently keep random tensors, so hard-fail
    missing = [k for k in patcher.model.diffusion_model.state_dict().keys() if k not in provided_keys]
    if missing:
        raise ValueError(
            f"Hunyuan3D 2.1 paint checkpoint is missing {len(missing)} weights "
            f"(e.g. {missing[:3]}); the file appears truncated or from a different "
            f"model revision than hunyuan3d-paintpbr-v2-1")
    return patcher
