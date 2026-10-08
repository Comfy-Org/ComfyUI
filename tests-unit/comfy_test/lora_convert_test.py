import logging

import torch

import comfy.lora
import comfy.lora_convert
from comfy.weight_adapter.lokr import LoKrAdapter

MAIN_PREFIX = "lycoris_transformer_blocks_0_attn_to_qkv"
TEXT_PREFIX = "lycoris_text_fusion_layerwise_blocks_0_attn_to_qkv"
MAIN_OUT, MAIN_IN = 9216, 6144  # 48 q heads + 2 * 12 kv heads, head dim 128
TEXT_OUT, TEXT_IN = 7680, 2560  # 20 q/k/v heads, head dim 128


def _fused_lokr(prefix, out_l, out_k, in_m, in_n):
    torch.manual_seed(0)
    return {
        prefix + ".lokr_w1": torch.randn(out_l, in_m),
        prefix + ".lokr_w2": torch.randn(out_k, in_n),
        prefix + ".alpha": torch.tensor(float(out_l)),
        prefix + ".dora_scale": torch.randn(out_l * out_k, 1),
    }


def _make_sd(out_l, out_k, in_m, in_n):
    # "text_fusion" marks the file as Krea2, the fused to_qkv keys select the converter.
    sd = _fused_lokr(MAIN_PREFIX, out_l, out_k, in_m, in_n)
    sd.update(_fused_lokr(TEXT_PREFIX, 24, 320, 20, 128))
    return sd


def _apply(sd, base, out_dim, in_dim):
    adapter = LoKrAdapter.load(base, sd, float(sd[base + ".alpha"].item()), sd[base + ".dora_scale"], set())
    return comfy.lora.calculate_weight([(1.0, adapter, 1.0, None, None)], torch.zeros(out_dim, in_dim), base)


def _assert_projections_match(converted, reference, fused_prefix, out_dim, in_dim):
    assert fused_prefix not in converted
    sizes = [in_dim, (out_dim - in_dim) // 2, (out_dim - in_dim) // 2]
    offset = 0
    for proj, size in zip("qkv", sizes):
        base = fused_prefix.replace("qkv", "") + proj
        assert torch.allclose(_apply(converted, base, size, in_dim), reference[offset:offset + size], atol=1e-4)
        offset += size


def _capture_warnings(fn):
    """Run fn and return the warnings it logs and its result."""
    root = logging.getLogger()
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        result = fn()
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
    return messages, result


def test_krea2_fused_qkv_lokr_splits_into_projections():
    sd = _make_sd(24, 384, 24, 256)  # 24 * 384 = 9216 out, 24 * 256 = 6144 in
    reference = _apply(sd, MAIN_PREFIX, MAIN_OUT, MAIN_IN)
    converted = comfy.lora_convert.convert_lora(sd)

    # The aligned factorization is split on the outer factor, no row offsets needed.
    assert converted[MAIN_PREFIX.replace("qkv", "") + "q.lokr_w1"].shape == (16, 24)
    assert converted[MAIN_PREFIX.replace("qkv", "") + "k.lokr_w1"].shape == (4, 24)
    assert MAIN_PREFIX.replace("qkv", "") + "q.row_offset" not in converted
    _assert_projections_match(converted, reference, MAIN_PREFIX, MAIN_OUT, MAIN_IN)


def test_krea2_fused_qkv_lokr_with_unaligned_factorization_uses_row_offset():
    sd = _make_sd(16, 576, 16, 384)  # 576 does not divide the 6144/1536 projection widths
    reference = _apply(sd, MAIN_PREFIX, MAIN_OUT, MAIN_IN)
    converted = comfy.lora_convert.convert_lora(sd)

    assert converted[MAIN_PREFIX.replace("qkv", "") + "q.row_offset"].item() == 0
    assert converted[MAIN_PREFIX.replace("qkv", "") + "k.row_offset"].item() == MAIN_IN
    assert converted[MAIN_PREFIX.replace("qkv", "") + "v.row_offset"].item() == MAIN_IN + 1536
    _assert_projections_match(converted, reference, MAIN_PREFIX, MAIN_OUT, MAIN_IN)


def test_krea2_fused_qkv_lora_splits_up_rows():
    sd = {
        MAIN_PREFIX + ".lora_A.weight": torch.randn(8, MAIN_IN),
        MAIN_PREFIX + ".lora_B.weight": torch.randn(MAIN_OUT, 8),
        MAIN_PREFIX + ".lora_mid.weight": torch.randn(8, 8),
        MAIN_PREFIX + ".alpha": torch.tensor(8.0),
        MAIN_PREFIX + ".dora_scale": torch.randn(MAIN_OUT, 1),
        TEXT_PREFIX + ".lora_A.weight": torch.randn(8, TEXT_IN),
        TEXT_PREFIX + ".lora_B.weight": torch.randn(TEXT_OUT, 8),
    }
    converted = comfy.lora_convert.convert_lora(sd)

    assert MAIN_PREFIX not in converted
    assert converted[MAIN_PREFIX.replace("qkv", "") + "q.lora_B.weight"].shape == (MAIN_IN, 8)
    assert converted[MAIN_PREFIX.replace("qkv", "") + "k.lora_B.weight"].shape == (1536, 8)
    assert converted[MAIN_PREFIX.replace("qkv", "") + "v.lora_B.weight"].shape == (1536, 8)
    assert converted[MAIN_PREFIX.replace("qkv", "") + "q.lora_A.weight"].shape == (8, MAIN_IN)
    assert MAIN_PREFIX + ".lora_mid.weight" not in converted
    assert torch.equal(converted[MAIN_PREFIX.replace("qkv", "") + "v.lora_mid.weight"], sd[MAIN_PREFIX + ".lora_mid.weight"])


def test_krea2_unused_per_projection_qkv_placeholders_are_skipped():
    sd = _make_sd(24, 384, 24, 256)
    fused_alpha = sd[MAIN_PREFIX + ".alpha"].item()
    fused_w2 = sd[MAIN_PREFIX + ".lokr_w2"]
    fused_dora = sd[MAIN_PREFIX + ".dora_scale"]
    q = MAIN_PREFIX.replace("qkv", "q")
    # simpletuner also writes per-projection entries next to the fused adapter. They belong to the
    # unused unfused modules: no trained delta, only an initialized DoRA scale.
    sd[q + ".lokr_w1"] = torch.randn(24, 24)
    sd[q + ".lokr_w2"] = torch.zeros(256, 256)
    sd[q + ".alpha"] = torch.tensor(3.0)
    sd[q + ".dora_scale"] = torch.randn(MAIN_IN, 1)

    messages, converted = _capture_warnings(lambda: comfy.lora_convert.convert_lora(sd))

    assert converted[q + ".lokr_w1"].shape == (16, 24)
    assert converted[q + ".lokr_w2"].shape == fused_w2.shape
    assert converted[q + ".alpha"].item() == fused_alpha
    assert torch.equal(converted[q + ".dora_scale"], fused_dora[:MAIN_IN])
    assert messages == []


def test_krea2_trained_per_projection_qkv_adapter_is_reported():
    sd = _make_sd(24, 384, 24, 256)
    q = MAIN_PREFIX.replace("qkv", "q")
    sd[q + ".lokr_w1"] = torch.randn(24, 24)
    sd[q + ".lokr_w2"] = torch.randn(256, 256)
    sd[q + ".alpha"] = torch.tensor(24.0)

    messages, converted = _capture_warnings(lambda: comfy.lora_convert.convert_lora(sd))

    assert converted[q + ".lokr_w2"].shape == sd[MAIN_PREFIX + ".lokr_w2"].shape
    assert len(messages) == 1 and q in messages[0] and "not applied" in messages[0]


def test_load_lora_converts_krea2_fused_qkv_lora():
    # comfy.hooks and custom nodes call comfy.lora.load_lora without comfy.lora_convert, so the
    # fused to_qkv adapter has to be split for them as well.
    sd = _make_sd(16, 576, 16, 384)  # not q/k/v aligned, so the projections need row offsets
    prefix = MAIN_PREFIX.replace("qkv", "")
    for proj, rows in zip("qkv", (MAIN_IN, 1536, 1536)):
        # the unused per-projection placeholders LyCORIS writes next to the fused adapter
        sd[prefix + proj + ".lokr_w1"] = torch.randn(16, 16)
        sd[prefix + proj + ".lokr_w2"] = torch.zeros(rows // 16, 384)
        sd[prefix + proj + ".alpha"] = sd[MAIN_PREFIX + ".alpha"]
        sd[prefix + proj + ".dora_scale"] = torch.randn(rows, 1)

    fused = LoKrAdapter.load(MAIN_PREFIX, sd, float(sd[MAIN_PREFIX + ".alpha"].item()), sd[MAIN_PREFIX + ".dora_scale"], set())
    reference = comfy.lora.calculate_weight([(1.0, fused, 1.0, None, None)], torch.zeros(MAIN_OUT, MAIN_IN), MAIN_PREFIX)

    key_map = {prefix + proj: "diffusion_model.blocks.0.attn." + p + ".weight" for proj, p in zip("qkv", ("wq", "wk", "wv"))}
    messages, patches = _capture_warnings(lambda: comfy.lora.load_lora(sd, key_map))

    assert len(patches) == 3
    assert not any(message.startswith("lora key not loaded: " + MAIN_PREFIX) for message in messages)
    offset = 0
    for proj, size in zip("qkv", (MAIN_IN, 1536, 1536)):
        key = "diffusion_model.blocks.0.attn." + {"q": "wq", "k": "wk", "v": "wv"}[proj] + ".weight"
        result = comfy.lora.calculate_weight([(1.0, patches[key], 1.0, None, None)], torch.zeros(size, MAIN_IN), key)
        assert torch.allclose(result, reference[offset:offset + size], atol=1e-4)
        offset += size


def test_lokr_fused_qkv_kept_per_projection_picks_projection_rows():
    # Tools and loaders that rename a fused to_qkv adapter per projection leave the factors as
    # wide as the whole module, so the rows of the named projection still have to be applied.
    torch.manual_seed(0)
    w1 = torch.randn(16, 16)
    w2 = torch.randn(576, 384)  # 16 * 576 = 9216 out, 16 * 384 = 6144 in, not q/k/v aligned
    alpha = torch.tensor(16.0)
    dora = torch.randn(MAIN_OUT, 1)
    prefix = MAIN_PREFIX.replace("qkv", "")

    lora = {}
    for proj in "qkv":
        lora[prefix + proj + ".lokr_w1"] = w1
        lora[prefix + proj + ".lokr_w2"] = w2
        lora[prefix + proj + ".alpha"] = alpha
        lora[prefix + proj + ".dora_scale"] = dora

    fused = LoKrAdapter.load("fused", {"fused.lokr_w1": w1, "fused.lokr_w2": w2}, float(alpha.item()), dora, set())
    reference = comfy.lora.calculate_weight([(1.0, fused, 1.0, None, None)], torch.zeros(MAIN_OUT, MAIN_IN), "fused")

    key_map = {prefix + proj: "diffusion_model.blocks.0.attn." + p + ".weight" for proj, p in zip("qkv", ("wq", "wk", "wv"))}
    messages, patches = _capture_warnings(lambda: comfy.lora.load_lora(lora, key_map, log_missing=False))

    assert len(patches) == 3
    x = torch.randn(2, MAIN_IN)
    fused_h = fused.h(x, torch.randn(2, MAIN_OUT))
    offset = 0
    for proj, size in zip("qkv", (MAIN_IN, 1536, 1536)):
        key = "diffusion_model.blocks.0.attn." + {"q": "wq", "k": "wk", "v": "wv"}[proj] + ".weight"
        patch = patches[key]
        result = comfy.lora.calculate_weight([(1.0, patch, 1.0, None, None)], torch.zeros(size, MAIN_IN), key)
        assert torch.allclose(result, reference[offset:offset + size], atol=1e-4)
        # the bypass path slices the same rows, so its output matches the base output width
        assert torch.allclose(patch.h(x, torch.zeros(2, size)), fused_h[:, offset:offset + size], atol=1e-4)
        offset += size


def test_lokr_fused_qkv_mapped_to_one_projection_picks_projection_rows():
    # A loader that maps the fused to_qkv entry itself onto a projection keeps the fused factors
    # under the fused name, so the rows have to come from the target key.
    torch.manual_seed(0)
    w1 = torch.randn(16, 16)
    w2 = torch.randn(576, 384)  # 16 * 576 = 9216 out, 16 * 384 = 6144 in, not q/k/v aligned
    dora = torch.randn(MAIN_OUT, 1)
    lora = {
        MAIN_PREFIX + ".lokr_w1": w1,
        MAIN_PREFIX + ".lokr_w2": w2,
        MAIN_PREFIX + ".alpha": torch.tensor(16.0),
        MAIN_PREFIX + ".dora_scale": dora,
    }

    fused = LoKrAdapter.load("fused", {"fused.lokr_w1": w1, "fused.lokr_w2": w2}, 16.0, dora, set())
    reference = comfy.lora.calculate_weight([(1.0, fused, 1.0, None, None)], torch.zeros(MAIN_OUT, MAIN_IN), "fused")

    key = "diffusion_model.blocks.0.attn.wk.weight"
    messages, patches = _capture_warnings(lambda: comfy.lora.load_lora(lora, {MAIN_PREFIX: key}, log_missing=False))

    result = comfy.lora.calculate_weight([(1.0, patches[key], 1.0, None, None)], torch.zeros(1536, MAIN_IN), key)
    assert torch.allclose(result, reference[MAIN_IN:MAIN_IN + 1536], atol=1e-4)


def test_load_lora_maps_fused_qkv_key_to_the_split_projection():
    # convert_lora splits a fused to_qkv adapter per projection, so a loader that maps the fused
    # key itself is loaded for the projection of the target weight. Keys the converter keeps under
    # the fused name must not stop that.
    sd = _make_sd(16, 576, 16, 384)  # not q/k/v aligned, so the projections need row offsets
    sd[MAIN_PREFIX + ".reshape_weight"] = torch.tensor([1, 1])
    reference = _apply(sd, MAIN_PREFIX, MAIN_OUT, MAIN_IN)

    key = "diffusion_model.blocks.0.attn.wk.weight"
    messages, patches = _capture_warnings(lambda: comfy.lora.load_lora(sd, {MAIN_PREFIX: key}))

    assert list(patches) == [key]
    result = comfy.lora.calculate_weight([(1.0, patches[key], 1.0, None, None)], torch.zeros(1536, MAIN_IN), key)
    assert torch.allclose(result, reference[MAIN_IN:MAIN_IN + 1536], atol=1e-4)


def test_convert_lora_leaves_other_fused_qkv_models_alone():
    # Audio models keep a real fused to_qkv module; their adapters must not be rewritten.
    sd = {
        "diffusion_model.decoder.layers.0.self_attn.to_qkv.lokr_w1": torch.randn(4, 4),
        "diffusion_model.decoder.layers.0.self_attn.to_qkv.lokr_w2": torch.randn(4, 4),
    }

    assert comfy.lora_convert.convert_lora(sd) is sd
