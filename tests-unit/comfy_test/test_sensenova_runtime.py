from types import SimpleNamespace
import json

import pytest
import torch
import comfy_kitchen
import comfy_aimdo.host_buffer
from safetensors import safe_open
from safetensors.torch import save_file

from comfy.cli_args import args

args.cpu = True

import comfy.hooks
import comfy.checkpoint_convert
import comfy.sd
import comfy.model_base
import comfy.model_detection
import comfy.supported_models
import comfy.model_management
import comfy.model_patcher
import comfy.quant_ops
import comfy.model_prefetch
import comfy.ops
import comfy.utils
import comfy.text_encoders.sensenova
from comfy.ldm.sensenova import model as sensenova_model
from comfy_extras.nodes_textgen import TextGenerateV2
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from comfy_extras.nodes_sensenova import SenseNovaConditioning
from sensenova_test_utils import (
    PatchEmbedding,
    _generation_input_ids,
    _run_prepared_generation,
    _text_clip,
)


def _prefix_dtype_clip(monkeypatch, storage_dtype, compute_dtype, offloaded):
    model = _text_clip([42, 43, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    model.cond_stage_model.to(dtype=compute_dtype)
    diffusion = model.cond_stage_model
    backbone = diffusion.language_model.model
    for layer in backbone.layers:
        for module in (layer.input_layernorm, layer.self_attn.k_proj, layer.self_attn.v_proj,
                       layer.self_attn.k_norm, layer.self_attn.k_norm_hw):
            module.weight.detach().fill_(1)
    weight = backbone.embed_tokens.weight.detach().to(storage_dtype)
    weight[198].copy_(weight[1])
    embed = comfy.ops.manual_cast.Embedding(*weight.shape, dtype=storage_dtype, device="cpu")
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    if offloaded:
        # Simulate a table beyond the allocator watermark, using the real host-row op.
        embed._v = (SimpleNamespace(base_addr=0, get_watermark=lambda: 0), 0, 1)
        embed.weight_lowvram_function = None
    backbone.embed_tokens = embed
    diffusion.vision_model = PatchEmbedding(3, weight.shape[1], kernel_size=32, stride=32).to(compute_dtype)
    diffusion.vision_model.weight.detach().zero_()
    diffusion.vision_model.bias.detach().zero_()
    monkeypatch.setattr(comfy.text_encoders.sensenova, "generate_dtype", lambda device: compute_dtype)
    return model


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_sampler_prefix_uses_inference_dtype(
    monkeypatch, storage_dtype, compute_dtype, offloaded, with_reference,
):
    model = _prefix_dtype_clip(monkeypatch, storage_dtype, compute_dtype, offloaded)
    metadata = {
        "text_input_ids": _generation_input_ids(),
        "reference_latents": [torch.ones(1, 32, 32, 3)] if with_reference else [],
    }
    conds = model.encode_from_tokens_scheduled(metadata)[0][1]

    assert model.cond_stage_model.language_model.model.embed_tokens.weight.dtype == storage_dtype
    for kind in ("prefix_keys", "prefix_values"):
        assert len(conds[kind]) == 2
        for tensor in conds[kind]:
            assert tensor.dtype == compute_dtype
            assert torch.isfinite(tensor).all()
            assert torch.count_nonzero(tensor) > 0


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
def test_sensenova_generate_uses_inference_dtype(
    monkeypatch, compute_dtype, offloaded, with_reference, interleave,
):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _prefix_dtype_clip(monkeypatch, torch.float8_e4m3fn, compute_dtype, offloaded)
    if interleave:
        head = model.cond_stage_model.language_model.lm_head.weight.detach()
        head[151670].copy_(head[151668])
        head[151668].zero_()
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "reference_latents": [torch.ones(1, 32, 32, 3)] if with_reference else [],
        "sensenova_interleave": interleave,
    }]]
    result = _run_prepared_generation(
        generation_mode="interleave" if interleave else "image",
        clip=model, conditioning=positive,
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]], max_think_tokens=8,
    )

    assert result.args[1] == "KL"
    assert result.args[3] is True
    expected = [42, 43, 151670] if interleave else [42, 43, 151668, 271, 151670]
    assert result.args[0][0][1]["text_input_ids"][0, -len(expected):].tolist() == expected
    conds = result.args[0][0][1]
    for kind in ("prefix_keys", "prefix_values"):
        assert all(value.dtype == compute_dtype and torch.isfinite(value).all() for value in conds[kind])


def _add_identity_decoder_layers(monkeypatch, model):
    backbone = model.cond_stage_model.language_model.model
    for name, value in {
        "HIDDEN_SIZE": backbone.embed_tokens.embedding_dim,
        "INTERMEDIATE_SIZE": 4, "NUM_LAYERS": 2, "NUM_HEADS": 1, "NUM_KV_HEADS": 1, "HEAD_DIM": 8,
    }.items():
        monkeypatch.setattr(sensenova_model, name, value)
    backbone.layers = torch.nn.ModuleList([
        sensenova_model.DecoderLayer(operations=torch.nn) for _ in range(2)
    ])
    for parameter in backbone.layers.parameters():
        torch.nn.init.zeros_(parameter)


@pytest.mark.parametrize(
    "storage_dtype,compute_dtype",
    [(torch.float32, torch.float32), (torch.float8_e4m3fn, torch.bfloat16)],
)
@pytest.mark.parametrize("configuration", ["module", "global"])
@pytest.mark.parametrize("rotated", [False, True])
def test_sensenova_decode_attention_uses_configured_backend(monkeypatch, storage_dtype, compute_dtype, configuration, rotated):
    monkeypatch.setattr(comfy_kitchen, "flash_attention_decode_is_available", lambda device: True)
    monkeypatch.setattr(comfy_kitchen.registry, "is_available", lambda backend: True)
    attention = sensenova_model.Attention(
        device=torch.device("cpu"), dtype=storage_dtype, operations=comfy.ops.manual_cast,
    )
    for parameter in attention.parameters():
        parameter.detach().zero_()
    attention.q_proj.weight.detach().fill_(1 / 64)
    attention.k_proj.weight.detach().fill_(1 / 128)
    attention.v_proj.weight.detach().fill_(1 / 128)
    attention.q_norm.weight.detach().fill_(2)
    attention.q_norm_hw.weight.detach().fill_(3)
    attention.k_norm.weight.detach().fill_(1)
    attention.k_norm_hw.weight.detach().fill_(1)
    attention.o_proj.weight.detach().copy_(torch.eye(4096))
    visible_lengths = iter((3, 4))

    def configured_backend(query, key, value, heads, **kwargs):
        assert isinstance(query, sensenova_model.AttentionTensorContainer)
        assert isinstance(key, sensenova_model.AttentionTensorContainer)
        assert isinstance(value, sensenova_model.AttentionTensorContainer)
        queries = query.take()
        assert queries.shape == (1, 32, 1, 128)
        torch.testing.assert_close(queries, expected_query, atol=2e-2, rtol=0)
        length = next(visible_lengths)
        assert key.take().shape == (1, 8, 5, 128)
        values = value.take()
        assert values.dtype == compute_dtype
        assert values.shape == (1, 8, 5, 128)
        assert torch.isfinite(values).all()
        assert kwargs["mask"][0, 0, 0].tolist() == [0.0] * length + [float("-inf")] * (5 - length)
        assert heads == 32
        assert kwargs["enable_gqa"] is True
        return values.new_ones((1, 1, 4096))

    if configuration == "module":
        attention.comfy_attention.function = configured_backend
    else:
        monkeypatch.setattr(sensenova_model, "optimized_attention", configured_backend)
    hidden = torch.ones(1, 1, 4096, dtype=compute_dtype)
    angles = ((0, 1), (0, -1), (-1, 0)) if rotated else ((1, 0),) * 3
    rope = tuple(
        (torch.full((1, 1, 1, dim), cosine, dtype=compute_dtype),
         torch.full((1, 1, 1, dim), sine, dtype=compute_dtype))
        for dim, (cosine, sine) in zip((64, 32, 32), angles)
    )
    expected_query = hidden.new_tensor(
        [-2] * 32 + [2] * 32 + [3] * 16 + [-3] * 16 + [-3] * 32
        if rotated else [2] * 64 + [3] * 64
    ).view(1, 1, 1, 128).expand(1, 32, 1, 128)
    expected_key = hidden.new_tensor(
        [-1] * 32 + [1] * 32 + [1] * 16 + [-1] * 16 + [-1] * 32 if rotated else [1] * 128
    ).view(1, 1, 128).expand(1, 8, 128)
    prefix = torch.ones(1, 8, 2, 128, dtype=compute_dtype)
    cache = sensenova_model.FixedKV(
        torch.zeros(1, 5, 8, 128, dtype=compute_dtype), torch.zeros(1, 5, 8, 128, dtype=compute_dtype),
        2, torch.empty(1, dtype=torch.int64), torch.tensor([2], dtype=torch.int32),
    )
    mask = hidden.new_full((1, 1, 1, 5), float("-inf"))
    mask[..., :2].zero_()
    cache.key[:, :2].copy_(prefix.transpose(1, 2))
    cache.value[:, :2].copy_(prefix.transpose(1, 2))
    for position in (2, 3):
        hidden.fill_(position - 1)
        cache.prepare(1)
        mask[..., cache.index].zero_()
        output = attention.forward_decode(hidden, rope, cache, mask, {})
        cache.advance(1)

        assert output.dtype == compute_dtype
        assert torch.equal(output, torch.ones_like(hidden))
        torch.testing.assert_close(cache.key[:, position], expected_key, atol=2e-2, rtol=0)
        assert torch.all(cache.value[:, position] != 0)
        assert torch.count_nonzero(cache.key[:, position + 1:]) == 0
        assert torch.count_nonzero(cache.value[:, position + 1:]) == 0
    assert cache.key.shape == cache.value.shape == (1, 5, 8, 128)
    assert cache.key.dtype == cache.value.dtype == compute_dtype
    assert torch.equal(cache.key[:, :2], prefix.transpose(1, 2))
    assert torch.equal(cache.value[:, :2], prefix.transpose(1, 2))
    assert torch.all(cache.key[:, 2:4] != 0)
    assert torch.equal(cache.value[:, 2], torch.full_like(cache.value[:, 2], 32))
    assert torch.equal(cache.value[:, 3], torch.full_like(cache.value[:, 3], 64))


@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_sensenova_plain_cpu_fp8_uses_native_embedding_for_prefill_and_decode(monkeypatch, storage_dtype):
    embed = comfy.ops.manual_cast.Embedding(8, 4, dtype=storage_dtype, device="cpu")
    embed.weight = torch.nn.Parameter(torch.arange(32).reshape(8, 4).to(storage_dtype), requires_grad=False)
    embed._v = (SimpleNamespace(base_addr=0, get_watermark=lambda: 0), 0, 1)
    embed.weight_lowvram_function = None
    monkeypatch.setattr(embed, "host_rows", lambda *args, **kwargs: pytest.fail("Plain CPU FP8 requires native embedding lookup"))
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed, layers=[])))
    ids = torch.tensor([[1, 2]])
    prefix, _, _, _ = sensenova_model.SenseNovaTextModel._prepare_prefix(model, ids, None, None, None, compute_dtype=torch.bfloat16)
    expected = torch.nn.functional.embedding(ids, embed.weight.float()).bfloat16()
    assert torch.equal(prefix, expected)
    hidden = torch.empty(1, 1, 4, dtype=torch.bfloat16)
    sensenova_model.SenseNovaTextModel._decode_text_token(model, ids[:, 0], [], hidden, None, None)
    assert torch.equal(hidden, expected[:, :1])
    assert embed.weight.dtype == storage_dtype


@pytest.mark.parametrize("quant_format,convrot", [
    ("float8_e4m3fn", False), ("float8_e5m2", False),
    ("int8_tensorwise", False), ("int8_tensorwise", True),
])
@pytest.mark.parametrize("offloaded", [False, True])
def test_sensenova_quantized_embedding_preserves_native_lookup(quant_format, convrot, offloaded):
    weight = torch.linspace(-1, 1, 16 * 256).reshape(16, 256).bfloat16()
    config = {"format": quant_format}
    quant_kwargs = {}
    if quant_format == "int8_tensorwise":
        quant_kwargs = dict(per_channel=True, convrot=convrot, convrot_groupsize=256)
        config.update(convrot=convrot, convrot_groupsize=256)
    layout = comfy.quant_ops.QUANT_ALGOS[quant_format]["comfy_tensor_layout"]
    quantized = comfy.quant_ops.QuantizedTensor.from_float(weight, layout, **quant_kwargs)
    embed = comfy.ops.mixed_precision_ops({}).Embedding(16, 256, device="cpu", dtype=torch.bfloat16)
    embed.load_state_dict({
        "weight": quantized._qdata, "weight_scale": quantized._params.scale,
        "comfy_quant": torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8),
    })
    ids = torch.tensor([[1, 2, 1]])
    expected = embed(ids, out_dtype=torch.bfloat16)
    if offloaded:
        embed._v = (SimpleNamespace(base_addr=0, get_watermark=lambda: 0), 0, 1)
        embed.weight_lowvram_function = None
        assert sensenova_model._use_host_rows(embed)
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed, layers=[])))
    prefix, _, _, _ = sensenova_model.SenseNovaTextModel._prepare_prefix(model, ids, None, None, None, compute_dtype=torch.bfloat16)
    assert torch.equal(prefix, expected)
    hidden = torch.empty(1, 1, 256, dtype=torch.bfloat16)
    sensenova_model.SenseNovaTextModel._decode_text_token(model, ids[:, 0], [], hidden, None, None)
    assert torch.equal(hidden, expected[:, :1])
    assert isinstance(embed.weight, comfy.quant_ops.QuantizedTensor)




@pytest.fixture
def small_sensenova(monkeypatch):
    for name, value in {"HIDDEN_SIZE": 16, "INTERMEDIATE_SIZE": 32, "NUM_LAYERS": 2,
                        "NUM_HEADS": 2, "NUM_KV_HEADS": 1, "HEAD_DIM": 8}.items():
        monkeypatch.setattr(sensenova_model, name, value)


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_sensenova_checkpoint_and_clip_round_trip(small_sensenova, monkeypatch, tmp_path, prefix, caplog):
    text = sensenova_model.SenseNovaTextModel(has_lm_head=True, operations=torch.nn)
    image = sensenova_model.SenseNovaU15(dtype=torch.float32, operations=torch.nn)
    text_sd, image_sd = text.state_dict(), image.state_dict()
    assert not text_sd.keys() & image_sd.keys()
    state = {prefix + key.replace("vision_model.", "vision_model.embeddings.", 1): value for key, value in text_sd.items()}
    for key, value in image_sd.items():
        if key.startswith("layers."):
            parts = key.split(".")
            parts[3 if parts[2] == "self_attn" else 2] += "_mot_gen"
            name = "language_model.model." + ".".join(parts)
        elif key.startswith("norm."):
            name = "language_model.model.norm_mot_gen." + key[len("norm."):]
        elif key.startswith("patch_embed."):
            name = "fm_modules.vision_model_mot_gen.embeddings." + key[len("patch_embed."):]
        elif key.startswith("head."):
            name = "fm_modules.fm_head." + key[len("head."):]
        else:
            name = "fm_modules." + key
        state[prefix + name] = value
    destination = tmp_path / "comfy.safetensors"
    save_file(comfy.checkpoint_convert.convert_checkpoint(state), destination, metadata={"format": "sensenova-u1.5-mot"})
    with safe_open(destination, framework="pt") as converted:
        assert converted.metadata() == {"format": "sensenova-u1.5-mot"}
    state = comfy.utils.load_torch_file(str(destination))
    model_config = comfy.supported_models.SenseNovaU15({"image_model": "sensenova_u15"})
    # Detection dimensions are covered separately with full-size checkpoint shapes.
    monkeypatch.setattr(comfy.model_detection, "model_config_from_unet", lambda *a, **kw: model_config)
    options = {"load_device": torch.device("cpu"), "dtype": torch.float32}
    model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
        state, model_options=options, te_model_options=options, disable_dynamic=True,
    )
    assert model is not None and clip is not None and vae is not None
    for layer in model.model.diffusion_model.layers:
        assert layer.self_attn.combined_qkv is None and layer.mlp.combined_gate_up is None
    assert state == {}
    assert set(model.model.diffusion_model.state_dict()) == set(image_sd)
    assert set(clip.get_sd()) == set(text_sd)
    assert not {p.data_ptr() for p in model.model.parameters()} & {p.data_ptr() for p in clip.cond_stage_model.parameters()}
    saved = model.model.state_dict_for_saving(model.model.diffusion_model.state_dict(), clip.get_sd(), vae.get_sd())
    torch.testing.assert_close(saved["vae.pixel_space_vae"], torch.tensor(1.0))
    for prefix, weights in (("model.diffusion_model.", image_sd), ("text_encoders.", text_sd)):
        for key, value in weights.items():
            torch.testing.assert_close(saved[prefix + key], value, rtol=0, atol=0)
    reloaded_model, reloaded_clip, reloaded_vae, _ = comfy.sd.load_state_dict_guess_config(
        saved, model_options=options, te_model_options=options, disable_dynamic=True,
    )
    assert set(reloaded_model.model.diffusion_model.state_dict()) == set(image_sd)
    assert set(reloaded_vae.get_sd()) == {"pixel_space_vae"}
    standalone = comfy.sd.load_text_encoder_state_dicts([clip.get_sd()], model_options=options, disable_dynamic=True)
    tokens = clip.tokenize("an apple", skip_template=False)
    expected = clip.generate(tokens, do_sample=False, max_length=2)
    assert reloaded_clip.generate(tokens, do_sample=False, max_length=2) == expected
    assert standalone.generate(tokens, do_sample=False, max_length=2) == expected
    assert "missing:" not in caplog.text and "unexpected:" not in caplog.text

    text_key = "language_model.model.layers.0.self_attn.q_proj"
    image_key = "layers.0.self_attn.q_proj"
    lora = {prefix + key + suffix: value for prefix, key in (("text_encoders.", text_key), ("diffusion_model.", image_key)) for suffix, value in (
        (".lora_up.weight", torch.ones(16, 1)), (".lora_down.weight", torch.ones(1, 16)),
    )}
    patched_model, patched_clip = comfy.sd.load_lora_for_models(model, clip, lora, 1.0, 0.5)
    assert set(patched_model.patches) == {"diffusion_model." + image_key + ".weight"}
    assert set(patched_clip.patcher.patches) == {text_key + ".weight"}
    text_weight = clip.get_sd()[text_key + ".weight"].clone()
    image_weight = model.model.diffusion_model.state_dict()[image_key + ".weight"].clone()
    try:
        comfy.model_management.load_models_gpu([patched_model, patched_clip.patcher], force_full_load=True)
        torch.testing.assert_close(patched_model.model.diffusion_model.state_dict()[image_key + ".weight"], image_weight + 1)
        torch.testing.assert_close(patched_clip.get_sd()[text_key + ".weight"], text_weight + 0.5)
    finally:
        patched_model.unpatch_model()
        patched_clip.patcher.unpatch_model()


@pytest.mark.parametrize("key_format", ["current", "legacy", "legacy_flat"])
def test_sensenova_generation_lora_legacy_keys(small_sensenova, key_format, caplog):
    config = comfy.supported_models.SenseNovaU15({"image_model": "sensenova_u15"})
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    modules = (
        ("self_attn.q_proj", "self_attn.q_proj_mot_gen"),
        ("self_attn.k_proj", "self_attn.k_proj_mot_gen"),
        ("self_attn.v_proj", "self_attn.v_proj_mot_gen"),
        ("self_attn.o_proj", "self_attn.o_proj_mot_gen"),
        ("mlp.gate_proj", "mlp_mot_gen.gate_proj"),
        ("mlp.up_proj", "mlp_mot_gen.up_proj"),
        ("mlp.down_proj", "mlp_mot_gen.down_proj"),
    )
    weights = base.state_dict()
    expected = set()
    lora = {}
    for layer in range(sensenova_model.NUM_LAYERS):
        for current, legacy in modules:
            target = f"diffusion_model.layers.{layer}.{current}.weight"
            weights[target].zero_()
            expected.add(target)
            if key_format == "current":
                key = target.removesuffix(".weight")
            else:
                key = f"language_model.model.layers.{layer}.{legacy}"
                key = "lora_unet_" + key.replace(".", "_") if key_format == "legacy_flat" else "diffusion_model." + key
            rows, columns = weights[target].shape
            lora[key + ".lora_up.weight"] = torch.ones(rows, 1)
            lora[key + ".lora_down.weight"] = torch.ones(1, columns)
            lora[key + ".alpha"] = torch.tensor(2.0)
    patched, _ = comfy.sd.load_lora_for_models(model, None, lora, 0.25, 0.0)
    assert set(patched.patches) == expected
    assert not model.patches
    assert "lora key not loaded" not in caplog.text
    try:
        comfy.model_management.load_models_gpu([patched], force_full_load=True)
        for key in expected:
            torch.testing.assert_close(base.state_dict()[key], torch.full_like(weights[key], 0.5))
    finally:
        patched.unpatch_model()


@pytest.mark.parametrize("component", ["model", "clip", "vae", "diffusion"])
def test_sensenova_partial_checkpoint_loading(small_sensenova, monkeypatch, component, caplog):
    text = sensenova_model.SenseNovaTextModel(has_lm_head=True, operations=torch.nn)
    image = sensenova_model.SenseNovaU15(dtype=torch.float32, operations=torch.nn)
    text_sd, image_sd = text.state_dict(), image.state_dict()
    state = {"model.diffusion_model." + key: value for key, value in image_sd.items()}
    state.update({"text_encoders." + key: value for key, value in text_sd.items()})
    vae_sd = {"pixel_space_vae": torch.tensor(1.0)}
    state.update({"vae." + key: value for key, value in vae_sd.items()})
    model_config = comfy.supported_models.SenseNovaU15({"image_model": "sensenova_u15"})
    monkeypatch.setattr(comfy.model_detection, "model_config_from_unet", lambda *a, **kw: model_config)
    options = {"load_device": torch.device("cpu"), "dtype": torch.float32}
    if component == "diffusion":
        model = comfy.sd.load_diffusion_model_state_dict(state, model_options=options, disable_dynamic=True)
    else:
        model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
            state, output_model=component == "model", output_clip=component == "clip", output_vae=component == "vae",
            model_options=options, te_model_options=options, disable_dynamic=True,
        )
    if component == "clip":
        actual, expected = clip.get_sd(), text_sd
    elif component == "vae":
        actual, expected = vae.get_sd(), vae_sd
    else:
        actual, expected = model.model.diffusion_model.state_dict(), image_sd
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)
    assert "missing:" not in caplog.text and "unexpected:" not in caplog.text


def test_sensenova_native_sampling_and_generation_node():
    clip = _text_clip([42, 43, 151645])
    tokens = {"text_input_ids": torch.tensor([[1]])}
    assert clip.generate(tokens, do_sample=False, max_length=8) == [42, 43, 151645]
    assert clip.generate(tokens, top_k=1, max_length=8, seed=7) == [42, 43, 151645]
    sampled = dict(max_length=8, seed=7, temperature=0.8, top_k=7, top_p=0.9, min_p=0.01, repetition_penalty=1.2, presence_penalty=0.1)
    assert clip.generate(tokens, **sampled) == clip.generate(tokens, **sampled)
    embed = clip.cond_stage_model.language_model.model.embed_tokens.weight.detach()
    embed[271].copy_(embed[1])
    result = TextGenerateV2.execute(clip=clip, prompt="Say hello", max_length=8, sampling_mode={"sampling_mode": "off"})
    assert result.args == ("KL", "")
    raw = clip.tokenize("literal <img>", skip_template=True)
    assert clip.decode([pair[0] for pair in raw["sensenova_u15"][0]], skip_special_tokens=False) == "literal <img>"
    custom = clip.tokenize("hello", skip_template=False, system_prompt="Be concise.")
    assert "system\nBe concise.<|im_end|>" in clip.decode([pair[0] for pair in custom["sensenova_u15"][0]], skip_special_tokens=False)


@pytest.mark.parametrize(("generated", "limit", "expected"), [
    ([42, 43, 151645], 8, [42, 43, 151645]),
    ([151645], 8, [151645]),
    ([42, 151668], 8, [42, 151668]),
    ([42, 43, 44], 2, [42, 43]),
])
def test_sensenova_text_progress_reaches_ui_and_console(monkeypatch, capsys, generated, limit, expected):
    clip = _text_clip(generated)
    updates = []
    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", lambda value, total, preview, **kw: updates.append((value, total)))
    monkeypatch.setattr(comfy.utils, "PROGRESS_THROTTLE_MIN_INTERVAL", 0)
    monkeypatch.setattr(comfy.utils, "PROGRESS_THROTTLE_MIN_PERCENT", 0)

    result = clip.generate({"text_input_ids": torch.tensor([[1]]), "stop_token_ids": (151668,)}, do_sample=False, max_length=limit)

    assert result == expected
    count = len(expected)
    assert updates == [(value, limit) for value in range(count + 1)] + [(count, count)]
    console = capsys.readouterr().err
    assert "Generating tokens" in console and "100%" in console
    assert f"{count}/{count}" in console


@pytest.mark.parametrize("phase", ["prefill", "decode", "encode"])
def test_sensenova_clip_cancellation_releases_prefetch_and_can_retry(monkeypatch, phase):
    clip = _text_clip([42, 43, 151668])
    _add_identity_decoder_layers(monkeypatch, clip)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *a, **kw: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy.text_encoders.sensenova, "generate_dtype", lambda device: torch.float32)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    tokens = {"text_input_ids": torch.tensor([[1]]), "stop_token_ids": (151668,)}

    def cancel(*args, **kwargs):
        assert comfy.model_prefetch.PREFETCH_QUEUES
        raise comfy.model_management.InterruptProcessingException()

    handle = None
    if phase == "decode":
        monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", lambda value, *a, **kw: cancel() if value else None)
    else:
        handle = clip.cond_stage_model.language_model.model.layers[0].mlp.register_forward_pre_hook(cancel)
    try:
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            if phase == "encode":
                clip.encode_from_tokens_scheduled(tokens)
            else:
                clip.generate(tokens, do_sample=False, max_length=8)
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
    finally:
        if handle is not None:
            handle.remove()
        monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    assert clip.generate(tokens, do_sample=False, max_length=8) == [42, 43, 151668]
    assert comfy.model_prefetch.PREFETCH_QUEUES == []


def test_sensenova_clip_uses_native_weight_hooks():
    clip = _text_clip([42, 43, 151668])
    embed = clip.cond_stage_model.language_model.model.embed_tokens.weight.detach().clone()
    delta = torch.zeros_like(embed)
    delta[1] = embed[42] - embed[1]
    hook = comfy.hooks.WeightHook()
    hook.need_weight_init = False
    group = comfy.hooks.HookGroup()
    group.add(hook)
    patched = clip.clone()
    patched.patcher.add_hook_patches(hook, {"language_model.model.embed_tokens.weight": (delta,)})
    patched.patcher.forced_hooks = group
    tokens = {"text_input_ids": torch.tensor([[1]]), "stop_token_ids": (151668,)}
    try:
        assert patched.generate(tokens, do_sample=False, max_length=8) == [43, 151668]
        assert clip.generate(tokens, do_sample=False, max_length=8) == [42, 43, 151668]
    finally:
        patched.patcher.unpatch_model()


def test_sensenova_sampler_requires_reencoding_late_references():
    model = object.__new__(comfy.model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.diffusion_model = SimpleNamespace(dtype=torch.float32)
    keys = [torch.ones(1, 1, 3, 8)]
    metadata = {"prefix_keys": keys, "prefix_values": keys, "prefix_time": torch.tensor([3]), "device": torch.device("cpu")}
    conds = model.extra_conds(**metadata)
    assert conds["prefix_keys"].cond[0] is keys[0]
    reference = torch.zeros(1, 32, 32, 3)
    with pytest.raises(ValueError, match="Encode reference images"):
        model.extra_conds(**metadata, reference_latents=[reference])
    conds = model.extra_conds(**metadata, reference_latents=[reference], sensenova_reference_count=1)
    assert conds["prefix_keys"].cond[0] is keys[0]
    model.manual_cast_dtype = torch.bfloat16
    conds = model.extra_conds(**metadata)
    assert conds["prefix_keys"].cond[0].dtype == torch.bfloat16
    assert conds["prefix_keys"].process_cond(batch_size=3).cond[0] is conds["prefix_keys"].cond[0]
    assert keys[0].dtype == torch.float32


@pytest.mark.parametrize("quant_format", ["int8_tensorwise", "float8_e4m3fn"])
@pytest.mark.parametrize("bundled", [False, True])
def test_sensenova_clip_loader_preserves_quantized_weights(small_sensenova, monkeypatch, quant_format, bundled):
    text = sensenova_model.SenseNovaTextModel(has_lm_head=True, operations=torch.nn)
    state = text.state_dict()
    quant_keys = ("language_model.model.embed_tokens", "language_model.model.layers.0.self_attn.q_proj")
    for key in quant_keys:
        layout = comfy.quant_ops.QUANT_ALGOS[quant_format]["comfy_tensor_layout"]
        options = {"per_channel": True} if quant_format == "int8_tensorwise" else {}
        weight = comfy.quant_ops.QuantizedTensor.from_float(state[key + ".weight"], layout, **options)
        state[key + ".weight"] = weight._qdata
        state[key + ".weight_scale"] = weight._params.scale
        state[key + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
    options = {"load_device": torch.device("cpu")}
    if not bundled:
        clip = comfy.sd.load_text_encoder_state_dicts([state], model_options=options, disable_dynamic=True)
    else:
        image = sensenova_model.SenseNovaU15(dtype=torch.float32, operations=torch.nn)
        state = {"text_encoders." + key: value for key, value in state.items()}
        state.update({"model.diffusion_model." + key: value for key, value in image.state_dict().items()})
        monkeypatch.setattr(comfy.model_detection, "detect_unet_config", lambda *a, **kw: {"image_model": "sensenova_u15"})
        model, clip, _, _ = comfy.sd.load_state_dict_guess_config(
            state, output_vae=False, model_options=options, te_model_options=options, disable_dynamic=True,
        )
        assert model.model.model_config.quant_config is None
    for key in quant_keys:
        weight = comfy.utils.get_attr(clip.cond_stage_model, key + ".weight")
        assert isinstance(weight, comfy.quant_ops.QuantizedTensor)
    for layer in clip.cond_stage_model.language_model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        assert attn.combined_qkv.linears == (attn.q_proj, attn.k_proj, attn.v_proj)
        assert mlp.combined_gate_up.linears == (mlp.gate_proj, mlp.up_proj)
    conditioning = clip.encode_from_tokens_scheduled({"text_input_ids": torch.tensor([[1, 2, 3]])})
    assert all(torch.isfinite(key).all() for key in conditioning[0][1]["prefix_keys"])


def test_sensenova_image_conditioning_encodes_both_reference_branches(small_sensenova):
    text = sensenova_model.SenseNovaTextModel(has_lm_head=True, operations=torch.nn)
    clip = comfy.sd.load_text_encoder_state_dicts([text.state_dict()], model_options={"load_device": torch.device("cpu")}, disable_dynamic=True)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("an apple"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize("blurry"))
    positive, negative = HiDreamO1ReferenceImages.execute(
        positive=positive, negative=negative, images={"image_1": torch.full((1, 33, 65, 3), 0.5)},
    ).args
    result = SenseNovaConditioning.execute(clip=clip, conditioning=positive, negative=negative, task={"task": "image"}, max_tokens=8).args
    expected = clip.encode_from_tokens_scheduled(clip.tokenize("an apple", images=[torch.full((1, 33, 65, 3), 0.5)]))
    for actual, reference in zip(result[0][0][1]["prefix_keys"], expected[0][1]["prefix_keys"]):
        torch.testing.assert_close(actual, reference)
    assert result[0][0][1]["sensenova_reference_count"] == result[2][0][1]["sensenova_reference_count"] == 1
    assert result[2][0][1]["prefix_keys"][0].shape[2] == 17
    assert positive[0][1]["sensenova_reference_count"] == negative[0][1]["sensenova_reference_count"] == 0
