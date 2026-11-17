from types import SimpleNamespace
import json

import pytest
import torch
import comfy_aimdo.host_buffer
import comfy_kitchen

from comfy.cli_args import args

args.cpu = True

import comfy.hooks
import comfy.model_management
import comfy.model_patcher
import comfy.quant_ops
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.sensenova import model as sensenova_model
from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy.text_encoders.sensenova import SenseNovaQwenTokenizer
from comfy_extras.nodes_sensenova import TextGenerateSenseNova
from sensenova_test_utils import (
    PatchEmbedding,
    _generation_input_ids,
    _run_prepared_generation,
    _text_model_patcher,
)


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("max_tokens,compute_dtype,text_padding,text_bytes,reference_bytes", [
    (32, torch.float32, 0, 16_171_908, 19_269_888),
    (8192, torch.float32, 0, 2_823_734_148, 2_826_832_128),
    (32, torch.bfloat16, 0, 8_085_954, 9_634_944),
    (8192, torch.bfloat16, 0, 1_411_867_074, 1_413_416_064),
    (32, torch.bfloat16, 4096, 746_529_218, 748_225_664),
])
def test_sensenova_generation_reserves_prefix_and_decode_memory(
    monkeypatch, with_reference, max_tokens, compute_dtype, text_padding, text_bytes, reference_bytes,
):
    budgets = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, memory_required: budgets.append(memory_required))
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 2)
    model = _text_model_patcher([42, 151668])
    model.model.manual_cast_dtype = compute_dtype
    diffusion = model.model.diffusion_model
    diffusion.vision_model = PatchEmbedding(3, 2, kernel_size=32, stride=32)
    model.model.to(compute_dtype)
    embeddings = diffusion.language_model.model.embed_tokens.weight.detach()
    embeddings[198].copy_(embeddings[1])
    input_ids = _generation_input_ids()[:, :-1]
    input_ids = torch.cat((input_ids[:, :3], input_ids.new_full((1, text_padding), 2), input_ids[:, 3:]), dim=1)

    output = TextGenerateSenseNova._execute_image(
        model=model, clip=SenseNovaQwenTokenizer(), max_tokens=max_tokens,
        conditioning=[[torch.empty(1), {
            "text_input_ids": input_ids,
            "reference_latents": [torch.ones(1, 33, 65, 3)] if with_reference else [],
        }]],
    )

    # 15 text positions plus padding; a 2x3 reference grid adds 9 prefix positions.
    assert budgets == [reference_bytes if with_reference else text_bytes]
    assert output.args[1] == "K"
    assert model.model.current_patcher is None


def test_sensenova_interleave_memory_includes_sampled_images(monkeypatch):
    budgets = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, memory_required: budgets.append(memory_required))
    monkeypatch.setattr(sensenova_model, "HIDDEN_SIZE", 3)
    clip = SenseNovaQwenTokenizer()
    first = TextGenerateSenseNova._execute_interleave(
        model=_text_model_patcher([42, 151670]), clip=clip,
        positive=[[torch.empty(1), {"text_input_ids": torch.tensor([[7, 1]])}]],
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]],
        samples=None, max_text_tokens=8,
    ).args
    model = _text_model_patcher([151671, 43, 151645])
    model.model.diffusion_model.vision_model = PatchEmbedding(3, 3, kernel_size=32, stride=32)
    second = TextGenerateSenseNova._execute_interleave(
        model=model, clip=clip, positive=first[0], negative=first[2],
        samples={"samples": torch.zeros(1, 3, 32, 64)}, max_text_tokens=8,
    ).args

    assert first[3] is True and second[3] is False
    assert (first[1], second[1]) == ("K", "L")
    assert budgets == [3_440_656, 5_161_156]
    assert second[0][0][1]["sensenova_interleave_finished"] is True
    assert model.model.current_patcher is None


def _prefix_dtype_model(monkeypatch, storage_dtype, compute_dtype, offloaded):
    model = _text_model_patcher([42, 43, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    model.model.to(dtype=compute_dtype)
    diffusion = model.model.diffusion_model
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
    diffusion.dtype = storage_dtype
    model.model.manual_cast_dtype = compute_dtype
    model.model.concat_keys = ()
    return model


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_sampler_prefix_uses_inference_dtype(
    monkeypatch, storage_dtype, compute_dtype, offloaded, with_reference,
):
    model = _prefix_dtype_model(monkeypatch, storage_dtype, compute_dtype, offloaded)
    metadata = {
        "text_input_ids": _generation_input_ids(),
        "reference_latents": [torch.ones(1, 32, 32, 3)] if with_reference else [],
    }
    conds = model.model.extra_conds(**metadata, device=model.load_device)

    assert model.model.diffusion_model.language_model.model.embed_tokens.weight.dtype == storage_dtype
    for kind in ("prefix_keys", "prefix_values"):
        assert len(conds[kind].cond) == 2
        for tensor in conds[kind].cond:
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
    model = _prefix_dtype_model(monkeypatch, torch.float8_e4m3fn, compute_dtype, offloaded)
    if interleave:
        head = model.model.diffusion_model.language_model.lm_head.weight.detach()
        head[151670].copy_(head[151668])
        head[151668].zero_()
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "reference_latents": [torch.ones(1, 32, 32, 3)] if with_reference else [],
        "sensenova_interleave": interleave,
    }]]
    result = _run_prepared_generation(
        generation_mode="interleave" if interleave else "image",
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive,
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]], max_think_tokens=8,
    )

    assert result.args[1] == "KL"
    assert result.args[3] is True
    expected = [42, 43, 151670] if interleave else [42, 43, 151668, 271, 151670]
    assert result.args[0][0][1]["text_input_ids"][0, -len(expected):].tolist() == expected
    conds = model.model.extra_conds(**result.args[0][0][1], device=model.load_device)
    for kind in ("prefix_keys", "prefix_values"):
        assert all(value.dtype == compute_dtype and torch.isfinite(value).all() for value in conds[kind].cond)


@pytest.mark.parametrize("patch_dtype,expected_bytes", [
    (torch.float32, 4_472_932), (torch.bfloat16, 2_236_466), (None, 2_236_466),
])
def test_sensenova_generate_uses_pending_compute_dtype_patch(monkeypatch, patch_dtype, expected_bytes):
    compute_dtype = patch_dtype or torch.bfloat16
    model = _prefix_dtype_model(monkeypatch, torch.bfloat16, compute_dtype, False)
    model.model.manual_cast_dtype = torch.bfloat16 if compute_dtype == torch.float32 else torch.float32
    model.set_model_compute_dtype(patch_dtype)
    memory_budgets = []

    def load_model(models, memory_required=0):
        memory_budgets.append(memory_required)
        for patcher in models:
            patcher.patch_model(load_weights=False)

    monkeypatch.setattr(comfy.model_management, "load_models_gpu", load_model)
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "reference_latents": [torch.ones(1, 32, 32, 3)],
    }]]
    result = _run_prepared_generation(
        generation_mode="image",
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive, max_think_tokens=8,
    )

    assert result.args[1] == "KL"
    assert memory_budgets == [expected_bytes]
    assert model.model.get_dtype_inference() == compute_dtype
    conds = model.model.extra_conds(**result.args[0][0][1], device=model.load_device)
    for kind in ("prefix_keys", "prefix_values"):
        assert all(value.dtype == compute_dtype and torch.isfinite(value).all() for value in conds[kind].cond)


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_reference", [False, True])
def test_sensenova_hooked_sampling_uses_inference_dtype(monkeypatch, compute_dtype, offloaded, with_reference):
    model = _prefix_dtype_model(monkeypatch, torch.float8_e4m3fn, compute_dtype, offloaded).model
    diffusion = model.diffusion_model
    width = diffusion.language_model.model.embed_tokens.embedding_dim
    diffusion.language_model.model.norm_mot_gen = torch.nn.Identity()
    diffusion.fm_modules = torch.nn.ModuleDict({
        "vision_model_mot_gen": PatchEmbedding(3, width, kernel_size=32, stride=32),
        "timestep_embedder": sensenova_model.TimestepEmbedder(width, operations=comfy.ops.manual_cast),
        "noise_scale_embedder": sensenova_model.TimestepEmbedder(width, operations=comfy.ops.manual_cast),
        "fm_head": torch.nn.ConvTranspose2d(width, 3, kernel_size=32, stride=32),
    }).to(compute_dtype)
    for parameter in diffusion.fm_modules.parameters():
        parameter.detach().fill_(0.1)
    model.model_sampling = SenseNovaModelSampling()
    metadata = {
        "text_input_ids": _generation_input_ids(),
        "reference_latents": [torch.ones(1, 32, 32, 3)] if with_reference else [],
        "device": torch.device("cpu"),
    }
    image = torch.ones(1, 3, 32, 32)
    sigma = torch.tensor([0.5])
    prepared = model.extra_conds(**metadata)
    expected = model.apply_model(image, sigma, **{key: value.cond for key, value in prepared.items()})
    hooked = model.extra_conds(**metadata, hooks=object())
    actual = model.apply_model(image, sigma, **{key: value.cond for key, value in hooked.items()})

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("interleave", [False, True])
def test_sensenova_generate_initialization_failure_cleans_up_and_can_run_again(monkeypatch, interleave):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 151645 if interleave else 151668])
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_interleave": interleave,
    }]]
    kwargs = dict(
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive,
        negative=[[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]],
        max_think_tokens=8,
    )

    def fail_initialization(patcher):
        assert patcher.model.current_patcher is patcher
        raise RuntimeError("initialization failed")

    model.add_callback_with_key("on_pre_run", "test_failure", fail_initialization)
    with pytest.raises(RuntimeError, match="initialization failed"):
        _run_prepared_generation(generation_mode="interleave" if interleave else "image", **kwargs)

    assert model.model.current_patcher is None
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]
    model.remove_callbacks_with_key("on_pre_run", "test_failure")
    assert _run_prepared_generation(generation_mode="interleave" if interleave else "image", **kwargs).args[1] == "K"
    assert model.model.current_patcher is None


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("cancel_during_prefill", [False, True])
def test_sensenova_generate_cancel_cleans_up_and_can_run_again(monkeypatch, interleave, dynamic, cancel_during_prefill):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 43, 151645 if interleave else 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    positive = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
        "sensenova_interleave": interleave,
    }]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]])}]]
    kwargs = dict(
        model=model, clip=SenseNovaQwenTokenizer(), conditioning=positive,
        negative=negative, max_think_tokens=8,
    )

    def cancel_on_progress(value, total, preview, node_id=None):
        assert model.model.current_patcher is model
        assert bool(comfy.model_prefetch.PREFETCH_QUEUES) == dynamic
        comfy.model_management.interrupt_current_processing()

    layers = model.model.diffusion_model.language_model.model.layers
    completed_prefill_layers = []

    def cancel_after_layer(module, inputs, output):
        completed_prefill_layers.append(module)
        cancel_on_progress(0, 0, None)

    handles = [layer.mlp.register_forward_hook(cancel_after_layer) for layer in layers] if cancel_during_prefill else []
    if not cancel_during_prefill:
        monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", cancel_on_progress)
    try:
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            _run_prepared_generation(generation_mode="interleave" if interleave else "image", **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        comfy.model_management.interrupt_current_processing(False)
    if cancel_during_prefill:
        assert completed_prefill_layers == [layers[0].mlp]
    assert comfy.model_prefetch.PREFETCH_QUEUES == []
    assert model.model.current_patcher is None
    assert positive[0][1]["text_input_ids"].tolist() == [[1]]

    monkeypatch.setattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
    completed = _run_prepared_generation(generation_mode="interleave" if interleave else "image", **kwargs)
    expected = [[1, 42, 43]] if interleave else [[1, 42, 43, 151668, 271, 151670]]
    assert completed.args[0][0][1]["text_input_ids"].tolist() == expected
    assert model.model.current_patcher is None
    assert comfy.model_prefetch.PREFETCH_QUEUES == []


@pytest.mark.parametrize("dynamic", [False, True])
def test_sensenova_generate_uses_native_prefetch_for_dynamic_models(monkeypatch, dynamic):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 43, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    options = {"prefetch_dynamic_vbars": not dynamic}
    model.model_options["transformer_options"] = options
    active_prefetch = []
    monkeypatch.setattr(
        comfy.utils, "PROGRESS_BAR_HOOK",
        lambda *args, **kwargs: active_prefetch.append(bool(comfy.model_prefetch.PREFETCH_QUEUES)),
    )
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
    }]]
    try:
        output = _run_prepared_generation(
            generation_mode="image",
            model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
            max_think_tokens=8,
        )
        assert output.args[0][0][1]["text_input_ids"].tolist() == [[1, 42, 43, 151668, 271, 151670]]
        assert output.args[1] == "KL"
        assert active_prefetch and all(active == dynamic for active in active_prefetch)
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model_options["transformer_options"] == {"prefetch_dynamic_vbars": not dynamic}
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()


@pytest.mark.parametrize("dynamic", [False, True])
def test_sensenova_sampler_prefix_uses_current_patcher_prefetch(monkeypatch, dynamic):
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    if dynamic:
        model = comfy.model_patcher.ModelPatcherDynamic(
            model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
        )
    model.model.concat_keys = ()
    model.pre_run()
    try:
        conds = model.model.extra_conds(
            text_input_ids=torch.tensor([[1]]), device=model.load_device,
        )
        assert conds["prefix_time"].cond.tolist() == [1]
        assert len(conds["prefix_keys"].cond) == 2
        for key, value in zip(conds["prefix_keys"].cond, conds["prefix_values"].cond):
            assert key.shape == value.shape == (1, 1, 1, 8)
            assert torch.count_nonzero(key) == torch.count_nonzero(value) == 0
        assert bool(comfy.model_prefetch.PREFETCH_QUEUES) == dynamic
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
        model.cleanup()


def _add_identity_decoder_layers(monkeypatch, model):
    backbone = model.model.diffusion_model.language_model.model
    for name, value in {
        "HIDDEN_SIZE": backbone.embed_tokens.embedding_dim,
        "INTERMEDIATE_SIZE": 4, "NUM_HEADS": 1, "NUM_KV_HEADS": 1, "HEAD_DIM": 8,
    }.items():
        monkeypatch.setattr(sensenova_model, name, value)
    backbone.layers = torch.nn.ModuleList([
        sensenova_model.DecoderLayer(operations=torch.nn) for _ in range(2)
    ])
    for parameter in backbone.layers.parameters():
        torch.nn.init.zeros_(parameter)


def test_sensenova_generate_layer_failure_releases_prefetch_and_can_retry(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    model = _text_model_patcher([42, 151668])
    _add_identity_decoder_layers(monkeypatch, model)
    model = comfy.model_patcher.ModelPatcherDynamic(
        model.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
    )
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
    }]]
    kwargs = dict(model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning, max_think_tokens=8)

    def fail_layer(module, inputs):
        assert comfy.model_prefetch.PREFETCH_QUEUES
        raise RuntimeError("layer failed")

    handle = model.model.diffusion_model.language_model.model.layers[0].mlp.register_forward_pre_hook(fail_layer)
    try:
        with pytest.raises(RuntimeError, match="layer failed"):
            _run_prepared_generation(generation_mode="image", **kwargs)
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model.current_patcher is None
        assert conditioning[0][1]["text_input_ids"].tolist() == [[1]]
    finally:
        handle.remove()
        comfy.model_prefetch.cleanup_prefetch_queues()
    assert _run_prepared_generation(generation_mode="image", **kwargs).args[1] == "K"
    assert comfy.model_prefetch.PREFETCH_QUEUES == []


def test_sensenova_generate_releases_native_prefetch_on_initialization_failure(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", 1)
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "device_supports_non_blocking", lambda device: True)
    model = _text_model_patcher([42, 151668])
    conditioning = [[torch.empty(1), {
        "text_input_ids": torch.tensor([[1]]),
    }]]

    def fail_initialization(patcher):
        queue = comfy.model_prefetch.make_prefetch_queue(
            [], patcher.load_device, {"prefetch_dynamic_vbars": True},
        )
        assert queue is not None
        raise RuntimeError("initialization failed after prefetch")

    model.add_callback_with_key("on_pre_run", "prefetch_failure", fail_initialization)
    try:
        with pytest.raises(RuntimeError, match="initialization failed after prefetch"):
            _run_prepared_generation(
                generation_mode="image",
                model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
                max_think_tokens=8,
            )
        assert comfy.model_prefetch.PREFETCH_QUEUES == []
        assert model.model.current_patcher is None
        model.remove_callbacks_with_key("on_pre_run", "prefetch_failure")
        assert _run_prepared_generation(
            generation_mode="image",
            model=model, clip=SenseNovaQwenTokenizer(), conditioning=conditioning,
            max_think_tokens=8,
        ).args[1] == "K"
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()


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
        torch.zeros(1, 5, 8, 128, dtype=compute_dtype), torch.zeros(1, 5, 8, 128, dtype=compute_dtype), 2,
        torch.empty(1, dtype=torch.int64), torch.tensor([2], dtype=torch.int32),
    )
    cache.key[:, :2].copy_(prefix.transpose(1, 2))
    cache.value[:, :2].copy_(prefix.transpose(1, 2))
    for position in (2, 3):
        hidden.fill_(position - 1)
        cache.prepare(1)
        output = attention.forward_decode(hidden, rope, cache, {})
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


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("prior_hook", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_sensenova_generation_applies_conditioning_weights_and_restores_sampling(monkeypatch, interleave, prior_hook, interrupt):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 43, 151670 if interleave else 151668])
    embed = model.model.diffusion_model.language_model.model.embed_tokens
    base = embed.weight.detach().clone()
    key = "diffusion_model.language_model.model.embed_tokens.weight"

    def weight_group(column):
        delta = torch.zeros_like(base)
        delta[1] = -base[1]
        delta[1, column] += 1
        hook = comfy.hooks.WeightHook()
        hook.need_weight_init = False
        hook.weights = {key: (delta,)}
        group = comfy.hooks.HookGroup()
        group.add(hook)
        return group

    generation_hooks, sampling_hooks = weight_group(1), weight_group(2)
    if prior_hook:
        hook = sampling_hooks.hooks[0]
        model.add_hook_patches(hook, hook.weights)
        model.apply_hooks(sampling_hooks)
    previous_hooks = model.current_hooks
    previous_weights = embed.weight.detach().clone()
    previous_patches = model.hook_patches
    previous_uuid = model.patches_uuid
    seen = []

    def inspect_prefill(module, inputs, output):
        if not seen:
            seen.append(output.clone())
            assert torch.equal(output[0, 0], base[42])
            if interrupt:
                raise comfy.model_management.InterruptProcessingException()

    handle = embed.register_forward_hook(inspect_prefill)
    positive = [[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "hooks": generation_hooks}]]
    negative = [[torch.empty(1), {"text_input_ids": torch.tensor([[9]]), "hooks": sampling_hooks}]]
    try:
        if interrupt:
            with pytest.raises(comfy.model_management.InterruptProcessingException):
                _run_prepared_generation(generation_mode="interleave" if interleave else "image", model=model,
                                         clip=SenseNovaQwenTokenizer(), conditioning=positive, negative=negative, max_think_tokens=8)
        else:
            result = _run_prepared_generation(generation_mode="interleave" if interleave else "image", model=model,
                                             clip=SenseNovaQwenTokenizer(), conditioning=positive, negative=negative, max_think_tokens=8)
            assert result.args[1] == "L"
            assert result.args[0][0][1]["hooks"] is generation_hooks
            if interleave:
                assert result.args[2][0][1]["hooks"] is sampling_hooks
    finally:
        handle.remove()
    assert seen
    assert torch.equal(embed.weight, previous_weights)
    assert model.current_hooks is previous_hooks
    assert model.hook_patches is previous_patches
    assert model.patches_uuid == previous_uuid
    assert model.model.current_patcher is None
    assert comfy.model_prefetch.PREFETCH_QUEUES == []

    model.register_all_hook_patches(sampling_hooks, comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model))
    model.apply_hooks(sampling_hooks)
    assert torch.equal(embed(torch.tensor([[1]]))[0, 0], base[43])
    model.cleanup()


@pytest.mark.parametrize("storage_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_sensenova_plain_cpu_fp8_uses_native_embedding_for_prefill_and_decode(monkeypatch, storage_dtype):
    embed = comfy.ops.manual_cast.Embedding(8, 4, dtype=storage_dtype, device="cpu")
    embed.weight = torch.nn.Parameter(torch.arange(32).reshape(8, 4).to(storage_dtype), requires_grad=False)
    embed._v = (SimpleNamespace(base_addr=0, get_watermark=lambda: 0), 0, 1)
    embed.weight_lowvram_function = None
    monkeypatch.setattr(embed, "host_rows", lambda *args, **kwargs: pytest.fail("Plain CPU FP8 requires native embedding lookup"))
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed, layers=[])))
    ids = torch.tensor([[1, 2]])
    prefix, _, _, _ = sensenova_model.SenseNovaU15._prepare_prefix(model, ids, None, None, None, compute_dtype=torch.bfloat16)
    expected = torch.nn.functional.embedding(ids, embed.weight.float()).bfloat16()
    assert torch.equal(prefix, expected)
    hidden = torch.empty(1, 1, 4, dtype=torch.bfloat16)
    sensenova_model.SenseNovaU15._decode_text_token(model, ids[:, 0], [], hidden, None)
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
    prefix, _, _, _ = sensenova_model.SenseNovaU15._prepare_prefix(model, ids, None, None, None, compute_dtype=torch.bfloat16)
    assert torch.equal(prefix, expected)
    hidden = torch.empty(1, 1, 256, dtype=torch.bfloat16)
    sensenova_model.SenseNovaU15._decode_text_token(model, ids[:, 0], [], hidden, None)
    assert torch.equal(hidden, expected[:, :1])
    assert isinstance(embed.weight, comfy.quant_ops.QuantizedTensor)


def test_sensenova_generation_weight_hook_uses_native_dynamic_delegate(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", lambda *args: None)
    original = _text_model_patcher([42, 43, 151668])
    model = comfy.model_patcher.ModelPatcherDynamic(_text_model_patcher([42, 43, 151668]).model,
                                                 load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    model.non_dynamic_delegate_model = original.get_clone_model_override()
    weights = original.model.diffusion_model.language_model.model.embed_tokens.weight.detach()
    delta = torch.zeros_like(weights)
    delta[1] = weights[42] - weights[1]
    hook = comfy.hooks.WeightHook()
    hook.need_weight_init = False
    hook.weights = {"diffusion_model.language_model.model.embed_tokens.weight": (delta,)}
    group = comfy.hooks.HookGroup()
    group.add(hook)
    loaded = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: loaded.extend(models))
    result = _run_prepared_generation(generation_mode="image", model=model, clip=SenseNovaQwenTokenizer(),
                                     conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "hooks": group}]],
                                     max_think_tokens=8)
    assert result.args[1] == "L"
    assert len(loaded) == 1 and not loaded[0].is_dynamic()
    assert model.is_dynamic()
    assert model.current_hooks is None
    assert model.hook_patches == {}


def test_sensenova_generation_weight_hook_preserves_transformer_patches(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 151668])
    patch = lambda *args: None
    model.set_model_attn1_patch(patch)
    hook = comfy.hooks.WeightHook()
    hook.need_weight_init = False
    hook.weights = {}
    group = comfy.hooks.HookGroup()
    group.add(hook)
    diffusion = model.model.diffusion_model
    prefill = diffusion._preprocess_prefix_state
    seen = []

    def inspect_prefill(ids, references, indexes, mask, transformer_options=None, **kwargs):
        seen.append(transformer_options["patches"]["attn1_patch"])
        return prefill(ids, references, indexes, mask, transformer_options=transformer_options, **kwargs)

    monkeypatch.setattr(diffusion, "_preprocess_prefix_state", inspect_prefill)
    result = _run_prepared_generation(generation_mode="image", model=model, clip=SenseNovaQwenTokenizer(),
                                     conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "hooks": group}]],
                                     max_think_tokens=8)
    assert result.args[1] == "K"
    assert seen == [[patch]]
    assert model.model_options["transformer_options"] == {"patches": {"attn1_patch": [patch]}}


@pytest.mark.parametrize("scope", list(comfy.hooks.EnumHookScope))
@pytest.mark.parametrize("with_weight_hook", [False, True])
def test_sensenova_generation_registers_transformer_hooks(monkeypatch, scope, with_weight_hook):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 151668])
    override = lambda *args, **kwargs: None
    hook = comfy.hooks.TransformerOptionsHook({"optimized_attention_override": override}, hook_scope=scope)
    hooks = comfy.hooks.HookGroup()
    hooks.add(hook)
    if with_weight_hook:
        weight_hook = comfy.hooks.WeightHook()
        weight_hook.need_weight_init = False
        weight_hook.weights = {}
        hooks.add(weight_hook)
    diffusion = model.model.diffusion_model
    prefill = diffusion._preprocess_prefix_state
    seen = []

    def inspect_prefill(ids, references, indexes, mask, transformer_options=None, **kwargs):
        seen.append(transformer_options["optimized_attention_override"])
        return prefill(ids, references, indexes, mask, transformer_options=transformer_options, **kwargs)

    monkeypatch.setattr(diffusion, "_preprocess_prefix_state", inspect_prefill)
    result = _run_prepared_generation(generation_mode="image", model=model, clip=SenseNovaQwenTokenizer(),
                                     conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "hooks": hooks}]],
                                     max_think_tokens=8)
    assert result.args[1] == "K"
    assert seen == [override]
    assert model.model_options["transformer_options"] == {}
    assert not hook._skip_adding


@pytest.mark.parametrize("initial_strength", [0.75, 1.0])
def test_sensenova_text_stages_use_initial_hook_keyframe(monkeypatch, initial_strength):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: None)
    model = _text_model_patcher([42, 43, 151668])
    model.model.model_sampling = SenseNovaModelSampling()
    weight = model.model.diffusion_model.language_model.model.embed_tokens.weight.detach()
    delta = torch.zeros_like(weight)
    delta[1] = weight[42] - weight[1]
    hook = comfy.hooks.WeightHook()
    hook.need_weight_init = False
    hook.weights = {"diffusion_model.language_model.model.embed_tokens.weight": (delta,)}
    hook.hook_keyframe.add(comfy.hooks.HookKeyframe(initial_strength, 0.0))
    hook.hook_keyframe.add(comfy.hooks.HookKeyframe(0.0, 0.5))
    hooks = comfy.hooks.HookGroup()
    hooks.add(hook)
    kwargs = dict(generation_mode="image", model=model, clip=SenseNovaQwenTokenizer(),
                  conditioning=[[torch.empty(1), {"text_input_ids": torch.tensor([[1]]), "hooks": hooks}]],
                  max_think_tokens=8)
    assert _run_prepared_generation(**kwargs).args[1] == "L"
    model.register_all_hook_patches(hooks, comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model))
    hook.initialize_timesteps(model.model)
    sigmas = torch.tensor([1.0, 0.0])
    for sigma in sigmas:
        model.prepare_hook_patches_current_keyframe(sigma[None], hooks, {"transformer_options": {"sample_sigmas": sigmas}})
        model.apply_hooks(hooks)
    model.cleanup()
    assert hook.strength == 0.0
    assert _run_prepared_generation(**kwargs).args[1] == "L"
    assert hook.strength == 0.0
