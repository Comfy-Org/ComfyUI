from types import SimpleNamespace

import pytest
import torch

import comfy.model_base
from comfy import model_management, ops
from comfy.cli_args import parser
from comfy.ldm.krea2 import sm120
from comfy.ldm.krea2.model import RMSNorm
from comfy.quant_ops import QuantizedTensor


@pytest.fixture
def square_profile(monkeypatch):
    class Krea2:
        current_weight_patches_uuid = "loaded"
        diffusion_model = SimpleNamespace(blocks=[
            SimpleNamespace(attn=SimpleNamespace(wq=SimpleNamespace(quant_format="float8_e4m3fn")))
        ])

        def get_dtype_inference(self):
            return torch.float16

    patcher = SimpleNamespace(
        model=Krea2(), patches_uuid="loaded", loaded_size=lambda: 1, hook_patches={},
        object_patches={}, injections={}, wrappers={}, callbacks={}, model_options={"transformer_options": {}},
    )
    sampler = SimpleNamespace(
        model=patcher, device="cuda:0", sampler="euler", scheduler="simple", steps=8,
        denoise=1.0, model_options={"transformer_options": {}},
    )
    monkeypatch.setattr(comfy.model_base, "Krea2", Krea2)
    monkeypatch.setattr(sm120, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(sm120, "torch", SimpleNamespace(
        device=torch.device, float16=torch.float16, is_tensor=torch.is_tensor,
        is_inference_mode_enabled=lambda: True,
        cuda=SimpleNamespace(get_device_capability=lambda _: (12, 0),
                             get_device_name=lambda _: "NVIDIA GeForce RTX 5090"),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_fp16_accumulation=True))),
    ))
    monkeypatch.setattr(model_management.args, "use_sage_attention", True)
    monkeypatch.setattr(model_management.args, "use_pytorch_cross_attention", False)
    monkeypatch.setattr(model_management.args, "use_flash_attention", False)
    monkeypatch.setattr(model_management.args, "use_ck_attention", False)
    monkeypatch.setattr(model_management, "ENABLE_PYTORCH_ATTENTION", True)
    monkeypatch.setattr(model_management, "xformers_enabled", lambda: False)
    noise = torch.empty((1, 16, 1, 256, 256), device="meta")
    positive = [[torch.empty((1, 478, 30720), device="meta"), {}]]
    negative = [[torch.empty_like(positive[0][0]), {}]]
    return sampler, noise, positive, negative, 1, torch.empty(9), None


def test_admits_verified_square_profile(square_profile):
    assert sm120.eligible(*square_profile)


@pytest.mark.parametrize("convrot", [False, True], ids=["ordinary-int8", "convrot"])
def test_only_convrot_int8_weights_can_use_sm120(square_profile, convrot):
    projection = square_profile[0].model.model.diffusion_model.blocks[0].attn.wq
    projection.quant_format = "int8_tensorwise"
    projection.weight = QuantizedTensor.from_float(
        torch.ones((16, 256), dtype=torch.float16),
        "TensorWiseINT8Layout", per_channel=True, convrot=convrot,
    )
    assert sm120.eligible(*square_profile) is convrot


def test_installed_sm120_package_can_be_enabled_without_a_path():
    assert parser.parse_args([]).krea2_sm120 is None
    assert parser.parse_args(["--krea2-sm120"]).krea2_sm120 is True


@pytest.mark.parametrize("force_pytorch", [False, True], ids=["automatic", "explicit"])
def test_sm120_can_keep_pytorch_attention(square_profile, monkeypatch, force_pytorch):
    monkeypatch.setattr(model_management.args, "use_sage_attention", False)
    monkeypatch.setattr(model_management.args, "use_pytorch_cross_attention", force_pytorch)
    assert sm120.eligible(*square_profile)


@pytest.mark.parametrize("sampler_name", ["euler", "heun", "dpmpp_2m"])
@pytest.mark.parametrize("steps", [1, 4, 8, 12])
def test_admits_supported_samplers_with_variable_steps(square_profile, sampler_name, steps):
    sampler, noise, positive, negative, cfg, _, mask = square_profile
    sampler.sampler = sampler_name
    sampler.steps = steps
    assert sm120.eligible(sampler, noise, positive, negative, cfg, torch.empty(steps + 1), mask)


@pytest.mark.parametrize("sampler_name", ["euler_cfg_pp", "dpmpp_2m_cfg_pp"])
def test_cfg_pp_keeps_stock_sampling_at_cfg_one(square_profile, sampler_name):
    square_profile[0].sampler = sampler_name
    assert not sm120.eligible(*square_profile)


def test_partial_sigma_schedule_keeps_stock_sampling(square_profile):
    sampler, noise, positive, negative, cfg, sigmas, mask = square_profile
    assert not sm120.eligible(sampler, noise, positive, negative, cfg, sigmas[2:], mask)


@pytest.mark.parametrize("flag", ["use_flash_attention", "use_ck_attention"])
def test_explicit_attention_backend_takes_precedence_over_automatic_pytorch(square_profile, monkeypatch, flag):
    monkeypatch.setattr(model_management.args, "use_sage_attention", False)
    monkeypatch.setattr(model_management.args, flag, True)
    assert not sm120.eligible(*square_profile)


def test_xformers_takes_precedence_over_automatic_pytorch(square_profile, monkeypatch):
    monkeypatch.setattr(model_management.args, "use_sage_attention", False)
    monkeypatch.setattr(model_management, "xformers_enabled", lambda: True)
    assert not sm120.eligible(*square_profile)


def test_missing_sm120_package_keeps_stock_sampling(caplog, monkeypatch):
    def missing_package(name):
        raise sm120.PackageNotFoundError(name)

    monkeypatch.setattr(sm120, "distribution", missing_package)
    sm120.load_backend.cache_clear()
    try:
        assert sm120.load_backend(True) is None
        assert "requirements-krea2-sm120.txt" in caplog.text
    finally:
        sm120.load_backend.cache_clear()


@pytest.mark.parametrize(("noise_shape", "tokens"), [
    ((1, 16, 1, 256, 256), 13),
    ((1, 16, 1, 255, 255), 478),
    ((1, 16, 320, 240), 478),
    ((1, 16, 1, 125, 96), 645),
])
def test_admits_variable_resolution_and_prompt_length(square_profile, noise_shape, tokens):
    sampler, _, _, negative, cfg, sigmas, mask = square_profile
    noise = torch.empty(noise_shape, device="meta")
    positive = [[torch.empty((1, tokens, 30720), device="meta"), {}]]
    assert sm120.eligible(sampler, noise, positive, negative, cfg, sigmas, mask)


@pytest.mark.parametrize("noise_shape", [
    (2, 16, 320, 240),
    (1, 4, 320, 240),
    (1, 16, 2, 320, 240),
    (1, 16, 0, 240),
])
def test_other_latent_layouts_keep_stock_sampling(square_profile, noise_shape):
    sampler, _, positive, negative, cfg, sigmas, mask = square_profile
    noise = torch.empty(noise_shape, device="meta")
    assert not sm120.eligible(sampler, noise, positive, negative, cfg, sigmas, mask)


@pytest.mark.parametrize("context_shape", [(13, 30720), (2, 13, 30720), (1, 0, 30720), (1, 13, 2560)])
def test_other_conditioning_layouts_keep_stock_sampling(square_profile, context_shape):
    sampler, noise, _, negative, cfg, sigmas, mask = square_profile
    positive = [[torch.empty(context_shape, device="meta"), {}]]
    assert not sm120.eligible(sampler, noise, positive, negative, cfg, sigmas, mask)


@pytest.mark.parametrize("key", ["optimized_attention_override", "patches_replace"])
def test_standard_attention_extensions_keep_the_stock_path(square_profile, key):
    square_profile[0].model.model_options["transformer_options"][key] = object()
    assert not sm120.eligible(*square_profile)


def test_alternate_condition_batcher_keeps_the_stock_path(square_profile):
    square_profile[0].model.model_options["sampler_calc_cond_batch_function"] = object()
    assert not sm120.eligible(*square_profile)


def test_negative_condition_hook_keeps_the_stock_path_at_cfg_one(square_profile):
    square_profile[3][0][1]["hooks"] = object()
    assert not sm120.eligible(*square_profile)


def test_uses_patcher_attention_options_when_sampler_options_are_empty(square_profile):
    sampler = square_profile[0]
    sampler.model_options = {}
    sampler.model.model_options["transformer_options"]["optimized_attention_override"] = object()
    assert not sm120.eligible(*square_profile)


@pytest.mark.parametrize("is_eligible", [False, True], ids=["ineligible", "backend-unavailable"])
def test_nested_fallback_sampling_suspends_outer_profile(monkeypatch, is_eligible):
    monkeypatch.setattr(sm120, "eligible", lambda *_: is_eligible)
    monkeypatch.setattr(sm120, "load_backend", lambda _: None)
    outer_profile = object()
    token = sm120.active_sample.set(outer_profile)
    try:
        with pytest.raises(RuntimeError, match="nested sample failed"):
            with sm120.sample_scope("kernels", None, None, None, None, 1, None, None):
                assert sm120.current_sample() is None
                raise RuntimeError("nested sample failed")
        assert sm120.current_sample() is outer_profile
    finally:
        sm120.active_sample.reset(token)


@pytest.fixture
def unreadable_sample_context(monkeypatch):
    def fail_if_read():
        raise AssertionError("Compiled paths must skip SM120 context reads")

    monkeypatch.setattr(sm120, "active_sample", SimpleNamespace(get=fail_if_read))


def test_sm120_linear_can_compile_as_a_full_graph(unreadable_sample_context):
    layer = ops.mixed_precision_ops().Linear(4, 4)
    x = torch.ones(2, 4)
    weight = torch.eye(4)
    forward = torch.compile(layer._forward, backend="eager", fullgraph=True)
    torch.testing.assert_close(forward(x, weight, None), x)


def test_krea_rms_norm_can_compile_as_a_full_graph(unreadable_sample_context):
    norm = RMSNorm(4)
    torch.nn.init.zeros_(norm.scale)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = torch.nn.functional.rms_norm(x, (4,), weight=torch.ones(4), eps=norm.eps)
    forward = torch.compile(norm, backend="eager", fullgraph=True)
    torch.testing.assert_close(forward(x), expected)
