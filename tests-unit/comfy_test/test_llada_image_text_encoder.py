import hashlib
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

import comfy.ops
import comfy.supported_models
import comfy.ldm.modules.attention
import comfy.text_encoders.llada_image as llada_image
from comfy.text_encoders.llada_image import (
    LLaDA2Backbone,
    LLaDA2Config,
    LLaDA2Experts,
    LLaDA2Gate,
    LLaDAImageClipModel,
    LLaDAImageRawTokenizer,
    LLaDAImageTEModel,
    _rms_norm,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_vq_memory_estimation_includes_cfg_logits_and_block_padding(dtype):
    config = LLaDA2Config()
    model = SimpleNamespace(llada2=SimpleNamespace(config=config, dtype=dtype))
    estimate = LLaDAImageTEModel.vq_memory_estimation
    memory = estimate(model, 1050)
    two_block_logits = 2 * 2 * 32 * config.vocab_size * dtype.itemsize

    assert isinstance(memory, int)
    assert memory > two_block_logits
    assert memory == estimate(model, 1056)
    assert memory > estimate(model, 1024)
    assert estimate(model, 2048) > memory


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_rms_norm_rounds_before_nonunit_affine_weight(dtype):
    inputs = torch.linspace(-17, 29, 2048, dtype=dtype).reshape(2, 1024)
    norm = comfy.ops.manual_cast.RMSNorm(1024, eps=1e-6, dtype=dtype)
    norm.weight.data.copy_(torch.linspace(0.1, 3.4, 1024, dtype=dtype))
    full_precision = inputs.float()
    normalized = full_precision * torch.rsqrt(full_precision.square().mean(-1, keepdim=True) + norm.eps)
    expected = normalized.to(dtype) * norm.weight
    torch.testing.assert_close(_rms_norm(inputs, norm), expected, rtol=0 if dtype == torch.bfloat16 else 1e-5, atol=0 if dtype == torch.bfloat16 else 1e-6)


def test_rms_norm_uses_patched_affine_weight():
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    norm = comfy.ops.manual_cast.RMSNorm(4, eps=1e-6)
    norm.weight.data.fill_(1.0)
    norm.weight_function = [lambda weight: weight * 2]
    expected = torch.nn.functional.rms_norm(inputs, (4,), eps=norm.eps) * 2
    torch.testing.assert_close(_rms_norm(inputs, norm), expected)


def load_official_text_encoder_module():
    source = os.environ.get("LLADA_IMAGE_TEXT_ENCODER_CODE")
    if not source:
        pytest.skip(
            "set LLADA_IMAGE_TEXT_ENCODER_CODE to the pinned official text encoder directory"
        )
    root = Path(source)
    if root.is_file():
        root = root.parent
    required = (
        "configuration_llada2uni_moe.py",
        "fused_moe_ops.py",
        "modeling_llada2uni_moe.py",
    )
    if any(not (root / name).is_file() for name in required):
        pytest.skip("the pinned official text encoder code fixture is incomplete")
    expected_sha256 = {
        "configuration_llada2uni_moe.py": "e9c88818e03e390cdfa3b683a5235f6d4a7132b56aa6af7d6db9c1bbd4dfd6fe",
        "fused_moe_ops.py": "e20ed7ec34ae2d57cfcb40ea34b83a669aac3ff12615c86eac517cc3597e8a08",
        "modeling_llada2uni_moe.py": "c74fdf1183d411a5f7a33621725d45fd8fc98161e586c5a159b83708c081382f",
    }
    for name, expected in expected_sha256.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"official text encoder fixture hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )

    package_name = "_llada_image_official_text_encoder"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    for stem in ("configuration_llada2uni_moe", "fused_moe_ops"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{stem}", root / f"{stem}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    stem = "modeling_llada2uni_moe"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{stem}", root / f"{stem}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parity_device():
    device = torch.device(os.environ.get("LLADA_IMAGE_PARITY_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("LLADA_IMAGE_PARITY_DEVICE=cuda requires a CUDA PyTorch build")
    return device


def matching_finite_values(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(actual)
    assert finite.any()
    return finite


def assert_bfloat16_parity(actual, expected):
    finite = matching_finite_values(actual, expected)
    absolute_error = (actual.float() - expected.float()).abs()[finite]
    assert float(absolute_error.max()) <= 1.0 / 64.0
    assert float(absolute_error.mean()) <= 1.0 / 256.0


def small_backbone():
    config = LLaDA2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        pad_token_id=0,
        max_position_embeddings=64,
    )
    operations = comfy.ops.mixed_precision_ops(
        {}, torch.float32, full_precision_mm=True
    )
    model = LLaDA2Backbone(
        config, torch.float32, torch.device("cpu"), operations
    ).eval()
    torch.manual_seed(20)
    state_dict = {
        key: torch.randn_like(value) * 0.02 for key, value in model.state_dict().items()
    }
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if hasattr(module, "_orig_shape"):
            state_dict[f"{prefix}weight"] = torch.randn(module._orig_shape) * 0.02
        elif (
            hasattr(module, "in_features")
            and hasattr(module, "out_features")
            and getattr(module, "weight", None) is None
        ):
            state_dict[f"{prefix}weight"] = (
                torch.randn(module.out_features, module.in_features) * 0.02
            )
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return model


def test_bfloat16_moe_matches_pinned_official_eager_path(monkeypatch):
    official = load_official_text_encoder_module()
    monkeypatch.setenv("LLADA_MOE_BACKEND", "eager")
    device = parity_device()
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "rms_norm_eps": 1e-6,
        "rope_theta": 600000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
    }
    official_config = official.LLaDA2MoeConfig(
        **values,
        attention_dropout=0.0,
        use_cache=False,
        use_bias=False,
        use_qkv_bias=False,
        use_qk_norm=True,
        output_router_logits=False,
    )
    native_config = LLaDA2Config(**values)
    operations = comfy.ops.mixed_precision_ops(
        {}, torch.bfloat16, full_precision_mm=True
    )
    official_gate = official.LLaDA2MoeGate(official_config).to(
        device=device, dtype=torch.bfloat16
    )
    official_experts = official.LLaDA2MoeExperts(official_config).to(
        device=device, dtype=torch.bfloat16
    )
    native_gate = LLaDA2Gate(native_config, torch.bfloat16, device)
    native_experts = LLaDA2Experts(
        native_config,
        torch.bfloat16,
        device,
        operations,
    )

    torch.manual_seed(37)
    with torch.no_grad():
        official_gate.weight.normal_(std=0.2)
        official_gate.expert_bias.normal_(std=0.05)
        official_experts.gate_proj.normal_(std=0.2)
        official_experts.up_proj.normal_(std=0.2)
        official_experts.down_proj.normal_(std=0.2)
    native_gate.load_state_dict(official_gate.state_dict(), strict=True)
    native_experts.load_state_dict(
        {
            "gate_proj.weight": official_experts.gate_proj,
            "up_proj.weight": official_experts.up_proj,
            "down_proj.weight": official_experts.down_proj,
        },
        strict=True,
    )
    hidden_states = torch.randn(3, 16, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        expected_indices, expected_weights, _ = official_gate(hidden_states)
        actual_indices, actual_weights = native_gate(hidden_states)
        expected = official_experts(
            hidden_states, expected_weights, expected_indices
        )
        actual = native_experts(hidden_states, actual_weights, actual_indices)

    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.bfloat16), ids=("float32", "bfloat16")
)
def test_text_backbone_matches_pinned_official_eager_path(monkeypatch, dtype):
    official = load_official_text_encoder_module()
    monkeypatch.setenv("LLADA_MOE_BACKEND", "eager")
    # Compare the same SDPA contract, independent of Core's ambient CPU backend.
    monkeypatch.setattr(
        llada_image, "optimized_attention", comfy.ldm.modules.attention.attention_pytorch
    )
    device = torch.device("cpu") if dtype == torch.float32 else parity_device()
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "rms_norm_eps": 1e-6,
        "rope_theta": 600000.0,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
    }
    official_config = official.LLaDA2MoeConfig(
        **values,
        attention_dropout=0.0,
        use_cache=False,
        use_bias=False,
        use_qkv_bias=False,
        use_qk_norm=True,
        output_router_logits=False,
    )
    official_config._attn_implementation = "sdpa"
    native_config = LLaDA2Config(**values)
    operations = comfy.ops.mixed_precision_ops({}, dtype, full_precision_mm=True)
    torch.manual_seed(38)
    expected_model = (
        official.LLaDA2MoeBackbone(official_config)
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        for name, parameter in expected_model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1.0)
            elif name.endswith("expert_bias"):
                parameter.zero_()
            else:
                parameter.normal_(std=0.02)
    actual_model = LLaDA2Backbone(
        native_config,
        dtype,
        device,
        operations,
    ).eval()
    state_dict = {}
    expert_suffixes = (
        ".mlp.experts.gate_proj",
        ".mlp.experts.up_proj",
        ".mlp.experts.down_proj",
    )
    for key, value in expected_model.state_dict().items():
        state_dict[f"{key}.weight" if key.endswith(expert_suffixes) else key] = value
    incompatible = actual_model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 0, 0], [8, 7, 6, 5, 4, 3, 2]],
        dtype=torch.long,
        device=device,
    )
    key_valid = input_ids != 0
    causal = torch.ones(7, 7, dtype=torch.bool, device=device).tril()
    attention_mask = key_valid[:, None, None, :] & causal[None, None]
    position_ids = key_valid.long().cumsum(dim=1) - 1
    position_ids.clamp_min_(0)

    with torch.inference_mode():
        expected = expected_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        ).last_hidden_state
        actual = actual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    if dtype == torch.bfloat16:
        assert_bfloat16_parity(actual, expected)
    else:
        assert torch.isfinite(actual).all()
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("image_token_count", (17, 65))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("near_threshold", (False, True))
def test_vq_block_diffusion_matches_pinned_official_logic(image_token_count, dtype, near_threshold):
    official = load_official_text_encoder_module()

    class OfficialTiny:
        device = torch.device("cpu")
        _top_k_logits = staticmethod(official.LLaDA2MoeModelLM._top_k_logits)
        _top_p_logits = staticmethod(official.LLaDA2MoeModelLM._top_p_logits)
        _sample_with_temperature_topk_topp = (
            official.LLaDA2MoeModelLM._sample_with_temperature_topk_topp
        )
        _get_num_transfer_tokens = staticmethod(
            official.LLaDA2MoeModelLM._get_num_transfer_tokens
        )
        generate = official.LLaDA2MoeModelLM.generate_bd_image_logic

        def __call__(self, input_ids, attention_mask, position_ids):
            del attention_mask, position_ids
            if near_threshold:
                logits = torch.full((*input_ids.shape, 12), -100.0, dtype=dtype)
                chosen = 5 if torch.any(input_ids == 4) else 4
                logits[..., 4:6] = 0.0
                logits[..., chosen] = 2.953125
                return SimpleNamespace(logits=logits)
            logits = torch.full((*input_ids.shape, 12), -20.0, dtype=dtype)
            positions = torch.arange(input_ids.shape[1])
            selected = 4 + positions.remainder(4)
            logits.scatter_(
                -1,
                selected[None, :, None].expand(input_ids.shape[0], -1, -1),
                20.0,
            )
            return SimpleNamespace(logits=logits)

    class NativeTiny(LLaDAImageClipModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                mask_token_id=3,
                end_of_image_token_id=11,
                image_token_offset=4,
            )

        def forward_logits(self, input_ids, attention_mask, position_ids, logits_to_keep=0):
            assert logits_to_keep == 32
            return OfficialTiny()(input_ids, attention_mask, position_ids).logits[:, -logits_to_keep:]

    input_ids = torch.tensor([[1, 2, 3]])
    unconditional_ids = torch.tensor([2, 3])
    expected_output = OfficialTiny().generate(
        data={"input_ids": input_ids, "uncond_ids": unconditional_ids},
        block_length=32,
        steps=8,
        gen_length=image_token_count,
        threshold=0.95,
        mask_id=3,
        cfg_scale=2.0,
        mode="eoi",
    )
    expected = expected_output[
        :, input_ids.shape[1] : input_ids.shape[1] + image_token_count
    ] - 4
    actual = NativeTiny().generate_vq_tokens(
        input_ids, unconditional_ids, image_token_count, cfg_scale=2.0
    )

    assert torch.equal(actual, expected)


def test_moe_expert_banks_are_never_resident_together():
    config = LLaDA2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
    )
    experts = LLaDA2Experts(
        config,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.mixed_precision_ops({}, torch.float32),
    )
    for bank in (experts.gate_proj, experts.up_proj, experts.down_proj):
        bank.weight = torch.nn.Parameter(torch.randn(bank._orig_shape))

    residency = {"active": 0, "maximum": 0}
    for bank in (experts.gate_proj, experts.up_proj, experts.down_proj):
        original = bank.bank_resident

        @contextmanager
        def tracked(input_tensor, _original=original):
            residency["active"] += 1
            residency["maximum"] = max(residency["maximum"], residency["active"])
            try:
                with _original(input_tensor) as resident:
                    yield resident
            finally:
                residency["active"] -= 1

        bank.bank_resident = tracked

    hidden_states = torch.randn(6, config.hidden_size)
    selected_experts = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3]])
    routing_weights = torch.rand(6, config.num_experts_per_tok)

    output = experts(hidden_states, routing_weights, selected_experts)

    assert output.shape == hidden_states.shape
    assert residency == {"active": 0, "maximum": 1}


def test_backbone_bool_mask_blocks_padded_tokens_from_valid_outputs():
    model = small_backbone()
    first = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    second = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    allowed = valid[:, None, None, :].expand(-1, 1, 7, -1).repeat(2, 1, 1, 1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 4, 4]]).repeat(2, 1)

    with torch.inference_mode():
        output = model.language_model(
            torch.cat((first, second)),
            attention_mask=allowed,
            position_ids=position_ids,
        )

    torch.testing.assert_close(output[0, :5], output[1, :5], atol=2e-5, rtol=2e-5)
    assert torch.isfinite(output).all()


def test_aio_clip_processing_adapts_official_expert_tensor_names():
    config = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": "base"}
    )
    state_dict = {
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.gate_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.up_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.down_proj": torch.empty(
            8, 16, 4
        ),
    }

    processed = config.process_clip_state_dict(state_dict)

    assert set(processed) == {
        "llada2.model.language_model.layers.1.mlp.experts.gate_proj.weight",
        "llada2.model.language_model.layers.1.mlp.experts.up_proj.weight",
        "llada2.model.language_model.layers.1.mlp.experts.down_proj.weight",
    }


def test_aio_clip_component_prefixes_roundtrip_for_saving():
    config = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": "base"}
    )
    checkpoint_state = {
        "text_encoders.llada2.model.language_model.word_embeddings.weight": torch.empty(
            64, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.gate_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.queryformer.meta_queries": torch.empty(5, 16),
        "text_encoders.text_projection.projector.weight": torch.empty(8, 16),
        "text_encoders.sigvq.prior_token_embedding.weight": torch.empty(8, 10),
        "text_encoders.tokenizer_json": torch.empty(128, dtype=torch.uint8),
    }

    processed = config.process_clip_state_dict(dict(checkpoint_state))

    assert set(processed) == {
        "llada2.model.language_model.word_embeddings.weight",
        "llada2.model.language_model.layers.1.mlp.experts.gate_proj.weight",
        "llada2.queryformer.meta_queries",
        "llada2.text_projection.projector.weight",
        "llada2.sigvq.prior_token_embedding.weight",
        "tokenizer_json",
    }

    saved = config.process_clip_state_dict_for_saving(processed)

    assert set(saved) == set(checkpoint_state)
    for key in checkpoint_state:
        assert saved[key] is checkpoint_state[key]


class TinyVQGenerator(LLaDAImageClipModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            mask_token_id=3,
            end_of_image_token_id=11,
            image_token_offset=4,
        )

    def forward_logits(self, input_ids, attention_mask, position_ids, logits_to_keep=0):
        logits = torch.full((input_ids.shape[0], logits_to_keep, 12), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        return logits


@pytest.mark.parametrize("block_length", (1, 17, 32, 65))
@pytest.mark.parametrize("steps", (1, 3, 8, 64))
def test_vq_transfer_schedule_uses_python_counts(block_length, steps):
    expected = torch.full((steps,), block_length // steps, dtype=torch.int64)
    expected[: block_length % steps] += 1
    actual = LLaDAImageClipModel._num_transfer_tokens(block_length, steps)

    assert all(type(count) is int for count in actual)
    assert actual == expected.tolist()
    assert sum(actual) == block_length


@pytest.mark.parametrize("shape", ((2,), (1, 2)))
def test_vq_unconditional_tokens_stay_tensors(monkeypatch, shape):
    def reject_list_conversion(tensor):
        raise AssertionError("VQ token IDs must not round-trip through Python lists")

    class InspectUnconditional(TinyVQGenerator):
        def forward_logits(self, input_ids, attention_mask, position_ids, logits_to_keep=0):
            assert torch.equal(input_ids[1, :4], torch.tensor([3, 3, 7, 8]))
            assert torch.equal(position_ids[1, :4], torch.tensor([0, 0, 0, 1]))
            assert not attention_mask[1, :, :, :2].any()
            return super().forward_logits(input_ids, attention_mask, position_ids, logits_to_keep)

    monkeypatch.setattr(torch.Tensor, "tolist", reject_list_conversion)
    tokens = InspectUnconditional().generate_vq_tokens(
        torch.tensor([[1, 2, 5, 6]]), torch.tensor([7, 8]).reshape(shape), 4
    )
    assert torch.equal(tokens, torch.zeros(1, 4, dtype=torch.long))


def test_block_diffusion_vq_generation_is_greedy_and_offset_normalized():
    model = TinyVQGenerator()
    input_ids = torch.tensor([[1, 2]])
    unconditional_ids = torch.tensor([2])

    token_ids = model.generate_vq_tokens(input_ids, unconditional_ids, 4)

    assert token_ids.shape == (1, 4)
    assert torch.equal(token_ids, torch.zeros_like(token_ids))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_vq_logits_project_only_active_block(dtype):
    torch.manual_seed(42)
    hidden = torch.randn(2, 96, 16, dtype=dtype)
    projected_shapes = []

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(16, 24, bias=False, dtype=dtype)

        def forward(self, **kwargs):
            return hidden

    backbone = Backbone()
    model = SimpleNamespace(model=backbone)
    hook = backbone.lm_head.register_forward_pre_hook(lambda module, inputs: projected_shapes.append(inputs[0].shape))
    with torch.inference_mode():
        full = LLaDAImageClipModel.forward_logits(model, None, None, None)
        block = LLaDAImageClipModel.forward_logits(model, None, None, None, logits_to_keep=32)
    hook.remove()

    assert projected_shapes == [(2, 96, 16), (2, 32, 16)]
    torch.testing.assert_close(block, full[:, -32:], rtol=0, atol=0)


def test_official_tokenizer_golden_ids_when_reference_is_available():
    tokenizer_path = os.environ.get("LLADA_IMAGE_TOKENIZER_JSON")
    if not tokenizer_path or not os.path.isfile(tokenizer_path):
        pytest.skip(
            "set LLADA_IMAGE_TOKENIZER_JSON to run the pinned tokenizer parity fixture"
        )
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer_bytes = tokenizer_file.read()
    assert hashlib.sha256(tokenizer_bytes).hexdigest() == (
        "2197aeddaf09785316673451ca6fb86dcfcfdb108972a3145d106b8fa4c927e6"
    )
    tokenizer = LLaDAImageRawTokenizer(
        tokenizer_data={
            "tokenizer_json": torch.frombuffer(
                bytearray(tokenizer_bytes), dtype=torch.uint8
            )
        }
    )
    cases = {
        "": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            13,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "A red fox sleeping beneath a glass tree.": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            355,
            3862,
            46998,
            20666,
            24772,
            259,
            7966,
            6919,
            13,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "一只白猫坐在月亮上。": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            6097,
            1030,
            1474,
            8187,
            8986,
            33413,
            520,
            311,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "literal <IMAGE1> and <|image|> tokens": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            31567,
            220,
            157185,
            301,
            220,
            156901,
            21255,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
    }
    for prompt, expected in cases.items():
        actual = [pair[0] for pair in tokenizer.tokenize_with_weights(prompt)[0]]
        assert actual == expected

    long_ids = [
        pair[0]
        for pair in tokenizer.tokenize_with_weights("绘画 " + "blue mountain " * 1500)[
            0
        ]
    ]
    assert len(long_ids) == 2048
    assert hashlib.sha256(",".join(map(str, long_ids)).encode()).hexdigest() == (
        "521e55c839c1429a3e0525bea573025ca76b8a9cb5406f78b5ae896ae1c579bc"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("attention_dropout", 0.1),
        ("embedding_dropout", 0.1),
        ("output_dropout", 0.1),
        ("output_router_logits", True),
        ("rope_scaling", {"rope_type": "linear", "factor": 2.0}),
        ("score_function", "softmax"),
        ("sliding_window", 4096),
        ("use_cache", True),
    ],
)
def test_unsupported_conditioning_config_fails_instead_of_approximating(key, value):
    with pytest.raises(ValueError, match=key):
        LLaDAImageClipModel(
            dtype=torch.float32,
            model_options={"llada2_config": {key: value}},
        )
