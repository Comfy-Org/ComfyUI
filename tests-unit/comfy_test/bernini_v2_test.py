import pytest
import torch

import comfy.ops
import comfy.ldm.bernini_v2.qwen as bernini_qwen
from comfy.ldm.bernini_v2.guidance import (
    apg_delta,
    compose_denoised_guidance,
    compose_velocity_guidance,
    guidance_chunks,
    unipc_flow_sigmas,
)
from comfy.ldm.bernini_v2.media import fit_media_size, ordered_renderer_sources
from comfy.ldm.bernini_v2.planner import (
    maskgit_order,
    split_reference_images,
    validate_generation_request,
)
from comfy.ldm.bernini_v2.planner_model import (
    DiffLossFM,
    FlowMatchScheduler,
    MLPConnector,
)
from comfy.ldm.bernini_v2.presets import task_preset
from comfy.ldm.bernini_v2.qwen import (
    _explicit_gqa_attention,
    planner_video_frame_indices,
    process_qwen2vl_video,
    qwen_grid_for_media,
)
from comfy.ldm.bernini_v2.runtime import _embedded_bytes, _embedded_json
from comfy.ldm.bernini_v2.template import (
    BerniniTemplate,
    build_conversation,
    build_custom_attention_mask,
)
from comfy.ldm.bernini_v2.unipc import sample_flow_unipc_bh2
from comfy.text_encoders.qwen_vl import apply_rotary_pos_emb_vision


def test_guidance_chunks_default_to_low_memory_order():
    names = ["base", "source", "text", "target"]
    assert guidance_chunks(names, "auto") == [
        ["base"],
        ["source"],
        ["text"],
        ["target"],
    ]
    assert guidance_chunks(names, "2") == [["base", "source"], ["text", "target"]]
    assert guidance_chunks(names, "all") == [names]


def test_standard_and_rv2v_guidance_composition():
    predictions = {
        "base": torch.tensor([[1.0, 0.0]]),
        "source": torch.tensor([[2.0, 1.0]]),
        "text": torch.tensor([[2.0, 3.0]]),
        "target": torch.tensor([[4.0, 3.0]]),
    }
    expected = (
        predictions["base"]
        + 2.0
        * apg_delta(predictions["source"] - predictions["base"], predictions["source"])
        + 3.0
        * apg_delta(predictions["text"] - predictions["source"], predictions["text"])
        + 4.0
        * apg_delta(predictions["target"] - predictions["text"], predictions["target"])
    )
    torch.testing.assert_close(
        compose_velocity_guidance(
            predictions,
            omega_video=0.0,
            omega_image=2.0,
            omega_text=3.0,
            omega_target=4.0,
            rv2v=False,
        ),
        expected,
    )

    rv2v = {
        name: torch.tensor([[value]])
        for name, value in zip(
            ("base", "video", "image", "text", "target"),
            (1.0, 2.0, 4.0, 7.0, 11.0),
            strict=True,
        )
    }
    torch.testing.assert_close(
        compose_velocity_guidance(
            rv2v,
            omega_video=2.0,
            omega_image=3.0,
            omega_text=4.0,
            omega_target=5.0,
            rv2v=True,
        ),
        torch.tensor([[41.0]]),
    )


def test_denoised_guidance_round_trip():
    sample = torch.tensor([[[[[8.0]]]]])
    sigma = torch.tensor([0.5])
    velocities = {
        name: torch.tensor([[[[[value]]]]])
        for name, value in {
            "base": 1.0,
            "text": 2.0,
            "target": 3.0,
        }.items()
    }
    denoised = {name: sample - sigma * value for name, value in velocities.items()}
    guided = compose_velocity_guidance(
        velocities,
        omega_video=0.0,
        omega_image=0.0,
        omega_text=2.0,
        omega_target=3.0,
        rv2v=False,
    )
    actual = compose_denoised_guidance(
        denoised,
        sample,
        sigma,
        omega_video=0.0,
        omega_image=0.0,
        omega_text=2.0,
        omega_target=3.0,
        rv2v=False,
    )
    torch.testing.assert_close(actual, sample - sigma * guided)


def test_media_geometry_and_source_order():
    assert fit_media_size(1080, 1920, max_size=848) == (480, 848)
    assert fit_media_size(1920, 1080, max_size=848) == (848, 480)
    assert max(fit_media_size(1000, 1000, max_size=842)) <= 842
    assert ordered_renderer_sources(
        image_sources=["i0", "i1"], video_sources=["v0"]
    ) == ["i0", "i1", "v0"]


def test_planner_preflight_accepts_two_second_non_square_video():
    values = {
        "task": "t2v",
        "width": 640,
        "height": 368,
        "length": 33,
        "source_fps": 16.0,
        "planning_steps": 25,
        "vit_denoising_steps": 1,
    }
    validate_generation_request(**values)
    validate_generation_request(**{**values, "width": 368, "height": 640})
    with pytest.raises(ValueError, match="4n\\+1"):
        validate_generation_request(**{**values, "length": 32})
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_generation_request(**{**values, "width": 639})


def test_reference_inputs_use_numeric_order():
    references = {
        "reference_image_10": torch.full((1, 1, 1, 1), 10),
        "reference_image_2": torch.full((1, 1, 1, 1), 2),
        "reference_image_1": torch.full((1, 1, 1, 1), 1),
    }
    assert [int(item.item()) for item in split_reference_images(references)] == [
        1,
        2,
        10,
    ]


def test_small_planner_modules_and_weight_layout():
    operations = comfy.ops.disable_weight_init
    connector = MLPConnector(
        in_dim=8,
        out_dim_for_gen=6,
        out_dim_for_vit=8,
        operations=operations,
    )
    assert len(connector.state_dict()) == 12
    x = torch.randn(2, 3, 8)
    assert connector.for_gen(x).shape == (2, 3, 6)
    assert connector.for_vit(x).shape == (2, 3, 8)

    decoder = DiffLossFM(
        target_channels=8,
        z_channels=8,
        depth=2,
        width=16,
        operations=operations,
    )
    output = decoder.net(torch.randn(4, 8), torch.ones(1), torch.randn(4, 8))
    assert output.shape == (4, 8)

    branches = decoder.sample(
        torch.randn(3, 8),
        cfg=0.5,
        img_cfg=0.5,
        num_inference_steps=1,
        seed=1,
    )
    assert branches.shape == (3, 8)
    assert not hasattr(decoder, "scheduler")


def test_planner_flow_scheduler_lands_at_zero():
    scheduler = FlowMatchScheduler(shift=2.0)
    scheduler.set_timesteps(3)
    sample = torch.ones(1)
    result = scheduler.step(torch.ones(1), scheduler.timesteps[-1], sample)
    torch.testing.assert_close(result, sample - scheduler.sigmas[-1])


def test_planner_flow_scheduler_keeps_high_step_schedule_unique_and_float32():
    scheduler = FlowMatchScheduler(shift=2.0, extra_one_step=True)
    scheduler.set_timesteps(100)
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.timesteps.dtype == torch.float32
    assert torch.unique(scheduler.sigmas).numel() == 100
    assert torch.unique(scheduler.timesteps).numel() == 100


def test_qwen_patchification_and_frame_policy():
    frames = torch.rand(3, 28, 28, 3)
    patches, grid = process_qwen2vl_video(
        frames, min_pixels=28 * 28, max_pixels=28 * 28
    )
    assert grid.tolist() == [[2, 2, 2]]
    assert patches.shape == (8, 3 * 2 * 14 * 14)
    assert qwen_grid_for_media(10, 480, 832).tolist() == [[5, 12, 20]]
    indices = planner_video_frame_indices(
        81, source_fps=16, planner_fps=2, max_frames=81
    )
    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 80


def test_qwen_gqa_expands_kv_heads_in_official_order():
    captured = {}

    def attention(query, key, value, heads, **kwargs):
        captured.update(key=key, value=value, heads=heads, kwargs=kwargs)
        return query

    query = torch.zeros(1, 4, 2, 3)
    key = torch.arange(12).reshape(1, 2, 2, 3)
    value = key + 100
    result = _explicit_gqa_attention(attention)(
        query,
        key,
        value,
        4,
        enable_gqa=True,
        mask="mask",
    )

    assert result is query
    assert captured["key"].shape == query.shape
    assert captured["key"][0, :, 0, 0].tolist() == [0, 0, 6, 6]
    assert captured["value"][0, :, 0, 0].tolist() == [100, 100, 106, 106]
    assert captured["kwargs"] == {"mask": "mask"}


def test_plan_forward_uses_input_dtype_rope_and_explicit_gqa(monkeypatch):
    captured = {}

    def attention(query, key, value, heads, **kwargs):
        captured.update(key=key, value=value, heads=heads, kwargs=kwargs)
        return query.transpose(1, 2).flatten(2)

    monkeypatch.setattr(
        bernini_qwen,
        "optimized_attention_for_device",
        lambda *args, **kwargs: attention,
    )

    class Layer:
        def __call__(self, x, freqs_cis, optimized_attention, **kwargs):
            captured["rope_dtypes"] = tuple(value.dtype for value in freqs_cis)
            query = torch.zeros(1, 4, 1, 2, dtype=x.dtype)
            key = torch.zeros(1, 2, 1, 2, dtype=x.dtype)
            optimized_attention(
                query,
                key,
                key,
                4,
                enable_gqa=True,
                mask=kwargs["attention_mask"],
                skip_reshape=True,
            )
            return x, None

    class Model:
        layers = [Layer()]

        @staticmethod
        def compute_freqs_cis(position_ids, device):
            del position_ids
            return tuple(torch.ones(1, device=device) for _ in range(3))

    inputs = torch.ones(1, 1, 4, dtype=torch.bfloat16)
    result = bernini_qwen.plan_forward(
        Model(),
        inputs,
        torch.zeros(3, 1, 1, dtype=torch.long),
        torch.zeros(1, 1, 1, dtype=torch.bfloat16),
        intermediate_output=0,
    )

    assert torch.equal(result, inputs)
    assert captured["rope_dtypes"] == (torch.bfloat16,) * 3
    assert captured["key"].shape == (1, 4, 1, 2)
    assert captured["kwargs"]["skip_reshape"] is True
    assert torch.equal(
        captured["kwargs"]["mask"],
        torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen_vision_rope_preserves_attention_dtype(dtype):
    query = torch.randn(8, 2, 4, dtype=dtype)
    key = torch.randn_like(query)
    cos = torch.randn(8, 4)
    sin = torch.randn(8, 4)
    rotated_query, rotated_key = apply_rotary_pos_emb_vision(query, key, cos, sin)
    assert rotated_query.dtype == dtype
    assert rotated_key.dtype == dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()

    def rotate_half(value):
        midpoint = value.shape[-1] // 2
        return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)

    expected_query = (query.float() * cos + rotate_half(query.float()) * sin).to(dtype)
    expected_key = (key.float() * cos + rotate_half(key.float()) * sin).to(dtype)
    torch.testing.assert_close(rotated_query, expected_query)
    torch.testing.assert_close(rotated_key, expected_key)


def test_official_task_presets_and_sigma_spacing():
    assert task_preset("t2i")["steps"] == 50
    assert task_preset("t2v")["planning_steps"] == 50
    sigmas = unipc_flow_sigmas(16, 5.0)
    expected = torch.tensor(
        [0.9997998476, 0.9866341949, 0.9720060229, 0.9556572437, 0.9372654557]
    )
    torch.testing.assert_close(sigmas[:5], expected)
    assert sigmas[-1] == 0


class _ToyComfyModel:
    def __call__(self, sample, sigma, **_kwargs):
        sigma = sigma.reshape(sigma.shape + (1,) * (sample.ndim - sigma.ndim))
        velocity = 0.2 * sample + 0.1 * sigma
        return sample - sigma * velocity


def test_flow_unipc_is_deterministic_and_finite():
    noise = torch.tensor([[[[0.25, -0.5], [1.0, -1.5]]]])
    sigmas = unipc_flow_sigmas(8, 5.0)
    scaled_noise = noise * sigmas[0]
    first = sample_flow_unipc_bh2(_ToyComfyModel(), scaled_noise, sigmas)
    second = sample_flow_unipc_bh2(_ToyComfyModel(), scaled_noise, sigmas)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize(
    "sigmas",
    [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.01, 0.0]),
        torch.tensor([-0.01, 0.0]),
        torch.tensor([0.5, torch.nan]),
    ],
)
def test_flow_unipc_rejects_non_flow_sigmas(sigmas):
    with pytest.raises(ValueError, match=r"sigmas in \[0, 1\)"):
        sample_flow_unipc_bh2(_ToyComfyModel(), torch.ones(1), sigmas)


@pytest.mark.parametrize(
    "sigmas",
    [torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.5, 0.75, 0.0])],
)
def test_flow_unipc_rejects_non_decreasing_sigmas(sigmas):
    with pytest.raises(ValueError, match="strictly decreasing"):
        sample_flow_unipc_bh2(_ToyComfyModel(), torch.ones(1), sigmas)


def test_flow_unipc_preserves_short_schedule_early_return():
    noise = torch.randn(2, 3)
    assert sample_flow_unipc_bh2(_ToyComfyModel(), noise, torch.tensor([1.0])) is noise


def test_standalone_planner_embedded_payload_contract():
    payload = b'{"hidden_size":3584}'
    state_dict = {
        "config_json": torch.tensor(list(payload), dtype=torch.uint8),
        "tokenizer_json": torch.tensor([1, 2, 3], dtype=torch.uint8),
    }
    assert _embedded_json(state_dict, "config_json") == {"hidden_size": 3584}
    assert _embedded_bytes(state_dict, "tokenizer_json") == b"\x01\x02\x03"
    with pytest.raises(ValueError, match="missing embedded tensor"):
        _embedded_bytes(state_dict, "tokenizer_config")


def test_standalone_planner_embedded_payload_rejects_wrong_dtype():
    state_dict = {"config_json": torch.ones(4, dtype=torch.float32)}
    with pytest.raises(ValueError, match="one-dimensional U8"):
        _embedded_json(state_dict, "config_json")


def test_maskgit_order_is_seeded_and_scatter_compatible():
    order = maskgit_order(8, 42)
    assert torch.equal(order, maskgit_order(8, 42))
    assert not torch.equal(order, maskgit_order(8, 43))
    assert sorted(order.tolist()) == list(range(8))
    selected = torch.zeros(8, dtype=torch.bool)
    selected.scatter_(0, order[:3], True)
    assert selected.sum() == 3
    assert torch.equal(
        maskgit_order(8, 0xFFFFFFFFFFFFFFFF),
        maskgit_order(8, 0xFFFFFFFF),
    )


class _RecordingTokenizer:
    def __init__(self):
        self.texts = []
        self.ids = {}

    def convert_tokens_to_ids(self, value):
        if isinstance(value, list):
            return [self.convert_tokens_to_ids(item) for item in value]
        if value not in self.ids:
            self.ids[value] = len(self.ids) + 100
        return self.ids[value]

    def add_special_tokens(self, values):
        for token in values["additional_special_tokens"]:
            self.convert_tokens_to_ids(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        self.texts.append(text)
        return [len(self.texts)]


def test_template_mask_dtype_negative_prompt_and_visual_limit():
    token_type = torch.tensor([[0, 3]])
    segment_ids = torch.tensor([[0, 1]])
    mask = build_custom_attention_mask(token_type, segment_ids, dtype=torch.bfloat16)
    assert mask.dtype == torch.bfloat16

    tokenizer = _RecordingTokenizer()
    template = BerniniTemplate(tokenizer)
    conversation = build_conversation(
        "replace the sky",
        source_videos=0,
        source_images=0,
        output_is_image=True,
    )
    encoded = template.encode(
        conversation,
        num_tokens={"video": [], "image": [4]},
        task="t2i",
        drop_text=True,
        negative_prompt="坏",
        mask_dtype=torch.bfloat16,
    )
    assert encoded["attention_mask_4d"].dtype == torch.bfloat16
    assert any("坏" in text for text in tokenizer.texts)
    assert not any("replace the sky" in text for text in tokenizer.texts)
    with pytest.raises(ValueError, match="at most 64 visual items, got 65"):
        template._visual_pattern(1, 64, output=False)
