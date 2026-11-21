import pytest
import torch

from comfy.tensorfold.moss_case import conditioning, digest, dimensions, make_case, schedule, unpack_latents


def encoded(rows=2):
    return [[torch.arange(rows * 4096, dtype=torch.float32).reshape(1, rows, 4096), {"pooled_output": None}]]


@pytest.mark.parametrize("steps,sigmas,timesteps", [
    (1, [1065353216, 0], [1148846080]),
    (2, [1065353216, 1019517863, 0], [1148846080, 1103101953]),
    (4, [1065353216, 1063843266, 1060605264, 1019517863, 0],
     [1148846080, 1147371519, 1144209408, 1103101953]),
    (8, [1065353216, 1064816345, 1064111702, 1063188955, 1061846778, 1059833512, 1055723094, 1019517863, 0],
     [1148846080, 1148321792, 1147633664, 1146732544, 1145421824, 1143455744, 1139245056, 1103101953]),
])
def test_publisher_v2_schedule_bits(steps, sigmas, timesteps):
    # Publisher 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3, canonical scalar float32 linspace.
    got_times, got_sigmas = schedule(steps)
    assert got_times.view(torch.int32).tolist() == timesteps
    assert got_sigmas.view(torch.int32).tolist() == sigmas


@pytest.mark.parametrize("steps,effective,expected", [
    (50, 50, "b874381e0cb6f4fdfe2f5e8994b6f7c6249d28c860e82719132ec5d48e7182b5"),
    (1000, 615, "ed5d5b6f1a53d5bc557a11b17b2e7a02077686c835e25088430bb0c3a156b831"),
])
def test_publisher_default_and_deduplicated_schedules(steps, effective, expected):
    times, sigmas = schedule(steps)
    assert times.numel() == effective
    assert digest(sigmas) == expected
    assert torch.all(sigmas[:-1] > sigmas[1:])


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 1001])
def test_invalid_step_counts(value):
    with pytest.raises(ValueError, match="steps"):
        schedule(value)


def test_seed_is_local_and_case_has_only_generation_inputs():
    state = torch.random.get_rng_state().clone()
    first = make_case(encoded(), encoded(1), 32, 16, 4, 42)
    second = make_case(encoded(), encoded(1), 32, 16, 4, 42)
    other = make_case(encoded(), encoded(1), 32, 16, 4, 43)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first["latents"], second["latents"])
    assert not torch.equal(first["latents"], other["latents"])
    assert set(first) == {"prompt", "negative", "latents", "timesteps", "sigmas"}
    assert first["latents"].shape == (1, 2, 128)
    assert first["latents"].dtype == torch.float32
    assert first["prompt"].shape == (2, 4096)
    assert first["prompt"].dtype == torch.bfloat16
    assert first["negative"].shape == (1, 4096)
    generator = torch.Generator("cpu").manual_seed(42)
    native_order = torch.randn((1, 128, 1, 2), generator=generator)
    assert first["latents"][0, 1, 7] == native_order[0, 7, 0, 1]


def test_attention_mask_compacts_valid_rows_without_changing_input():
    value = encoded(3)
    original = value[0][0].clone()
    value[0][1]["attention_mask"] = torch.tensor([[1, 0, 1]])
    result = conditioning(value, "positive")
    assert torch.equal(result[0], original[0, 0].to(torch.bfloat16))
    assert torch.equal(result[1], original[0, 2].to(torch.bfloat16))
    assert torch.equal(value[0][0], original)


@pytest.mark.parametrize("metadata", [
    {"area": (1, 2, 3, 4)}, {"start_percent": 0.2}, {"control": object()},
    {"reference_latents": [torch.zeros(1)]}, {"context_attention_bias": torch.ones(2)},
])
def test_conditioning_semantics_are_not_silently_discarded(metadata):
    value = encoded()
    value[0][1].update(metadata)
    with pytest.raises(ValueError, match="conditioning options"):
        conditioning(value, "positive")


@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(1, 3), torch.tensor([[1.0, 0.5]]), torch.zeros(1, 2)])
def test_unrepresentable_masks_are_refused(mask):
    value = encoded()
    value[0][1]["attention_mask"] = mask
    with pytest.raises(ValueError):
        conditioning(value, "negative")


@pytest.mark.parametrize("shape", [(2, 2, 4096), (1, 2, 768), (2, 4096), (1, 0, 4096), (1, 4097, 4096)])
def test_wrong_encoder_batch_or_length_is_refused(shape):
    with pytest.raises(ValueError):
        conditioning([[torch.zeros(shape), {}]], "positive")


def test_scheduled_and_nonfinite_conditioning_are_refused():
    with pytest.raises(ValueError, match="unscheduled"):
        conditioning(encoded() + encoded(), "positive")
    value = encoded()
    value[0][0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        conditioning(value, "negative")


@pytest.mark.parametrize("size", [(0, 16), (17, 16), (16.0, 16), (2**32, 2**32)])
def test_dimensions_fail_before_noise_allocation(size):
    with pytest.raises(ValueError):
        dimensions(*size)


def test_dimensions_match_native_signed_kernel_address_limit():
    assert dimensions(16, 126975 * 16) == (126975, 1)
    with pytest.raises(ValueError, match="address limits"):
        dimensions(16, 126976 * 16)


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.0])
def test_seed_contract(seed):
    with pytest.raises(ValueError, match="seed"):
        make_case(encoded(), encoded(), 16, 16, 1, seed)


def test_output_axes_match_standard_flux2_latents():
    raw = torch.arange(2 * 3 * 128, dtype=torch.float32)
    result = unpack_latents(raw, 48, 32)["samples"]
    assert result.shape == (1, 128, 2, 3)
    assert result.is_contiguous()
    assert result[0, 7, 1, 2] == (1 * 3 + 2) * 128 + 7
    assert result[0, 127, 0, 0] == 127


def test_bad_output_size_and_finiteness():
    with pytest.raises(ValueError, match="invalid MOSS latent"):
        unpack_latents(torch.zeros(127), 16, 16)
    with pytest.raises(ValueError, match="non-finite"):
        unpack_latents(torch.full((128,), float("inf")), 16, 16)
