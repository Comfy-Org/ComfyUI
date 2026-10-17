import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy import samplers  # noqa: E402


@pytest.mark.parametrize("name", ["gligen", "control"])
def test_empty_regional_destinations_are_left_unchanged(name):
    source_payloads = [object(), object()]
    conds = [{name: payload} for payload in source_payloads]
    identities = [object(), object()]
    uncond = [
        {
            "area": (8, 8, 0, 0),
            "strength": 0.75,
            "marker": "region-a",
            "identity": identities[0],
        },
        {
            "area": (4, 4, 4, 4),
            "strength": 0.25,
            "marker": "region-b",
            "identity": identities[1],
        },
    ]
    original = [entry.copy() for entry in uncond]
    original_entry_ids = [id(entry) for entry in uncond]

    samplers.apply_empty_x_to_equal_area(
        conds, uncond, name, lambda payloads, index: payloads[index]
    )

    assert uncond == original
    assert [id(entry) for entry in uncond] == original_entry_ids
    assert all("area" in entry and name not in entry for entry in uncond)
    assert [entry["identity"] for entry in uncond] == identities


def test_global_fill_wraps_round_robin_without_changing_destination_count():
    conds = [{"gligen": payload} for payload in ("A", "B", "C")]
    uncond = [
        {"marker": "destination-0"},
        {"marker": "destination-1"},
    ]

    samplers.apply_empty_x_to_equal_area(
        conds, uncond, "gligen", lambda payloads, index: payloads[index]
    )

    assert [entry["gligen"] for entry in uncond] == ["C", "B"]
    assert [entry["marker"] for entry in uncond] == [
        "destination-0",
        "destination-1",
    ]
    assert len(uncond) == 2


@pytest.mark.parametrize("name", ["gligen", "control"])
def test_existing_global_destination_is_not_overwritten(name):
    existing_payload = object()
    conds = [{name: object()}]
    uncond = [
        {name: existing_payload, "marker": "existing"},
        {"marker": "empty"},
    ]
    original = [entry.copy() for entry in uncond]

    samplers.apply_empty_x_to_equal_area(
        conds, uncond, name, lambda payloads, index: payloads[index]
    )

    assert uncond == original
    assert uncond[0][name] is existing_payload
    assert uncond[1] == {"marker": "empty"}
