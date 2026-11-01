import pytest

from comfy_execution.caching import cache_ram_from_prompt


def settings(active, inactive, unit="percent"):
    return {"class_type": "CacheRAMSettings", "inputs": {"active": active, "inactive": inactive, "unit": unit}}


SAMPLER = {"class_type": "KSampler", "inputs": {"seed": 1}}


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ({"1": SAMPLER}, None),
        ({"1": SAMPLER, "2": settings(10, 25)}, (12.8, 32.0)),
        ({"2": settings(10, 25, "GB")}, (10.0, 25.0)),
        ({"2": settings(10, 100)}, (12.8, 128.0)),
        ({"2": settings(20, 5, "GB")}, (20.0, 20.0)),
        ({"2": settings(10, 500, "GB")}, (10.0, 128.0)),
        ({"2": settings(10, 25), "3": settings(5, 50)}, (12.8, 64.0)),
        ({"2": settings("x", 25)}, None),
        ({"2": settings(["4", 0], 25)}, None),
    ],
)
def test_cache_ram_from_prompt(prompt, expected):
    result = cache_ram_from_prompt(prompt, 128.0)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)
