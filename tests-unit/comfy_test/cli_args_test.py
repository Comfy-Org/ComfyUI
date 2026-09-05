import subprocess
import sys

import pytest


PARSE_ARGS = "from comfy.options import enable_args_parsing; enable_args_parsing(); import comfy.cli_args"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--cache-score", "-1"],
        ["--cache-score", "1", "-1"],
        ["--cache-score", "nan"],
        ["--cache-score", "inf"],
        ["--cache-score=-inf"],
    ],
)
def test_cache_score_rejects_invalid_thresholds(arguments):
    result = subprocess.run(
        [sys.executable, "-c", PARSE_ARGS, *arguments], capture_output=True, text=True
    )

    assert result.returncode == 2
    assert "--cache-score values must be finite and non-negative" in result.stderr


def test_cache_score_rejects_more_than_two_thresholds():
    result = subprocess.run(
        [sys.executable, "-c", PARSE_ARGS, "--cache-score", "1", "2", "3"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert (
        "--cache-score accepts at most two values: active GB and inactive GB"
        in result.stderr
    )


@pytest.mark.parametrize("values", [[], ["0"], ["1", "2"]])
def test_cache_score_accepts_valid_thresholds(values):
    result = subprocess.run(
        [sys.executable, "-c", PARSE_ARGS, "--cache-score", *values],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
