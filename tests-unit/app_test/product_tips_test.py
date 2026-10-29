import argparse

import pytest

from app import product_tips


def _args(**overrides):
    flags = dict(disable_product_tips=False, dont_print_server=False, windows_standalone_build=False, quick_test_for_ci=False)
    flags.update(overrides)
    return argparse.Namespace(**flags)


def test_startup_tip_is_single_ascii_line():
    assert product_tips.COMFY_API_STARTUP_TIP.isascii()
    assert "\n" not in product_tips.COMFY_API_STARTUP_TIP


def test_tips_enabled_by_default():
    assert product_tips.tips_enabled(_args(), {})
    assert product_tips.tips_enabled(_args(), {"CI": "0", "COMFYUI_DISABLE_PRODUCT_TIPS": "0"})


@pytest.mark.parametrize("flag", ["disable_product_tips", "dont_print_server", "windows_standalone_build", "quick_test_for_ci"])
def test_tips_disabled_by_flag(flag):
    assert not product_tips.tips_enabled(_args(**{flag: True}), {})


@pytest.mark.parametrize("environ", [
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "1"},
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "True"},
    {"COMFYUI_DISABLE_PRODUCT_TIPS": "YES"},
    {"CI": "1"},
    {"CI": "true"},
])
def test_tips_disabled_by_env(environ):
    assert not product_tips.tips_enabled(_args(), environ)


def test_job_hint_once_and_only_for_headless_success(monkeypatch):
    monkeypatch.setattr(product_tips, "_shown_job_hints", set())
    get = product_tips.get_relevant_job_hint
    assert get(_args(disable_product_tips=True), {}, True, {}) is None
    assert get(_args(), {"CI": "1"}, True, {}) is None
    assert get(_args(), {}, False, {}) is None
    assert get(_args(), {}, True, {"client_id": "abc"}) is None
    assert get(_args(), {}, True, {}) == product_tips.COMFY_API_JOB_HINT
    assert get(_args(), {}, True, {}) is None
    assert product_tips.COMFY_API_JOB_HINT.isascii()
