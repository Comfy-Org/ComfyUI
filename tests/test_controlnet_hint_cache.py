from types import SimpleNamespace

import torch

import comfy.utils
from comfy.controlnet import ControlNet, T2IAdapter


class FakeControlModel:
    dtype = torch.float32

    def __call__(self, **kw):
        self.last_hint = kw["hint"]
        return torch.zeros((1, 4))


def make_cn():
    cn = ControlNet()
    cn.control_model = FakeControlModel()
    cn.model_sampling_current = SimpleNamespace(
        timestep=lambda t: t,
        calculate_input=lambda t, x: x,
    )
    # focus these tests on hint preparation, not output merging
    cn.control_merge = lambda control, control_prev, output_dtype=None: control
    cn.cond_hint_original = torch.zeros((1, 3, 16, 16))
    return cn


def call(cn, h, w, batch=1, batched_number=1, dtype=torch.float32):
    x = torch.zeros((batch, 4, h, w), dtype=dtype)
    cond = {"c_crossattn": torch.zeros((1, 1, 1))}
    return cn.get_control(x, torch.tensor([999.0]), cond, batched_number, {})


def test_alternating_area_sizes_prepare_hint_once_per_size(monkeypatch):
    prep_calls = []
    real_upscale = comfy.utils.common_upscale

    def spy_upscale(samples, width, height, *a, **k):
        prep_calls.append((width, height))
        return real_upscale(samples, width, height, *a, **k)

    monkeypatch.setattr(comfy.utils, "common_upscale", spy_upscale)
    cn = make_cn()

    call(cn, 8, 8)
    call(cn, 4, 4)
    call(cn, 8, 8)

    # Area composition alternates two sizes every step; each size should be
    # prepared once instead of re-upscaling (and re-VAE-encoding) per step.
    assert len(prep_calls) == 2, f"hint prepared {len(prep_calls)} times"


def test_cached_hints_are_bounded():
    cn = make_cn()
    call(cn, 8, 8)
    call(cn, 4, 4)
    call(cn, 6, 6)

    assert len(cn.cond_hints) <= 2


def test_cleanup_releases_cached_hints():
    cn = make_cn()
    call(cn, 8, 8)

    cn.cleanup()

    assert len(cn.cond_hints) == 0


def test_batch_broadcast_does_not_pollute_cache(monkeypatch):
    prep_calls = []
    real_upscale = comfy.utils.common_upscale

    def spy_upscale(samples, width, height, *a, **k):
        prep_calls.append((width, height))
        return real_upscale(samples, width, height, *a, **k)

    monkeypatch.setattr(comfy.utils, "common_upscale", spy_upscale)
    cn = make_cn()
    call(cn, 8, 8, batch=1)
    call(cn, 8, 8, batch=2)

    assert len(prep_calls) == 1
    # the cached copy keeps its original pre-broadcast batch size so later
    # calls with any batch count can reuse it
    assert cn.cond_hints[(8, 8)].shape[0] == cn.cond_hint_original.shape[0]


class FakeT2IModel:
    unshuffle_amount = 8

    def __init__(self):
        self.calls = []

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, hint):
        self.calls.append(tuple(hint.shape))
        return {"output": [hint]}


def make_t2i():
    adapter = object.__new__(T2IAdapter)
    adapter.cond_hint_original = torch.zeros((1, 3, 64, 64))
    adapter.compression_ratio = 8
    adapter.upscale_algorithm = "nearest-exact"
    adapter.channels_in = 3
    adapter.t2i_model = FakeT2IModel()
    adapter.device = torch.device("cpu")
    adapter.previous_controlnet = None
    adapter.timestep_range = None
    adapter.cond_hints = {}
    adapter.control_inputs = {}
    adapter.control_merge = lambda control, control_prev, output_dtype=None: control
    return adapter


def test_t2i_alternating_sizes_recompute_adapter():
    adapter = make_t2i()

    out_1 = call(adapter, 8, 8)
    out_2 = call(adapter, 4, 4)
    out_3 = call(adapter, 8, 8)

    # the adapter runs once per size; the second 8x8 step reuses the cached
    # result instead of rerunning t2i_model
    assert adapter.t2i_model.calls == [((1, 3, 64, 64)), ((1, 3, 32, 32))]
    assert len(adapter.control_inputs) == 2
    assert out_1["output"][0].shape == (1, 3, 64, 64)
    assert out_2["output"][0].shape == (1, 3, 32, 32)
    assert out_3["output"][0].shape == (1, 3, 64, 64)


def test_t2i_same_size_different_batch_recomputes():
    adapter = make_t2i()
    adapter.cond_hint_original = torch.zeros((2, 3, 16, 16))

    call(adapter, 8, 8, batch=2)
    call(adapter, 8, 8, batch=4, batched_number=2)

    # same spatial size, but cond_hint is broadcast differently per call, so
    # the cached control_input from the first shape must not be reused
    assert adapter.t2i_model.calls == [((2, 3, 64, 64)), ((4, 3, 64, 64))]
    assert len(adapter.control_inputs) == 2

    # repeating the first configuration is served from the cache again
    call(adapter, 8, 8, batch=2)
    assert adapter.t2i_model.calls == [((2, 3, 64, 64)), ((4, 3, 64, 64))]


def test_t2i_same_size_different_dtype_recomputes():
    adapter = make_t2i()

    call(adapter, 8, 8, dtype=torch.float32)
    call(adapter, 8, 8, dtype=torch.float16)

    # the model is cast to x_noisy.dtype on a miss; a cached fp32 result must
    # not be served when the next step runs in fp16
    assert len(adapter.t2i_model.calls) == 2
    assert {k[4] for k in adapter.control_inputs} == {torch.float32, torch.float16}


def test_set_cond_hint_invalidates_cache(monkeypatch):
    prep_calls = []
    real_upscale = comfy.utils.common_upscale

    def spy_upscale(samples, width, height, *a, **k):
        prep_calls.append((width, height))
        return real_upscale(samples, width, height, *a, **k)

    monkeypatch.setattr(comfy.utils, "common_upscale", spy_upscale)
    cn = make_cn()

    call(cn, 8, 8)
    assert len(prep_calls) == 1

    cn.set_cond_hint(torch.zeros((1, 3, 32, 32)))
    call(cn, 8, 8)

    # a new hint image must not be served from the per-size cache
    assert len(prep_calls) == 2
