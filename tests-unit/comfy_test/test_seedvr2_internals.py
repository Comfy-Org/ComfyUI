"""SeedVR2 internals regression tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.seedvr.vae import (  # noqa: E402
    causal_norm_wrapper,
    set_norm_limit,
)
from comfy.ldm.seedvr.attention import var_attention_optimized_split  # noqa: E402


_NUM_CHANNELS = 8
_NUM_GROUPS = 4
_TENSOR_SHAPE = (1, 8, 2, 4, 4)


@pytest.fixture(autouse=True)
def _isolate_weight_patches():
    """Keep these tests independent of leaked weight patches.

    ``comfy.ops.CastWeightBiasOp`` holds ``weight_function``/``bias_function`` as *class* level
    lists, so any code that appends to one without first giving the module its own list mutates the
    shared default and every comfy layer created afterwards silently inherits the patch. That is
    what happens when these tests run after tests-unit/comfy_quant in the same process.
    """
    cls = comfy_ops.CastWeightBiasOp
    saved = (list(cls.weight_function), list(cls.bias_function))
    cls.weight_function.clear()
    cls.bias_function.clear()
    try:
        yield
    finally:
        cls.weight_function[:] = saved[0]
        cls.bias_function[:] = saved[1]


_GROUPNORM_SUBCLASSES = [
    pytest.param(comfy_ops.disable_weight_init.GroupNorm, id="disable_weight_init"),
    pytest.param(comfy_ops.manual_cast.GroupNorm, id="manual_cast"),
]


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(1e-9)
    try:
        gn = groupnorm_cls(num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS)
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({"num_groups": int(num_groups_arg)})
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS)

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE
        assert full_calls == 0, (
            f"low-limit GroupNorm gate must NOT take the full-forward path; got full_calls={full_calls}"
        )
        assert chunked_calls > 0, (
            f"low-limit GroupNorm gate must take the chunked path; got chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)


def test_seedvr2_7b_swin_attention_forward_uses_optimized_var_attention(monkeypatch):
    dim = 8
    heads = 2
    head_dim = 4
    attn = seedvr_model.NaSwinAttention(
        vid_dim=dim,
        txt_dim=dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=False,
        qk_norm=comfy_ops.disable_weight_init.RMSNorm,
        qk_norm_eps=1e-6,
        rope_type=None,
        rope_dim=head_dim,
        shared_weights=False,
        window=(2, 1, 1),
        window_method="720pwin_by_size_bysize",
        version=True,
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    generator = torch.Generator(device="cpu").manual_seed(11)
    vid = torch.randn(8, dim, generator=generator)
    txt = torch.randn(3, dim, generator=generator)
    vid_shape = torch.tensor([[2, 2, 2]], dtype=torch.long)
    txt_shape = torch.tensor([[3]], dtype=torch.long)
    calls = []

    def fake_optimized_var_attention(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(seedvr_model, "optimized_var_attention", fake_optimized_var_attention)

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape, seedvr_model.Cache(disable=True))

    assert tuple(vid_out.shape) == (8, dim)
    assert tuple(txt_out.shape) == (3, dim)
    assert len(calls) == 1
    call = calls[0]
    assert tuple(call["q"].shape) == (14, heads, head_dim)
    assert tuple(call["k"].shape) == (14, heads, head_dim)
    assert tuple(call["v"].shape) == (14, heads, head_dim)
    assert call["heads"] == heads
    assert call["skip_reshape"] is True
    assert call["skip_output_reshape"] is True
    assert call["cu_seqlens_q"] == [0, 7, 14]
    assert call["cu_seqlens_k"] == [0, 7, 14]


def _make_swin_attention(rope_type, dim=16, heads=2, head_dim=8):
    torch.manual_seed(0)
    attn = seedvr_model.NaSwinAttention(
        vid_dim=dim,
        txt_dim=dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=False,
        qk_norm=comfy_ops.disable_weight_init.RMSNorm,
        qk_norm_eps=1e-6,
        rope_type=rope_type,
        rope_dim=head_dim,
        shared_weights=False,
        window=(2, 2, 2),
        window_method="720pwin_by_size_bysize",
        version=(rope_type == "rope3d"),
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    for param in attn.parameters():
        torch.nn.init.normal_(param, std=0.5)
    return attn


@pytest.mark.parametrize("rope_type", [None, "rope3d", "mmrope3d"])
def test_seedvr2_swin_attention_batched_samples_match_one_at_a_time(rope_type):
    """A batched cond+uncond forward must give each sample its own text tokens."""
    attn = _make_swin_attention(rope_type)
    generator = torch.Generator(device="cpu").manual_seed(3)
    vids = [torch.randn(2, 6, 6, 16, generator=generator), torch.randn(3, 6, 8, 16, generator=generator)]
    txts = [torch.randn(5, 16, generator=generator), torch.randn(7, 16, generator=generator)]

    vid, vid_shape = seedvr_model.flatten(vids)
    txt, txt_shape = seedvr_model.flatten(txts)
    batched_vid, batched_txt = attn(vid, txt, vid_shape, txt_shape, seedvr_model.Cache())

    single_vid, single_txt = [], []
    for one_vid, one_txt in zip(vids, txts):
        vid_i, vid_shape_i = seedvr_model.flatten([one_vid])
        txt_i, txt_shape_i = seedvr_model.flatten([one_txt])
        out_vid, out_txt = attn(vid_i, txt_i, vid_shape_i, txt_shape_i, seedvr_model.Cache())
        single_vid.append(out_vid)
        single_txt.append(out_txt)

    torch.testing.assert_close(batched_vid, torch.cat(single_vid), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(batched_txt, torch.cat(single_txt), rtol=1e-4, atol=1e-5)


def test_var_attention_optimized_split_batches_equal_length_windows(monkeypatch):
    heads = 2
    head_dim = 3
    q = torch.arange(36, dtype=torch.float32).reshape(6, heads, head_dim)
    k = q + 100
    v = q + 200
    cu = [0, 2, 4, 6]
    calls = []

    def fake_optimized_attention(q_arg, k_arg, v_arg, heads_arg, **kwargs):
        calls.append(tuple(q_arg.shape))
        return q_arg + v_arg

    monkeypatch.setattr(attention, "optimized_attention", fake_optimized_attention)

    out = var_attention_optimized_split(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert calls == [(3, heads, 2, head_dim)], (
        f"equal-length windows must share one batched attention call; got {calls}"
    )
    torch.testing.assert_close(out, q + v, rtol=0, atol=0)


def test_var_attention_optimized_split_calls_dense_backend_per_window(monkeypatch):
    heads = 2
    head_dim = 3
    q = torch.arange(30, dtype=torch.float32).reshape(5, heads, head_dim)
    k = q + 100
    v = q + 200
    cu = [0, 2, 5]
    calls = []

    def fake_optimized_attention(q_arg, k_arg, v_arg, heads_arg, **kwargs):
        calls.append(
            {
                "q_shape": tuple(q_arg.shape),
                "k_shape": tuple(k_arg.shape),
                "v_shape": tuple(v_arg.shape),
                "heads": heads_arg,
                "kwargs": kwargs,
            }
        )
        return q_arg + v_arg

    monkeypatch.setattr(attention, "optimized_attention", fake_optimized_attention)

    out = var_attention_optimized_split(
        q,
        k,
        v,
        heads,
        cu,
        cu,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    assert tuple(out.shape) == (5, heads, head_dim)
    assert len(calls) == 2
    assert calls[0]["q_shape"] == (1, heads, 2, head_dim)
    assert calls[1]["q_shape"] == (1, heads, 3, head_dim)
    assert all(call["heads"] == heads for call in calls)
    assert all(call["kwargs"]["skip_reshape"] is True for call in calls)
    assert all(call["kwargs"]["skip_output_reshape"] is True for call in calls)
    torch.testing.assert_close(out, q + v, rtol=0, atol=0)


class _AlwaysPackedCache(vae_mod.CausalMemoryCache):
    """Drops the size gate so small tails exercise the packing."""

    def _packable(self, value):
        return torch.is_tensor(value) and value.is_cuda and value.is_contiguous(memory_format=torch.channels_last_3d)


def _cache_tail(channels, frames=2):
    torch.manual_seed(0)
    tail = torch.randn(1, channels, frames, 48, 64, device="cuda") * 2
    tail[:, ::5] *= 20.0  # the per-channel outliers the rotation exists to spread
    return tail.half().contiguous(memory_format=torch.channels_last_3d)


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_causal_memory_cache_holds_unpackable_tails_as_they_are():
    cache = vae_mod.CausalMemoryCache()
    assert not cache._packable(torch.zeros(1, 128, 2, 4, 4)), "cpu tails stay unpacked"
    assert not cache._packable(torch.zeros(4, 8)) and not cache._packable("not a tensor")
    tail = torch.randn(1, 96, 2, 4, 4)
    cache["conv"] = tail
    assert cache["conv"] is tail and "conv" in cache
    assert cache.pop("conv") is tail and "conv" not in cache
    assert cache.pop("conv", "fallback") == "fallback"
    with pytest.raises(KeyError):
        cache["missing"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tails pack on CUDA")
@pytest.mark.parametrize("channels", [128, 256, 512])
def test_causal_memory_cache_packing_round_trips(channels):
    tail = _cache_tail(channels)
    cache = _AlwaysPackedCache()
    cache["conv"] = tail
    assert "conv" in cache.packed
    restored = cache["conv"]
    assert restored.shape == tail.shape and restored.dtype == tail.dtype
    assert restored.is_contiguous(memory_format=torch.channels_last_3d)
    assert _rel_err(restored, tail) < 1e-2
    plain = torch.randn(1, 96, 2, 4, 4)
    cache["conv"] = plain   # replaces the packed entry
    assert cache["conv"] is plain and "conv" not in cache.packed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload moves tails between CUDA and host memory")
def test_causal_memory_cache_offload_round_trips_across_slices():
    """Offloaded tails come back exactly as the resident packing returns them, in the order the
    next slice reads them (which the one-ahead prefetch assumes), and free() unpins the host buffers."""
    keys = [f"conv{i}" for i in range(6)]
    tails = {k: _cache_tail(128) for k in keys}
    pinned_before = vae_mod.comfy.model_management.TOTAL_PINNED_MEMORY
    resident = _AlwaysPackedCache()
    offloaded = _AlwaysPackedCache(offload=True)
    for slice_idx in range(3):
        for k in keys:
            resident[k] = tails[k] * (slice_idx + 1)
            offloaded[k] = tails[k] * (slice_idx + 1)
        assert not any(q.is_cuda for q, *_ in offloaded.packed.values()), "offloaded tails stay off the GPU between slices"
        for k in keys:
            a, b = resident[k], offloaded[k]
            assert b.is_contiguous(memory_format=torch.channels_last_3d)
            assert torch.equal(a, b), f"{k} slice {slice_idx}: offloaded tail differs"
    assert offloaded.pop("conv0") is not None and "conv0" not in offloaded
    offloaded.free()
    assert vae_mod.comfy.model_management.TOTAL_PINNED_MEMORY == pinned_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ring entries pack and offload on CUDA")
def test_causal_memory_cache_frame_ring_keeps_the_last_frames():
    """push_frames extends a key's tail one frame at a time; get() returns the last `keep`
    frames in order, retired frames are dropped, and a plain assignment replaces the ring."""
    cache = _AlwaysPackedCache(offload=True)
    frames = [f.contiguous(memory_format=torch.channels_last_3d) for f in _cache_tail(128, frames=4).split(1, dim=2)]
    cache.push_frames("conv", torch.cat(frames[:2], dim=2), keep=2)
    assert "conv" in cache
    assert _rel_err(cache["conv"], torch.cat(frames[:2], dim=2)) < 1e-2
    cache.push_frames("conv", frames[2], keep=2)          # one new frame: one packed entry
    assert len(cache._rings["conv"]) == 2
    assert _rel_err(cache["conv"], torch.cat(frames[1:3], dim=2)) < 1e-2
    cache.push_frames("conv", frames[3], keep=2)
    assert len(cache.plain) + len(cache.packed) == 2, "retired frames must be dropped, not accumulated"
    cache["conv"] = frames[0]                              # plain assignment replaces the ring
    assert "conv" not in cache._rings and cache["conv"].shape == frames[0].shape
    cache.free()


def _make_block(is_last_layer, dim=16, heads=2, head_dim=8):
    torch.manual_seed(0)
    block = seedvr_model.NaMMSRTransformerBlock(
        vid_dim=dim, txt_dim=dim, emb_dim=dim * 6, heads=heads, head_dim=head_dim,
        expand_ratio=2, norm=comfy_ops.disable_weight_init.RMSNorm, norm_eps=1e-6,
        ada=seedvr_model.AdaSingle, qk_bias=False, qk_norm=comfy_ops.disable_weight_init.RMSNorm,
        mlp_type="normal", shared_weights=False, rope_type="mmrope3d", rope_dim=head_dim,
        is_last_layer=is_last_layer, window=(2, 2, 2), window_method="720pwin_by_size_bysize",
        version=False, device="cpu", dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    for param in block.parameters():
        torch.nn.init.normal_(param, std=0.5)
    return block


@pytest.mark.parametrize("is_last_layer", [False, True])
def test_seedvr2_norm_ada_in_matches_unfused_norm_then_modulate(is_last_layer):
    """The fused norm+modulate must agree with norm() then ada() on BOTH branches.

    The last block's attn_norm normalizes txt while its ada modulates vid alone, so a fused path
    that lets ada carry the norm hands attention an unnormalized txt to read as K/V.
    """
    block = _make_block(is_last_layer)
    generator = torch.Generator(device="cpu").manual_seed(5)
    vid = torch.randn(12, 16, generator=generator) * 30.0
    txt = torch.randn(7, 16, generator=generator) * 30.0
    emb = torch.randn(1, 16 * 6, generator=generator)

    for norm, layer in ((block.attn_norm, "attn"), (block.mlp_norm, "mlp")):
        ada_kwargs = {
            "emb": emb,
            "hid_len": seedvr_model.MMArg(torch.tensor([12]), torch.tensor([7])),
            "cache": seedvr_model.Cache(),
            "branch_tag": seedvr_model.MMArg("vid", "txt"),
        }
        fused_vid, fused_txt = block._norm_ada_in(norm, vid.clone(), txt.clone(), layer, ada_kwargs)

        ada_kwargs["cache"] = seedvr_model.Cache()
        ref_vid, ref_txt = norm(vid.clone(), txt.clone())
        ref_vid, ref_txt = block.ada(ref_vid, ref_txt, layer=layer, mode="in", **ada_kwargs)

        torch.testing.assert_close(fused_vid, ref_vid, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            fused_txt, ref_txt, rtol=1e-4, atol=1e-4,
            msg=lambda m: f"{layer} txt branch diverged (is_last_layer={is_last_layer}):\n{m}",
        )


def test_seedvr2_vid_out_ada_reuses_the_block_attn_modulation():
    """``vid_out_ada`` has layers=["out"], so its own slice of a 6*dim embedding is twice as wide
    as hid. Upstream survives that because its cache key collides with the blocks' "attn" entry and
    it silently takes theirs; the released 3B weights were exported against that aliasing."""
    dim = 16
    torch.manual_seed(0)
    block_ada = seedvr_model.AdaSingle(dim=dim, emb_dim=dim * 6, layers=["attn", "mlp"])
    out_ada = seedvr_model.AdaSingle(dim=dim, emb_dim=dim * 6, layers=["out"], modes=["in"])
    for module in (block_ada, out_ada):
        for param in module.parameters():
            torch.nn.init.normal_(param, std=0.1)

    generator = torch.Generator(device="cpu").manual_seed(1)
    emb = torch.randn(1, dim * 6, generator=generator)
    hid_len = torch.tensor([9])
    cache = seedvr_model.Cache()

    block_ada(torch.randn(9, dim, generator=generator), emb=emb, layer="attn", mode="in",
              cache=cache, branch_tag="vid", hid_len=hid_len)
    assert "emb_repeat_0_vid" in cache.cache

    hid = torch.randn(9, dim, generator=generator)
    out = out_ada(hid.clone(), emb=emb, layer="out", mode="in",
                  cache=cache, branch_tag="vid", hid_len=hid_len)

    shiftA, scaleA, _ = cache.cache["emb_repeat_0_vid"].unbind(-1)
    expected = hid * (scaleA + out_ada.out_scale) + (shiftA + out_ada.out_shift)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("padding", [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 0, 0), (1, 2, 0, 1, 3, 0)])
@pytest.mark.parametrize("concat_dim", [2, 3])
@pytest.mark.parametrize("with_cache", [False, True])
def test_seedvr2_padded_input_matches_cat_then_pad(padding, concat_dim, with_cache):
    """``_padded_input`` replaces cat-then-F.pad with one buffer; it must be value-identical, a
    no-op when there is nothing to add, and cast a cache of another dtype where cat would raise."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(1, 8, 4, 5, 6, generator=generator)
    cache = None
    if with_cache:
        shape = list(x.shape)
        shape[concat_dim] = 2
        cache = torch.randn(*shape, generator=generator).to(torch.float16)

    got = vae_mod.InflatedCausalConv3d._padded_input(x, cache, padding, concat_dim)

    reference = x if cache is None else torch.cat([cache.to(x.dtype), x], dim=concat_dim)
    reference = torch.nn.functional.pad(reference, padding, mode="constant", value=0.0)
    assert got.dtype == x.dtype and got.shape == reference.shape
    torch.testing.assert_close(got, reference, rtol=0, atol=0)
    if cache is None and not any(padding):
        assert got is x


def _reference_group_norm(norm, x, silu):
    """What causal_norm_wrapper computes: per-frame statistics via the (b*t, c, h, w) view."""
    b, c, t, h, w = x.shape
    out = torch.nn.functional.group_norm(
        x.transpose(1, 2).reshape(b * t, c, h, w), norm.num_groups, norm.weight, norm.bias, norm.eps,
    )
    if silu:
        out = torch.nn.functional.silu(out)
    return out.reshape(b, t, c, h, w).transpose(1, 2)


@pytest.mark.parametrize("silu", [False, True])
def test_seedvr2_causal_norm_wrapper_silu_matches_norm_then_silu(silu):
    """The silu flag must fold in exactly, on whichever path this machine takes."""
    torch.manual_seed(0)
    norm = comfy_ops.disable_weight_init.GroupNorm(4, 16, eps=1e-6, affine=True)
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.2)
    torch.nn.init.normal_(norm.bias, std=0.2)
    x = torch.randn(1, 16, 3, 5, 7)

    got = causal_norm_wrapper(norm, x, silu=silu)

    torch.testing.assert_close(got, _reference_group_norm(norm, x, silu), rtol=1e-5, atol=1e-6)


def _decode_estimate(frames, height, width):
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(vae_mod.VideoAutoencoderKLWrapper)
    latent_t = (frames - 1) // 4 + 1
    return wrapper.comfy_memory_used_decode((1, 16, latent_t, height // 8, width // 8))


def test_seedvr2_decode_estimate_is_flat_in_clip_length():
    """Slicing bounds the working set, so a longer clip must not multiply the estimate.

    Charging the peak per output pixel (frames included) made a 21-frame 1080p decode look like
    6.5 GiB when it really needs about 27, and ComfyUI then frees too little and falls back to
    tiling that costs roughly 1.75x per frame.
    """
    short = _decode_estimate(9, 1080, 1920)
    long = _decode_estimate(81, 1080, 1920)
    assert long > short, "the decoded frames themselves still accumulate"
    assert long < short * 1.5, (
        f"estimate grew {long / short:.1f}x for 9x the frames; it should track one frame's area"
    )


@pytest.mark.parametrize("frames, height, width, measured_gib", [
    (9, 480, 864, 3.52),
    (21, 720, 1280, 6.07),
    (9, 864, 1536, 8.38),
    (21, 1080, 1920, 12.66),
])
def test_seedvr2_decode_estimate_tracks_measured_peak(frames, height, width, measured_gib):
    """Measured on an RTX 5090 with the caches offloaded and a one-frame tail; the estimate
    should sit just above reality, never under it."""
    estimate_gib = _decode_estimate(frames, height, width) / 1024 ** 3
    assert estimate_gib >= measured_gib, (
        f"estimate {estimate_gib:.2f} GiB under-reports the measured {measured_gib:.2f} GiB peak"
    )
    assert estimate_gib < measured_gib * 1.35, (
        f"estimate {estimate_gib:.2f} GiB is far above the measured {measured_gib:.2f} GiB peak"
    )


def test_seedvr2_encode_accepts_the_chunked_io_device_kwarg():
    """``comfy_has_chunked_io`` is one flag for both directions: sd.py leaves the pixels where they
    are and calls ``encode(x, device=...)``, so encode must take it and move the data itself."""
    import inspect
    sig = inspect.signature(vae_mod.VideoAutoencoderKLWrapper.encode)
    assert "device" in sig.parameters, "encode must accept the chunked-io device kwarg"
    assert sig.parameters["device"].default is None, "device must be optional"
    # the wrapper claims the protocol, so both sides of it have to exist
    assert vae_mod.VideoAutoencoderKLWrapper.comfy_has_chunked_io is True
    assert hasattr(vae_mod.VideoAutoencoderKLWrapper, "decode_output_shape")
    assert "output_buffer" in inspect.signature(vae_mod.VideoAutoencoderKLWrapper.decode).parameters


def test_seedvr2_decode_output_shape_matches_decode():
    """sd.py preallocates from this; it must equal what decode() returns (frames from the 4n+1
    rule, spatial 8x, cropped to even) for both latent layouts."""
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(vae_mod.VideoAutoencoderKLWrapper)
    wrapper.spatial_downsample_factor = 8
    assert wrapper.decode_output_shape((1, 16, 6, 90, 160)) == (1, 3, 21, 720, 1280)
    assert wrapper.decode_output_shape((1, 16, 1, 135, 240)) == (1, 3, 1, 1080, 1920)
    assert wrapper.decode_output_shape((2, 16 * 3, 45, 81)) == (2, 3, 9, 360, 648)
    assert wrapper.decode_output_shape((1, 16, 2, 13, 13)) == (1, 3, 5, 104, 104)
    assert vae_mod.VideoAutoencoderKLWrapper.comfy_has_chunked_io is True


def _tile_side(free_gib):
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(vae_mod.VideoAutoencoderKLWrapper)
    wrapper.spatial_downsample_factor = 8
    return wrapper.preferred_decode_tile(free_gib * 1024 ** 3)


def test_seedvr2_tile_side_tracks_free_memory_within_bounds():
    """Grows with free memory, stays inside [min, max] on whole latent blocks, and the tile it
    picks is predicted to fit the memory it was sized against."""
    assert _tile_side(0.1) == vae_mod.SEEDVR2_MIN_TILE_LATENT
    assert _tile_side(10_000) == vae_mod.SEEDVR2_MAX_TILE_LATENT
    assert _tile_side(6) < _tile_side(30)
    assert _tile_side(30) * 8 >= 512
    for free in (2, 4, 8, 16, 24, 32, 80):
        side = _tile_side(free)
        assert vae_mod.SEEDVR2_MIN_TILE_LATENT <= side <= vae_mod.SEEDVR2_MAX_TILE_LATENT and side % 8 == 0
        if side != vae_mod.SEEDVR2_MIN_TILE_LATENT:
            predicted = (side * 8) ** 2 * vae_mod.SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL + vae_mod.SEEDVR2_DECODE_FIXED_BYTES
            assert predicted <= free * 1024 ** 3
