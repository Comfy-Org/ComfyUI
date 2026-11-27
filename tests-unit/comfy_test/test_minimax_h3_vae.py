import time

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.multigpu
import comfy.quant_ops
import comfy.sd
from comfy.ldm.minimax.vae import Attention, MiniMaxH3VideoVAE


class _OffloadedScale:
    """Stands in for a buffer that comfy's dynamic-VRAM offload left on another
    device: it only tracks its device label and can be moved with .to(), like
    the real qk_norm_scale tensor the bug report is about."""

    def __init__(self, device):
        self.device = torch.device(device)

    def to(self, device):
        return _OffloadedScale(device)


class _FakeCK:
    def rms_rope_split_half(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    def rms_rope_split_half_(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    @staticmethod
    def _check(query, key, scale):
        if scale.device != query.device:
            # mirrors comfy_kitchen's eager backend, which does no device
            # alignment and refuses to run the fused kernel on mixed devices
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got weight is on "
                f"{scale.device}, different from other tensors on {query.device}"
            )
        return query, key


def test_attention_moves_offloaded_qk_norm_scale_to_input_device(monkeypatch):
    monkeypatch.setattr(comfy.quant_ops, "ck", _FakeCK(), raising=False)

    heads, dim_head = 2, 4
    attn = Attention(heads=heads, dim_head=dim_head)
    # simulate a partially-offloaded VAE: qk_norm_scale sits on a different
    # device than the tensors flowing through the fused rope kernel
    attn._buffers["qk_norm_scale"] = _OffloadedScale("meta")

    batch_size, seq_len = 1, 3
    x = torch.arange(1, 25, dtype=torch.float32).reshape(batch_size, seq_len, heads * dim_head)
    rotary_pos_emb = torch.randn(batch_size, 1, seq_len, dim_head // 2, 2, 2)

    pre_norm = comfy.ops.manual_cast.RMSNorm(heads * dim_head, elementwise_affine=False, eps=1e-5)
    qkv_outputs = []
    hook = attn.to_qkv.register_forward_hook(lambda module, inputs, output: qkv_outputs.append(output))
    with torch.no_grad():
        attn.to_qkv.weight.copy_(torch.eye(heads * dim_head).repeat(3, 1))
        attn.to_qkv.bias.zero_()
        attn.to_out.weight.copy_(torch.eye(heads * dim_head))
        attn.to_out.bias.zero_()
        out = attn.forward(
            x, rotary_pos_emb, pre_norm=pre_norm, residual=None, residual_scale=None
        )
    hook.remove()

    assert out.device == x.device
    normalized = torch.nn.functional.rms_norm(x, (heads * dim_head,), eps=pre_norm.eps)
    expected = torch.nn.functional.linear(normalized, attn.to_qkv.weight, attn.to_qkv.bias)
    unnormalized = torch.nn.functional.linear(x, attn.to_qkv.weight, attn.to_qkv.bias)
    assert len(qkv_outputs) == 1
    torch.testing.assert_close(qkv_outputs[0], expected)
    assert not torch.allclose(qkv_outputs[0], unnormalized)



def test_still_decodes_as_first_latent_of_a_full_clip():
    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    seen = []

    def decode_pixels(z):  # every output frame holds its own index / 100
        seen.append(z.shape[2])
        frames = torch.arange(z.shape[2] * vae.vae_ratio_t, dtype=torch.float32) / 100
        return frames.view(1, 1, -1, 1, 1).expand(z.shape[0], 3, -1, z.shape[-2] * vae.vae_ratio, z.shape[-1] * vae.vae_ratio)

    vae._decode_pixels = decode_pixels
    out = vae.decode(torch.zeros(1, 24, 1, 40, 24))  # 640x384, several tiles
    assert out.shape == (1, 3, 1, 640, 384)
    assert set(seen) == {vae.tokens_chunk_size}
    torch.testing.assert_close(out, torch.full_like(out, vae.frame_pre_padding / 100))


def test_decode_temporal_chunk_map_matches_sequential(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)

    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)

    def fake_adaptive_decode(clip_z):
        # Deterministic position-dependent values to verify temporal overlap blending
        t_frames = clip_z.shape[2] * vae.vae_ratio_t
        base = clip_z[:, :3, :1, :1, :1].repeat(1, 1, t_frames, vae.vae_ratio, vae.vae_ratio)
        ramp = torch.linspace(0.0, 0.25, t_frames, dtype=clip_z.dtype).view(1, 1, t_frames, 1, 1)
        return base + ramp

    vae._adaptive_decode = fake_adaptive_decode
    z = torch.randn(1, 24, 12, 1, 1)
    out_seq = vae.decode(z)

    d0 = torch.device("cpu:0")
    d1 = torch.device("cpu:1")
    pool = comfy.multigpu.MultiGPUThreadPool([d0, d1])
    try:
        def slow_first(clip):
            time.sleep(0.02)
            return fake_adaptive_decode(clip)

        out_parallel = vae.decode(z, chunk_map=lambda clips: pool.map_ordered(clips, {d0: slow_first, d1: fake_adaptive_decode}))
        torch.testing.assert_close(out_parallel, out_seq, rtol=0, atol=0)
    finally:
        pool.shutdown()


def test_encode_temporal_chunk_map_matches_sequential(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)

    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)

    def fake_adaptive_encode(clip_x):
        return clip_x[:, :1, ::vae.vae_ratio_t, ::vae.vae_ratio, ::vae.vae_ratio].repeat(1, 48, 1, 1, 1)

    vae._adaptive_encode = fake_adaptive_encode
    x = torch.randn(1, 3, 40, 16, 16)
    out_seq = vae.encode(x, device=torch.device("cpu"))

    d0 = torch.device("cpu:0")
    d1 = torch.device("cpu:1")
    pool = comfy.multigpu.MultiGPUThreadPool([d0, d1])
    try:
        out_parallel = vae.encode(
            x,
            device=torch.device("cpu"),
            chunk_map=lambda clips: pool.map_ordered(clips, {d0: lambda c: vae._encode_temporal_chunk(c, torch.device("cpu")), d1: lambda c: vae._encode_temporal_chunk(c, torch.device("cpu"))}),
        )
        torch.testing.assert_close(out_parallel, out_seq, rtol=0, atol=0)
    finally:
        pool.shutdown()


def test_vae_wrapper_dispatches_minimax_chunked_multigpu(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)

    def _make_stub_vae(noncausal=False):
        with torch.device("meta"):
            model = MiniMaxH3VideoVAE(noncausal_decoder=noncausal)
        model.latents_mean, model.latents_std = torch.zeros(24), torch.ones(24)
        model.pixel_mean, model.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
        model._adaptive_decode = lambda clip_z, frames=slice(None): torch.zeros(
            (clip_z.shape[0], 3, clip_z.shape[2] * model.vae_ratio_t, clip_z.shape[3] * model.vae_ratio, clip_z.shape[4] * model.vae_ratio),
            dtype=torch.float32,
        )[:, :, frames]
        model._adaptive_encode = lambda clip_x: torch.zeros(
            (clip_x.shape[0], 48, max(1, clip_x.shape[2] // model.vae_ratio_t), clip_x.shape[3] // model.vae_ratio, clip_x.shape[4] // model.vae_ratio),
            dtype=torch.float32,
        )
        patcher = type("StubPatcher", (), {
            "load_device": torch.device("cpu"),
            "pre_run": lambda self: None,
            "cleanup": lambda self: None,
            "get_free_memory": lambda self, dev=None: 16 * 1024 ** 3,
        })()
        v = comfy.sd.VAE(no_init=True)
        v.first_stage_model = model
        v.patcher = patcher
        v.device = torch.device("cpu")
        v.vae_dtype = torch.float32
        v.output_device = torch.device("cpu")
        v.disable_offload = False
        v.not_video = False
        v.crop_input = False
        v.output_channels = 3
        v.pad_channel_value = None
        v.format_encoded = None
        v.latent_dim = 3
        v.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        v.process_input = lambda x: x
        v.process_output = lambda x: x
        v.memory_used_decode = lambda shape, dtype: 1024
        v.memory_used_encode = lambda shape, dtype: 1024
        v.multigpu_clones = {}
        return v

    primary = _make_stub_vae(noncausal=False)
    clone = _make_stub_vae(noncausal=False)
    clone.device = torch.device("cpu:1")
    clone.patcher.load_device = torch.device("cpu:1")
    primary.multigpu_clones = {torch.device("cpu:1"): clone}

    calls = []
    orig_session = primary._multigpu_session

    def tracked_session(mem):
        calls.append("session")
        return orig_session(mem)

    primary._multigpu_session = tracked_session

    # Multi-chunk video decode uses _multigpu_session
    out_vid = primary.decode(torch.zeros(1, 24, 12, 1, 1))
    assert calls == ["session"]
    assert out_vid.shape[0] == 1

    # Multi-clip video encode uses _multigpu_session
    out_lat = primary.encode(torch.zeros(1, 40, 16, 16, 3))
    assert calls == ["session", "session"]
    assert out_lat.shape[1] == 24

    # Single-frame still bypasses chunked multigpu
    primary.decode(torch.zeros(1, 24, 1, 1, 1))
    assert len(calls) == 2

    # Noncausal (Sol-H3-VAE) bypasses chunked multigpu
    sol = _make_stub_vae(noncausal=True)
    sol.multigpu_clones = {torch.device("cpu:1"): _make_stub_vae(noncausal=True)}
    sol_calls = []
    sol._multigpu_session = lambda mem: sol_calls.append("session") or orig_session(mem)
    sol.first_stage_model.decode = lambda z, **kw: torch.zeros(1, 3, 5, 16, 16)
    sol.first_stage_model.decode_output_shape = lambda s, **kw: (1, 3, 5, 16, 16)
    sol.decode(torch.zeros(1, 24, 2, 1, 1))
    assert sol_calls == []
