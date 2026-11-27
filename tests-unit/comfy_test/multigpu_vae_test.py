import time
import uuid

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.multigpu
import comfy.sd
import comfy.utils
from comfy_extras import nodes_model_advanced, nodes_multigpu


class FakeVAEPatcher:
    def __init__(self, model, load_device, clone_base_uuid=None, has_init=True):
        self.model = model
        self.load_device = load_device
        self.offload_device = torch.device("cpu")
        self.clone_base_uuid = clone_base_uuid or uuid.uuid4()
        self.is_multigpu_base_clone = False
        self.cached_patcher_init = (lambda: None, ()) if has_init else None
        self.object_patches = {}
        self.pre_run_calls = 0
        self.cleanup_calls = 0

    def is_dynamic(self):
        return False

    def clone(self, disable_dynamic=False):
        c = FakeVAEPatcher(
            self.model,
            self.load_device,
            clone_base_uuid=self.clone_base_uuid,
            has_init=self.cached_patcher_init is not None,
        )
        c.offload_device = self.offload_device
        c.is_multigpu_base_clone = self.is_multigpu_base_clone
        c.object_patches = dict(self.object_patches)
        return c

    def deepclone_multigpu(self, new_load_device=None):
        if self.cached_patcher_init is None:
            raise RuntimeError("cached_patcher_init is not initialized")
        cloned_model = FakeVAEModel(scale=self.model.scale, temporal_scale=self.model.temporal_scale)
        return FakeVAEPatcher(
            cloned_model,
            new_load_device or self.load_device,
            clone_base_uuid=self.clone_base_uuid,
            has_init=True,
        )

    def set_model_compute_dtype(self, dtype):
        self.object_patches["manual_cast_dtype"] = dtype

    def get_free_memory(self, dev=None):
        return 16 * 1024 ** 3

    def pre_run(self):
        self.pre_run_calls += 1

    def cleanup(self):
        self.cleanup_calls += 1


class FakeVAEModel(torch.nn.Module):
    def __init__(self, scale=2, temporal_scale=1):
        super().__init__()
        self.scale = scale
        self.temporal_scale = temporal_scale
        self.current_patcher = None
        self.seen_devices = []

    def decode(self, z):
        self.seen_devices.append(z.device)
        out = z[:, :1].mean(dim=1, keepdim=True).repeat(1, 3, *([1] * (z.ndim - 2)))
        if z.ndim == 5:
            return out.repeat_interleave(self.temporal_scale, dim=2).repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4).float().clone()
        return out.repeat_interleave(self.scale, dim=2).repeat_interleave(self.scale, dim=3).float().clone()

    def encode(self, x):
        self.seen_devices.append(x.device)
        if x.ndim == 5:
            return x[:, :1, ::self.temporal_scale, ::self.scale, ::self.scale].repeat(1, 4, 1, 1, 1).float().clone()
        return x[:, :1, ::self.scale, ::self.scale].repeat(1, 4, 1, 1).float().clone()


def make_fake_vae(load_device, has_init=True, temporal=False):
    vae = comfy.sd.VAE(no_init=True)
    model = FakeVAEModel(scale=2, temporal_scale=2 if temporal else 1)
    vae.patcher = FakeVAEPatcher(model, load_device, has_init=has_init)
    vae.first_stage_model = model
    vae.first_stage_model.current_patcher = vae.patcher
    vae.device = load_device
    vae.vae_dtype = torch.float32
    vae.output_device = torch.device("cpu")
    vae.working_dtypes = [torch.float16, torch.float32]
    vae.disable_offload = False
    vae.not_video = False
    vae.size = 1024
    vae.latent_channels = 4
    vae.output_channels = 3
    vae.latent_dim = 3 if temporal else 2
    vae.downscale_ratio = (lambda a: a // 2, 2, 2) if temporal else 2
    vae.upscale_ratio = (lambda a: a * 2, 2, 2) if temporal else 2
    vae.downscale_index_formula = (2, 2, 2) if temporal else None
    vae.upscale_index_formula = (2, 2, 2) if temporal else None
    vae.extra_1d_channel = None
    vae.crop_input = False
    vae.handles_tiling = False
    vae.format_encoded = None
    vae.audio_sample_rate = 44100
    vae.pad_channel_value = None
    vae.process_input = lambda image: image * 2.0 - 1.0
    vae.process_output = lambda image: image
    vae.memory_used_encode = lambda shape, dtype: 1024
    vae.memory_used_decode = lambda shape, dtype: 1024
    vae.multigpu_clones = {}
    return vae


def test_map_ordered_preserves_order_and_handles_asymmetric_workers(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)

    d0 = torch.device("cpu:0")
    d1 = torch.device("cpu:1")
    pool = comfy.multigpu.MultiGPUThreadPool([d0, d1])
    try:
        items = list(range(8))

        def slow_first(x):
            if x == 0:
                time.sleep(0.05)
            return x * 10

        def fast(x):
            return x * 10

        results = list(pool.map_ordered(items, {d0: slow_first, d1: fast}, max_in_flight=4))
        assert results == [x * 10 for x in items]

        def fail_on_three(x):
            if x == 3:
                raise ValueError("boom")
            return x

        with pytest.raises(ValueError, match="boom"):
            list(pool.map_ordered(items, {d0: fail_on_three, d1: fail_on_three}))

        # Pool remains usable after a worker error.
        assert list(pool.map_ordered([1, 2], {d0: fast, d1: fast})) == [10, 20]
    finally:
        pool.shutdown()


def test_tiled_scale_multidim_multigpu_matches_single_device_1d_2d_3d(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)

    d0 = torch.device("cpu:0")
    d1 = torch.device("cpu:1")

    # 1D non-integer scale with edge clipping
    s_1d = torch.linspace(0.1, 1.1, 11).reshape(1, 1, 11)
    fn_1d = lambda a: a.repeat(1, 1, 1)[:, :, :round(a.shape[-1] * 1.1)]
    expected_1d = comfy.utils.tiled_scale_multidim(s_1d, fn_1d, tile=(5,), overlap=2, upscale_amount=1.1, out_channels=1)
    actual_1d = comfy.multigpu.tiled_scale_multidim_multigpu(s_1d, {d0: fn_1d, d1: fn_1d}, tile=(5,), overlap=2, upscale_amount=1.1, out_channels=1)
    torch.testing.assert_close(actual_1d, expected_1d, rtol=0, atol=0)

    # 2D
    s_2d = torch.randn(1, 4, 18, 20)
    fn_2d = lambda a: a[:, :3].repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    expected_2d = comfy.utils.tiled_scale_multidim(s_2d, fn_2d, tile=(8, 8), overlap=2, upscale_amount=2, out_channels=3)
    actual_2d = comfy.multigpu.tiled_scale_multidim_multigpu(s_2d, {d0: fn_2d, d1: fn_2d}, tile=(8, 8), overlap=2, upscale_amount=2, out_channels=3)
    torch.testing.assert_close(actual_2d, expected_2d, rtol=0, atol=0)

    # 3D with callable temporal upscale formula
    s_3d = torch.randn(1, 4, 7, 14, 14)
    up_t = lambda a: a * 2
    fn_3d = lambda a: a[:, :3].repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
    expected_3d = comfy.utils.tiled_scale_multidim(s_3d, fn_3d, tile=(4, 6, 6), overlap=(1, 2, 2), upscale_amount=(up_t, 2, 2), index_formulas=(2, 2, 2), out_channels=3)
    actual_3d = comfy.multigpu.tiled_scale_multidim_multigpu(s_3d, {d0: fn_3d, d1: fn_3d}, tile=(4, 6, 6), overlap=(1, 2, 2), upscale_amount=(up_t, 2, 2), index_formulas=(2, 2, 2), out_channels=3)
    torch.testing.assert_close(actual_3d, expected_3d, rtol=0, atol=0)


def test_multigpu_vae_node_and_device_composition(monkeypatch):
    d0 = torch.device("cuda", 0)
    d1 = torch.device("cuda", 1)
    monkeypatch.setattr(comfy.model_management, "get_all_torch_devices", lambda exclude_current=False: [d0, d1])
    monkeypatch.setattr(comfy.model_management, "loaded_models", lambda: [])
    monkeypatch.setattr(comfy.model_management, "supports_dtype", lambda dev, dt: True)

    vae = make_fake_vae(d0, has_init=True)
    out_vae = nodes_multigpu.MultiGPUVAENode.execute(vae, max_gpus=2).result[0]
    assert out_vae.device == d0
    assert list(out_vae.multigpu_clones.keys()) == [d1]
    assert out_vae.multigpu_clones[d1].patcher.is_multigpu_base_clone is True
    assert not vae.multigpu_clones

    # Lowering max_gpus=1 prunes clones
    pruned = nodes_multigpu.MultiGPUVAENode.execute(out_vae, max_gpus=1).result[0]
    assert pruned.multigpu_clones == {}

    # reuse_loaded=True reuses already-loaded multigpu clone
    existing_clone_patcher = out_vae.multigpu_clones[d1].patcher
    monkeypatch.setattr(comfy.model_management, "loaded_models", lambda: [existing_clone_patcher])
    reused = comfy.multigpu.create_vae_multigpu_deepclones(vae, max_gpus=2, reuse_loaded=True)
    assert reused.multigpu_clones[d1].patcher.model is existing_clone_patcher.model

    # VAEComputeDtype propagates to clones
    dt_vae = nodes_model_advanced.VAEComputeDtype.execute(out_vae, "fp16").result[0]
    assert dt_vae.vae_dtype == torch.float16
    assert dt_vae.multigpu_clones[d1].vae_dtype == torch.float16
    assert dt_vae.multigpu_clones[d1].patcher.object_patches["manual_cast_dtype"] == torch.float16

    # SelectVAEDeviceNode prunes colliding clone when primary moves to d1
    monkeypatch.setattr(nodes_multigpu.comfy.model_management, "resolve_gpu_device_option", lambda _: d1)
    moved = nodes_multigpu.SelectVAEDeviceNode.execute(out_vae, "gpu:1").result[0]
    assert moved.device == d1
    assert d1 not in moved.multigpu_clones

    # Missing cached_patcher_init falls back cleanly
    no_init_vae = make_fake_vae(d0, has_init=False)
    fallback = nodes_multigpu.MultiGPUVAENode.execute(no_init_vae, max_gpus=2).result[0]
    assert fallback.multigpu_clones == {}


def test_vae_decode_and_encode_multigpu_paths(monkeypatch):
    d0 = torch.device("cpu:0")
    d1 = torch.device("cpu:1")
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda dev: False)
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda dev: None)
    monkeypatch.setattr(comfy.model_management, "synchronize", lambda: None)
    monkeypatch.setattr(comfy.model_management, "get_all_torch_devices", lambda exclude_current=False: [d0, d1])
    monkeypatch.setattr(comfy.model_management, "loaded_models", lambda: [])
    monkeypatch.setattr(comfy.model_management, "supports_dtype", lambda dev, dt: True)

    loaded_batches = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, **kwargs: loaded_batches.append([m.load_device for m in models]))

    # 2D tiled decode + B>1 untiled decode/encode
    vae = make_fake_vae(d0, has_init=True, temporal=False)
    mg_vae = nodes_multigpu.MultiGPUVAENode.execute(vae, max_gpus=2).result[0]

    latents_2d = torch.randn(1, 4, 16, 16)
    expected_tiled = vae.decode_tiled(latents_2d, tile_x=8, tile_y=8, overlap=2)
    actual_tiled = mg_vae.decode_tiled(latents_2d, tile_x=8, tile_y=8, overlap=2)
    torch.testing.assert_close(actual_tiled, expected_tiled, rtol=0, atol=0)
    assert loaded_batches[-1] == [d0, d1]
    assert mg_vae.patcher.pre_run_calls == 1 and mg_vae.patcher.cleanup_calls == 1
    assert mg_vae.multigpu_clones[d1].patcher.pre_run_calls == 1 and mg_vae.multigpu_clones[d1].patcher.cleanup_calls == 1

    # B > 1 untiled decode & encode
    latents_b = torch.randn(4, 4, 8, 8)
    expected_dec = vae.decode(latents_b)
    actual_dec = mg_vae.decode(latents_b)
    torch.testing.assert_close(actual_dec, expected_dec, rtol=0, atol=0)

    imgs_b = torch.randn(4, 16, 16, 3)
    expected_enc = vae.encode(imgs_b)
    actual_enc = mg_vae.encode(imgs_b)
    torch.testing.assert_close(actual_enc, expected_enc, rtol=0, atol=0)

    # 3D tiled decode
    vae_3d = make_fake_vae(d0, has_init=True, temporal=True)
    mg_vae_3d = nodes_multigpu.MultiGPUVAENode.execute(vae_3d, max_gpus=2).result[0]
    latents_3d = torch.randn(1, 4, 6, 12, 12)
    expected_3d = vae_3d.decode_tiled(latents_3d, tile_t=4, tile_x=6, tile_y=6, overlap=2)
    actual_3d = mg_vae_3d.decode_tiled(latents_3d, tile_t=4, tile_x=6, tile_y=6, overlap=2)
    torch.testing.assert_close(actual_3d, expected_3d, rtol=0, atol=0)
