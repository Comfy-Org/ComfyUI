"""Unit tests for the native Hunyuan3D 2.1 PBR paint model port.

These construct a small randomly-initialized UNet2p5DConditionModel (no weights
required) and verify architecture, state_dict/detection parity through core model
detection, the DDIM v-prediction schedule against core model sampling, the packed
multiview forward, and an end-to-end run through core's standard sampling loop.
"""

from __future__ import annotations

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_base  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.hunyuan3d.paint.unet import UNet2p5DConditionModel, PaintReferenceBank  # noqa: E402
from comfy.ldm.hunyuan3d.paint.loader import detect_paint_config, load_paint_unet, _HEAD_DIM  # noqa: E402

OPS = comfy_ops.disable_weight_init

# Small config using the real model's SD-2 head dim (64) so detection round-trips.
SMALL = dict(
    in_channels=12, ref_in_channels=4, out_channels=4,
    block_out_channels=(64, 64, 128, 128), layers_per_block=2, cross_attention_dim=64,
    num_attention_heads=(1, 1, 2, 2), transformer_layers_per_block=1, norm_num_groups=32,
    pbr_setting=("albedo", "mr"), pbr_token_channels=7, dino_embeddings_dim=32, use_dino=True,
)
CROSS = SMALL["cross_attention_dim"]
DINO_DIM = SMALL["dino_embeddings_dim"]
TOKENS = SMALL["pbr_token_channels"]


def _build(dtype=torch.float32, device="cpu", **over):
    cfg = {**SMALL, **over}
    return UNet2p5DConditionModel(dtype=dtype, device=device, operations=OPS, **cfg), cfg


def _init_weights(model, seed=0, scale=0.03):
    """Small seeded init: disable_weight_init leaves parameters uninitialized
    (torch.empty garbage), which is fine for shape tests but overflows through a
    deep net; tests asserting finite numerics need real (small) weights."""
    with torch.no_grad():
        # Index into the sorted parameter list, not hash(name): str hashing is salted per
        # process unless PYTHONHASHSEED is pinned, which would make these weights - and so
        # any numerics failure below - differ from run to run.
        for i, (_, p) in enumerate(sorted(model.named_parameters())):
            g = torch.Generator().manual_seed(seed + i)
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * scale)
    return model


def _inputs(B, n_pbr, V, H, dtype=torch.float32, dino=True, position=True):
    """Packed-layout forward inputs: x carries the channel-concatenated
    [latent, normal, position] groups with views on the non-batch axis."""
    x = torch.randn(B, 12, n_pbr * V, H, H, dtype=dtype)
    context = torch.randn(B, n_pbr, TOKENS, CROSS, dtype=dtype)
    ref = torch.randn(B, 1, 4, H, H, dtype=dtype)
    pos_map = torch.rand(B, V, 3, H, H, dtype=dtype) if position else None
    dino_h = torch.randn(B, 5, DINO_DIM, dtype=dtype) if dino else None
    return x, context, ref, pos_map, dino_h


def _reference_alphas_cumprod():
    """The paint scheduler's alphas_cumprod, re-derived with the diffusers formulas
    (scaled-linear betas 0.00085..0.012, 1000 steps, rescale_zero_terminal_snr) -
    the contract the deleted bespoke DDIM scheduler implemented."""
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    a0 = alphas_bar_sqrt[0].clone()
    aT = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= aT
    alphas_bar_sqrt *= a0 / (a0 - aT)
    return alphas_bar_sqrt ** 2


def _nodes():
    import comfy_extras.nodes_hunyuan3d_paint as N
    return N


def test_forward_output_shape_and_dtype():
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 1, 2, 3, 16
    x, context, ref, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        out = model(x, torch.tensor([500]), context=context, ref_latents=ref,
                    dino_features=dino_h, position_maps=pos_map)
    assert tuple(out.shape) == (B, 4, n_pbr * V, H, H)
    assert out.dtype == torch.float32


def test_forward_without_dino_or_position():
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 1, 2, 3, 16
    x, context, ref, _, _ = _inputs(B, n_pbr, V, H, dino=False, position=False)
    with torch.no_grad():
        out = model(x, torch.tensor([10]), context=context, ref_latents=ref,
                    dino_features=None, position_maps=None)
    assert tuple(out.shape) == (B, 4, n_pbr * V, H, H)


def test_forward_flattened_context_matches_material_layout():
    """comfy conditioning carries the per-material context flattened to
    (B, n_pbr*L, cross); the forward must reproduce the 4D layout exactly."""
    model, _ = _build()
    _init_weights(model)
    model.eval()
    B, n_pbr, V, H = 1, 2, 2, 16
    x, context, ref, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        a = model(x, torch.tensor([500]), context=context, ref_latents=ref,
                  dino_features=dino_h, position_maps=pos_map)
        b = model(x, torch.tensor([500]), context=context.reshape(B, n_pbr * TOKENS, CROSS),
                  ref_latents=ref, dino_features=dino_h, position_maps=pos_map)
    assert torch.equal(a, b)


def test_forward_cfg_batch_with_tensor_ref_scale():
    """Core CFG batches cond/uncond along B; a precomputed reference bank rides
    both with a per-batch-item ref_scale tensor (0 on the uncond)."""
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 3, 2, 2, 16
    x, context, ref, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        bank = PaintReferenceBank(model.compute_reference_bank(ref[:1]))
        out = model(x, torch.tensor([500]), context=context, ref_bank=bank,
                    dino_features=dino_h, position_maps=pos_map,
                    ref_scale=torch.tensor([0.0, 1.0, 1.0]))
    assert tuple(out.shape) == (B, 4, n_pbr * V, H, H)


def test_bfloat16_forward():
    model, _ = _build(dtype=torch.bfloat16)
    model.eval()
    B, n_pbr, V, H = 1, 2, 2, 16
    x, context, ref, pos_map, dino_h = _inputs(B, n_pbr, V, H, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(x, torch.tensor([500]), context=context, ref_latents=ref,
                    dino_features=dino_h, position_maps=pos_map)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (B, 4, n_pbr * V, H, H)


def test_detect_config_roundtrip():
    model, cfg = _build()
    detected = detect_paint_config(model.state_dict())
    assert detected is not None
    for key in ("in_channels", "ref_in_channels", "out_channels", "block_out_channels",
                "layers_per_block", "cross_attention_dim", "transformer_layers_per_block",
                "pbr_setting", "pbr_token_channels", "dino_embeddings_dim", "use_dino"):
        assert detected[key] == cfg[key], (key, detected[key], cfg[key])
    # heads are recovered assuming SD-2 head dim
    assert detected["num_attention_heads"] == tuple(c // _HEAD_DIM for c in cfg["block_out_channels"])


def test_detect_returns_none_for_non_paint_state_dict():
    assert detect_paint_config({"foo.weight": torch.zeros(1)}) is None
    assert detect_paint_config({}) is None


def test_core_model_detection_picks_paint_config():
    """The paint checkpoint routes through comfy.model_detection into the
    Hunyuan3DPaint model config: v-prediction, zsnr schedule, packed latents."""
    import comfy.supported_models
    model, cfg = _build()
    sd = model.state_dict()
    model_config = comfy.model_detection.model_config_from_unet(sd, "")
    assert isinstance(model_config, comfy.supported_models.Hunyuan3DPaint)
    assert model_config.unet_config["image_model"] == "hunyuan3d_paint"
    assert model_config.unet_config["block_out_channels"] == cfg["block_out_channels"]
    assert model_config.sampling_settings["zsnr"] is True
    assert model_config.latent_format.latent_dimensions == 3  # packed view axis


def test_loader_rejects_ldm_checkpoint_with_family_diagnosis():
    """A wrong-family checkpoint must fail with a clear "this looks like X, expected
    hunyuan3d-paintpbr-v2-1" ValueError - never a KeyError."""
    sd = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
          "model.diffusion_model.out.2.weight": torch.zeros(4, 320, 3, 3)}
    with pytest.raises(ValueError, match=r"looks like a ComfyUI/LDM diffusion checkpoint"):
        load_paint_unet(sd)


def test_loader_rejects_plain_diffusers_unet_with_family_diagnosis():
    sd = {"conv_in.weight": torch.zeros(320, 4, 3, 3),
          "down_blocks.0.resnets.0.conv1.weight": torch.zeros(320, 320, 3, 3)}
    with pytest.raises(ValueError, match="plain diffusers UNet"):
        load_paint_unet(sd)


def test_loader_rejects_single_stream_checkpoint():
    """Paint-style keys without the dual-stream reference UNet are named as such."""
    model, _ = _build()
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("unet_dual.")}
    with pytest.raises(ValueError, match="dual-stream reference UNet"):
        load_paint_unet(sd)


def test_loader_rejects_truncated_checkpoint_not_keyerror():
    """A paint-family checkpoint missing derivation keys raises an informative
    ValueError (truncated), not a KeyError from a raw dict lookup."""
    model, _ = _build()
    sd = dict(model.state_dict())
    del sd["unet.conv_out.weight"]
    with pytest.raises(ValueError, match="truncated"):
        detect_paint_config(sd)


def test_loader_rejects_checkpoint_with_missing_weights():
    """Detection can pass while weights are still missing; the load must hard-fail
    with a truncation message instead of silently leaving random weights."""
    model, _ = _build()
    sd = dict(model.state_dict())
    # not read by detect_paint_config, but required by the model
    del sd["unet.mid_block.resnets.0.conv1.weight"]
    with pytest.raises(ValueError, match="missing 1 weights"):
        load_paint_unet(sd, model_options={"dtype": torch.float32})


def test_detected_config_rebuilds_with_strict_key_parity():
    """Detection -> config -> rebuild must reproduce the exact state_dict keys/shapes,
    and loading the weights must give a numerically identical model."""
    torch.manual_seed(0)
    model, _ = _build()
    model.eval()
    sd = model.state_dict()

    detected = detect_paint_config(sd)
    twin = UNet2p5DConditionModel(
        dtype=torch.float32, device="cpu", operations=OPS,
        in_channels=detected["in_channels"], ref_in_channels=detected["ref_in_channels"],
        out_channels=detected["out_channels"], block_out_channels=detected["block_out_channels"],
        layers_per_block=detected["layers_per_block"], cross_attention_dim=detected["cross_attention_dim"],
        num_attention_heads=detected["num_attention_heads"],
        transformer_layers_per_block=detected["transformer_layers_per_block"],
        norm_num_groups=detected["norm_num_groups"], pbr_setting=detected["pbr_setting"],
        pbr_token_channels=detected["pbr_token_channels"], dino_embeddings_dim=detected["dino_embeddings_dim"],
        use_dino=detected["use_dino"])
    twin.eval()

    missing, unexpected = twin.load_state_dict(sd, strict=False)
    assert missing == [], missing
    assert unexpected == [], unexpected

    B, n_pbr, V, H = 1, 2, 2, 16
    x, context, ref, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        a = model(x, torch.tensor([500]), context=context, ref_latents=ref,
                  dino_features=dino_h, position_maps=pos_map)
        b = twin(x, torch.tensor([500]), context=context, ref_latents=ref,
                 dino_features=dino_h, position_maps=pos_map)
    # The rebuilt model must reproduce the reference exactly (equal_nan: random-init
    # deep nets can overflow, but both models must overflow identically).
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_load_paint_unet_returns_core_model_patcher():
    """The loader goes through core's diffusion-model path: a ModelPatcher wrapping
    a model_base.Hunyuan3DPaint (v-prediction + zsnr) around the UNet."""
    from comfy.model_patcher import ModelPatcher
    model, cfg = _build()
    sd = model.state_dict()
    patcher = load_paint_unet(sd, model_options={"dtype": torch.float32})
    assert isinstance(patcher, ModelPatcher)
    assert isinstance(patcher.model, comfy.model_base.Hunyuan3DPaint)
    assert patcher.model.model_type == comfy.model_base.ModelType.V_PREDICTION
    assert isinstance(patcher.model.diffusion_model, UNet2p5DConditionModel)
    assert set(patcher.model.diffusion_model.state_dict().keys()) == set(sd.keys())
    # zsnr schedule: finite terminal sigma at core's universal clamp (~4519)
    assert float(patcher.model.model_sampling.sigma_max) > 4000.0
    assert patcher.model.latent_format.scale_factor == pytest.approx(0.18215)


def test_state_dict_has_expected_special_keys():
    model, _ = _build()
    keys = set(model.state_dict().keys())
    assert "unet.conv_in.weight" in keys
    assert "unet_dual.conv_in.weight" in keys
    assert "unet.learned_text_clip_albedo" in keys
    assert "unet.learned_text_clip_mr" in keys
    assert "unet.learned_text_clip_ref" in keys
    assert "unet.image_proj_model_dino.proj.weight" in keys
    # material and reference PBR heads
    assert any(k.endswith("transformer.attn1.processor.to_q_mr.weight") for k in keys)
    assert any("attn_refview.processor.to_v_mr.weight" in k for k in keys)
    assert any("attn_multiview.to_q.weight" in k for k in keys)
    assert any("attn_dino.to_k.weight" in k for k in keys)
    # conv_in is 12 channels on the main stream, 4 on the dual/reference stream
    assert model.state_dict()["unet.conv_in.weight"].shape[1] == 12
    assert model.state_dict()["unet_dual.conv_in.weight"].shape[1] == 4


def test_trailing_timesteps_zero_terminal_snr():
    N = _nodes()
    ts = N.trailing_timesteps(15)
    assert len(ts) == 15
    assert int(ts[0]) == 999            # trailing spacing starts at the last train step
    assert all(int(a) > int(b) for a, b in zip(ts[:-1], ts[1:]))  # strictly decreasing
    assert list(ts) == [999, 932, 866, 799, 732, 666, 599, 532, 466, 399, 332, 266, 199, 132, 66]
    # the schedule the model was trained with is zero terminal SNR
    assert _reference_alphas_cumprod()[-1].item() < 1e-6


def test_ddim_schedule_matches_core_model_sampling():
    """Rewire-parity contract: core's ModelSamplingDiscrete(zsnr=True), which now
    drives the paint model, must reproduce the reference DDIM v-pred/zero-SNR
    schedule (diffusers scaled-linear 0.00085..0.012 + rescale_zero_terminal_snr).

    The only intended difference is the terminal step, where core clamps
    alpha_cumprod[-1] to 4.897e-8 (finite sigma_max ~4519) while the reference
    scheduler keeps an exact 0 (infinite sigma). A full 15-step DDIM loop under
    either convention agrees to ~2e-4 (see PR notes)."""
    import comfy.model_sampling as cms
    N = _nodes()

    ms = cms.ModelSamplingDiscrete(model_config=None, zsnr=True)
    ac = _reference_alphas_cumprod()
    sigmas = ((1 - ac[:-1]) / ac[:-1]) ** 0.5
    torch.testing.assert_close(ms.sigmas[:999].double(), sigmas, rtol=1e-6, atol=1e-4)
    # documented divergence at the terminal step
    assert float(ac[-1]) == 0.0
    assert float(ms.sigmas[-1]) > 4000.0
    # the trailing timesteps round-trip exactly through core's sigma<->timestep maps
    ts = [int(t) for t in N.trailing_timesteps(15)]
    rt = ms.timestep(ms.sigma(torch.tensor(ts, dtype=torch.float32))).tolist()
    assert rt == ts
    # the scheduler node emits exactly those sigmas (plus the terminal 0)
    model, _ = _build()
    patcher = load_paint_unet(model.state_dict(), model_options={"dtype": torch.float32})
    node_sigmas = N.Hunyuan3DPaintScheduler.execute(patcher, 15)[0]
    pm = patcher.model.model_sampling
    torch.testing.assert_close(node_sigmas[:-1], pm.sigma(torch.tensor(ts, dtype=torch.float32)))
    assert float(node_sigmas[-1]) == 0.0


def test_cam_mapping_view_scale():
    N = _nodes()
    assert N.view_scale_mapping(0) == pytest.approx(1.0)
    assert N.view_scale_mapping(90) == pytest.approx(2.0)
    assert N.view_scale_mapping(180) == pytest.approx(2.0)
    assert N.view_scale_mapping(360) == pytest.approx(1.0)


def test_view_scale_pre_cfg_matches_reference_composition():
    """The reference triple-batch CFG telescopes: uncond + s*vs*(ref - uncond) +
    s*vs*(full - ref) == uncond + s*vs*(full - uncond) for ANY middle prediction.
    The pre-CFG view-scale patch must reproduce that composition through core's
    scalar cfg_function."""
    N = _nodes()
    azims = [0.0, 90.0, 180.0, 270.0]
    n_pbr, scale = 2, 3.0
    g = torch.Generator().manual_seed(3)
    cond = torch.randn(1, 4, n_pbr * len(azims), 8, 8, generator=g, dtype=torch.float64)
    uncond = torch.randn(1, 4, n_pbr * len(azims), 8, 8, generator=g, dtype=torch.float64)
    ref_mid = torch.randn(1, 4, n_pbr * len(azims), 8, 8, generator=g, dtype=torch.float64)

    out = N._view_scale_pre_cfg(azims)({"conds_out": [cond, uncond]})
    # core cfg_function: uncond + (cond' - uncond) * scale
    result = out[1] + (out[0] - out[1]) * scale

    vs = torch.tensor([N.view_scale_mapping(a) for a in azims], dtype=torch.float64)
    vs = vs.repeat(n_pbr).reshape(1, 1, -1, 1, 1)
    reference = uncond + scale * vs * (ref_mid - uncond) + scale * vs * (cond - ref_mid)
    torch.testing.assert_close(result, reference)
    # cfg 1.0 path: uncond is skipped, conds pass through untouched
    out = N._view_scale_pre_cfg(azims)({"conds_out": [cond, None]})
    assert out[0] is cond and out[1] is None


def _sampling_setup(patcher, V, H, dino=True, seed=0):
    """Hand-build the conditioning the cond-prep node produces (random geometry)."""
    base = patcher.model
    comfy.model_management.load_models_gpu([patcher])
    device = patcher.load_device
    dm = base.diffusion_model
    n_pbr = len(dm.pbr_setting)
    g = torch.Generator().manual_seed(seed)
    normal = torch.randn(V, 4, H, H, generator=g)
    position = torch.randn(V, 4, H, H, generator=g)
    geo = torch.cat([normal, position], dim=1).movedim(0, 1).unsqueeze(0).repeat(1, 1, n_pbr, 1, 1)
    ref = torch.randn(1, 1, 4, H, H, generator=g).to(device)
    with torch.no_grad():
        bank = PaintReferenceBank(dm.compute_reference_bank(base.process_latent_in(ref)))
        context = dm.material_context(1).detach().float()
    context = context.reshape(1, -1, context.shape[-1])
    pos_maps = torch.rand(1, V, 3, H, H, generator=g)

    cond = {"concat_latent_image": geo, "ref_bank": bank, "position_maps": pos_maps, "ref_scale": 1.0}
    uncond = dict(cond)
    uncond["ref_scale"] = 0.0
    if dino:
        d = torch.randn(1, 5, DINO_DIM, generator=g)
        cond["dino_features"] = d
        uncond["dino_features"] = torch.zeros_like(d)
    positive = [[context, cond]]
    negative = [[context, uncond]]
    latent = torch.zeros(1, 4, n_pbr * V, H, H)
    return positive, negative, latent


def _core_sample(patcher, positive, negative, latent, steps, cfg, seed, azims=None):
    import comfy.sample
    import comfy.samplers
    N = _nodes()
    m = patcher.clone()
    if azims is not None:
        m.set_model_sampler_pre_cfg_function(N._view_scale_pre_cfg(azims))
    noise = comfy.sample.prepare_noise(latent, seed)
    sigmas = N.Hunyuan3DPaintScheduler.execute(m, steps)[0]
    sampler = comfy.samplers.sampler_object("euler")
    return comfy.sample.sample_custom(m, noise, cfg, sampler, sigmas, positive, negative,
                                      latent, disable_pbar=True, seed=seed)


def test_core_sampling_end_to_end_shapes():
    """The standard comfy loop (prepare_noise -> euler -> CFG with the view-scale
    patch) drives the paint model end to end; cond batching keeps the packed
    material/view groups intact."""
    model, _ = _build()
    _init_weights(model)
    patcher = load_paint_unet(model.state_dict(), model_options={"dtype": torch.float32})
    V, H = 4, 16
    positive, negative, latent = _sampling_setup(patcher, V, H, dino=True)
    out = _core_sample(patcher, positive, negative, latent, steps=2, cfg=3.0, seed=0,
                       azims=[0.0, 90.0, 180.0, 270.0])
    assert tuple(out.shape) == (1, 4, 2 * V, H, H)
    assert torch.isfinite(out).all()


def test_core_sampling_without_cfg_or_dino():
    """cfg=1.0 skips the uncond batch entirely (core's optimization); the model
    must run from the positive cond alone, without DINO features."""
    model, _ = _build()
    _init_weights(model)
    patcher = load_paint_unet(model.state_dict(), model_options={"dtype": torch.float32})
    V, H = 2, 16
    positive, negative, latent = _sampling_setup(patcher, V, H, dino=False, seed=1)
    out = _core_sample(patcher, positive, negative, latent, steps=2, cfg=1.0, seed=1)
    assert tuple(out.shape) == (1, 4, 2 * V, H, H)
    assert torch.isfinite(out).all()
