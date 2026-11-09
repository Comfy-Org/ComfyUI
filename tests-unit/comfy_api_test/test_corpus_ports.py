"""Corpus packs on the rest of the API, each checked against its original V1 code from the pack
corpus (overlay-r2/pack-db), on a real (tiny) Wan model:

- Detail-Daemon's LyingSigmaSampler: a custom sampler that wraps another SAMPLER (sampler.custom);
- smZNodes' dpmpp_2m_alt: a named sampler a pack registers (ComfyExtension.get_samplers);
- ppm's ays / ays+ / beta_1_1: schedulers a pack registers (ComfyExtension.get_schedulers);
  the last two also under pack isolation, registered from the pack's manifest at startup and run
  in the pack's own process.
"""
import asyncio
import json
import os
import pathlib
import shutil
import sys
import textwrap

import pytest
import torch

from comfy_api._impl import registration

import engine_loopback as loopback
from corpus_ports import corpus_file, original

PROOF = pathlib.Path(__file__).resolve().parents[3] / "poc-packs" / "proof"
PORTS = pathlib.Path(__file__).resolve().parent / "corpus_ports"


@pytest.fixture(scope="module")
def tiny():
    if not (PROOF / "tiny_models.py").is_file():
        pytest.skip("the POC proof kit (tiny engine models) isn't next to this ComfyUI")
    sys.path.insert(0, str(PROOF))
    import tiny_models
    return tiny_models


def _trange(n, disable=None):
    """tqdm's trange without the bar, for originals that print progress."""
    return range(n)


def _corpus_or_skip(pack, relative):
    try:
        return corpus_file(pack, relative)
    except FileNotFoundError:
        pytest.skip(f"{pack} isn't in the pack corpus here")


@pytest.fixture(params=sorted(loopback.MODES))
def run(request):
    mode = loopback.MODES[request.param]

    def go(node, **inputs):
        return asyncio.run(mode(node, "corpus", inputs))[0]
    return go


@pytest.fixture
def clean_lists():
    """Registering samplers and schedulers changes ComfyUI's lists; put them back after."""
    import comfy.k_diffusion.sampling as kds
    import comfy.samplers as samplers
    from comfy_api._impl import scheduler_providers as sp
    saved = (dict(samplers.SCHEDULER_HANDLERS), list(samplers.SCHEDULER_NAMES), list(samplers.KSAMPLER_NAMES),
             list(samplers.SAMPLER_NAMES), set(vars(kds)))
    owners, handlers = dict(sp._OWNERS), dict(sp._HANDLERS)
    yield
    sp._OWNERS.clear(), sp._OWNERS.update(owners), sp._HANDLERS.clear(), sp._HANDLERS.update(handlers)
    samplers.SCHEDULER_HANDLERS.clear()
    samplers.SCHEDULER_HANDLERS.update(saved[0])
    samplers.SCHEDULER_NAMES[:] = saved[1]
    samplers.KSAMPLER_NAMES[:] = saved[2]
    samplers.SAMPLER_NAMES[:] = saved[3]
    for name in set(vars(kds)) - saved[4]:
        delattr(kds, name)


def _sample(model, sampler, sigmas=None, steps=4):
    import comfy.sample
    import comfy.samplers

    cond = [[torch.ones(1, 4, 4096) * 0.01, {}]]
    latent = torch.zeros(1, 16, 1, 8, 8)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(3))
    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps)
    return comfy.sample.sample_custom(model, noise, 1.0, sampler, sigmas, cond, cond, latent, disable_pbar=True, seed=3)


# --------------------------------------------------------------------------- #
# Detail-Daemon: a custom sampler wrapping another SAMPLER
# --------------------------------------------------------------------------- #
def test_detail_daemon_lying_sigma_sampler_matches_the_original(run, tiny):
    import comfy.samplers
    from corpus_ports.detail_daemon import LyingSigmaSamplerNode

    source = _corpus_or_skip("comfyui-detail-daemon", "detail_daemon_node.py")
    v1 = original(source, ["lying_sigma_sampler", "LyingSigmaSamplerNode"], {"KSAMPLER": comfy.samplers.KSAMPLER})
    model = tiny.load_model()
    euler = comfy.samplers.ksampler("euler")
    (expected_sampler,) = v1["LyingSigmaSamplerNode"].go(euler, -0.3, start_percent=0.0, end_percent=1.0)
    ported = run(LyingSigmaSamplerNode, sampler=euler, dishonesty_factor=-0.3, start_percent=0.0, end_percent=1.0)
    expected, got = _sample(model, expected_sampler), _sample(model, ported.result[0])
    assert torch.equal(got, expected)
    assert not torch.equal(got, _sample(model, euler))  # it changed the sampling


# --------------------------------------------------------------------------- #
# smZNodes / ppm: what a pack registers, in ComfyUI's process
# --------------------------------------------------------------------------- #
def test_smz_dpmpp_2m_alt_registers_and_samples_as_the_original(tiny, clean_lists):
    import comfy.samplers
    from corpus_ports.smz_samplers import SmzSamplers

    source = _corpus_or_skip("comfyui_smznodes", "smZNodes.py")
    v1 = original(source, ["sample_dpmpp_2m_alt"], {"torch": torch, "trange": _trange})
    declared = asyncio.run(registration.collect(SmzSamplers()))
    registration.install(*registration.from_extension(declared), "smz")
    assert "dpmpp_2m_alt" in comfy.samplers.KSampler.SAMPLERS
    model = tiny.load_model()
    expected = _sample(model, comfy.samplers.KSAMPLER(v1["sample_dpmpp_2m_alt"]))
    assert torch.equal(_sample(model, comfy.samplers.ksampler("dpmpp_2m_alt")), expected)


def _ppm_original():
    source = _corpus_or_skip("comfyui-ppm", "src/schedulers.py")
    import comfy.samplers
    from comfy_extras.nodes_align_your_steps import loglinear_interp
    from functools import partial
    namespace = {"torch": torch, "partial": partial, "simple_scheduler": comfy.samplers.simple_scheduler,
                 "beta_scheduler": comfy.samplers.beta_scheduler, "loglinear_interp": loglinear_interp,
                 "SchedulerHandler": comfy.samplers.SchedulerHandler, "GITS_NOISE_LEVELS": {}}
    return original(source, ["_ays_scheduler", "_gits_scheduler", "CUSTOM_HANDLERS"], namespace)["CUSTOM_HANDLERS"]


@pytest.mark.parametrize("name", ["ays", "ays+", "beta_1_1"])
def test_ppm_schedulers_register_and_compute_the_originals_sigmas(tiny, clean_lists, name):
    import comfy.samplers
    from corpus_ports.ppm_schedulers import PpmSchedulers

    v1 = _ppm_original()
    registration.install(*registration.from_extension(asyncio.run(registration.collect(PpmSchedulers()))), "ppm")
    assert name in comfy.samplers.KSampler.SCHEDULERS
    sampling = tiny.load_model().get_model_object("model_sampling")
    for steps in (4, 10, 30):
        expected = v1[name].handler(sampling, steps)
        assert torch.equal(comfy.samplers.calculate_sigmas(sampling, name, steps), expected), (name, steps)


# --------------------------------------------------------------------------- #
# The same registrations under pack isolation: from the manifest, run in the pack's process
# --------------------------------------------------------------------------- #
PACK_INIT = '''
from comfy_api.index import ComfyExtension
from .smz_samplers import sample_dpmpp_2m_alt
from .ppm_schedulers import SCHEDULERS


class Extension(ComfyExtension):
    async def get_node_list(self):
        return []

    async def get_samplers(self):
        return {"dpmpp_2m_alt": sample_dpmpp_2m_alt}

    async def get_schedulers(self):
        return dict(SCHEDULERS)


async def comfy_entrypoint():
    return Extension()
'''


def test_registrations_from_a_manifest_run_in_the_packs_process(tiny, clean_lists, tmp_path, monkeypatch):
    import comfy.samplers
    from comfy_api._impl import _engine, _runtime, manifest
    from comfy_api._impl import isolation
    from comfy_api._impl.isolation import packs, server

    _corpus_or_skip("comfyui-ppm", "src/schedulers.py")
    pack = tmp_path / "custom_nodes" / "corpus_registrations"
    pack.mkdir(parents=True)
    for name in ("smz_samplers.py", "ppm_schedulers.py"):
        shutil.copy(PORTS / name, pack / name)
    (pack / "__init__.py").write_text(textwrap.dedent(PACK_INIT))
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    data = server.PackServer.manifest(str(pack))
    assert data["samplers"] == ["dpmpp_2m_alt"] and data["schedulers"] == ["ays", "ays+", "beta_1_1"]
    (pack / manifest.FILE).write_text(json.dumps(data))

    monkeypatch.setenv(isolation.ENV, "true")
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(tmp_path / "index"))
    os.makedirs(tmp_path / "index")
    saved = _runtime.providers.execution_backend
    assert _runtime.load_execution_provider() is True
    backend = _runtime.providers.execution_backend
    try:
        backend.setup_packs([str(tmp_path / "custom_nodes")])
        assert not backend.packs.started  # listed at startup; nothing imported or started
        assert "dpmpp_2m_alt" in comfy.samplers.KSampler.SAMPLERS and "ays" in comfy.samplers.KSampler.SCHEDULERS
        v1 = _ppm_original()
        model = tiny.load_model()
        sampling = model.get_model_object("model_sampling")
        for name in ("ays", "beta_1_1"):
            assert torch.equal(comfy.samplers.calculate_sigmas(sampling, name, 10), v1[name].handler(sampling, 10))
        assert sorted(backend.packs.started) == ["corpus_registrations"]  # the scheduler ran there
        source = _corpus_or_skip("comfyui_smznodes", "smZNodes.py")
        expected = _sample(model, comfy.samplers.KSAMPLER(original(source, ["sample_dpmpp_2m_alt"],
                                                                   {"torch": torch, "trange": _trange})["sample_dpmpp_2m_alt"]))
        assert torch.equal(_sample(model, comfy.samplers.ksampler("dpmpp_2m_alt")), expected)
    finally:
        backend.stop()
        _runtime.providers.execution_backend = saved


# --------------------------------------------------------------------------- #
# vslinx: a model wrapper (MultiDiffusion tiling), alone and chained
# --------------------------------------------------------------------------- #
def _vslinx_original_wrapper(latent_h, latent_w, rows, columns, overlap, old_wrapper):
    """The original's wrapper (multidiffusion_tiled_hires.py, its tile setup and ``wrapper``), with the
    original's helpers taken from the corpus and its own chaining of ``old_wrapper``."""
    source = _corpus_or_skip("comfyui-vslinx-nodes", "nodes/anima_lllite_tiled_sampler.py")
    helpers = original(source, ["_md_spans", "_md_weight_1d"], {"torch": torch})
    _md_spans, _md_weight_1d = helpers["_md_spans"], helpers["_md_weight_1d"]
    tile_hl, tile_wl = max(1, latent_h // rows), max(1, latent_w // columns)
    ov_hl = 0 if rows == 1 else min(int(tile_hl * overlap), tile_hl // 2)
    ov_wl = 0 if columns == 1 else min(int(tile_wl * overlap), tile_wl // 2)
    ys, xs = _md_spans(latent_h, rows, ov_hl), _md_spans(latent_w, columns, ov_wl)

    def build_tiles(device, dtype):
        tiles = []
        for i, (y0, y1) in enumerate(ys):
            wy = _md_weight_1d(y1 - y0, (ys[i - 1][1] - y0) if i > 0 else 0, (y1 - ys[i + 1][0]) if i < rows - 1 else 0, device, dtype)
            for j, (x0, x1) in enumerate(xs):
                wx = _md_weight_1d(x1 - x0, (xs[j - 1][1] - x0) if j > 0 else 0, (x1 - xs[j + 1][0]) if j < columns - 1 else 0, device, dtype)
                tiles.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1, "weight": wy[:, None] * wx[None, :]})
        return tiles

    def call_model(apply_model, xt, t, c):
        if old_wrapper is not None:
            return old_wrapper(apply_model, {"input": xt, "timestep": t, "c": c})
        return apply_model(xt, t, **c)

    def wrapper(apply_model, args):
        x_in, t, c = args["input"], args["timestep"], args["c"]
        tiles = build_tiles(x_in.device, x_in.dtype)
        acc = torch.zeros_like(x_in)
        lead = (1,) * (x_in.ndim - 2)
        wsum = torch.zeros(lead + (x_in.shape[-2], x_in.shape[-1]), device=x_in.device, dtype=x_in.dtype)
        for td in tiles:
            y0, y1, x0, x1 = td["y0"], td["y1"], td["x0"], td["x1"]
            eps = call_model(apply_model, x_in[..., y0:y1, x0:x1], t, c)
            wb = td["weight"].view(lead + td["weight"].shape)
            acc[..., y0:y1, x0:x1] += eps * wb
            wsum[..., y0:y1, x0:x1] += wb
        return acc / wsum.clamp(min=1e-6)
    return wrapper


def _sample16(model):
    import comfy.sample
    import comfy.samplers
    cond = [[torch.ones(1, 4, 4096) * 0.01, {}]]
    latent = torch.zeros(1, 16, 1, 16, 16)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(5))
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 3)
    return comfy.sample.sample_custom(model, noise, 1.0, comfy.samplers.ksampler("euler"), sigmas, cond, cond,
                                      latent, disable_pbar=True, seed=5)


def _shift(apply_model, args):  # an earlier node's wrapper
    return apply_model(args["input"], args["timestep"], **args["c"]) + 0.1


@pytest.mark.parametrize("chained", [False, True])
def test_vslinx_multidiffusion_wrapper_samples_as_the_original(run, tiny, chained):
    from corpus_ports.vslinx_multidiffusion import MultiDiffusionModel

    model = tiny.load_model()
    if chained:
        model = model.clone()
        model.set_model_unet_function_wrapper(_shift)
    reference = model.clone()
    reference.set_model_unet_function_wrapper(_vslinx_original_wrapper(16, 16, 2, 2, 0.25, _shift if chained else None))
    ported = run(MultiDiffusionModel, model=model, latent={"samples": torch.zeros(1, 16, 1, 16, 16)},
                 rows=2, columns=2, overlap=0.25).result[0]
    assert torch.equal(_sample16(ported), _sample16(reference))



def test_ppms_own_conversion_registers_its_schedulers_from_its_manifest(tiny, clean_lists, tmp_path, monkeypatch):
    """The corpus conversion itself (ComfyUI-ppm's v2/secure-nodes.json scheduler_providers, the secure
    overlay's format), under pack isolation: registered at startup without importing the pack, each
    call in a short-lived process that loads only the provider's module."""
    import comfy.samplers
    from comfy_api._impl import _runtime
    from comfy_api._impl import isolation
    from comfy_api._impl.isolation import packs, server
    from corpus_ports import CORPUS

    import glob
    found = glob.glob(os.path.join(CORPUS, "comfyui-ppm", "*", "ComfyUI-ppm-HEAD"))
    if not found:
        pytest.skip("ComfyUI-ppm isn't in the pack corpus here")
    folder = tmp_path / "custom_nodes"
    folder.mkdir()
    (folder / "ComfyUI-ppm").symlink_to(found[0])
    monkeypatch.setenv(isolation.ENV, "true")
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    monkeypatch.setenv("COMFY_PACK_ENVS", "off")  # its requirements are ComfyUI's own here
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(tmp_path / "index"))
    os.makedirs(tmp_path / "index")
    saved = _runtime.providers.execution_backend
    assert _runtime.load_execution_provider() is True
    backend = _runtime.providers.execution_backend
    try:
        backend.setup_packs([str(folder)])
        assert not backend.packs.started
        assert {"ays", "ays+", "ays_30", "ays_30+", "beta_1_1"} <= set(comfy.samplers.KSampler.SCHEDULERS)
        v1 = _ppm_original()
        sampling = tiny.load_model().get_model_object("model_sampling")
        for name in ("ays", "ays_30+", "beta_1_1"):
            for steps in (4, 10):
                expected = v1[name].handler(sampling, steps)
                got = comfy.samplers.calculate_sigmas(sampling, name, steps)
                assert torch.allclose(got, expected, rtol=0, atol=1e-6), (name, steps)
        # The conversion's ays+ and ays_30 differ from the original pack: the original maps ays+ to the
        # 30-step table and ays_30 to the standard one; the conversion maps them by name. A corpus
        # conversion bug, recorded here so it's visible: the registration reproduces what's declared.
        for name in ("ays+", "ays_30"):
            assert not torch.allclose(comfy.samplers.calculate_sigmas(sampling, name, 4), v1[name].handler(sampling, 4),
                                      rtol=0, atol=1e-6), f"{name} now matches the original: the conversion was fixed"
        assert not backend.packs.started  # each call ran in its own short-lived process
    finally:
        backend.stop()
        _runtime.providers.execution_backend = saved
