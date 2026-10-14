"""Real SDK policy dispatch with an untrained decoder boundary fixture.

Original checkpoint/model and guest/gVisor comparisons remain separate gates.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from comfy_api.latest import _sdk


METHODS = ("decode", "decode_tensor", "decode_tiled", "decode_tensor_tiled")


class BoundaryVae:
    def __init__(self, dtype=torch.float32):
        self.raw = torch.tensor([-3, -1, 0, 1, 3], dtype=dtype).reshape(1, 1, 5, 1).repeat(1, 1, 1, 3)
        self.process_output = lambda image: image.add_(1).div_(2).clamp_(0, 1)
        self.calls = []
        self.fail = False
        self.barrier = None

    def decode(self, _samples):
        self.calls.append((id(self), "normal", {}))
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        result = self.process_output(self.raw.clone())
        if self.fail:
            raise RuntimeError("decoder failed after output processing")
        return result

    def decode_tiled(self, samples, **kwargs):
        self.calls.append((id(self), "tiled", kwargs))
        return self.decode(samples)

    def spacial_compression_decode(self):
        return 8

    def temporal_compression_decode(self):
        return None


async def handles(value):
    refs = _sdk.InProcessRefResolver()
    vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
    latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": torch.zeros(1, 4, 1, 1)}))
    return refs, vae, latent


async def decode(value, method, flag="omitted"):
    refs, vae, latent = await handles(value)
    options = {} if flag == "omitted" else {"allow_extended_range": flag}
    with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
        result = await getattr(vae, method)(latent, **options)
    return await refs.resolve(result)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("flag", ["omitted", False, True])
def test_default_and_extended_values_normal_and_tiled(method, flag):
    value = BoundaryVae()
    original = value.process_output
    result = asyncio.run(decode(value, method, flag))
    expected = value.raw.clone().add_(1).div_(2)
    if flag is not True:
        expected.clamp_(0, 1)
    assert torch.equal(result, expected)
    assert result.dtype == value.raw.dtype and result.device == value.raw.device
    assert value.process_output is original
    assert all(owner != id(value) for owner, _, _ in value.calls) == (flag is True)
    if "tiled" in method:
        assert value.calls[0][2] == {
            "tile_x": 64, "tile_y": 64, "overlap": 8,
            "tile_t": None, "overlap_t": None,
        }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extended_preserves_output_dtype(dtype):
    value = BoundaryVae(dtype)
    result = asyncio.run(decode(value, "decode", True))
    assert result.dtype == dtype
    assert torch.equal(result, value.raw.clone().add_(1).div_(2))


@pytest.mark.parametrize("method", METHODS)
def test_false_preserves_custom_existing_processing(method):
    value = BoundaryVae()
    value.process_output = lambda image: image.mul_(3).add_(7)
    original = value.process_output
    result = asyncio.run(decode(value, method, False))
    assert torch.equal(result, value.raw.clone().mul_(3).add_(7))
    assert value.process_output is original


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [None, 0, 1, "true", lambda image: image])
def test_invalid_policy_refused_at_public_and_provider_boundaries(method, bad):
    async def run():
        value = BoundaryVae()
        refs, vae, latent = await handles(value)
        ops = _sdk.InProcessOps()
        with _sdk.bind_runtime(refs, None, ops):
            with pytest.raises(TypeError, match="boolean"):
                await getattr(vae, method)(latent, allow_extended_range=bad)
            with pytest.raises(TypeError, match="boolean"):
                await ops.apply("vae." + method, vae, {"latent": latent, "allow_extended_range": bad})
        assert value.calls == []
    asyncio.run(run())


@pytest.mark.parametrize("method", METHODS)
def test_failure_leaves_shared_vae_processing_untouched(method):
    value = BoundaryVae()
    original = value.process_output
    value.fail = True
    with pytest.raises(RuntimeError, match="decoder failed"):
        asyncio.run(decode(value, method, True))
    assert value.process_output is original
    value.fail = False
    result = asyncio.run(decode(value, method, False))
    assert torch.equal(result, value.raw.clone().add_(1).div_(2).clamp_(0, 1))


@pytest.mark.parametrize("method", ["decode", "decode_tiled"])
def test_concurrent_default_and_extended_calls_keep_independent_processing(method):
    value = BoundaryVae()
    original = value.process_output
    value.barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        extended = pool.submit(asyncio.run, decode(value, method, True))
        default = pool.submit(asyncio.run, decode(value, method, False))
        actual_extended, actual_default = extended.result(timeout=15), default.result(timeout=15)
    assert torch.equal(actual_extended, value.raw.clone().add_(1).div_(2))
    assert torch.equal(actual_default, value.raw.clone().add_(1).div_(2).clamp_(0, 1))
    assert value.process_output is original


@pytest.mark.parametrize("mode", ["foreign", "released", "closed", "relabelled"])
def test_owned_current_refs_are_required(mode):
    async def run():
        value = BoundaryVae()
        refs, vae, latent = await handles(value)
        if mode == "foreign":
            refs = _sdk.InProcessRefResolver()
        elif mode == "released":
            await refs.release(vae)
        elif mode == "closed":
            refs.clear()
        else:
            vae = _sdk.VaeRef._wrap(await refs.create("IMAGE", value))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises((KeyError, TypeError)):
                await vae.decode(latent, allow_extended_range=True)
        assert value.calls == []
    asyncio.run(run())


def test_unsafe_wrapper_copy_refused_before_mutation():
    class UnsafeVae(BoundaryVae):
        def __copy__(self):
            return self
    value = UnsafeVae()
    original = value.process_output
    with pytest.raises(TypeError, match="independent wrapper"):
        asyncio.run(decode(value, "decode", True))
    assert value.process_output is original
    assert value.calls == []


def test_tiled_budget_is_unchanged_for_extended_policy():
    async def run():
        value = BoundaryVae()
        refs, vae, latent = await handles(value)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match="tile_size"):
                await vae.decode_tiled(latent, tile_size=32, allow_extended_range=True)
        assert value.calls == []
    asyncio.run(run())
