"""Default-provider contract tests; synthetic tensors are not model proof.

Source-only proposal: this module has not been imported or executed by its author.
Run only in a separately selected exact Core/SDK environment.
"""
import asyncio
import gc
import hashlib
from pathlib import Path
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from comfy_api.latest import _sdk as sdk


def run(body):
    assert hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest() == (
        "19945795d5a6188e8d122e96ea82d7a8887c752bced9c776cdf79aba3afc700d")
    async def scoped():
        refs = sdk.InProcessRefResolver()
        try:
            with sdk.bind_runtime(refs, SimpleNamespace(), sdk.InProcessOps()):
                await body(refs)
        finally:
            refs.clear()
            assert not refs._table
    asyncio.run(scoped())


def bits(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def numeric(case):
    if case == "empty":
        return torch.empty((0, 2, 1, 2, 2), dtype=torch.float32)
    if case == "strided":
        return torch.arange(24, dtype=torch.float32).reshape(4, 6).t()[::2]
    if case == "bf16-bits":
        return torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    if case == "special":
        return torch.tensor([0x7fc01234, 0x7f800000, -8388608, -2147483648],
                            dtype=torch.int32).view(torch.float32)
    if case == "numpy":
        return np.arange(24, dtype=np.float64).reshape(4, 6).T
    dtype = {"bool": torch.bool, "int64": torch.int64,
             "fp32": torch.float32, "fp64": torch.float64}[case]
    return torch.tensor([0, 1, 0, 1], dtype=dtype)


@pytest.mark.parametrize("case", ["bool", "int64", "fp32", "fp64", "bf16-bits",
                                  "special", "numpy", "strided", "empty"])
def test_public_numeric_preserves_values_dtype_layout_and_native_identity(case):
    async def body(refs):
        original = numeric(case)
        expected = torch.from_numpy(original) if isinstance(original, np.ndarray) else original
        value = await sdk.TensorRef.from_value(original)
        assert type(value) is sdk.TensorRef and value.kind == "TENSOR"
        actual = await refs.resolve(value)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert torch.equal(bits(actual), bits(expected))
        if isinstance(original, torch.Tensor):
            assert actual is original
        elif original.size:
            assert actual.data_ptr() == expected.data_ptr()
    run(body)


@pytest.mark.parametrize("case", ["complex", "sparse", "meta", "numpy-object",
                                  "numpy-complex", "rank", "axis", "elements", "bytes"])
def test_numeric_rejects_without_materializing_large_storage(case):
    async def body(refs):
        small = torch.ones(1)
        values = {
            "complex": lambda: torch.ones(2, dtype=torch.complex64),
            "sparse": lambda: torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones(1), (2,)),
            "meta": lambda: torch.empty(2, device="meta"),
            "numpy-object": lambda: np.array([object()], dtype=object),
            "numpy-complex": lambda: np.ones(2, dtype=np.complex64),
            "rank": lambda: small.reshape((1,) * 9),
            "axis": lambda: small.expand(16385),
            "elements": lambda: small.expand(4096, 4097),
            "bytes": lambda: torch.ones(1, dtype=torch.float64).expand(4096, 2049),
        }
        before = len(refs._table)
        with pytest.raises((TypeError, ValueError)):
            await sdk.TensorRef.from_value(values[case]())
        assert len(refs._table) == before
    run(body)


def test_metadata_boundaries_without_allocating_boundary_buffers():
    assert hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest() == (
        "19945795d5a6188e8d122e96ea82d7a8887c752bced9c776cdf79aba3afc700d")
    # Deliberately tests only the metadata helper, not transport admission.
    for shape, size in [([16384], 65536), ([4096, 4096], 64 * 1024**2),
                        ([0, 16384], 0), ([1] * 8, 8), ([], 8)]:
        sdk._numeric_tensor_shape(shape, size)
    for shape, size in [([16385], 0), ([1] * 9, 1), ([True], 1),
                        ([4096, 4097], 0), ([1], 64 * 1024**2 + 1), ([1], -1)]:
        with pytest.raises(ValueError):
            sdk._numeric_tensor_shape(shape, size)


class Poison:
    def __iter__(self):
        raise AssertionError("ignored metadata iterated")

    def __repr__(self):
        raise AssertionError("ignored metadata represented")

    def __deepcopy__(self, memo):
        raise AssertionError("ignored metadata copied")


def test_scheduled_projection_and_latent_ignore_metadata_and_keep_views():
    async def body(refs):
        base = torch.arange(48, dtype=torch.float64).reshape(4, 12)
        first, second = base[:, ::2], base[:, 1::2]
        metadata = Poison()
        scheduled = [[first, {"pooled_output": metadata, "hooks": metadata,
                              "clip_start_percent": 0.0}],
                     [second, {"pooled_output": metadata, "clip_end_percent": 1.0}]]
        cond = sdk.CondRef._wrap(await refs.create("CONDITIONING", scheduled))
        latent = sdk.LatentRef._wrap(await refs.create("LATENT", {
            "samples": first, "noise_mask": metadata, "ignored": metadata}))
        assert await refs.resolve(await cond.embedding()) is first
        assert await refs.resolve(await cond.embedding(1)) is second
        assert await refs.resolve(await latent.samples()) is first
        assert await refs.resolve(cond) is scheduled
        assert scheduled[0][1]["hooks"] is metadata
    run(body)


@pytest.mark.parametrize("kind,value,error,message", [
    ("LATENT", {}, KeyError, "'samples'"),
    ("CONDITIONING", [], IndexError, "list index out of range"),
    ("CONDITIONING", [[]], IndexError, "list index out of range"),
])
def test_missing_native_fields_preserve_errors(kind, value, error, message):
    async def body(refs):
        cls = sdk.LatentRef if kind == "LATENT" else sdk.CondRef
        ref = cls._wrap(await refs.create(kind, value))
        with pytest.raises(error) as caught:
            await (ref.samples() if kind == "LATENT" else ref.embedding())
        assert str(caught.value) == message
    run(body)


@pytest.mark.parametrize("index", [True, -1, 256, 0.5, "0"])
def test_projection_index_domain(index):
    async def body(refs):
        cond = sdk.CondRef._wrap(await refs.create("CONDITIONING", [[torch.ones(1), {}]]))
        with pytest.raises(ValueError):
            await cond.embedding(index)
    run(body)


@pytest.mark.parametrize("kind", ["LATENT", "CONDITIONING"])
@pytest.mark.parametrize("attack", ["foreign", "released", "relabel"])
def test_reference_owner_release_and_authoritative_kind(kind, attack):
    async def body(refs):
        other = sdk.InProcessRefResolver()
        try:
            value = {"samples": torch.ones(1)} if kind == "LATENT" else [[torch.ones(1), {}]]
            cls = sdk.LatentRef if kind == "LATENT" else sdk.CondRef
            table = other if attack == "foreign" else refs
            actual_kind = "IMAGE" if attack == "relabel" else kind
            ref = cls(kind=kind, id=(await table.create(actual_kind, value)).id)
            if attack == "released":
                await table.release(ref)
            with pytest.raises(TypeError if attack == "relabel" else KeyError):
                await (ref.samples() if kind == "LATENT" else ref.embedding())
        finally:
            other.clear()
    run(body)


def test_projection_lifetime_is_owned_until_last_ref_is_released():
    async def body(refs):
        tensor = torch.ones(2)
        observed = weakref.ref(tensor)
        latent = sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": tensor}))
        projected = await latent.samples()
        del tensor
        await latent.release()
        gc.collect()
        assert observed() is not None
        await projected.release()
        gc.collect()
        assert observed() is None
    run(body)


def test_logical_helper_bytes_are_not_backing_storage_or_host_quota():
    async def body(refs):
        backing = torch.ones(1024 * 1024)
        view = backing[:1]
        ref = await sdk.TensorRef.from_value(view)
        actual = await refs.resolve(ref)
        assert actual is view
        assert actual.numel() * actual.element_size() == 4
        assert actual.untyped_storage().nbytes() == 4 * 1024 * 1024
    run(body)


def test_scheduled_row_count_admission_edge():
    async def body(refs):
        tensor = torch.ones(1)
        good = sdk.CondRef._wrap(await refs.create("CONDITIONING", [[tensor, {}]] * 256))
        assert await refs.resolve(await good.embedding(255)) is tensor
        bad = sdk.CondRef._wrap(await refs.create("CONDITIONING", [[tensor, {}]] * 257))
        with pytest.raises(TypeError):
            await bad.embedding()
    run(body)


@pytest.mark.parametrize("case", ["chars", "utf8", "template", "two_images",
                                  "tuple_images", "wrong_ref", "foreign_image", "image_shape"])
def test_multimodal_admission_precedes_genuine_tokenization(case):
    class NoCalls:
        def tokenize(self, *args, **kwargs):
            raise AssertionError("rejected request reached tokenizer")

        def encode_from_tokens_scheduled(self, *args, **kwargs):
            raise AssertionError("rejected request reached encoder")

    async def body(refs):
        other = sdk.InProcessRefResolver()
        try:
            clip = sdk.ClipRef._wrap(await refs.create("CLIP", NoCalls()))
            kwargs = {"images": []}
            text = "short"
            if case == "chars":
                text = "a" * 32769
            elif case == "utf8":
                text = "\U0001f600" * 16385
            elif case == "template":
                kwargs["llama_template"] = "t" * 4097
            elif case == "two_images":
                kwargs["images"] = [None, None]
            elif case == "tuple_images":
                kwargs["images"] = ()
            elif case == "wrong_ref":
                kwargs["images"] = [sdk.TensorRef._wrap(await refs.create("TENSOR", torch.ones(1)))]
            elif case == "foreign_image":
                kwargs["images"] = [sdk.ImageRef._wrap(await other.create("IMAGE", torch.ones(1, 2, 2, 3)))]
            else:
                kwargs["images"] = [sdk.ImageRef._wrap(await refs.create("IMAGE", torch.ones(2, 2, 3)))]
            with pytest.raises((KeyError, TypeError, ValueError)):
                await clip.encode(text, **kwargs)
        finally:
            other.clear()
    run(body)


def test_old_text_request_and_old_error_unchanged():
    # A forwarding observer, explicitly not a replacement for model qualification.
    class ClipContract:
        def __init__(self):
            self.calls = []
            self.tokens = object()
            self.result = [[torch.ones(1), {}]]

        def tokenize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            if text == "error":
                raise LookupError("original contract error")
            return self.tokens

        def encode_from_tokens_scheduled(self, tokens):
            assert tokens is self.tokens
            return self.result

    class RequestObserver:
        def __init__(self):
            self.real = sdk.InProcessOps()
            self.requests = []

        async def apply(self, op, subject, params):
            self.requests.append((op, params.copy()))
            return await self.real.apply(op, subject, params)

    async def body(refs):
        core = ClipContract()
        observer = RequestObserver()
        clip = sdk.ClipRef._wrap(await refs.create("CLIP", core))
        with sdk.bind_runtime(refs, SimpleNamespace(), observer):
            cond = await clip.encode("old text")
            assert await refs.resolve(cond) is core.result
            with pytest.raises(LookupError, match="^original contract error$"):
                await clip.encode("error")
        assert observer.requests == [("clip.encode", {"text": "old text"}),
                                     ("clip.encode", {"text": "error"})]
        assert core.calls == [("old text", {}), ("error", {})]
    run(body)
