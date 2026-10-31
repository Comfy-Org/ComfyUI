"""Authored records remain bounded data and preserve live asset authority."""
import asyncio
import math
from types import SimpleNamespace

import pytest

from comfy_api.latest import _sdk, sdk
from comfy_api.latest._asset_metadata import (
    MAX_METADATA_BYTES,
    MAX_METADATA_DEPTH,
    decode_catalogue_metadata,
    encode_catalogue_metadata,
)


class Records:
    """Inert authorized-source fixture, not a real repository producer."""

    def __init__(self, values):
        self.values = values
        self.calls = []

    async def catalogue_metadata(self, ref):
        self.calls.append(ref)
        return self.values[ref.id]


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    monkeypatch.setattr(_sdk.providers, "asset_catalogue_metadata_provider", None)


def binding():
    refs = _sdk.InProcessRefResolver()
    assets = _sdk._InProcessAssets()
    return refs, assets, _sdk.bind_runtime(refs, SimpleNamespace(assets=assets))


async def admitted(refs, path="/fixture/approved-model.safetensors"):
    return _sdk.AssetRef._wrap(await refs.create("ASSET", path))


def test_closed_record_roundtrip_preserves_unicode_and_reserved_data_keys():
    value = {"model_name": "café", "trainedWords": ["a", "b"],
             "__ref__": {"kind": "ASSET", "id": "inert text"},
             "nested": {"__cb__": None, "__wire_container__": [1, True, 0.5]}}
    assert decode_catalogue_metadata(encode_catalogue_metadata(value)) == value
    assert decode_catalogue_metadata(encode_catalogue_metadata(value)) is not value


def test_absence_and_empty_authored_record_are_distinct():
    assert encode_catalogue_metadata(None) is None
    assert decode_catalogue_metadata(None) is None
    assert decode_catalogue_metadata(encode_catalogue_metadata({})) == {}


@pytest.mark.parametrize("value", [[], "text", 1, b"bytes", (1,),
    {1: "nonstring key"}, {"bad": (1, 2)}, {"bad": b"bytes"},
    {"bad": object()}, {"bad": math.nan}, {"bad": math.inf}])
def test_non_closed_json_refuses(value):
    with pytest.raises((TypeError, ValueError)):
        encode_catalogue_metadata(value)


def test_custom_container_hooks_are_never_called():
    class Custom(dict):
        def items(self):
            raise AssertionError("custom iteration must not run")
    with pytest.raises(TypeError, match="JSON object"):
        encode_catalogue_metadata(Custom())
    with pytest.raises(TypeError, match="closed JSON"):
        encode_catalogue_metadata({"nested": Custom()})


def test_exact_encoded_byte_ceiling_and_escaped_expansion():
    overhead = len(encode_catalogue_metadata({"value": ""}))
    exact = {"value": "x" * (MAX_METADATA_BYTES - overhead)}
    encoded = encode_catalogue_metadata(exact)
    assert len(encoded) == MAX_METADATA_BYTES
    assert decode_catalogue_metadata(encoded) == exact
    with pytest.raises(ValueError, match="1 MiB"):
        encode_catalogue_metadata({"value": exact["value"] + "x"})
    with pytest.raises(ValueError, match="1 MiB"):
        encode_catalogue_metadata({"value": '"' * (MAX_METADATA_BYTES // 2)})
    with pytest.raises(ValueError, match="1 MiB"):
        decode_catalogue_metadata(b" " * (MAX_METADATA_BYTES + 1))


def test_cyclic_and_deep_records_refuse():
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycles"):
        encode_catalogue_metadata(cyclic)
    nested = {}
    for _ in range(MAX_METADATA_DEPTH + 2):
        nested = {"next": nested}
    with pytest.raises(ValueError, match="structural bound"):
        encode_catalogue_metadata(nested)


@pytest.mark.parametrize("raw", [b'null', b'[]', b'{"x":1,"x":2}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":"\xff"}'])
def test_malformed_or_non_object_transport_refuses(raw):
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        decode_catalogue_metadata(raw)


def test_missing_provider_refuses_instead_of_claiming_record_absence():
    async def run():
        refs, assets, scope = binding()
        ref = await admitted(refs)
        with scope, pytest.raises(NotImplementedError, match="source is unavailable"):
            await assets.catalogue_metadata(ref)
    asyncio.run(run())


def test_default_protocol_implementation_does_not_claim_absence():
    class Unimplemented(_sdk.AssetCatalogueMetadataProvider):
        pass
    async def run():
        refs, assets, scope = binding()
        ref = await admitted(refs)
        _sdk.providers.register_asset_catalogue_metadata_provider(Unimplemented())
        with scope, pytest.raises(NotImplementedError, match="source is unavailable"):
            await assets.catalogue_metadata(ref)
    asyncio.run(run())


def test_provider_registration_is_explicit_and_public_contract_matches():
    with pytest.raises(TypeError, match="catalogue_metadata"):
        _sdk.providers.register_asset_catalogue_metadata_provider(object())
    assert sdk.AssetCatalogueMetadataProvider is _sdk.AssetCatalogueMetadataProvider
    assert hasattr(sdk.AssetsDomain, "catalogue_metadata")
    _sdk.providers.register_asset_catalogue_metadata_provider(None)


def test_current_record_uses_exact_ref_without_digest_collapsing_or_cache():
    async def run():
        refs, assets, scope = binding()
        first = await admitted(refs, "/library/first.safetensors")
        second = await admitted(refs, "/library/second.safetensors")
        provider = Records({first.id: {"model_name": "first", "weight_sha": "a"*64},
                            second.id: {"model_name": "second", "weight_sha": "a"*64}})
        _sdk.providers.register_asset_catalogue_metadata_provider(provider)
        with scope:
            one = await assets.catalogue_metadata(first)
            two = await assets.catalogue_metadata(second)
            assert one["model_name"] == "first" and two["model_name"] == "second"
            provider.values[first.id] = {"trainedWords": ["new current words"]}
            assert await assets.catalogue_metadata(first) == provider.values[first.id]
            provider.values[first.id] = None  # Authorized fixture confirms absence.
            assert await assets.catalogue_metadata(first) is None
        assert provider.calls == [first, second, first, first]
    asyncio.run(run())


@pytest.mark.parametrize("problem", [PermissionError("denied"), OSError("backend down")])
def test_provider_failures_propagate(problem):
    class Failing:
        async def catalogue_metadata(self, ref):
            raise problem
    async def run():
        refs, assets, scope = binding()
        ref = await admitted(refs)
        _sdk.providers.register_asset_catalogue_metadata_provider(Failing())
        with scope, pytest.raises(type(problem), match=str(problem)):
            await assets.catalogue_metadata(ref)
    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["unknown", "foreign", "released", "relabeled"])
def test_live_ref_checks_run_before_provider(invalid):
    async def run():
        refs, assets, scope = binding()
        ref = await admitted(refs)
        if invalid == "unknown":
            ref = _sdk.AssetRef(kind="ASSET", id="not-admitted")
        elif invalid == "foreign":
            ref = await admitted(_sdk.InProcessRefResolver())
        elif invalid == "released":
            await refs.release(ref)
        else:
            wrong = await refs.create("IMAGE", object())
            ref = _sdk.AssetRef._wrap(wrong)
        provider = Records({})
        _sdk.providers.register_asset_catalogue_metadata_provider(provider)
        with scope, pytest.raises((KeyError, TypeError)):
            await assets.catalogue_metadata(ref)
        assert provider.calls == []
    asyncio.run(run())


@pytest.mark.parametrize("change", ["release", "replace-provider"])
def test_inflight_source_or_ref_invalidation_refuses(change):
    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()
        class Waiting:
            async def catalogue_metadata(self, ref):
                entered.set()
                await finish.wait()
                return {"trainedWords": ["must not escape invalidated read"]}
        refs, assets, scope = binding()
        ref = await admitted(refs)
        _sdk.providers.register_asset_catalogue_metadata_provider(Waiting())
        with scope:
            task = asyncio.create_task(assets.catalogue_metadata(ref))
            await entered.wait()
            if change == "release":
                await refs.release(ref)
            else:
                _sdk.providers.register_asset_catalogue_metadata_provider(None)
            finish.set()
            with pytest.raises((KeyError, RuntimeError)):
                await task
    asyncio.run(run())
