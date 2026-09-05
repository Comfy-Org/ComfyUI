import pytest

from comfy_api.latest._capabilities import (
    Capability,
    CapabilityRegistry,
    split_request,
)


def _registry(*entries):
    registry = CapabilityRegistry()
    for name, version in entries:
        registry.register(
            Capability(name=name, version=version), f"{name}@{version}")
    return registry


def test_bare_name_resolves_to_the_newest_contract():
    registry = _registry(("sam.segment", 1), ("sam.segment", 2))
    assert registry.resolve("sam.segment") == "sam.segment@2"


def test_pinned_version_resolves_exactly():
    """A node written against v1 keeps getting v1 after v2 ships."""
    registry = _registry(("sam.segment", 1), ("sam.segment", 2))
    assert registry.resolve("sam.segment@1") == "sam.segment@1"
    assert registry.resolve("sam.segment@2") == "sam.segment@2"


def test_versions_coexist_so_adding_one_breaks_nobody():
    registry = _registry(("sam.segment", 1))
    assert registry.resolve("sam.segment@1") == "sam.segment@1"
    registry.register(Capability("sam.segment", 2), "sam.segment@2")
    assert registry.resolve("sam.segment@1") == "sam.segment@1"


def test_unknown_capability_is_an_answer_not_a_failure():
    registry = _registry(("sam.segment", 1))
    assert registry.supports("nothing.here") is False
    assert registry.resolve("nothing.here") is None
    assert registry.supports("sam.segment@9") is False


def test_supports_tolerates_a_malformed_request():
    registry = _registry(("sam.segment", 1))
    assert registry.supports("sam.segment@notanumber") is False


def test_resolve_rejects_a_malformed_request_loudly():
    """supports() is a question; resolve() is an intent, so a bad version is a bug."""
    registry = _registry(("sam.segment", 1))
    with pytest.raises(ValueError):
        registry.resolve("sam.segment@notanumber")


def test_enumeration_lets_a_caller_discover_an_unfamiliar_runtime():
    registry = _registry(("b.op", 1), ("a.op", 2), ("a.op", 1))
    assert [c.id for c in registry.capabilities()] == [
        "a.op@1", "a.op@2", "b.op@1"]


def test_descriptor_carries_the_schema_for_validation_and_typing():
    registry = CapabilityRegistry()
    schema = {"attrs": {"description": "d"}, "inputs": [], "outputs": []}
    registry.register(Capability("x.op", 1, schema), object())
    assert registry.describe("x.op").schema == schema


def test_split_request():
    assert split_request("a.b") == ("a.b", None)
    assert split_request("a.b@3") == ("a.b", 3)
    with pytest.raises(ValueError):
        split_request("a.b@x")
