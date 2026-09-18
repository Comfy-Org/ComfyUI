"""Shared fixtures for the execution validation tests.

Forces CPU before any comfy import so the tests in this directory collect
and run on machines without CUDA/MPS; no GPU code path is exercised.
"""
import pytest

from comfy.cli_args import args as cli_args

cli_args.cpu = True

import nodes  # noqa: E402


class _StubBase:
    """Minimal node base for stubs registered in NODE_CLASS_MAPPINGS."""

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "_test"
    OUTPUT_NODE = False

    def run(self, **kwargs):
        """Never executed; validation stops before node execution."""
        return ()


def _make_stub(name, input_types, return_types=(), output_node=False, validate=None):
    """Build a legacy-style node class with the given validation surface."""
    attrs = {
        "RETURN_TYPES": return_types,
        "OUTPUT_NODE": output_node,
        "INPUT_TYPES": classmethod(lambda cls: input_types),
    }
    if validate is not None:
        attrs["VALIDATE_INPUTS"] = classmethod(validate)
    return type(name, (_StubBase,), attrs)


@pytest.fixture
def stub_nodes(monkeypatch):
    """Register stub node classes in nodes.NODE_CLASS_MAPPINGS; auto-clean."""

    def register(name, **kwargs):
        cls = _make_stub(name, **kwargs)
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
        return cls

    return register
