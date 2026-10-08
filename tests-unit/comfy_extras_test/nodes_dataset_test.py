"""Tests that dataset node config declared as class attributes reaches the schema.

``ImageProcessingNode`` and ``TextProcessingNode`` let subclasses configure
themselves with plain class attributes, and their shared ``define_schema()`` is
what forwards those attributes into ``io.Schema``. Anything it forgets to
forward is silently dropped from /object_info, so this pins the forwarding
itself rather than any single field.
"""

import dataclasses

import pytest

from comfy_api.latest import io
from comfy_extras import nodes_dataset

# Structural schema members, not per-node config; a node class would never
# declare these as class attributes.
IGNORED_FIELDS = {"inputs", "outputs", "hidden", "node_id"}

SCHEMA_FIELDS = [
    f.name for f in dataclasses.fields(io.Schema) if f.name not in IGNORED_FIELDS
]


def _node_classes():
    """Every concrete node defined in nodes_dataset."""
    found = []
    for obj in vars(nodes_dataset).values():
        if not isinstance(obj, type) or not issubclass(obj, io.ComfyNode):
            continue
        if obj.__module__ != nodes_dataset.__name__:
            continue
        if getattr(obj, "node_id", "") is None:
            continue  # abstract base class, define_schema() would raise
        found.append(obj)
    return sorted(found, key=lambda c: c.__name__)


@pytest.mark.parametrize("node_cls", _node_classes(), ids=lambda c: c.__name__)
def test_class_attributes_are_forwarded_to_schema(node_cls):
    schema = node_cls.define_schema()
    for name in SCHEMA_FIELDS:
        declared = getattr(node_cls, name, None)
        if not declared:
            continue  # unset, or left at the base class default
        assert getattr(schema, name) == declared, (
            f"{node_cls.__name__}.{name} is not forwarded into io.Schema by "
            f"define_schema(), so /object_info reports "
            f"{name}={getattr(schema, name)!r} instead of {declared!r}"
        )


def test_node_classes_are_discovered():
    """Guard against the parametrization above collapsing to zero cases."""
    assert _node_classes()
