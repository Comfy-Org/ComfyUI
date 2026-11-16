"""Comfy API v2 contract helpers for core tests.

The contract is ``<repo>/openapi-v2.yaml``, generated one-way from the canonical
cloud spec by ``project_spec.py``. ``COMFY_V2_SPEC_PATH`` overrides the path for
tests only; it is read once at import, so set it before pytest starts.
"""

import hashlib
import json
import os
from pathlib import Path

import jsonschema
import yaml

SPEC_PATH = Path(os.environ.get("COMFY_V2_SPEC_PATH") or Path(__file__).resolve().parents[2] / "openapi-v2.yaml")
SPEC = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

# (METHOD, path) exactly as written in the spec, e.g. ("GET", "/api/v2/jobs/{id}").
OPERATIONS = {(method.upper(), path) for path, item in SPEC["paths"].items() for method in item if method in HTTP_METHODS}

# Core v2 routes that are not contract operations. Empty: every core route is in the contract.
ROUTE_EXTENSIONS = set()

# Reviewed-contract marker: update only after reviewing a normative contract change.
REVIEWED_NORMATIVE_DIGEST = "f72efe0be562226e41225dbb11422b22bb7cdc402bbbbaa680e1c1d84fac4b88"

_PROSE_KEYS = {"description", "summary", "example", "examples"}
_OAS_ONLY_KEYS = {"example", "examples", "discriminator", "xml", "externalDocs", "nullable"}


def _drop_keys(node, keys):
    """Recursively drop annotation ``keys``. Entries of a ``properties`` map are
    schema property names (normative), never annotations, so they are kept."""
    if isinstance(node, list):
        return [_drop_keys(v, keys) for v in node]
    if not isinstance(node, dict):
        return node
    return {
        k: {name: _drop_keys(s, keys) for name, s in v.items()} if k == "properties" and isinstance(v, dict) else _drop_keys(v, keys)
        for k, v in node.items()
        if k not in keys
    }


def normative_digest(spec):
    """sha256 of the normative contract: paths, components, top-level security and
    x-* keys, with description/summary/example/examples annotations dropped."""
    kept = {k: v for k, v in spec.items() if k in ("paths", "components", "security") or k.startswith("x-")}
    return hashlib.sha256(json.dumps(_drop_keys(kept, _PROSE_KEYS), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _to_json_schema(node):
    """OpenAPI 3.0 -> JSON Schema: honour ``nullable: true``, drop OpenAPI-only keys
    (ported from comfy-api-proxy/tests/test_schema_conformance.py)."""
    if isinstance(node, list):
        return [_to_json_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {
        k: {name: _to_json_schema(s) for name, s in v.items()} if k == "properties" and isinstance(v, dict) else _to_json_schema(v)
        for k, v in node.items()
        if k not in _OAS_ONLY_KEYS
    }
    if node.get("nullable") is True:
        if isinstance(out.get("type"), str):
            out["type"] = [out["type"], "null"]
        else:
            out = {"anyOf": [out, {"type": "null"}]}
    return out


_CONVERTED = _to_json_schema(SPEC)


def resolve(ref, doc=SPEC):
    """Resolve a local ``#/a/b`` pointer against ``doc``; raises KeyError if missing."""
    if not ref.startswith("#/"):
        raise KeyError(f"non-local $ref {ref}")
    node = doc
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _inline(node, seen=()):
    if isinstance(node, list):
        return [_inline(v, seen) for v in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if ref is not None:
        if ref in seen:
            raise ValueError(f"recursive $ref {ref} cannot be inlined")
        return _inline(resolve(ref, _CONVERTED), seen + (ref,))
    return {k: _inline(v, seen) for k, v in node.items()}


def schema(name):
    """JSON Schema for ``#/components/schemas/<name>`` with every $ref inlined."""
    return _inline({"$ref": f"#/components/schemas/{name}"})


def validate(name, instance):
    """Validate ``instance`` against a component schema name (or an already-built schema dict)."""
    jsonschema.Draft202012Validator(schema(name) if isinstance(name, str) else name).validate(instance)


def response_schema(op_id, status):
    """Declared response schema for ``op_id``/``status`` (application/json preferred,
    else the only media type); None when the response has no body. Raises KeyError
    when the operation or the status is not declared."""
    for item in _CONVERTED["paths"].values():
        for method, op in item.items():
            if method in HTTP_METHODS and op.get("operationId") == op_id:
                if str(status) not in op["responses"]:
                    raise KeyError(f"{op_id} does not declare status {status}")
                content = _inline(op["responses"][str(status)]).get("content")
                if not content:
                    return None
                return (content.get("application/json") or next(iter(content.values()))).get("schema")
    raise KeyError(f"unknown operationId {op_id}")
