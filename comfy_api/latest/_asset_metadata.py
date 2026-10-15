"""Bounded authored catalogue records, separate from model-file headers."""
from __future__ import annotations

import json
import math
from typing import Any


MAX_METADATA_BYTES = 1024 * 1024
MAX_METADATA_DEPTH = 64


def _closed_copy(value: Any, depth: int, active: set[int], budget: list[int]):
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_METADATA_DEPTH:
        raise ValueError("authored asset metadata exceeds its structural bound")
    kind = type(value)
    if value is None or kind in (bool, int):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("authored asset metadata needs finite JSON numbers")
        return value
    if kind is str:
        if len(value) > MAX_METADATA_BYTES:
            raise ValueError("authored asset metadata exceeds 1 MiB")
        return value
    if kind not in (dict, list):
        raise TypeError("authored asset metadata must contain only closed JSON")
    identity = id(value)
    if identity in active:
        raise ValueError("authored asset metadata must not contain cycles")
    if len(value) > MAX_METADATA_BYTES:
        raise ValueError("authored asset metadata exceeds its structural bound")
    active.add(identity)
    try:
        if kind is list:
            return [_closed_copy(item, depth + 1, active, budget) for item in value]
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("authored asset metadata needs string JSON keys")
            if len(key) > MAX_METADATA_BYTES:
                raise ValueError("authored asset metadata exceeds 1 MiB")
            result[key] = _closed_copy(item, depth + 1, active, budget)
        return result
    finally:
        active.remove(identity)


def encode_catalogue_metadata(value: dict[str, Any] | None) -> bytes | None:
    """Snapshot a complete closed record, or the provider's explicit absence."""
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError("authored asset metadata must be a JSON object or None")
    copied = _closed_copy(value, 0, set(), [MAX_METADATA_BYTES])
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    output = bytearray()
    for chunk in encoder.iterencode(copied):
        encoded = chunk.encode("utf-8")
        if len(output) + len(encoded) > MAX_METADATA_BYTES:
            raise ValueError("authored asset metadata exceeds 1 MiB")
        output.extend(encoded)
    return bytes(output)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("authored asset metadata has duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("authored asset metadata needs finite JSON numbers")


def decode_catalogue_metadata(data: bytes | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if type(data) is not bytes:
        raise TypeError("authored asset metadata transport must contain bytes")
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("authored asset metadata exceeds 1 MiB")
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant)
    except RecursionError as error:
        raise ValueError("authored asset metadata exceeds its depth bound") from error
    if type(value) is not dict:
        raise TypeError("an authored asset metadata record must be a JSON object")
    # Validate parsed depth/types and keep one fresh, provider-independent tree.
    _closed_copy(value, 0, set(), [MAX_METADATA_BYTES])
    return value
