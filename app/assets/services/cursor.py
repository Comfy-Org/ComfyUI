"""Opaque keyset-pagination cursor for /api/assets.

Payload JSON uses short keys to keep the encoded length small:

    {"s": <sort_field>, "v": <value>, "id": <id>, "o": <order>}

The `o` key binds the cursor to the sort direction it was minted under,
so replaying a `desc` cursor against an `asc` request fails with
``INVALID_CURSOR`` rather than silently walking the wrong direction.
`o` is mandatory on every payload — a cursor without it is rejected as
malformed.

Encoding is base64url with no padding. Cursors are opaque tokens: the
payload format is internal to this server, and clients must treat a
cursor as a black box handed back via `next_cursor`. No byte-level
compatibility with any other implementation is required.

Time values are serialized as Unix microseconds (UTC) — microsecond
precision is sufficient to round-trip the timestamps stored by the
database without rounding rows in the same millisecond bucket.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


class InvalidCursorError(ValueError):
    """Raised on a malformed, oversized, or unsupported-sort-field cursor.

    Map to a 400 response with code ``INVALID_CURSOR`` at the handler.
    """


# Wire-format length caps. Cursors are user-controlled, so caps protect the
# decode path from oversized allocations and downstream SQL predicates from
# unbounded strings.
#
# MAX_CURSOR_VALUE_LENGTH is 512 to fit the `AssetReference.name` column max
# (`String(512)`) — otherwise a long-named asset would mint a cursor the same
# server then refuses on the next request.
#
# MAX_ENCODED_CURSOR_LENGTH is the decode-path guard, sized comfortably above
# the largest cursor the per-field caps can produce. Worst case is value + id
# at their caps with every character JSON-escaping to the six-byte `\uXXXX`
# form (control characters), which is ~5.2 KB once base64url-encoded. At 8192
# the encoder can never mint a cursor that exceeds it, so a freshly minted
# cursor always decodes on the next request and there is no user-visible
# "cursor too long" failure.
MAX_ENCODED_CURSOR_LENGTH = 8192
MAX_CURSOR_VALUE_LENGTH = 512
MAX_CURSOR_ID_LENGTH = 128


@dataclass(frozen=True)
class CursorPayload:
    sort_field: str
    value: str
    id: str
    order: str


_VALID_ORDERS = ("asc", "desc")


def encode_cursor(sort_field: str, value: str, id: str, order: str = "desc") -> str:
    """Encode a cursor payload as a base64url (no-padding) string.

    `order` binds the cursor to the sort direction it was minted under so a
    later request with a flipped `order` query parameter is rejected with
    ``INVALID_CURSOR`` rather than silently walking the wrong direction.
    """
    if order not in _VALID_ORDERS:
        raise InvalidCursorError(f"order must be one of {_VALID_ORDERS}, got {order!r}")
    # Symmetric input validation: the encoder must reject anything the
    # decoder rejects, or the same server will mint cursors it then 400s on
    # the next request.
    if not id:
        raise InvalidCursorError("id must be non-empty")
    if len(id) > MAX_CURSOR_ID_LENGTH:
        raise InvalidCursorError("id exceeds maximum length")
    if len(value) > MAX_CURSOR_VALUE_LENGTH:
        raise InvalidCursorError("value exceeds maximum length")
    payload = {"s": sort_field, "v": value, "id": id, "o": order}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    # No mint-time length guard is needed: the per-field caps above bound the
    # encoded length well below MAX_ENCODED_CURSOR_LENGTH (see its definition),
    # so the encoder can never produce a cursor the decode path would reject.
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("ascii")


def encode_cursor_from_time(sort_field: str, t: datetime, id: str, order: str = "desc") -> str:
    """Encode a time-typed cursor at Unix microsecond precision.

    Accepts an aware datetime (any timezone) and normalizes to UTC. Naive
    datetimes are rejected so callers can't accidentally encode the local
    wall-clock value of a UTC-stored timestamp.
    """
    if t.tzinfo is None:
        raise ValueError("encode_cursor_from_time requires an aware datetime")
    micros = _datetime_to_unix_micros(t.astimezone(timezone.utc))
    return encode_cursor(sort_field, str(micros), id, order=order)


def decode_cursor(
    cursor: str,
    allowed_sort_fields: Iterable[str],
    expected_order: str | None = None,
) -> CursorPayload:
    """Parse an opaque cursor.

    ``allowed_sort_fields`` is the endpoint's accepted sort-field list — a
    cursor carrying a field outside this set is rejected so a cursor minted
    for one column can't be replayed against another (e.g. a ``created_at``
    timestamp string compared against a ``name`` column).

    ``expected_order`` (``"asc"``/``"desc"``), when supplied, must match the
    payload's ``o`` field. ``o`` is required on every payload; a cursor
    missing it is rejected as malformed.

    Passing no allowed fields rejects every cursor.
    """
    if len(cursor) > MAX_ENCODED_CURSOR_LENGTH:
        raise InvalidCursorError("cursor exceeds maximum length")

    try:
        # urlsafe_b64decode requires correct padding; we strip on encode, so
        # restore the trailing '=' pad here.
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
    except (ValueError, base64.binascii.Error) as e:
        raise InvalidCursorError(f"encoding: {e}") from e

    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InvalidCursorError(f"payload: {e}") from e

    if not isinstance(decoded, dict):
        raise InvalidCursorError("payload: expected object")

    sort_field = decoded.get("s")
    value = decoded.get("v")
    id = decoded.get("id")
    order = decoded.get("o")

    if not isinstance(sort_field, str) or not isinstance(value, str) or not isinstance(id, str):
        raise InvalidCursorError("payload: missing or non-string s/v/id")

    if id == "":
        raise InvalidCursorError("missing id")
    if len(id) > MAX_CURSOR_ID_LENGTH:
        raise InvalidCursorError("id exceeds maximum length")
    if len(value) > MAX_CURSOR_VALUE_LENGTH:
        raise InvalidCursorError("value exceeds maximum length")

    if sort_field not in allowed_sort_fields:
        raise InvalidCursorError(f"unsupported sort field {sort_field!r}")

    if not isinstance(order, str):
        raise InvalidCursorError("missing or non-string o")
    if order not in _VALID_ORDERS:
        raise InvalidCursorError(f"unsupported order {order!r}")
    if expected_order is not None and order != expected_order:
        raise InvalidCursorError(
            f"cursor order {order!r} does not match request order {expected_order!r}"
        )

    return CursorPayload(sort_field=sort_field, value=value, id=id, order=order)


def decode_cursor_time(payload: Optional[CursorPayload]) -> datetime:
    """Parse a time-typed cursor value as Unix microseconds, returning UTC."""
    if payload is None:
        raise InvalidCursorError("nil cursor payload")
    try:
        micros = int(payload.value)
    except ValueError as e:
        raise InvalidCursorError(f"value is not a valid timestamp: {e}") from e
    try:
        return _unix_micros_to_datetime(micros)
    except (OverflowError, OSError, ValueError) as e:
        # Crafted out-of-range microseconds (e.g. > datetime.MAX_YEAR) blow up
        # in fromtimestamp / datetime construction. Map to 400, not 500.
        raise InvalidCursorError(f"value is out of representable range: {e}") from e


def decode_cursor_int(payload: Optional[CursorPayload]) -> int:
    """Parse a cursor value as a base-10 integer."""
    if payload is None:
        raise InvalidCursorError("nil cursor payload")
    try:
        return int(payload.value)
    except ValueError as e:
        raise InvalidCursorError(f"value is not a valid integer: {e}") from e


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime_to_unix_micros(t: datetime) -> int:
    """Convert an aware UTC datetime to Unix microseconds (integer math)."""
    delta = t - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _unix_micros_to_datetime(micros: int) -> datetime:
    """Convert Unix microseconds to a UTC datetime, preserving precision."""
    seconds, micro_remainder = divmod(micros, 1_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=micro_remainder)
