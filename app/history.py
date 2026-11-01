"""Opt-in terminal history storage; never serialize execution objects or credentials."""

import json
import math

from sqlalchemy import select, delete
from sqlalchemy.exc import SQLAlchemyError

from app.assets.event_log import error_kind
from app.database.db import create_write_session
from app.database.models import ExecutionHistory
from comfy_execution.history import HistoryStorageError

MAX_RECORD_BYTES = 100 * 1024 * 1024
FORMAT_VERSION = 1
SENSITIVE_KEYS = frozenset(("auth_token_comfy_org", "api_key_comfy_org"))
EVENT_FIELDS = {
    "execution_start": ("prompt_id", "timestamp"),
    "execution_cached": ("prompt_id", "timestamp", "nodes"),
    "execution_success": ("prompt_id", "timestamp"),
    "execution_interrupted": (
        "prompt_id",
        "timestamp",
        "node_id",
        "node_type",
        "executed",
    ),
    "execution_error": ("prompt_id", "timestamp", "node_id", "node_type", "executed"),
}


def _copy_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_json(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _copy_json(v) for k, v in value.items() if k not in SENSITIVE_KEYS}
    raise HistoryStorageError(
        "History contains unsupported JSON values", reason="unsupported_json"
    )


def _validate(prompt_id, record):
    if not isinstance(record, dict) or not isinstance(record.get("outputs"), dict):
        raise HistoryStorageError("Invalid history record", reason="invalid_record")
    prompt = record.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 5
        or prompt[1] != prompt_id
        or not isinstance(prompt[2], dict)
        or not isinstance(prompt[3], dict)
        or not isinstance(prompt[4], list)
    ):
        raise HistoryStorageError("Invalid history prompt", reason="invalid_record")
    extra = prompt[3]
    pnginfo = extra.get("extra_pnginfo", {})
    if not isinstance(pnginfo, dict) or not isinstance(
        pnginfo.get("workflow", {}), dict
    ):
        raise HistoryStorageError("Invalid history workflow", reason="invalid_record")
    status = record.get("status")
    if status is not None:
        if (
            not isinstance(status, dict)
            or status.get("status_str") not in ("success", "error")
            or type(status.get("completed")) is not bool
            or not isinstance(status.get("messages"), list)
        ):
            raise HistoryStorageError("Invalid history status", reason="invalid_record")
        for message in status["messages"]:
            if (
                not isinstance(message, (list, tuple))
                or len(message) != 2
                or not isinstance(message[0], str)
                or not isinstance(message[1], dict)
            ):
                raise HistoryStorageError(
                    "Invalid history message", reason="invalid_record"
                )
            timestamp = message[1].get("timestamp")
            if timestamp is not None and (
                type(timestamp) not in (int, float)
                or (type(timestamp) is float and not math.isfinite(timestamp))
            ):
                raise HistoryStorageError(
                    "Invalid history timestamp", reason="invalid_record"
                )


def project_record(prompt_id, record):
    try:
        projected = dict(record)
        projected["prompt"] = list(record["prompt"][:5])
        _validate(prompt_id, {**record, "prompt": projected["prompt"]})
        status = record.get("status")
        if status is not None:
            messages = []
            for event, payload in status.get("messages", []):
                if event in EVENT_FIELDS:
                    clean = {k: payload[k] for k in EVENT_FIELDS[event] if k in payload}
                    if event == "execution_error":
                        clean["exception_message"] = (
                            "Execution failed; detailed diagnostics were not retained."
                        )
                    messages.append([event, clean])
            projected["status"] = {
                "status_str": status["status_str"],
                "completed": status["completed"],
                "messages": messages,
            }
        projected = _copy_json(projected)
        _validate(prompt_id, projected)
        encoded = json.dumps(
            projected, allow_nan=False, separators=(",", ":"), ensure_ascii=True
        )
        if len(encoded) > MAX_RECORD_BYTES:
            raise HistoryStorageError(
                "History record exceeds 100 MiB", reason="record_too_large"
            )
        return projected, encoded
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        raise HistoryStorageError(
            "History could not be serialized", reason="serialization_failed"
        ) from exc


class HistoryStore:
    def __init__(self, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("History limit must be positive")
        self.limit = limit

    def _prune(self, session):
        keep = (
            select(ExecutionHistory.sequence)
            .order_by(ExecutionHistory.sequence.desc())
            .limit(self.limit)
        )
        removed = list(
            session.scalars(
                select(ExecutionHistory.prompt_id).where(
                    ExecutionHistory.sequence.not_in(keep)
                )
            )
        )
        session.execute(
            delete(ExecutionHistory).where(ExecutionHistory.sequence.not_in(keep))
        )
        return removed

    def load(self):
        try:
            with create_write_session() as session, session.begin():
                self._prune(session)
                history = {}
                for row in session.scalars(
                    select(ExecutionHistory)
                    .order_by(ExecutionHistory.sequence)
                    .limit(self.limit)
                ):
                    if (
                        row.format_version != FORMAT_VERSION
                        or len(row.record_json) > MAX_RECORD_BYTES
                    ):
                        raise HistoryStorageError(
                            "Unsupported history format", reason="unsupported_format"
                        )
                    record = json.loads(
                        row.record_json,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("Nonfinite JSON")
                        ),
                    )
                    json.dumps(record, allow_nan=False)
                    _validate(row.prompt_id, record)
                    history[row.prompt_id] = record
                return history
        except (SQLAlchemyError, ValueError, TypeError, RecursionError) as exc:
            raise HistoryStorageError(
                "History could not be restored",
                reason=error_kind(exc)
                if isinstance(exc, SQLAlchemyError)
                else "invalid_json",
            ) from exc

    def save(self, prompt_id, record):
        canonical, encoded = project_record(prompt_id, record)
        try:
            with create_write_session() as session, session.begin():
                row = session.scalar(
                    select(ExecutionHistory).where(
                        ExecutionHistory.prompt_id == prompt_id
                    )
                )
                if row is None:
                    session.add(
                        ExecutionHistory(
                            prompt_id=prompt_id,
                            format_version=FORMAT_VERSION,
                            record_json=encoded,
                        )
                    )
                else:
                    row.format_version = FORMAT_VERSION
                    row.record_json = encoded
                session.flush()
                removed = self._prune(session)
            return canonical, removed
        except SQLAlchemyError as exc:
            raise HistoryStorageError(
                "History could not be saved", canonical, reason=error_kind(exc)
            ) from exc

    def remove(self, ids, clear=False):
        try:
            with create_write_session() as session, session.begin():
                if clear:
                    session.execute(delete(ExecutionHistory))
                else:
                    for start in range(0, len(ids), 500):
                        session.execute(
                            delete(ExecutionHistory).where(
                                ExecutionHistory.prompt_id.in_(ids[start : start + 500])
                            )
                        )
        except SQLAlchemyError as exc:
            raise HistoryStorageError(
                "History could not be deleted", reason=error_kind(exc)
            ) from exc
