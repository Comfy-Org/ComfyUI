"""Tests for the telemetry-safe failure description (``app/assets/failures.py``)."""

import errno
import importlib.util
import json
import logging
import os
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError, IntegrityError

from app.assets.event_log import ALLOWED_FIELDS, TAG, emit_failure
from app.assets.failures import (
    EXTERNAL,
    NO_SITE,
    NO_WINERROR,
    Classification,
    classify_failure,
    describe_failure,
    exception_fingerprint,
)
from app.assets.services.file_utils import get_size_and_mtime_ns

SECRET_DIR = "/home/someone/secret-models"


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Shadow the conftest fixture that boots a server; nothing here needs one."""
    yield


def raised(exc: BaseException) -> BaseException:
    """``exc`` with a traceback, as a caught exception would have."""
    try:
        raise exc
    except BaseException as caught:
        return caught


def with_winerror(exc: OSError, winerror: int) -> OSError:
    # OSError only sets winerror on Windows; the classifier reads the attribute.
    exc.winerror = winerror  # type: ignore[attr-defined]
    return exc


def load_module(name: str, source: str, path: Path) -> ModuleType:
    """Import ``source`` from ``path`` as a module called ``name``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raise_from(module: ModuleType, function: str = "fail") -> BaseException:
    try:
        getattr(module, function)()
    except BaseException as caught:
        return caught
    raise AssertionError("expected the function to raise")


# --- classify_failure ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (PermissionError(errno.EACCES, "denied"), ("permission_denied", "EACCES", -1)),
        (OSError(errno.EPERM, "denied"), ("permission_denied", "EPERM", -1)),
        (FileNotFoundError(errno.ENOENT, "gone"), ("vanished", "ENOENT", -1)),
        (OSError(errno.EBUSY, "busy"), ("locked", "EBUSY", -1)),
        (OSError(errno.ESTALE, "stale"), ("network_unavailable", "ESTALE", -1)),
        (OSError(errno.EHOSTUNREACH, "host"), ("network_unavailable", "EHOSTUNREACH", -1)),
        (OSError(errno.ENODEV, "device"), ("device_unavailable", "ENODEV", -1)),
        (OSError(errno.EIO, "io"), ("io_error", "EIO", -1)),
        (OSError(errno.EILSEQ, "seq"), ("encoding", "EILSEQ", -1)),
        (OSError(errno.ENAMETOOLONG, "long"), ("name_too_long", "ENAMETOOLONG", -1)),
        (OSError(errno.ELOOP, "loop"), ("path_loop", "ELOOP", -1)),
        (OSError(errno.EFBIG, "big"), ("too_large", "EFBIG", -1)),
        (OSError(errno.ENOSPC, "full"), ("no_space", "ENOSPC", -1)),
        (OSError(errno.EROFS, "ro"), ("read_only", "EROFS", -1)),
        (OSError(errno.EMFILE, "fds"), ("fd_exhausted", "EMFILE", -1)),
        (OSError(errno.ENOMEM, "mem"), ("oom", "ENOMEM", -1)),
        (OSError(errno.EXDEV, "cross-device"), ("other", "EXDEV", -1)),
        (FileNotFoundError("raised without an errno"), ("vanished", "none", -1)),
        (ConnectionResetError("reset"), ("network_unavailable", "none", -1)),
        (MemoryError(), ("oom", "none", -1)),
        (TimeoutError("slow"), ("timeout", "none", -1)),
        (UnicodeEncodeError("utf-8", "\udcff", 0, 1, "surrogates"), ("encoding", "none", -1)),
        (json.JSONDecodeError("bad", "{", 0), ("corrupt", "none", -1)),
        (struct.error("short"), ("corrupt", "none", -1)),
        (ModuleNotFoundError("blake3"), ("dependency_missing", "none", -1)),
        (ValueError("anything"), ("other", "none", -1)),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else None,
)
def test_classification_is_keyed_on_codes_and_classes(exc, expected):
    assert classify_failure(exc) == Classification(*expected)


@pytest.mark.parametrize(
    ("exc", "winerror", "reason"),
    [
        # Python maps these onto ENOENT, so an offline share raises FileNotFoundError.
        (FileNotFoundError(errno.ENOENT, "bad netpath"), 53, "network_unavailable"),
        (FileNotFoundError(errno.ENOENT, "bad net name"), 67, "network_unavailable"),
        (FileNotFoundError(errno.ENOENT, "invalid drive"), 15, "device_unavailable"),
        (FileNotFoundError(errno.ENOENT, "path not found"), 3, "vanished"),
        (PermissionError(errno.EACCES, "sharing violation"), 32, "locked"),
        (OSError(errno.EINVAL, "cloud provider not running"), 362, "cloud_placeholder"),
        (OSError(errno.EINVAL, "crc"), 23, "io_error"),
        (FileNotFoundError(errno.ENOENT, "too long"), 206, "name_too_long"),
        (OSError(errno.EINVAL, "unmapped"), 9999, "other"),
    ],
)
def test_winerror_wins_over_the_errno_python_maps_it_onto(exc, winerror, reason):
    classification = classify_failure(with_winerror(exc, winerror))

    assert classification.reason == reason
    assert classification.winerror == winerror


def test_an_out_of_range_winerror_is_reported_as_absent():
    exc = with_winerror(OSError(errno.EINVAL, "hresult"), -2147024891)

    assert classify_failure(exc).winerror == NO_WINERROR


def test_a_cause_is_classified_when_the_raised_exception_is_not():
    try:
        try:
            raise OSError(errno.ESTALE, "stale")
        except OSError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert classify_failure(outer) == Classification("network_unavailable", "ESTALE", -1)


def test_a_sqlite_error_is_classified_by_its_error_code(tmp_path: Path):
    not_a_database = tmp_path / "garbage.sqlite3"
    not_a_database.write_bytes(b"x" * 4096)

    with pytest.raises(sqlite3.DatabaseError) as raw:
        sqlite3.connect(not_a_database).execute("SELECT 1 FROM sqlite_master")
    with pytest.raises(DatabaseError) as wrapped:
        with create_engine(f"sqlite:///{not_a_database}").connect() as connection:
            connection.execute(text("SELECT 1 FROM sqlite_master"))

    assert classify_failure(raw.value).reason == "db_corrupt"
    assert classify_failure(wrapped.value).reason == "db_corrupt"


def test_an_integrity_error_is_a_constraint_failure():
    exc = IntegrityError("INSERT", {}, sqlite3.IntegrityError("UNIQUE"))

    assert classify_failure(exc).reason == "db_constraint"


def test_an_unidentified_image_is_an_unsupported_format(tmp_path: Path):
    from PIL import Image, UnidentifiedImageError

    garbage = tmp_path / "not-really.png"
    garbage.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError) as caught:
        Image.open(garbage)

    assert classify_failure(caught.value).reason == "unsupported_format"


class _MessageTrap(OSError):
    def __str__(self) -> str:
        raise AssertionError("the message was read")

    def __repr__(self) -> str:
        raise AssertionError("the repr was read")


def test_neither_classification_nor_fingerprint_reads_the_message():
    exc = raised(_MessageTrap(errno.EIO, f"{SECRET_DIR}/model.safetensors"))

    description = describe_failure(exc)

    assert description.reason == "io_error"


def test_a_cyclic_cause_chain_terminates():
    first, second = ValueError("a"), ValueError("b")
    first.__context__, second.__context__ = second, first

    assert classify_failure(first).reason == "other"
    assert exception_fingerprint(first).exc_fp


# --- exception_fingerprint -----------------------------------------------------------

FAILING_ASSETS_MODULE = """
def fail():
    raise OSError(5, "{message}")
"""


def test_the_fingerprint_ignores_messages_line_numbers_and_file_names(tmp_path: Path):
    first = raise_from(
        load_module(
            "app.assets.fake", FAILING_ASSETS_MODULE.format(message="a"), tmp_path / "a" / "fake.py"
        )
    )
    moved = raise_from(
        load_module(
            "app.assets.fake",
            "\n\n\n" + FAILING_ASSETS_MODULE.format(message="b"),
            tmp_path / "secret-models" / "elsewhere.py",
        )
    )

    assert exception_fingerprint(first).exc_fp == exception_fingerprint(moved).exc_fp
    assert exception_fingerprint(first).exc_line != exception_fingerprint(moved).exc_line


def test_the_site_is_the_innermost_assets_frame_by_module_name():
    exc = raised_by(lambda: get_size_and_mtime_ns(f"{SECRET_DIR}/missing.safetensors"))

    fingerprint = exception_fingerprint(exc)

    assert fingerprint.exc_site == "assets.services.file_utils.get_size_and_mtime_ns"
    assert fingerprint.exc_line > 0
    assert fingerprint.exc_class == "FileNotFoundError"


def raised_by(operation: Callable[[], object]) -> BaseException:
    try:
        operation()
    except BaseException as caught:
        return caught
    raise AssertionError("expected the operation to raise")


def test_the_fingerprint_separates_code_paths():
    via_assets = raised_by(lambda: get_size_and_mtime_ns(f"{SECRET_DIR}/missing.bin"))
    direct = raised_by(lambda: os.stat(f"{SECRET_DIR}/missing.bin"))

    assert exception_fingerprint(via_assets).exc_fp != exception_fingerprint(direct).exc_fp


def test_the_fingerprint_includes_the_cause_chain(tmp_path: Path):
    module = load_module(
        "app.assets.fake",
        "def plain():\n"
        "    raise RuntimeError()\n"
        "def chained():\n"
        "    try:\n"
        "        raise OSError(5, 'x')\n"
        "    except OSError as e:\n"
        "        raise RuntimeError() from e\n",
        tmp_path / "fake.py",
    )

    plain = exception_fingerprint(raise_from(module, "plain")).exc_fp
    chained = exception_fingerprint(raise_from(module, "chained")).exc_fp

    assert plain != chained


def test_unknown_modules_and_their_classes_collapse_to_ext(tmp_path: Path):
    module = load_module(
        "custom_nodes.secret_node",
        "class SecretNodeError(Exception):\n"
        "    pass\n"
        "def fail():\n"
        "    raise SecretNodeError()\n",
        tmp_path / "secret_node.py",
    )

    fingerprint = exception_fingerprint(raise_from(module))

    assert fingerprint.exc_class == EXTERNAL
    assert fingerprint.exc_site == NO_SITE
    assert fingerprint.exc_line == 0


def test_a_known_package_class_carries_its_module():
    exc = IntegrityError("INSERT", {}, sqlite3.IntegrityError("UNIQUE"))

    assert exception_fingerprint(exc).exc_class == "sqlalchemy.exc.IntegrityError"


# --- emit_failure --------------------------------------------------------------------


def test_emit_failure_line_carries_only_validated_fields_and_nothing_user_derived(caplog):
    exc = raised_by(lambda: get_size_and_mtime_ns(f"{SECRET_DIR}/model.safetensors"))

    with caplog.at_level(logging.INFO):
        emit_failure("scanner.stat_failed", exc, site="reference")

    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith(TAG)]
    fields = dict(pair.split("=", 1) for pair in line.split()[2:])
    assert set(fields) <= set(ALLOWED_FIELDS)
    assert fields["reason"] == "vanished"
    assert fields["errno_name"] == "ENOENT"
    assert fields["exc_site"] == "assets.services.file_utils.get_size_and_mtime_ns"
    assert "secret" not in line
    assert "model" not in line
    assert os.sep not in line.removeprefix(TAG)
