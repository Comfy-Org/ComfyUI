"""Telemetry-safe descriptions of an exception: a bounded failure reason and a
fingerprint of the code that raised it.

Nothing here reads an exception's message, args, notes, locals or any file name.
``classify_failure`` is keyed on errno, winerror, sqlite error code and the
exception class. ``exception_fingerprint`` is keyed on module and function names,
where the module name comes from the frame's globals rather than its code
object's file name, which would embed the install path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import re
import struct
from types import TracebackType
from typing import NamedTuple

REASONS = frozenset({
    "permission_denied",
    "vanished",
    "locked",
    "cloud_placeholder",
    "network_unavailable",
    "device_unavailable",
    "io_error",
    "encoding",
    "name_too_long",
    "path_loop",
    "too_large",
    "no_space",
    "read_only",
    "fd_exhausted",
    "oom",
    "timeout",
    "corrupt",
    "unsupported_format",
    "db_busy",
    "db_locked",
    "db_corrupt",
    "db_full",
    "db_io",
    "db_readonly",
    "db_cantopen",
    "db_constraint",
    "dependency_missing",
    "other",
})
NO_WINERROR = -1
NO_SITE = "none"
EXTERNAL = "ext"

_MAX_CHAIN_LINKS = 3
_MAX_FRAMES_PER_LINK = 12
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,63}")
_SITE_PACKAGE = "app.assets"

# Top-level packages whose module and class names may appear in a fingerprint.
# Anything else, such as a custom node, collapses to EXTERNAL.
_KNOWN_PACKAGES = frozenset({
    "app", "comfy", "comfy_extras", "folder_paths", "server", "execution",
    "sqlalchemy", "sqlite3", "blake3", "PIL", "safetensors",
    "os", "io", "pathlib", "shutil", "json", "struct", "asyncio", "aiohttp",
    "genericpath", "ntpath", "posixpath",
})


def _errno_codes(*names: str) -> frozenset[int]:
    return frozenset(getattr(errno, name) for name in names if hasattr(errno, name))


_ERRNO_REASONS: tuple[tuple[frozenset[int], str], ...] = (
    (_errno_codes("EACCES", "EPERM"), "permission_denied"),
    # ENOTDIR: a directory on the path was replaced by a file, so the file is gone.
    (_errno_codes("ENOENT", "ENOTDIR"), "vanished"),
    (_errno_codes("EBUSY", "ETXTBSY"), "locked"),
    (
        _errno_codes(
            "ESTALE", "EHOSTDOWN", "EHOSTUNREACH", "ENETUNREACH", "ENETDOWN",
            "ETIMEDOUT", "ECONNRESET", "ECONNABORTED", "ECONNREFUSED", "ENOTCONN",
            "EREMOTEIO", "ENOLINK", "ECOMM",
        ),
        "network_unavailable",
    ),
    (_errno_codes("ENODEV", "ENXIO", "ENOMEDIUM"), "device_unavailable"),
    (_errno_codes("EIO"), "io_error"),
    (_errno_codes("EILSEQ"), "encoding"),
    (_errno_codes("ENAMETOOLONG"), "name_too_long"),
    (_errno_codes("ELOOP"), "path_loop"),
    (_errno_codes("EFBIG", "EOVERFLOW"), "too_large"),
    (_errno_codes("ENOSPC", "EDQUOT"), "no_space"),
    (_errno_codes("EROFS"), "read_only"),
    (_errno_codes("EMFILE", "ENFILE"), "fd_exhausted"),
    (_errno_codes("ENOMEM"), "oom"),
    (_errno_codes("EINTR"), "timeout"),
)

# Checked before errno: Python maps several of these onto ENOENT, so an offline
# share would otherwise read as a file that was deleted.
_WINERROR_REASONS: tuple[tuple[frozenset[int], str], ...] = (
    (frozenset({2, 3}), "vanished"),
    (frozenset({5}), "permission_denied"),
    (frozenset({32, 33}), "locked"),
    (frozenset({51, 53, 59, 64, 67, 121, 1231}), "network_unavailable"),
    (frozenset({15, 21}), "device_unavailable"),
    (frozenset({23, 1117}), "io_error"),
    (frozenset({206}), "name_too_long"),
    (frozenset({112}), "no_space"),
    (frozenset({19}), "read_only"),
    # The ERROR_CLOUD_FILE_* block: OneDrive and friends, file not available locally.
    (frozenset(range(358, 405)), "cloud_placeholder"),
)

_SQLITE_REASONS = {
    5: "db_busy",
    6: "db_locked",
    8: "db_readonly",
    10: "db_io",
    11: "db_corrupt",
    13: "db_full",
    14: "db_cantopen",
    19: "db_constraint",
    26: "db_corrupt",  # SQLITE_NOTADB
}


class Classification(NamedTuple):
    reason: str
    errno_name: str
    winerror: int


class Fingerprint(NamedTuple):
    exc_fp: str
    exc_class: str
    exc_site: str
    exc_line: int


UNKNOWN_FINGERPRINT = Fingerprint("0" * 12, EXTERNAL, NO_SITE, 0)


class FailureDescription(NamedTuple):
    """Every field an assets event may carry about one exception."""

    reason: str
    errno_name: str
    winerror: int
    exc_fp: str
    exc_class: str
    exc_site: str
    exc_line: int


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception followed by its causes, the way a traceback prints them."""
    links: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(links) < _MAX_CHAIN_LINKS and current not in links:
        links.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return links


def _errno_name(code: object) -> str:
    return errno.errorcode.get(code, "none") if isinstance(code, int) else "none"


def _winerror(exc: BaseException) -> int:
    code = getattr(exc, "winerror", None)
    if isinstance(code, int) and not isinstance(code, bool) and 0 <= code <= 0xFFFF:
        return code
    return NO_WINERROR


def _lookup(code: int, table: tuple[tuple[frozenset[int], str], ...]) -> str | None:
    return next((reason for codes, reason in table if code in codes), None)


def _class_reason(exc: BaseException) -> str | None:
    cls = type(exc)
    if cls.__name__ == "UnidentifiedImageError" and cls.__module__.startswith("PIL"):
        return "unsupported_format"
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (json.JSONDecodeError, struct.error)):
        return "corrupt"
    if isinstance(exc, UnicodeError):
        return "encoding"
    if isinstance(exc, ImportError):
        return "dependency_missing"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if cls.__name__ == "IntegrityError" and cls.__module__.split(".")[0] in {"sqlalchemy", "sqlite3"}:
        return "db_constraint"
    return None


def _classify_one(exc: BaseException) -> Classification | None:
    winerror = _winerror(exc)
    exc_errno = getattr(exc, "errno", None) if isinstance(exc, OSError) else None
    errno_name = _errno_name(exc_errno)
    reason = _lookup(winerror, _WINERROR_REASONS) if winerror != NO_WINERROR else None
    if reason is None:
        reason = _class_reason(exc)
    if reason is None and isinstance(exc_errno, int):
        reason = _lookup(exc_errno, _ERRNO_REASONS)
    if reason is None and isinstance(exc, FileNotFoundError):
        reason = "vanished"  # raised without an errno, as tests and some libraries do
    if reason is None and isinstance(exc, ConnectionError):
        reason = "network_unavailable"
    if reason is None and isinstance(exc, TimeoutError):
        reason = "timeout"  # after errno, so a share's ETIMEDOUT reads as network
    sqlite_code = getattr(exc, "sqlite_errorcode", None)
    if reason is None and isinstance(sqlite_code, int):
        reason = _SQLITE_REASONS.get(sqlite_code & 0xFF)
    if reason is None:
        return None
    return Classification(reason, errno_name, winerror)


def classify_failure(exc: BaseException) -> Classification:
    """Map an exception onto a bounded failure reason, never reading its message.

    The raised exception wins over its causes; a SQLAlchemy error defers to the
    driver error it wraps. When nothing matches, the reason is ``other`` and the
    raw errno and winerror of the raised exception are kept, so an unmapped code
    is still visible. This never raises: scan code branches on the result, and a
    telemetry bug must not break a scan.
    """
    try:
        return _classify(exc)
    except Exception:
        logging.debug("Failure classification failed", exc_info=True)
        reason = "vanished" if isinstance(exc, FileNotFoundError) else "other"
        return Classification(reason, "none", NO_WINERROR)


def _classify(exc: BaseException) -> Classification:
    for link in _chain(exc):
        for candidate in (link, getattr(link, "orig", None)):
            if isinstance(candidate, BaseException):
                classification = _classify_one(candidate)
                if classification is not None:
                    return classification
    exc_errno = getattr(exc, "errno", None) if isinstance(exc, OSError) else None
    return Classification("other", _errno_name(exc_errno), _winerror(exc))


def _is_known(module: str) -> bool:
    return module.split(".", 1)[0] in _KNOWN_PACKAGES


def _identifier(value: str) -> str:
    """``value`` if it is a dotted identifier that fits a field, else EXTERNAL."""
    return value if _IDENTIFIER.fullmatch(value) else EXTERNAL


def exception_class(exc: BaseException) -> str:
    cls = type(exc)
    if cls.__module__ == "builtins":
        return _identifier(cls.__qualname__)
    if not _is_known(cls.__module__):
        return EXTERNAL
    return _identifier(f"{cls.__module__}.{cls.__qualname__}")


def _frames(tb: TracebackType | None) -> list[tuple[str, str, int]]:
    """(module, function, line) per frame, outermost first; unknown modules become EXTERNAL."""
    frames: list[tuple[str, str, int]] = []
    while tb is not None:
        module = tb.tb_frame.f_globals.get("__name__")
        name = module if isinstance(module, str) and _is_known(module) else EXTERNAL
        function = tb.tb_frame.f_code.co_name.strip("<>") if name != EXTERNAL else EXTERNAL
        frames.append((name, function, tb.tb_lineno))
        tb = tb.tb_next
    return frames[-_MAX_FRAMES_PER_LINK:]


def _signature(frames: list[tuple[str, str, int]]) -> str:
    parts: list[str] = []
    for module, function, _line in frames:
        part = EXTERNAL if module == EXTERNAL else f"{module}.{function}"
        if not (part == EXTERNAL and parts and parts[-1] == EXTERNAL):
            parts.append(part)
    return ">".join(parts)


def exception_fingerprint(exc: BaseException) -> Fingerprint:
    """Identify the code path that raised ``exc`` without anything user-derived.

    ``exc_fp`` hashes the class, errno and module.function frames of each link in
    the cause chain, but no line numbers, so it survives unrelated edits.
    ``exc_site`` and ``exc_line`` locate the innermost ``app.assets`` frame of the
    raised exception; paired with the build's commit that is an exact location.
    """
    links = _chain(exc)
    canonical = ["v1"]
    for link in links:
        exc_errno = getattr(link, "errno", None) if isinstance(link, OSError) else None
        canonical.append(
            f"{exception_class(link)}|{_errno_name(exc_errno)}|{_signature(_frames(link.__traceback__))}"
        )
    exc_fp = hashlib.sha256("|".join(canonical).encode("utf-8")).hexdigest()[:12]

    exc_site, exc_line = NO_SITE, 0
    for module, function, line in reversed(_frames(exc.__traceback__)):
        if module == _SITE_PACKAGE or module.startswith(_SITE_PACKAGE + "."):
            short_module = module.removeprefix("app.")
            # A name too long for a field keeps its module rather than reading as external.
            for candidate in (f"{short_module}.{function}", short_module):
                if _IDENTIFIER.fullmatch(candidate):
                    exc_site = candidate
                    break
            exc_line = line
            break
    return Fingerprint(exc_fp, exception_class(exc), exc_site, exc_line)


def describe_failure(exc: BaseException) -> FailureDescription:
    """Every field an event carries about ``exc``. Never raises, like classify_failure."""
    try:
        fingerprint = exception_fingerprint(exc)
    except Exception:
        logging.debug("Exception fingerprinting failed", exc_info=True)
        fingerprint = UNKNOWN_FINGERPRINT
    return FailureDescription(*classify_failure(exc), *fingerprint)
