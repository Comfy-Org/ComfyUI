"""Tests for the Content-Disposition header format in ``server.py``.

The ``/view`` route used to set ``Content-Disposition`` as
``filename="<name>"`` with no disposition type, which fails RFC 2183
parsers (e.g. Go's ``mime.ParseMediaType``) and breaks third-party
downloaders that rely on a real ``attachment`` / ``inline`` prefix.

This test guards against a future contributor re-introducing a
malformed header by scanning ``server.py`` for every disposition
string literal and asserting it matches the
``<disposition-type>; filename=...`` shape.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PY = REPO_ROOT / "server.py"

# Matches an RFC 2183 / 6266 Content-Disposition value:
#   <type>; filename="..."  or  <type>; filename=...
DISPOSITION_PATTERN = re.compile(
    r'^(inline|attachment|form-data)\s*;\s*filename=(?:"[^"\r\n]*"|[^\s;]+)\s*$'
)

# String literal that starts the line: captures f-strings, regular strings, etc.
_STRING_LITERAL_RE = re.compile(
    r'''(?P<quote>["'])                              # opening quote
        (?P<value>(?:\\.|(?!(?P=quote)).)*?)         # body (no newlines)
        (?P=quote)                                    # closing quote
    ''',
    re.VERBOSE | re.DOTALL,
)


def _all_string_literals(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, literal_value)`` for every string literal
    in ``text`` that *looks like* a Content-Disposition value — i.e. it
    starts with a disposition type or the legacy ``filename=`` form."""
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _STRING_LITERAL_RE.finditer(line):
            value = match.group("value")
            # Must look like a disposition: either a leading
            # ``inline;`` / ``attachment;`` / ``form-data;`` prefix or
            # the legacy bare ``filename=...`` form. The bare-form
            # check is a safety net so we never silently drop a
            # regression back to the old style.
            if not (value.startswith(("inline;", "attachment;", "form-data;"))
                    or value.startswith("filename=")):
                continue
            out.append((lineno, value))
    return out


def test_all_disposition_string_literals_match_rfc2183() -> None:
    """Every ``Content-Disposition`` string literal in ``server.py``
    must match ``<type>; filename=...`` per RFC 2183.

    The legacy ``filename="<name>"`` form (no disposition type) trips
    up parsers in Go, Python's ``email``, and other tools that
    strictly follow RFC 2183 — see issue #8914.
    """
    text = SERVER_PY.read_text()
    found = _all_string_literals(text)
    assert found, "expected at least one disposition-like string literal in server.py"

    bad: list[str] = []
    for lineno, value in found:
        if not DISPOSITION_PATTERN.match(value):
            bad.append(f"server.py:{lineno} — {value!r}")
    assert not bad, (
        "Found malformed Content-Disposition string(s) in server.py:\n  "
        + "\n  ".join(bad)
    )


def test_inline_disposition_is_used_for_image_previews() -> None:
    """The ``/view`` preview paths (jpeg/webp preview, RGB channel,
    alpha channel, default-disposition) should all use ``inline`` so
    the browser renders the response rather than prompting a
    download. The dangerous-content-type branch still uses
    ``attachment`` for safety."""
    text = SERVER_PY.read_text()
    inlines = [
        lineno for lineno, value in _all_string_literals(text)
        if value.startswith("inline")
    ]
    assert len(inlines) >= 4, (
        f"expected at least 4 inline Content-Disposition strings "
        f"(3 preview branches + 1 default), found {len(inlines)}: {inlines}"
    )


def test_attachment_disposition_is_used_for_dangerous_content() -> None:
    """The dangerous-content-type branch must keep the
    ``attachment; filename="..."`` form so browsers save the file
    instead of rendering it inline. This guards against an
    accidental revert of the safe-render path."""
    text = SERVER_PY.read_text()
    attachments = [
        lineno for lineno, value in _all_string_literals(text)
        if value.startswith("attachment")
    ]
    assert len(attachments) >= 1, (
        "expected at least one attachment Content-Disposition string "
        "(the dangerous-content-type branch in /view)"
    )


def test_issue_8914_debug_log_value_is_now_rfc_compliant() -> None:
    """The exact value that #8914 reports in its debug log is no
    longer a format we emit; the same filename should now be served
    as ``inline; filename="vace14b_00003.mp4"``."""
    issue_log_value = 'filename="vace14b_00003.mp4"'  # the old, broken form
    compliant_value = 'inline; filename="vace14b_00003.mp4"'

    # The old form must not be a valid RFC 2183 disposition (it has
    # no disposition type). Asserting that here documents the
    # regression we are guarding against.
    assert not DISPOSITION_PATTERN.match(issue_log_value)
    assert DISPOSITION_PATTERN.match(compliant_value)
