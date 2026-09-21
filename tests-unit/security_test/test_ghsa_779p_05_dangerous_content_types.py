"""CI unit guard for FIX #5 of GHSA-779p-m5rp-r4h4 — the /view forced-download set.

Vuln #5 was stored XSS via SVG upload: the /view endpoint's Content-Type
blocklist covered text/html, text/javascript, etc. but was missing
image/svg+xml, so an uploaded SVG carrying an inline <script> was served as
image/svg+xml and executed in the page origin when rendered.

The /view forced-download decision lives in the view_image closure registered by
server.PromptServer.add_routes (server.py ~line 596), which calls
`folder_paths.is_dangerous_content_type(content_type)` — a normalising check that
strips charset/boundary parameters and casing and folds in the whole */xml and
*+xml dialect family — rather than a bypassable raw
`content_type in folder_paths.DANGEROUS_CONTENT_TYPES` membership test. On a match
it rewrites the response to application/octet-stream with a
Content-Disposition: attachment header. server.py cannot be imported in a unit
test (importing it spins up the full PromptServer/aiohttp app and its global side
effects), so these tests pin the underlying dangerous-content data
(folder_paths.DANGEROUS_CONTENT_TYPES) and the normalising is_dangerous_content_type()
helper that the closure actually calls.

The end-to-end /view assertion (upload an SVG, GET /view, confirm the response
is not served as image/svg+xml) lives in the live POC at
.security/pocs/test_security_ghsa_779p.py::TestViewSvgContentType, which
requires a running server. This file is the fast, server-free CI guard on the
set contents so the blocklist can't silently regress.
"""

import folder_paths


# Active/renderable content types that must be forced to download. Each of these
# can carry an inline <script> (or otherwise execute) in the page origin if a
# browser renders it. image/svg+xml is the original missing item that caused
# vuln #5.
DANGEROUS = [
    'image/svg+xml',
    'application/xml',
    'text/xml',
    'text/html',
    'text/html-sandboxed',
    'application/xhtml+xml',
    'text/javascript',
    'application/javascript',
    'application/x-javascript',
    'application/ecmascript',
    'text/css',
]

# Benign image types that browsers display inline and that must keep rendering;
# forcing these to download would break legitimate previews.
BENIGN_INLINE_IMAGES = [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/gif',
]


def test_dangerous_content_types_is_a_set():
    assert isinstance(folder_paths.DANGEROUS_CONTENT_TYPES, set)


def test_svg_is_in_the_blocklist():
    """The specific item whose absence caused vuln #5."""
    assert 'image/svg+xml' in folder_paths.DANGEROUS_CONTENT_TYPES, (
        "image/svg+xml missing from DANGEROUS_CONTENT_TYPES — this is exactly "
        "the regression that reopens GHSA-779p-m5rp-r4h4 vuln #5 (stored XSS "
        "via SVG upload on /view)."
    )


def test_all_dangerous_types_present():
    missing = [ct for ct in DANGEROUS if ct not in folder_paths.DANGEROUS_CONTENT_TYPES]
    assert not missing, (
        f"DANGEROUS_CONTENT_TYPES is missing required active/renderable types: "
        f"{missing}. The /view closure only forces a download for content types "
        f"in this set; anything missing here is served inline and can execute."
    )


def test_benign_inline_image_types_absent():
    leaked = [ct for ct in BENIGN_INLINE_IMAGES if ct in folder_paths.DANGEROUS_CONTENT_TYPES]
    assert not leaked, (
        f"Benign inline-displayable image types found in DANGEROUS_CONTENT_TYPES: "
        f"{leaked}. Forcing these to download would break legitimate image "
        f"previews in /view — they must keep rendering inline."
    )


# ---------------------------------------------------------------------------
# is_dangerous_content_type() — the normalising check the /view and /userdata
# handlers now call instead of a raw `in DANGEROUS_CONTENT_TYPES` membership
# test. An exact-string membership test was bypassable with a charset parameter
# or odd casing, and missed the wider XML dialect family; these tests pin the
# normalisation so that bypass can't reopen.
# ---------------------------------------------------------------------------

def test_function_matches_plain_dangerous_types():
    for ct in DANGEROUS:
        assert folder_paths.is_dangerous_content_type(ct) is True, ct


def test_function_strips_parameters_and_casing():
    """A charset/boundary parameter or casing must not slip a type past the check.

    This is the bypass surfaced by review: the /view blake3 branch can serve an
    attacker-controlled, unvalidated asset mime_type like 'text/html; charset=utf-8',
    which an exact-string set test missed.
    """
    for ct in (
        'text/html; charset=utf-8',
        'TEXT/HTML',
        'Text/HTML; charset=UTF-8',
        'image/svg+xml; charset=utf-8',
        '  text/html  ',
    ):
        assert folder_paths.is_dangerous_content_type(ct) is True, ct


def test_function_covers_xml_dialect_family():
    """Any *+xml / */xml dialect is dangerous without enumerating each one."""
    for ct in (
        'application/xslt+xml',
        'application/rss+xml',
        'application/atom+xml',
        'application/rdf+xml',
        'application/mathml+xml',
        'message/rfc822',
    ):
        assert folder_paths.is_dangerous_content_type(ct) is True, ct


def test_function_allows_benign_and_empty():
    for ct in BENIGN_INLINE_IMAGES + ['application/octet-stream', 'text/plain']:
        assert folder_paths.is_dangerous_content_type(ct) is False, ct
    # None / empty (mimetypes.guess_type miss) must not be treated as dangerous.
    assert folder_paths.is_dangerous_content_type(None) is False
    assert folder_paths.is_dangerous_content_type('') is False
