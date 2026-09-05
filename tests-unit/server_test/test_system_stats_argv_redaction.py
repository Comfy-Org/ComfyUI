"""CI unit guard for redacting sys.argv in the /system_stats endpoint.

The /system_stats endpoint (server.py, PromptServer.add_routes -> system_stats)
returns "argv": sys.argv verbatim in its JSON response. That endpoint has no
authentication, so any client that can reach it -- including a page open in a
browser tab, per ComfyUI's threat model for local-network deployments -- could
read back the full raw command line, including values passed to path-bearing
flags such as --extra-model-paths-config, --output-directory, --database-url,
etc. Those values commonly contain usernames, internal directory layouts, or
otherwise-private filesystem paths that have nothing to do with diagnostics.

The fix is comfy.cli_args.redact_sensitive_argv(), which replaces only the
*values* that follow a known path-bearing flag with "*", leaving every flag
name (and every non-path argument) untouched. server.py calls this helper
instead of exposing sys.argv directly.

server.py cannot be imported in a unit test (importing it pulls in nodes/torch
and spins up the full PromptServer/aiohttp app), so -- following the existing
tests-unit/security_test/test_ghsa_779p_05_dangerous_content_types.py pattern
-- this file tests the redaction helper directly rather than the route.

Preserving the flag names (not just truncating argv to [sys.argv[0]]) matters
because the frontend's system-stats panel and "Copy System Info" support
feature (ComfyUI_frontend's useCopySystemInfo.ts /
systemStatsColumns.ts) both render the full `argv` array for legitimate
debugging/bug-report purposes; blanking it out entirely would regress that.
"""

from comfy.cli_args import (
    SENSITIVE_ARGV_FLAGS,
    SENSITIVE_ARGV_MULTI_FLAGS,
    redact_sensitive_argv,
)


def test_no_sensitive_flags_is_unchanged():
    argv = ["main.py", "--cpu", "--listen", "0.0.0.0", "--port", "8188"]
    assert redact_sensitive_argv(argv) == argv


def test_single_value_flag_value_is_redacted():
    """The exact repro from the issue report."""
    argv = [
        "main.py",
        "--extra-model-paths-config",
        r"D:\Private\models.yaml",
        "--output-directory",
        r"D:\Confidential\Renders",
    ]
    redacted = redact_sensitive_argv(argv)
    assert redacted == [
        "main.py",
        "--extra-model-paths-config",
        "*",
        "--output-directory",
        "*",
    ]
    # None of the original path fragments survive.
    joined = " ".join(redacted)
    assert "Private" not in joined
    assert "Confidential" not in joined


def test_flag_names_are_preserved_for_frontend_display():
    """Flag names must stay intact -- the frontend's system-stats panel and
    "Copy System Info" feature display which flags were passed."""
    argv = ["main.py", "--base-directory", "/home/alice/comfy"]
    redacted = redact_sensitive_argv(argv)
    assert redacted[0] == "main.py"
    assert "--base-directory" in redacted
    assert "alice" not in " ".join(redacted)


def test_equals_syntax_is_redacted():
    argv = ["main.py", "--database-url=sqlite:////home/alice/user/comfyui.db"]
    redacted = redact_sensitive_argv(argv)
    assert redacted == ["main.py", "--database-url=*"]


def test_multi_value_flag_redacts_every_value_up_to_next_flag():
    """--extra-model-paths-config uses argparse nargs='+', so it can take
    several path values before the next flag."""
    argv = [
        "main.py",
        "--extra-model-paths-config",
        "/a/extra1.yaml",
        "/a/extra2.yaml",
        "--cpu",
    ]
    redacted = redact_sensitive_argv(argv)
    assert redacted == ["main.py", "--extra-model-paths-config", "*", "--cpu"]


def test_flag_with_missing_value_is_not_corrupted():
    """A malformed/truncated argv (flag as the last token) must not raise or
    swallow adjacent tokens -- just leave it as-is."""
    argv = ["main.py", "--tls-keyfile"]
    assert redact_sensitive_argv(argv) == ["main.py", "--tls-keyfile"]


def test_flag_with_missing_value_does_not_swallow_the_next_flag():
    """If a sensitive single-value flag has no value because another flag
    immediately follows it, that following flag must survive untouched --
    not be consumed and replaced with "*"."""
    argv = ["main.py", "--tls-keyfile", "--cpu"]
    assert redact_sensitive_argv(argv) == ["main.py", "--tls-keyfile", "--cpu"]


def test_multi_value_flag_equals_syntax_is_redacted():
    """--extra-model-paths-config=/path.yaml (argparse also accepts "=" for
    nargs='+' flags with a single value) must be redacted like the
    single-value flags' equals syntax already is."""
    argv = ["main.py", "--extra-model-paths-config=/home/alice/extra.yaml"]
    redacted = redact_sensitive_argv(argv)
    assert redacted == ["main.py", "--extra-model-paths-config=*"]
    assert "alice" not in " ".join(redacted)


def test_non_path_arguments_are_left_alone():
    argv = ["main.py", "--multi-user", "--fast", "--preview-method", "auto"]
    assert redact_sensitive_argv(argv) == argv


def test_empty_argv_is_handled():
    assert redact_sensitive_argv([]) == []


def test_original_argv_is_not_mutated():
    argv = ["main.py", "--output-directory", "/secret/out"]
    original = list(argv)
    redact_sensitive_argv(argv)
    assert argv == original


def test_all_documented_sensitive_flags_are_redacted():
    """Every flag in SENSITIVE_ARGV_FLAGS/SENSITIVE_ARGV_MULTI_FLAGS must
    actually get its value replaced -- guards against the set growing without
    the redaction logic being exercised for it."""
    for flag in SENSITIVE_ARGV_FLAGS:
        argv = ["main.py", flag, "/some/private/path"]
        redacted = redact_sensitive_argv(argv)
        assert redacted == ["main.py", flag, "*"], flag
        assert "private" not in " ".join(redacted)

    for flag in SENSITIVE_ARGV_MULTI_FLAGS:
        argv = ["main.py", flag, "/some/private/path.yaml"]
        redacted = redact_sensitive_argv(argv)
        assert redacted == ["main.py", flag, "*"], flag
