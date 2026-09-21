"""Security tests for GHSA-779p-m5rp-r4h4 — FIX #3.

Path traversal in folder_paths.get_annotated_filepath / exists_annotated_filepath,
plus the shared is_within_directory() containment helper.

These are pure-function tests (no running server). The input/output/temp
directories are pointed at tmp_path via the folder_paths setters, so a crafted
name containing `../`, an absolute path, or a symlink that escapes the base
directory must be rejected.

Reference: https://github.com/Comfy-Org/ComfyUI/security/advisories/GHSA-779p-m5rp-r4h4
"""
import os

import pytest

import folder_paths
from comfy.options import enable_args_parsing
enable_args_parsing()


@pytest.fixture
def sandbox(tmp_path):
    """Point folder_paths' input/output/temp dirs at a real temp sandbox.

    Yields the realpath'd base, input, output and temp directories. The original
    directory values are restored afterward so tests stay isolated.
    """
    base = os.path.realpath(str(tmp_path))
    input_dir = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    temp_dir = os.path.join(base, "temp")
    for d in (input_dir, output_dir, temp_dir):
        os.makedirs(d, exist_ok=True)

    orig_input = folder_paths.get_input_directory()
    orig_output = folder_paths.get_output_directory()
    orig_temp = folder_paths.get_temp_directory()

    folder_paths.set_input_directory(input_dir)
    folder_paths.set_output_directory(output_dir)
    folder_paths.set_temp_directory(temp_dir)

    yield {
        "base": base,
        "input": input_dir,
        "output": output_dir,
        "temp": temp_dir,
    }

    folder_paths.set_input_directory(orig_input)
    folder_paths.set_output_directory(orig_output)
    folder_paths.set_temp_directory(orig_temp)


# ---------------------------------------------------------------------------
# is_within_directory() — the shared containment helper
# ---------------------------------------------------------------------------

def test_is_within_directory_legit_child(sandbox):
    base = sandbox["input"]
    child = os.path.join(base, "sub", "image.png")
    assert folder_paths.is_within_directory(base, child) is True


def test_is_within_directory_dotdot_escape(sandbox):
    base = sandbox["input"]
    escape = os.path.join(base, "..", "..", "etc", "passwd")
    assert folder_paths.is_within_directory(base, escape) is False


def test_is_within_directory_symlink_escape(sandbox):
    """A symlink created INSIDE base that points OUTSIDE base must not pass.

    This is the key new hardening: is_within_directory realpath()s both operands,
    so a symlink planted in the base directory can't be used to read files
    elsewhere. We create a real on-disk symlink and a real secret target to
    verify the check actually resolves the link.
    """
    base = sandbox["input"]

    # A directory living outside the base, holding a secret file.
    outside = os.path.join(sandbox["base"], "outside_secret_dir")
    os.makedirs(outside, exist_ok=True)
    secret = os.path.join(outside, "secret.txt")
    with open(secret, "w") as f:
        f.write("top secret")

    # Plant a symlink inside base that points at the outside directory.
    # symlink creation can require elevated privileges / Developer Mode on
    # Windows, so skip cleanly where it isn't available (same guard as the
    # sibling test in test_ghsa_779p_02_preview_traversal.py).
    link = os.path.join(base, "escape_link")
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform/filesystem")

    # Accessing the secret "through" the in-base symlink must be rejected.
    target_via_link = os.path.join(link, "secret.txt")
    assert folder_paths.is_within_directory(base, target_via_link) is False


# ---------------------------------------------------------------------------
# get_annotated_filepath()
# ---------------------------------------------------------------------------

def test_get_annotated_filepath_legit_name(sandbox):
    result = folder_paths.get_annotated_filepath("image.png")
    assert result == os.path.join(sandbox["input"], "image.png")
    assert folder_paths.is_within_directory(sandbox["input"], result)


def test_get_annotated_filepath_input_annotation(sandbox):
    result = folder_paths.get_annotated_filepath("image.png [input]")
    assert result == os.path.join(sandbox["input"], "image.png")


def test_get_annotated_filepath_output_annotation(sandbox):
    result = folder_paths.get_annotated_filepath("image.png [output]")
    assert result == os.path.join(sandbox["output"], "image.png")


def test_get_annotated_filepath_temp_annotation(sandbox):
    result = folder_paths.get_annotated_filepath("image.png [temp]")
    assert result == os.path.join(sandbox["temp"], "image.png")


def test_get_annotated_filepath_dotdot_raises(sandbox):
    with pytest.raises(ValueError):
        folder_paths.get_annotated_filepath("../etc/passwd")


def test_get_annotated_filepath_dotdot_with_annotation_raises(sandbox):
    with pytest.raises(ValueError):
        folder_paths.get_annotated_filepath("../../etc/passwd [output]")


def test_get_annotated_filepath_absolute_escape_raises(sandbox):
    with pytest.raises(ValueError):
        folder_paths.get_annotated_filepath("/etc/passwd")


# ---------------------------------------------------------------------------
# exists_annotated_filepath()
# ---------------------------------------------------------------------------

def test_exists_annotated_filepath_existing_legit_file(sandbox):
    real = os.path.join(sandbox["input"], "real.png")
    with open(real, "w") as f:
        f.write("data")
    assert folder_paths.exists_annotated_filepath("real.png") is True


def test_exists_annotated_filepath_traversal_returns_false(sandbox):
    """A traversal name must return False without raising and without probing
    outside the base directory (must never reach os.path.exists for the escape).
    """
    # /etc/passwd exists on POSIX; the function must still report False because
    # the resolved path escapes the input directory.
    assert folder_paths.exists_annotated_filepath("../../../../../../etc/passwd") is False


def test_exists_annotated_filepath_absolute_returns_false(sandbox):
    assert folder_paths.exists_annotated_filepath("/etc/passwd") is False
