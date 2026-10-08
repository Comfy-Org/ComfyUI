from pathlib import Path
import sys

import pytest

from app import governance


GOLDEN_DIGEST = "blake3:ee9d4961b928fa3c0983f95fe2123d44d7cbd44158859d77ff31f7269b68cdba"
SINGLE_FILE_DIGEST = "blake3:5a98145e4f501a639d74720737b5bf4ad075a2ba8befb831d231e32cca56499d"
GOLDEN_FILES = {
    "__init__.py": b"ROOT = 1\n",
    "native/WINDOWS.PYD": b"\x00pyd\xff\n",
    "nested/cafe\u0301.py": b"UNICODE = 'cafe'\n",
    "nested/foo.cpython-311-x86_64-linux-gnu.so": b"\x7fELFfixture-so\n",
    "nested/module.py": b"VALUE = 'nested'\n",
    "README.txt": b"ignored text\n",
    "models/model.safetensors": b"ignored weights\n",
}


@pytest.fixture
def golden_pack(tmp_path: Path) -> Path:
    pack_path = tmp_path / "pack"
    for relative_path, contents in GOLDEN_FILES.items():
        file_path = pack_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(contents)
    return pack_path


def test_pack_digest_matches_cross_language_golden_fixture(golden_pack: Path) -> None:
    # Given the committed tree from golden_pack
    # When its pack digest is computed
    digest = governance.pack_digest(str(golden_pack))

    # Then the byte-exact cross-language vector matches
    assert digest == GOLDEN_DIGEST


def test_pack_digest_for_single_file_uses_basename_record(golden_pack: Path) -> None:
    # Given a top-level Python file
    single_file = golden_pack / "__init__.py"

    # When it is measured as a pack
    digest = governance.pack_digest(str(single_file))

    # Then exactly one basename record is encoded
    assert digest == SINGLE_FILE_DIGEST


def test_pack_digest_changes_when_included_file_changes(golden_pack: Path) -> None:
    # Given a golden pack whose nested Python file changes
    module_path = golden_pack / "nested" / "module.py"
    module_path.write_bytes(module_path.read_bytes() + b"# changed\n")

    # When its digest is recomputed
    digest = governance.pack_digest(str(golden_pack))

    # Then the included byte changes the pack identity
    assert digest != GOLDEN_DIGEST


def test_pack_digest_ignores_changes_to_excluded_file(golden_pack: Path) -> None:
    # Given a golden pack whose text file changes
    text_path = golden_pack / "README.txt"
    text_path.write_bytes(text_path.read_bytes() + b"changed\n")

    # When its digest is recomputed
    digest = governance.pack_digest(str(golden_pack))

    # Then the excluded byte does not change the pack identity
    assert digest == GOLDEN_DIGEST


@pytest.mark.parametrize("relative_path", ["__pycache__/root.cpython-311.pyc", "nested/module.pyc", "legacy.pyo"])
def test_pack_digest_rejects_compiled_bytecode(golden_pack: Path, relative_path: str) -> None:
    # Given a pack carrying bytecode that can execute without its source
    bytecode_path = golden_pack / relative_path
    bytecode_path.parent.mkdir(parents=True, exist_ok=True)
    bytecode_path.write_bytes(b"compiled\n")

    # When the pack is measured, then the unmeasurable artifact rejects the whole pack
    with pytest.raises(ValueError):
        governance.pack_digest(str(golden_pack))


def test_pack_digest_rejects_normalized_relative_path_collision(tmp_path: Path) -> None:
    # Given distinct filesystem entries that normalize to the same NFC path
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    (pack_path / "cafe\u0301.py").write_bytes(b"decomposed\n")
    (pack_path / "caf\u00e9.py").write_bytes(b"composed\n")
    if len(list(pack_path.iterdir())) < 2:
        pytest.skip("filesystem normalizes Unicode filenames")

    # When the pack is measured, then the ambiguous record name is rejected
    with pytest.raises(ValueError):
        governance.pack_digest(str(pack_path))


def test_pack_digest_rejects_symlinked_python_file(tmp_path: Path) -> None:
    # Given a pack containing a symlinked Python file
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    target = tmp_path / "outside.py"
    target.write_bytes(b"outside\n")
    (pack_path / "linked.py").symlink_to(target)

    # When the pack is measured, then the symlink rejects the whole pack
    with pytest.raises(ValueError):
        governance.pack_digest(str(pack_path))


def test_pack_digest_rejects_symlinked_directory(tmp_path: Path) -> None:
    # Given a pack containing a symlinked directory
    pack_path = tmp_path / "pack"
    pack_path.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (pack_path / "linked").symlink_to(target, target_is_directory=True)

    # When the pack is measured, then the symlink rejects the whole pack
    with pytest.raises(ValueError):
        governance.pack_digest(str(pack_path))


def test_pack_digest_rejects_symlinked_pack_root(tmp_path: Path) -> None:
    # Given a pack root that is itself a symlink
    target = tmp_path / "real-pack"
    target.mkdir()
    pack_path = tmp_path / "linked-pack"
    pack_path.symlink_to(target, target_is_directory=True)

    # When the pack is measured, then the root symlink is rejected
    with pytest.raises(ValueError):
        governance.pack_digest(str(pack_path))


# Windows drops a trailing dot from a file name, so "lib/libbar.so." is written as "lib/libbar.so", a measured library.
TRAILING_DOT_NAMES_KEPT = pytest.mark.skipif(sys.platform == "win32", reason="Windows strips a trailing dot from file names")


# Shared with Comfy-Org/cloud services/comfy-builder/packdigest/digest_test.go (sharedFixtureFiles), which asserts the same digest.
SHARED_FIXTURE_DIGEST = "blake3:60627b4c95dad13ffe3dd68e165ac182c605ab1f873a88d11c3b9c0b65d531e1"
SHARED_FIXTURE_FILES = {
    "__init__.py": b"ROOT = 1\n",
    "utils/__init__.pyw": b"WINDOWS_SOURCE = 1\n",
    "web/x.js": b"export const X = 1;\n",
    "web/x.mjs": b"export const Y = 2;\n",
    "lib/libfoo.so.1": b"\x7fELFversioned-so\n",
    "README.md": b"# ignored readme\n",
    "data/config.json": b'{"ignored": true}\n',
}


# The shared fixture plus names where suffix rules could split: Python's pathlib gives a dotfile ("web/.js") no suffix where
# Go's filepath.Ext would give ".js". Only UPPER.PY and Libs/LibCase.SO.2 are measured. Libs/ is not lib/ so the fixture
# also holds on a case-insensitive filesystem. cloud's digest_test.go (edgeNameFixtureFiles) asserts the same digest.
EDGE_NAME_FIXTURE_DIGEST = "blake3:1f34f2d453c9ad9d2aa5302fc0b3332bc60b88077bfd1841e48d6102aa23968a"
EDGE_NAME_FIXTURE_FILES = {
    **SHARED_FIXTURE_FILES,
    "web/.js": b"DOTFILE_JS = 1\n",
    "pkg/.py": b"DOTFILE_PY = 1\n",
    "lib/.so.1": b"\x7fELFdotfile-so\n",
    "lib/libbar.so.": b"\x7fELFtrailing-dot\n",
    "lib/libt.so.1a": b"\x7fELFnot-a-version\n",
    "web/v.js.map": b'{"version": 3}\n',
    "UPPER.PY": b"UPPER = 1\n",
    "Libs/LibCase.SO.2": b"\x7fELFupper-so\n",
}


def _write_pack(pack_path: Path, files: dict[str, bytes]) -> Path:
    for relative_path, contents in files.items():
        file_path = pack_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(contents)
    return pack_path


def test_pack_digest_matches_shared_fixture_with_windows_source_web_and_versioned_library(tmp_path: Path) -> None:
    # Given the fixture cloud's packdigest also measures
    pack_path = _write_pack(tmp_path / "pack", SHARED_FIXTURE_FILES)

    # When its pack digest is computed
    digest = governance.pack_digest(str(pack_path))

    # Then it is the digest both implementations assert
    assert digest == SHARED_FIXTURE_DIGEST


@TRAILING_DOT_NAMES_KEPT
def test_pack_digest_matches_shared_fixture_with_edge_names(tmp_path: Path) -> None:
    # Given the shared fixture plus dotfiles, trailing dots, non-numeric versions and upper-case suffixes
    pack_path = _write_pack(tmp_path / "pack", EDGE_NAME_FIXTURE_FILES)

    # When its pack digest is computed
    digest = governance.pack_digest(str(pack_path))

    # Then it is the digest both implementations assert
    assert digest == EDGE_NAME_FIXTURE_DIGEST


@pytest.mark.parametrize(
    "relative_path",
    [
        "extra.pyw",
        "web/extra.js",
        "web/extra.mjs",
        "native/EXTRA.JS",
        "lib/libbar.so.1",
        "lib/libbar.so.1.2",
        "lib/LIBBAR.SO.3",
        "native/extra.dll",
        "native/extra.dylib",
    ],
)
def test_pack_digest_measures_source_native_and_web_script_files(golden_pack: Path, relative_path: str) -> None:
    # Given a golden pack gaining Python source, a web script, or a native library
    _write_pack(golden_pack, {relative_path: b"RUNS = 1\n"})

    # When its digest is recomputed
    digest = governance.pack_digest(str(golden_pack))

    # Then the new file changes the pack identity
    assert digest != GOLDEN_DIGEST


@pytest.mark.parametrize(
    "relative_path",
    ["README.md", "data/extra.json", "lib/libbar.so.1a", pytest.param("lib/libbar.so.", marks=TRAILING_DOT_NAMES_KEPT), "lib/.so.1", "web/x.js.map"],
)
def test_pack_digest_ignores_files_that_do_not_run(golden_pack: Path, relative_path: str) -> None:
    # Given a golden pack gaining a data file, or a name that only resembles a versioned library
    _write_pack(golden_pack, {relative_path: b"DATA = 1\n"})

    # When its digest is recomputed
    digest = governance.pack_digest(str(golden_pack))

    # Then the pack identity is unchanged
    assert digest == GOLDEN_DIGEST
