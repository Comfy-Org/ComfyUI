"""Upload content reuse is scoped to the folder the upload is placed in."""

import os
import uuid

import pytest

import app.assets.mode as mode_module
import folder_paths
from app.assets.services.ingest import upload_from_temp_path

DATA = b"folder-scoped-reuse"
TAGS = {
    "input": ["input"],
    "output": ["output"],
    "checkpoints": ["models", "model_type:checkpoints"],
}


@pytest.fixture
def setup(mock_create_session, roots, monkeypatch):
    monkeypatch.setattr(folder_paths, "temp_directory", str(roots / "temp"))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {"checkpoints": ([str(roots / "models" / "checkpoints")], {".bin"})})

    def hashing(enabled: bool):
        mode_module.init(type("Args", (), {"enable_asset_hashing": enabled})())

    yield roots, hashing
    mode_module.init(None)


def upload(roots, folder, reuse=True):
    staged = os.path.join(roots, "temp", "uploads", uuid.uuid4().hex, ".upload.part")
    os.makedirs(os.path.dirname(staged))
    with open(staged, "wb") as f:
        f.write(DATA)
    return upload_from_temp_path(temp_path=staged, name="a.bin", client_filename="a.bin", tags=TAGS[folder], reuse_content=reuse)


def stored(roots):
    return sorted(str(p.relative_to(roots)) for p in roots.rglob("*.bin"))


@pytest.mark.parametrize("folder", ["input", "checkpoints"])
def test_same_folder_reuses(setup, folder):
    roots, hashing = setup
    hashing(True)
    first, again = upload(roots, folder), upload(roots, folder)
    assert (first.content_reused, again.content_reused) == (False, True)
    assert again.ref.file_path == first.ref.file_path
    assert len(stored(roots)) == 1


@pytest.mark.parametrize(
    "first,second",
    [("input", "output"), ("output", "input"), ("input", "checkpoints"), ("checkpoints", "input"), ("output", "checkpoints")],
)
def test_other_folder_stores_a_new_copy(setup, first, second):
    roots, hashing = setup
    hashing(True)
    earlier, later = upload(roots, first), upload(roots, second)
    assert later.content_reused is False
    dest = os.path.abspath(folder_paths.get_folder_paths("checkpoints")[0] if second == "checkpoints" else roots / second)
    assert os.path.dirname(later.ref.file_path) == dest
    assert os.path.isfile(earlier.ref.file_path) and os.path.isfile(later.ref.file_path)
    assert len(stored(roots)) == 2


def test_hashing_off_never_reuses(setup):
    roots, hashing = setup
    hashing(False)
    first = upload(roots, "input", reuse=False)
    again = upload(roots, "input", reuse=False)
    assert (first.content_reused, again.content_reused) == (False, False)


def test_tagless_upload_still_reuses_any_folder(setup):
    roots, hashing = setup
    hashing(True)
    earlier = upload(roots, "output")
    staged = os.path.join(roots, "temp", "t.part")
    with open(staged, "wb") as f:
        f.write(DATA)
    tagless = upload_from_temp_path(temp_path=staged, name="a.bin")
    assert tagless.content_reused is True and tagless.ref.file_path == earlier.ref.file_path
