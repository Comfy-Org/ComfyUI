import logging
import os
from pathlib import Path
import re

import pytest
import safetensors.torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torch

from app import governance
from app.assets.database.models import AssetContent, Base
from app.assets.services import hashing
from comfy.cli_args import LatentPreviewMethod
import app.database.db
import comfy.latent_formats
import comfy.sd1_clip
import comfy.utils
import latent_preview


GOVERNANCE_PATH = Path(governance.__file__)
EMPTY_FILE_DIGEST = "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"


@pytest.fixture(autouse=True)
def isolated_model_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(governance, "_policy", None, raising=False)
    monkeypatch.setattr(governance, "_disabled_nodes", frozenset(), raising=False)
    monkeypatch.setattr(governance, "_allowed_models", None, raising=False)
    monkeypatch.setattr(governance, "_model_digests", {}, raising=False)
    yield
    governance.set_custom_node_policy(None, frozenset(), {})


def _write_model(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({"weight": torch.full((4,), value)}, str(path))
    return path


def _apply_model_policy(*model_paths: Path) -> None:
    models = sorted({governance.model_digest(str(path)) for path in model_paths})
    governance._model_digests.clear()
    governance._apply_policy({"activeForms": ["model"], "models": models})


def _fail_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_hash(*_args, **_kwargs):
        pytest.fail("model was hashed")

    monkeypatch.setattr(hashing, "compute_blake3_hash", unexpected_hash)


def test_model_digest_is_whole_file_blake3_in_the_signed_form(tmp_path: Path) -> None:
    # Given an empty file, whose unkeyed BLAKE3 is the published test vector
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")

    # When its model digest is computed, then it is that vector in the builder's blake3:<hex> form
    assert governance.model_digest(str(empty)) == EMPTY_FILE_DIGEST


def test_listed_models_load_and_hand_copied_model_is_refused(tmp_path: Path) -> None:
    # Given a two-model allowlist and a third model copied into checkpoints by hand
    first = _write_model(tmp_path / "checkpoints" / "first.safetensors", 1.0)
    second = _write_model(tmp_path / "loras" / "second.safetensors", 2.0)
    copied = _write_model(tmp_path / "checkpoints" / "copied.safetensors", 3.0)
    _apply_model_policy(first, second)

    # When each is loaded, then the listed ones load and the third fails with the policy message
    assert torch.equal(comfy.utils.load_torch_file(str(first))["weight"], torch.full((4,), 1.0))
    assert torch.equal(comfy.utils.load_torch_file(str(second))["weight"], torch.full((4,), 2.0))
    with pytest.raises(RuntimeError, match="copied.safetensors.*organization's policy"):
        comfy.utils.load_torch_file(str(copied))


def test_renaming_keeps_the_verdict_and_replacing_bytes_changes_it(tmp_path: Path) -> None:
    # Given an allowed model
    model = _write_model(tmp_path / "checkpoints" / "allowed.safetensors", 1.0)
    _apply_model_policy(model)
    comfy.utils.load_torch_file(str(model))

    # When it is renamed, it still loads
    renamed = model.rename(model.with_name("renamed.safetensors"))
    comfy.utils.load_torch_file(str(renamed))

    # When its bytes are replaced, it is refused
    _write_model(renamed, 9.0)
    os.utime(renamed, ns=(renamed.stat().st_atime_ns, renamed.stat().st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="organization's policy"):
        comfy.utils.load_torch_file(str(renamed))


def test_active_model_form_with_no_models_refuses_every_model(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    governance._apply_policy({"activeForms": ["model"], "models": []})

    assert governance.model_allowed(str(model)) is False


def test_without_model_form_loads_are_not_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a policy whose model rules are "All" (no model form), as the builder signs it
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    governance._apply_policy({"activeForms": ["nodeId"], "disabledNodes": ["SomeNode"], "models": []})
    _fail_hashing(monkeypatch)

    # When the model loads, then nothing is hashed and nothing is refused
    assert torch.equal(comfy.utils.load_torch_file(str(model))["weight"], torch.full((4,), 1.0))


def test_repeat_load_reuses_the_digest_until_the_file_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an allowed model that has been loaded once
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    _apply_model_policy(model)
    comfy.utils.load_torch_file(str(model))
    hashed = []
    compute_blake3_hash = hashing.compute_blake3_hash
    monkeypatch.setattr(hashing, "compute_blake3_hash", lambda path: hashed.append(path) or compute_blake3_hash(path))

    # When it loads again, then it is not hashed again
    comfy.utils.load_torch_file(str(model))
    assert hashed == []

    # When its mtime moves on, then the next load hashes it again
    os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1_000_000_000))
    comfy.utils.load_torch_file(str(model))
    assert hashed == [os.path.abspath(model)]


def test_matching_database_row_does_not_let_an_unlisted_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an unlisted model and a user-writable assets database whose row claims the file has the allowed digest
    allowed = _write_model(tmp_path / "checkpoints" / "allowed.safetensors", 1.0)
    unlisted = _write_model(tmp_path / "checkpoints" / "unlisted.safetensors", 7.0)
    _apply_model_policy(allowed)
    allowed_digest = governance.model_digest(str(allowed))
    governance._model_digests.clear()
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(app.database.db, "Session", sessionmaker(bind=engine))
    stat = unlisted.stat()
    with app.database.db.create_session() as session:
        session.add(AssetContent(path=os.path.abspath(unlisted), hash=allowed_digest, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        session.commit()

    # When the unlisted model loads, then its own bytes decide and it is refused
    with pytest.raises(RuntimeError, match="unlisted.safetensors.*organization's policy"):
        comfy.utils.load_torch_file(str(unlisted))


def test_relative_and_absolute_paths_share_one_cache_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a model reached once by its absolute path and once relative to the working directory
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    monkeypatch.chdir(tmp_path)
    hashed = []
    compute_blake3_hash = hashing.compute_blake3_hash
    monkeypatch.setattr(hashing, "compute_blake3_hash", lambda path: hashed.append(path) or compute_blake3_hash(path))

    # When both are digested, then the file is hashed once and cached under one absolute key
    assert governance.model_digest(str(model)) == governance.model_digest(os.path.join("checkpoints", "model.safetensors"))
    assert hashed == [os.path.abspath(model)]
    assert [key[0] for key in governance._model_digests] == [os.path.abspath(model)]


def test_file_replaced_under_the_same_size_and_mtime_is_hashed_again(tmp_path: Path) -> None:
    # Given an allowed model that has been loaded once
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    _apply_model_policy(model)
    comfy.utils.load_torch_file(str(model))

    # When another file of the same size is moved over it with the old mtime restored
    stat = model.stat()
    swapped = _write_model(tmp_path / "swapped.safetensors", 9.0)
    assert swapped.stat().st_size == stat.st_size
    os.utime(swapped, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(swapped, model)

    # Then the new file is hashed rather than trusted from the cache, and refused
    with pytest.raises(RuntimeError, match="organization's policy"):
        comfy.utils.load_torch_file(str(model))


def test_policy_refusal_is_a_dedicated_runtime_error(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "checkpoints" / "model.safetensors", 1.0)
    governance._apply_policy({"activeForms": ["model"], "models": []})

    with pytest.raises(governance.ModelNotPermittedError) as refused:
        comfy.utils.load_torch_file(str(model))
    assert isinstance(refused.value, RuntimeError)
    assert str(refused.value) == "Model 'model.safetensors' is not permitted by your organization's policy."


def _taesd_preview_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder_name: str) -> Path:
    decoder = _write_model(tmp_path / "vae_approx" / (decoder_name + ".safetensors"), 0.0)
    monkeypatch.setattr(latent_preview.folder_paths, "get_filename_list", lambda folder: [decoder.name] if folder == "vae_approx" else [])
    monkeypatch.setattr(latent_preview.folder_paths, "get_full_path", lambda folder, name: str(decoder) if folder == "vae_approx" and name == decoder.name else None)
    monkeypatch.setattr(latent_preview.args, "preview_method", LatentPreviewMethod.TAESD)
    return decoder


@pytest.mark.parametrize("latent_format", [comfy.latent_formats.SD15, comfy.latent_formats.HunyuanVideo])
def test_taesd_preview_falls_back_to_latent2rgb_when_the_decoder_is_not_listed(
    latent_format, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Given TAESD previews and a model policy that does not list the TAESD decoder
    fmt = latent_format()
    _taesd_preview_setup(tmp_path, monkeypatch, fmt.taesd_decoder_name)
    governance._apply_policy({"activeForms": ["model"], "models": ["blake3:" + "a" * 64]})

    # When a sampler builds its previewer, then it gets the latent2rgb preview and one warning, not an error
    with caplog.at_level(logging.WARNING):
        previewer = latent_preview.get_previewer(torch.device("cpu"), fmt)
    assert isinstance(previewer, latent_preview.Latent2RGBPreviewer)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "organization's policy" in warnings[0].getMessage()


def test_taesd_preview_still_raises_on_errors_other_than_a_policy_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a listed TAESD decoder file whose weights do not fit the decoder
    fmt = comfy.latent_formats.SD15()
    decoder = _taesd_preview_setup(tmp_path, monkeypatch, fmt.taesd_decoder_name)
    _apply_model_policy(decoder)

    # When the previewer is built, then the load error surfaces as before
    with pytest.raises(RuntimeError) as failed:
        latent_preview.get_previewer(torch.device("cpu"), fmt)
    assert not isinstance(failed.value, governance.ModelNotPermittedError)


def test_unlisted_embedding_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    embeddings = tmp_path / "embeddings"
    allowed = _write_model(embeddings / "allowed.safetensors", 1.0)
    _write_model(embeddings / "copied.safetensors", 2.0)
    _apply_model_policy(allowed)

    assert comfy.sd1_clip.load_embed("allowed", str(embeddings), 4) is not None
    with caplog.at_level(logging.WARNING):
        assert comfy.sd1_clip.load_embed("copied", str(embeddings), 4) is None
    assert "Embedding 'copied' is not permitted by your organization's policy." in caplog.text
    assert "error loading embedding" not in caplog.text


def test_unreadable_embedding_is_skipped_with_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Given a model policy and an embedding the process cannot read
    embeddings = tmp_path / "embeddings"
    _write_model(embeddings / "locked.safetensors", 1.0)
    governance._apply_policy({"activeForms": ["model"], "models": []})

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(hashing, "compute_blake3_hash", unreadable)

    # When it is loaded, then it is skipped with the broken-embedding warning, as without a policy
    with caplog.at_level(logging.WARNING):
        assert comfy.sd1_clip.load_embed("locked", str(embeddings), 4) is None
    assert "error loading embedding, skipping loading: locked" in caplog.text
    assert "organization's policy" not in caplog.text


def test_enforced_forms_line_is_readable_without_running_python() -> None:
    # Given the one top-level _ENFORCED_FORMS line, read the way the builder reads
    # GOVERNANCE_* lines: unindented, name and value split at the first "="
    lines = [line for line in GOVERNANCE_PATH.read_text(encoding="utf-8").splitlines() if line[:1] not in ("", " ", "\t")]
    values = [line.split("=", 1)[1] for line in lines if "=" in line and line.split("=", 1)[0].strip() == "_ENFORCED_FORMS"]

    # Then its quoted names are exactly the forms this build enforces, model included
    assert len(values) == 1
    assert frozenset(re.findall(r'"([A-Za-z]+)"', values[0])) == governance._ENFORCED_FORMS
    assert "model" in governance._ENFORCED_FORMS
