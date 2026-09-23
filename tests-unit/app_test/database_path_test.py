import os

from app.database import db
from comfy.cli_args import database_default_path


def test_default_database_url_uses_effective_user_directory(monkeypatch, tmp_path):
    user_dir = tmp_path / "custom_user"
    user_dir.mkdir()

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr("folder_paths.get_user_directory", lambda: str(user_dir))

    assert db.get_database_url() == f"sqlite:///{user_dir / 'comfyui.db'}"


def test_default_db_path_matches_legacy_default_without_custom_user_directory(monkeypatch):
    import folder_paths

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr(
        "folder_paths.get_user_directory",
        lambda: os.path.join(folder_paths.base_path, "user"),
    )

    assert os.path.abspath(db.get_db_path()) == database_default_path


def test_explicit_database_url_is_preserved(monkeypatch):
    database_url = "sqlite:///:memory:"

    monkeypatch.setattr(db.args, "database_url", database_url)

    assert db.get_database_url() == database_url


def test_explicit_url_at_legacy_default_location_is_honoured(monkeypatch):
    url = f"sqlite:///{database_default_path}"

    monkeypatch.setattr(db.args, "database_url", url)

    assert db.get_database_url() == url
    assert db.get_db_path() == database_default_path


def test_explicit_database_url_skips_legacy_copy(monkeypatch, tmp_path):
    legacy_db = tmp_path / "install" / "user" / "comfyui.db"
    target_db = tmp_path / "target" / "comfyui.db"
    legacy_db.parent.mkdir(parents=True)
    target_db.parent.mkdir(parents=True)
    legacy_db.write_bytes(b"legacy db")

    monkeypatch.setattr(db.args, "database_url", f"sqlite:///{target_db}")
    monkeypatch.setattr(db, "get_legacy_default_db_path", lambda: str(legacy_db))

    db.copy_legacy_default_db(str(target_db))

    assert not target_db.exists()


def test_legacy_default_database_is_copied_to_effective_user_directory(monkeypatch, tmp_path):
    legacy_db = tmp_path / "install" / "user" / "comfyui.db"
    user_dir = tmp_path / "custom_user"
    legacy_db.parent.mkdir(parents=True)
    user_dir.mkdir()
    legacy_db.write_bytes(b"legacy db")

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr("folder_paths.get_user_directory", lambda: str(user_dir))
    monkeypatch.setattr(db, "get_legacy_default_db_path", lambda: str(legacy_db))

    db.copy_legacy_default_db(str(user_dir / "comfyui.db"))

    assert (user_dir / "comfyui.db").read_bytes() == b"legacy db"
    assert not legacy_db.exists()
    assert legacy_db.with_suffix(".db.bak").read_bytes() == b"legacy db"


def test_legacy_copy_skipped_when_backup_already_exists(monkeypatch, tmp_path):
    legacy_db = tmp_path / "install" / "user" / "comfyui.db"
    backup_db = tmp_path / "install" / "user" / "comfyui.db.bak"
    user_db = tmp_path / "custom_user" / "comfyui.db"
    legacy_db.parent.mkdir(parents=True)
    user_db.parent.mkdir(parents=True)
    legacy_db.write_bytes(b"live db")
    backup_db.write_bytes(b"old backup")

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr(db, "get_legacy_default_db_path", lambda: str(legacy_db))

    db.copy_legacy_default_db(str(user_db))

    assert not user_db.exists()
    assert legacy_db.read_bytes() == b"live db"
    assert backup_db.read_bytes() == b"old backup"


def test_legacy_default_database_does_not_overwrite_existing_effective_db(monkeypatch, tmp_path):
    legacy_db = tmp_path / "install" / "user" / "comfyui.db"
    user_db = tmp_path / "custom_user" / "comfyui.db"
    legacy_db.parent.mkdir(parents=True)
    user_db.parent.mkdir(parents=True)
    legacy_db.write_bytes(b"legacy db")
    user_db.write_bytes(b"user db")

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr(db, "get_legacy_default_db_path", lambda: str(legacy_db))

    db.copy_legacy_default_db(str(user_db))

    assert user_db.read_bytes() == b"user db"
    assert legacy_db.read_bytes() == b"legacy db"


def test_prepare_file_database_creates_parent_directory(monkeypatch, tmp_path):
    db_path = tmp_path / "nested" / "comfyui.db"

    monkeypatch.setattr(db.args, "database_url", None)
    monkeypatch.setattr(db, "copy_legacy_default_db", lambda path: None)

    db.prepare_file_db_path(str(db_path))

    assert db_path.parent.is_dir()


def test_prepare_file_database_accepts_relative_database_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db.args, "database_url", "sqlite:///relative.db")
    monkeypatch.setattr(db, "copy_legacy_default_db", lambda path: None)

    db.prepare_file_db_path("relative.db")

    assert os.getcwd() == str(tmp_path)
    assert not list(tmp_path.iterdir())
