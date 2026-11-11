import pytest


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """The real file database: WAL mode, both engines and their hooks, in ``tmp_path``."""
    from app.database import db as db_module

    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    db_module._init_file_db(db_module.args.database_url)
    yield db_path
    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    db_module._db_lock.release(force=True)
