import os
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

import app.assets.mode as mode_module
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content
from app.assets.services.lookup import (
    claim_qualified_content,
    is_temp_path as _is_temp_path,
    lookup_for_from_hash,
    lookup_for_view,
    refresh_qualified_content,
)
from app.database.models import Base


@pytest.fixture
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture(autouse=True)
def enable_hashing():
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _make_file(tmp_path, name: str, content: bytes = b"bytes") -> str:
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_temp_only_match_from_hash_returns_none(session, tmp_path):
    f = _make_file(tmp_path, "f.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = lookup_for_from_hash(session, "abc123")
    assert result is None


def test_temp_only_match_view_returns_none(session, tmp_path):
    f = _make_file(tmp_path, "f2.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = lookup_for_view(session, "abc123")
    assert result is None


def test_sibling_prefix_not_temp(tmp_path):
    with patch("folder_paths.get_temp_directory", return_value=str(tmp_path / "temp")):
        assert not _is_temp_path(str(tmp_path / "temp-other" / "f.png"))
        assert _is_temp_path(str(tmp_path / "temp" / "f.png"))


def test_off_mode_from_hash_returns_none(session, tmp_path):
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    f = _make_file(tmp_path, "f3.png")
    create_content(session, path=f, hash="abc123")
    session.commit()
    result = lookup_for_from_hash(session, "abc123")
    assert result is None


def test_upload_content_lookup_not_gated_on_hashing_flag(session, tmp_path):
    class FakeArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    f = _make_file(tmp_path, "f4.png")
    content = create_content(session, path=f, hash="abc123")
    session.commit()
    result = lookup_for_view(session, "abc123")
    assert result is not None
    assert result.id == content.id


def test_stale_older_newer_live_returns_newer(session, tmp_path):
    f_newer = _make_file(tmp_path, "newer.png")
    old_time = datetime(2020, 1, 1)
    new_time = datetime(2024, 1, 1)
    c_old = create_content(session, path="/nonexistent/old.png", hash="xyz")
    c_new = create_content(session, path=f_newer, hash="xyz")
    session.execute(update(AssetContent).where(AssetContent.id == c_old.id).values(created_at=old_time))
    session.execute(update(AssetContent).where(AssetContent.id == c_new.id).values(created_at=new_time))
    session.commit()

    result = lookup_for_from_hash(session, "xyz")
    assert result is not None
    assert result.id == c_new.id


def test_claim_qualified_content_true_for_a_live_matching_row(session, tmp_path):
    f = _make_file(tmp_path, "revalidate_ok.png")
    content = create_content(session, path=f, hash="dup")
    session.commit()

    assert claim_qualified_content(session, content.id, "dup") is True


def test_claim_qualified_content_false_once_retired(session, tmp_path):
    f = _make_file(tmp_path, "revalidate_retired.png")
    content = create_content(session, path=f, hash="dup")
    session.commit()

    session.execute(
        update(AssetContent).where(AssetContent.id == content.id).values(is_missing=True)
    )
    session.commit()

    assert claim_qualified_content(session, content.id, "dup") is False, (
        "a row retired after the lookup must not be reused"
    )


def test_claim_qualified_content_false_once_hash_changed(session, tmp_path):
    f = _make_file(tmp_path, "revalidate_rehashed.png")
    content = create_content(session, path=f, hash="dup")
    session.commit()

    session.execute(
        update(AssetContent).where(AssetContent.id == content.id).values(hash="different")
    )
    session.commit()

    assert claim_qualified_content(session, content.id, "dup") is False, (
        "a row whose recorded content changed identity must not be reused"
        " for the hash that used to describe it"
    )


def test_refresh_qualified_content_none_when_file_vanishes(session, tmp_path):
    f = _make_file(tmp_path, "revalidate_gone.png")
    content = create_content(session, path=f, hash="dup")
    session.commit()
    assert claim_qualified_content(session, content.id, "dup") is True

    os.unlink(f)

    assert refresh_qualified_content(session, content.id) is None
