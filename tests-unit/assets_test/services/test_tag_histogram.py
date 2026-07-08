"""Tests for list_tag_histogram service function."""
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import ensure_tags_exist, add_tags_to_reference
from app.assets.helpers import get_utc_now
from app.assets.services.tagging import list_tag_histogram


def _make_asset(session: Session, hash_val: str = "blake3:test") -> Asset:
    asset = Asset(hash=hash_val, size_bytes=1024)
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    name: str = "test",
    owner_id: str = "",
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()
    return ref


class TestListTagHistogram:
    def test_returns_counts_for_all_tags(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["alpha", "beta"])
        a1 = _make_asset(session, "blake3:aaa")
        r1 = _make_reference(session, a1, name="r1")
        add_tags_to_reference(session, reference_id=r1.id, tags=["alpha", "beta"])

        a2 = _make_asset(session, "blake3:bbb")
        r2 = _make_reference(session, a2, name="r2")
        add_tags_to_reference(session, reference_id=r2.id, tags=["alpha"])
        session.commit()

        result = list_tag_histogram()

        assert result["alpha"] == 2
        assert result["beta"] == 1

    def test_empty_when_no_assets(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["unused"])
        session.commit()

        result = list_tag_histogram()

        assert result == {}

    def test_include_tags_filter(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["models", "loras", "input"])
        a1 = _make_asset(session, "blake3:aaa")
        r1 = _make_reference(session, a1, name="r1")
        add_tags_to_reference(session, reference_id=r1.id, tags=["models", "loras"])

        a2 = _make_asset(session, "blake3:bbb")
        r2 = _make_reference(session, a2, name="r2")
        add_tags_to_reference(session, reference_id=r2.id, tags=["input"])
        session.commit()

        result = list_tag_histogram(include_tags=["models"])

        # Only r1 has "models", so only its tags appear
        assert "models" in result
        assert "loras" in result
        assert "input" not in result

    def test_exclude_tags_filter(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["models", "loras", "input"])
        a1 = _make_asset(session, "blake3:aaa")
        r1 = _make_reference(session, a1, name="r1")
        add_tags_to_reference(session, reference_id=r1.id, tags=["models", "loras"])

        a2 = _make_asset(session, "blake3:bbb")
        r2 = _make_reference(session, a2, name="r2")
        add_tags_to_reference(session, reference_id=r2.id, tags=["input"])
        session.commit()

        result = list_tag_histogram(exclude_tags=["models"])

        # r1 excluded, only r2's tags remain
        assert "input" in result
        assert "loras" not in result

    def test_name_contains_filter(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["alpha", "beta"])
        a1 = _make_asset(session, "blake3:aaa")
        r1 = _make_reference(session, a1, name="my_model.safetensors")
        add_tags_to_reference(session, reference_id=r1.id, tags=["alpha"])

        a2 = _make_asset(session, "blake3:bbb")
        r2 = _make_reference(session, a2, name="picture.png")
        add_tags_to_reference(session, reference_id=r2.id, tags=["beta"])
        session.commit()

        result = list_tag_histogram(name_contains="model")

        assert "alpha" in result
        assert "beta" not in result

    def test_limit_caps_results(self, mock_create_session, session: Session):
        tags = [f"tag{i}" for i in range(10)]
        ensure_tags_exist(session, tags)
        a = _make_asset(session, "blake3:aaa")
        r = _make_reference(session, a, name="r1")
        add_tags_to_reference(session, reference_id=r.id, tags=tags)
        session.commit()

        result = list_tag_histogram(limit=3)

        assert len(result) == 3
