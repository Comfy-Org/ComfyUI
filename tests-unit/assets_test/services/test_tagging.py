"""Tests for tagging services."""
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import ensure_tags_exist, add_tags_to_reference
from app.assets.helpers import get_utc_now
from app.assets.services import apply_tags, remove_tags, list_tags


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


class TestApplyTags:
    def test_adds_new_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        session.commit()

        result = apply_tags(
            reference_id=ref.id,
            tags=["alpha", "beta"],
        )

        assert set(result.added) == {"alpha", "beta"}
        assert result.already_present == []
        assert set(result.total_tags) == {"alpha", "beta"}

    def test_reports_already_present(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["existing"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["existing"])
        session.commit()

        result = apply_tags(
            reference_id=ref.id,
            tags=["existing", "new"],
        )

        assert result.added == ["new"]
        assert result.already_present == ["existing"]

    def test_raises_for_nonexistent_ref(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            apply_tags(reference_id="nonexistent", tags=["x"])

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            apply_tags(
                reference_id=ref.id,
                tags=["new"],
                owner_id="user2",
            )


class TestRemoveTags:
    def test_removes_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["a", "b", "c"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["a", "b", "c"])
        session.commit()

        result = remove_tags(
            reference_id=ref.id,
            tags=["a", "b"],
        )

        assert set(result.removed) == {"a", "b"}
        assert result.not_present == []
        assert result.total_tags == ["c"]

    def test_reports_not_present(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["present"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["present"])
        session.commit()

        result = remove_tags(
            reference_id=ref.id,
            tags=["present", "absent"],
        )

        assert result.removed == ["present"]
        assert result.not_present == ["absent"]

    def test_raises_for_nonexistent_ref(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            remove_tags(reference_id="nonexistent", tags=["x"])

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            remove_tags(
                reference_id=ref.id,
                tags=["x"],
                owner_id="user2",
            )


class TestListTags:
    def test_returns_tags_with_counts(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["used", "unused"])
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        add_tags_to_reference(session, reference_id=ref.id, tags=["used"])
        session.commit()

        rows, total = list_tags()

        tag_dict = {name: count for name, count in rows}
        assert tag_dict["used"] == 1
        assert tag_dict["unused"] == 0
        assert total == 2

    def test_excludes_zero_counts(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["used", "unused"])
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        add_tags_to_reference(session, reference_id=ref.id, tags=["used"])
        session.commit()

        rows, total = list_tags(include_zero=False)

        tag_names = {name for name, _ in rows}
        assert "used" in tag_names
        assert "unused" not in tag_names

    def test_prefix_filter(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["alpha", "beta", "alphabet"])
        session.commit()

        rows, _ = list_tags(prefix="alph")

        tag_names = {name for name, _ in rows}
        assert tag_names == {"alpha", "alphabet"}

    def test_order_by_name(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["zebra", "alpha", "middle"])
        session.commit()

        rows, _ = list_tags(order="name_asc")

        names = [name for name, _ in rows]
        assert names == ["alpha", "middle", "zebra"]

    def test_pagination(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["a", "b", "c", "d", "e"])
        session.commit()

        rows, total = list_tags(limit=2, offset=1, order="name_asc")

        assert total == 5
        assert len(rows) == 2
        names = [name for name, _ in rows]
        assert names == ["b", "c"]

    def test_clamps_limit(self, mock_create_session, session: Session):
        ensure_tags_exist(session, ["a"])
        session.commit()

        # Service should clamp limit to max 1000
        rows, _ = list_tags(limit=2000)
        assert len(rows) <= 1000
