import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference, AssetReferenceTag, AssetReferenceMeta, Tag
from app.assets.database.queries import (
    ensure_tags_exist,
    get_reference_tags,
    set_reference_tags,
    add_tags_to_reference,
    remove_tags_from_reference,
    add_missing_tag_for_asset_id,
    remove_missing_tag_for_asset_id,
    list_tags_with_usage,
    bulk_insert_tags_and_meta,
)
from app.assets.helpers import get_utc_now


def _make_asset(session: Session, hash_val: str | None = None) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=1024)
    session.add(asset)
    session.flush()
    return asset


def _make_reference(session: Session, asset: Asset, name: str = "test", owner_id: str = "") -> AssetReference:
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


class TestEnsureTagsExist:
    def test_creates_new_tags(self, session: Session):
        ensure_tags_exist(session, ["alpha", "beta"])
        session.commit()

        tags = session.query(Tag).all()
        assert {t.name for t in tags} == {"alpha", "beta"}

    def test_is_idempotent(self, session: Session):
        ensure_tags_exist(session, ["alpha"])
        ensure_tags_exist(session, ["alpha"])
        session.commit()

        assert session.query(Tag).count() == 1

    def test_normalizes_tags(self, session: Session):
        ensure_tags_exist(session, ["  ALPHA  ", "Beta", "alpha"])
        session.commit()

        tags = session.query(Tag).all()
        assert {t.name for t in tags} == {"alpha", "beta"}

    def test_empty_list_is_noop(self, session: Session):
        ensure_tags_exist(session, [])
        session.commit()
        assert session.query(Tag).count() == 0


class TestGetReferenceTags:
    def test_returns_empty_for_no_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        tags = get_reference_tags(session, reference_id=ref.id)
        assert tags == []

    def test_returns_tags_for_reference(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        ensure_tags_exist(session, ["tag1", "tag2"])
        session.add_all([
            AssetReferenceTag(asset_reference_id=ref.id, tag_name="tag1", origin="manual", added_at=get_utc_now()),
            AssetReferenceTag(asset_reference_id=ref.id, tag_name="tag2", origin="manual", added_at=get_utc_now()),
        ])
        session.flush()

        tags = get_reference_tags(session, reference_id=ref.id)
        assert set(tags) == {"tag1", "tag2"}


class TestSetReferenceTags:
    def test_adds_new_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        result = set_reference_tags(session, reference_id=ref.id, tags=["a", "b"])
        session.commit()

        assert set(result.added) == {"a", "b"}
        assert result.removed == []
        assert set(result.total) == {"a", "b"}

    def test_removes_old_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        set_reference_tags(session, reference_id=ref.id, tags=["a", "b", "c"])
        result = set_reference_tags(session, reference_id=ref.id, tags=["a"])
        session.commit()

        assert result.added == []
        assert set(result.removed) == {"b", "c"}
        assert result.total == ["a"]

    def test_replaces_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        set_reference_tags(session, reference_id=ref.id, tags=["a", "b"])
        result = set_reference_tags(session, reference_id=ref.id, tags=["b", "c"])
        session.commit()

        assert result.added == ["c"]
        assert result.removed == ["a"]
        assert set(result.total) == {"b", "c"}


class TestAddTagsToReference:
    def test_adds_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        result = add_tags_to_reference(session, reference_id=ref.id, tags=["x", "y"])
        session.commit()

        assert set(result.added) == {"x", "y"}
        assert result.already_present == []

    def test_reports_already_present(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        add_tags_to_reference(session, reference_id=ref.id, tags=["x"])
        result = add_tags_to_reference(session, reference_id=ref.id, tags=["x", "y"])
        session.commit()

        assert result.added == ["y"]
        assert result.already_present == ["x"]

    def test_raises_for_missing_reference(self, session: Session):
        with pytest.raises(ValueError, match="not found"):
            add_tags_to_reference(session, reference_id="nonexistent", tags=["x"])


class TestRemoveTagsFromReference:
    def test_removes_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        add_tags_to_reference(session, reference_id=ref.id, tags=["a", "b", "c"])
        result = remove_tags_from_reference(session, reference_id=ref.id, tags=["a", "b"])
        session.commit()

        assert set(result.removed) == {"a", "b"}
        assert result.not_present == []
        assert result.total_tags == ["c"]

    def test_reports_not_present(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)

        add_tags_to_reference(session, reference_id=ref.id, tags=["a"])
        result = remove_tags_from_reference(session, reference_id=ref.id, tags=["a", "x"])
        session.commit()

        assert result.removed == ["a"]
        assert result.not_present == ["x"]

    def test_raises_for_missing_reference(self, session: Session):
        with pytest.raises(ValueError, match="not found"):
            remove_tags_from_reference(session, reference_id="nonexistent", tags=["x"])


class TestMissingTagFunctions:
    def test_add_missing_tag_for_asset_id(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["missing"])

        add_missing_tag_for_asset_id(session, asset_id=asset.id)
        session.commit()

        tags = get_reference_tags(session, reference_id=ref.id)
        assert "missing" in tags

    def test_add_missing_tag_is_idempotent(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["missing"])

        add_missing_tag_for_asset_id(session, asset_id=asset.id)
        add_missing_tag_for_asset_id(session, asset_id=asset.id)
        session.commit()

        links = session.query(AssetReferenceTag).filter_by(asset_reference_id=ref.id, tag_name="missing").all()
        assert len(links) == 1

    def test_remove_missing_tag_for_asset_id(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["missing"])
        add_missing_tag_for_asset_id(session, asset_id=asset.id)

        remove_missing_tag_for_asset_id(session, asset_id=asset.id)
        session.commit()

        tags = get_reference_tags(session, reference_id=ref.id)
        assert "missing" not in tags


class TestListTagsWithUsage:
    def test_returns_tags_with_counts(self, session: Session):
        ensure_tags_exist(session, ["used", "unused"])

        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        add_tags_to_reference(session, reference_id=ref.id, tags=["used"])
        session.commit()

        rows, total = list_tags_with_usage(session)

        tag_dict = {name: count for name, count in rows}
        assert tag_dict["used"] == 1
        assert tag_dict["unused"] == 0
        assert total == 2

    def test_exclude_zero_counts(self, session: Session):
        ensure_tags_exist(session, ["used", "unused"])

        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        add_tags_to_reference(session, reference_id=ref.id, tags=["used"])
        session.commit()

        rows, total = list_tags_with_usage(session, include_zero=False)

        tag_names = {name for name, _ in rows}
        assert "used" in tag_names
        assert "unused" not in tag_names

    def test_prefix_filter(self, session: Session):
        ensure_tags_exist(session, ["alpha", "beta", "alphabet"])
        session.commit()

        rows, total = list_tags_with_usage(session, prefix="alph")

        tag_names = {name for name, _ in rows}
        assert tag_names == {"alpha", "alphabet"}

    def test_order_by_name(self, session: Session):
        ensure_tags_exist(session, ["zebra", "alpha", "middle"])
        session.commit()

        rows, _ = list_tags_with_usage(session, order="name_asc")

        names = [name for name, _ in rows]
        assert names == ["alpha", "middle", "zebra"]

    def test_owner_visibility(self, session: Session):
        ensure_tags_exist(session, ["shared-tag", "owner-tag"])

        asset = _make_asset(session, "hash1")
        shared_ref = _make_reference(session, asset, name="shared", owner_id="")
        owner_ref = _make_reference(session, asset, name="owned", owner_id="user1")

        add_tags_to_reference(session, reference_id=shared_ref.id, tags=["shared-tag"])
        add_tags_to_reference(session, reference_id=owner_ref.id, tags=["owner-tag"])
        session.commit()

        # Empty owner sees only shared
        rows, _ = list_tags_with_usage(session, owner_id="", include_zero=False)
        tag_dict = {name: count for name, count in rows}
        assert tag_dict.get("shared-tag", 0) == 1
        assert tag_dict.get("owner-tag", 0) == 0

        # User1 sees both
        rows, _ = list_tags_with_usage(session, owner_id="user1", include_zero=False)
        tag_dict = {name: count for name, count in rows}
        assert tag_dict.get("shared-tag", 0) == 1
        assert tag_dict.get("owner-tag", 0) == 1


class TestBulkInsertTagsAndMeta:
    def test_inserts_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["bulk-tag1", "bulk-tag2"])
        session.commit()

        now = get_utc_now()
        tag_rows = [
            {"asset_reference_id": ref.id, "tag_name": "bulk-tag1", "origin": "manual", "added_at": now},
            {"asset_reference_id": ref.id, "tag_name": "bulk-tag2", "origin": "manual", "added_at": now},
        ]
        bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=[])
        session.commit()

        tags = get_reference_tags(session, reference_id=ref.id)
        assert set(tags) == {"bulk-tag1", "bulk-tag2"}

    def test_inserts_meta(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        session.commit()

        meta_rows = [
            {
                "asset_reference_id": ref.id,
                "key": "meta-key",
                "ordinal": 0,
                "val_str": "meta-value",
                "val_num": None,
                "val_bool": None,
                "val_json": None,
            },
        ]
        bulk_insert_tags_and_meta(session, tag_rows=[], meta_rows=meta_rows)
        session.commit()

        meta = session.query(AssetReferenceMeta).filter_by(asset_reference_id=ref.id).all()
        assert len(meta) == 1
        assert meta[0].key == "meta-key"
        assert meta[0].val_str == "meta-value"

    def test_ignores_conflicts(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["existing-tag"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["existing-tag"])
        session.commit()

        now = get_utc_now()
        tag_rows = [
            {"asset_reference_id": ref.id, "tag_name": "existing-tag", "origin": "duplicate", "added_at": now},
        ]
        bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=[])
        session.commit()

        # Should still have only one tag link
        links = session.query(AssetReferenceTag).filter_by(asset_reference_id=ref.id, tag_name="existing-tag").all()
        assert len(links) == 1
        # Origin should be original, not overwritten
        assert links[0].origin == "manual"

    def test_empty_lists_is_noop(self, session: Session):
        bulk_insert_tags_and_meta(session, tag_rows=[], meta_rows=[])
        assert session.query(AssetReferenceTag).count() == 0
        assert session.query(AssetReferenceMeta).count() == 0
