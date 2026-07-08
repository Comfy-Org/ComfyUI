"""Tests for metadata filtering logic in asset_reference queries."""
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference, AssetReferenceMeta
from app.assets.database.queries import list_references_page
from app.assets.database.queries.asset_reference import convert_metadata_to_rows
from app.assets.helpers import get_utc_now


def _make_asset(session: Session, hash_val: str) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=1024)
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    name: str,
    metadata: dict | None = None,
    system_metadata: dict | None = None,
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        owner_id="",
        name=name,
        asset_id=asset.id,
        user_metadata=metadata,
        system_metadata=system_metadata,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()

    # Build merged projection: {**system_metadata, **user_metadata}
    merged = {**(system_metadata or {}), **(metadata or {})}
    if merged:
        for key, val in merged.items():
            for row in convert_metadata_to_rows(key, val):
                meta_row = AssetReferenceMeta(
                    asset_reference_id=ref.id,
                    key=row["key"],
                    ordinal=row.get("ordinal", 0),
                    val_str=row.get("val_str"),
                    val_num=row.get("val_num"),
                    val_bool=row.get("val_bool"),
                    val_json=row.get("val_json"),
                )
                session.add(meta_row)
        session.flush()

    return ref


class TestMetadataFilterByType:
    """Table-driven tests for metadata filtering by different value types."""

    @pytest.mark.parametrize(
        "match_meta,nomatch_meta,filter_key,filter_val",
        [
            # String matching
            ({"category": "models"}, {"category": "images"}, "category", "models"),
            # Integer matching
            ({"epoch": 5}, {"epoch": 10}, "epoch", 5),
            # Float matching
            ({"score": 0.95}, {"score": 0.5}, "score", 0.95),
            # Boolean True matching
            ({"enabled": True}, {"enabled": False}, "enabled", True),
            # Boolean False matching
            ({"enabled": False}, {"enabled": True}, "enabled", False),
        ],
        ids=["string", "int", "float", "bool_true", "bool_false"],
    )
    def test_filter_matches_correct_value(
        self, session: Session, match_meta, nomatch_meta, filter_key, filter_val
    ):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "match", match_meta)
        _make_reference(session, asset, "nomatch", nomatch_meta)
        session.commit()

        refs, _, total = list_references_page(
            session, metadata_filter={filter_key: filter_val}
        )
        assert total == 1
        assert refs[0].name == "match"

    @pytest.mark.parametrize(
        "stored_meta,filter_key,filter_val",
        [
            # String no match
            ({"category": "models"}, "category", "other"),
            # Int no match
            ({"epoch": 5}, "epoch", 99),
            # Float no match
            ({"score": 0.5}, "score", 0.99),
        ],
        ids=["string_no_match", "int_no_match", "float_no_match"],
    )
    def test_filter_returns_empty_when_no_match(
        self, session: Session, stored_meta, filter_key, filter_val
    ):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "item", stored_meta)
        session.commit()

        refs, _, total = list_references_page(
            session, metadata_filter={filter_key: filter_val}
        )
        assert total == 0


class TestMetadataFilterNull:
    """Tests for null/missing key filtering."""

    @pytest.mark.parametrize(
        "match_name,match_meta,nomatch_name,nomatch_meta,filter_key",
        [
            # Null matches missing key
            ("missing_key", {}, "has_key", {"optional": "value"}, "optional"),
            # Null matches explicit null
            ("explicit_null", {"nullable": None}, "has_value", {"nullable": "present"}, "nullable"),
        ],
        ids=["missing_key", "explicit_null"],
    )
    def test_null_filter_matches(
        self, session: Session, match_name, match_meta, nomatch_name, nomatch_meta, filter_key
    ):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, match_name, match_meta)
        _make_reference(session, asset, nomatch_name, nomatch_meta)
        session.commit()

        refs, _, total = list_references_page(session, metadata_filter={filter_key: None})
        assert total == 1
        assert refs[0].name == match_name


class TestMetadataFilterList:
    """Tests for list-based (OR) filtering."""

    def test_filter_by_list_matches_any(self, session: Session):
        """List values should match ANY of the values (OR)."""
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "cat_a", {"category": "a"})
        _make_reference(session, asset, "cat_b", {"category": "b"})
        _make_reference(session, asset, "cat_c", {"category": "c"})
        session.commit()

        refs, _, total = list_references_page(session, metadata_filter={"category": ["a", "b"]})
        assert total == 2
        names = {r.name for r in refs}
        assert names == {"cat_a", "cat_b"}


class TestMetadataFilterMultipleKeys:
    """Tests for multiple filter keys (AND semantics)."""

    def test_multiple_keys_must_all_match(self, session: Session):
        """Multiple keys should ALL match (AND)."""
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "match", {"type": "model", "version": 2})
        _make_reference(session, asset, "wrong_type", {"type": "config", "version": 2})
        _make_reference(session, asset, "wrong_version", {"type": "model", "version": 1})
        session.commit()

        refs, _, total = list_references_page(
            session, metadata_filter={"type": "model", "version": 2}
        )
        assert total == 1
        assert refs[0].name == "match"


class TestMetadataFilterEmptyDict:
    """Tests for empty filter behavior."""

    def test_empty_filter_returns_all(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "a", {"key": "val"})
        _make_reference(session, asset, "b", {})
        session.commit()

        refs, _, total = list_references_page(session, metadata_filter={})
        assert total == 2


class TestSystemMetadataProjection:
    """Tests for system_metadata merging into the filter projection."""

    def test_system_metadata_keys_are_filterable(self, session: Session):
        """system_metadata keys should appear in the merged projection."""
        asset = _make_asset(session, "hash1")
        _make_reference(
            session, asset, "with_sys",
            system_metadata={"source": "scanner"},
        )
        _make_reference(session, asset, "without_sys")
        session.commit()

        refs, _, total = list_references_page(
            session, metadata_filter={"source": "scanner"}
        )
        assert total == 1
        assert refs[0].name == "with_sys"

    def test_user_metadata_overrides_system_metadata(self, session: Session):
        """user_metadata should win when both have the same key."""
        asset = _make_asset(session, "hash1")
        _make_reference(
            session, asset, "overridden",
            metadata={"origin": "user_upload"},
            system_metadata={"origin": "auto_scan"},
        )
        session.commit()

        # Should match the user value, not the system value
        refs, _, total = list_references_page(
            session, metadata_filter={"origin": "user_upload"}
        )
        assert total == 1
        assert refs[0].name == "overridden"

        # Should NOT match the system value (it was overridden)
        refs, _, total = list_references_page(
            session, metadata_filter={"origin": "auto_scan"}
        )
        assert total == 0
