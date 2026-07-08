import uuid

import pytest
from sqlalchemy.orm import Session

from app.assets.helpers import get_utc_now
from app.assets.database.models import Asset
from app.assets.database.queries import (
    asset_exists_by_hash,
    get_asset_by_hash,
    upsert_asset,
    bulk_insert_assets,
    update_asset_hash_and_mime,
)


class TestAssetExistsByHash:
    @pytest.mark.parametrize(
        "setup_hash,query_hash,expected",
        [
            (None, "nonexistent", False),  # No asset exists
            ("blake3:abc123", "blake3:abc123", True),  # Asset exists with matching hash
            (None, "", False),  # Null hash in DB doesn't match empty string
        ],
        ids=["nonexistent", "existing", "null_hash_no_match"],
    )
    def test_exists_by_hash(self, session: Session, setup_hash, query_hash, expected):
        if setup_hash is not None or query_hash == "":
            asset = Asset(hash=setup_hash, size_bytes=100)
            session.add(asset)
            session.commit()

        assert asset_exists_by_hash(session, asset_hash=query_hash) is expected


class TestGetAssetByHash:
    @pytest.mark.parametrize(
        "setup_hash,query_hash,should_find",
        [
            (None, "nonexistent", False),
            ("blake3:def456", "blake3:def456", True),
        ],
        ids=["nonexistent", "existing"],
    )
    def test_get_by_hash(self, session: Session, setup_hash, query_hash, should_find):
        if setup_hash is not None:
            asset = Asset(hash=setup_hash, size_bytes=200, mime_type="image/png")
            session.add(asset)
            session.commit()

        result = get_asset_by_hash(session, asset_hash=query_hash)
        if should_find:
            assert result is not None
            assert result.size_bytes == 200
            assert result.mime_type == "image/png"
        else:
            assert result is None


class TestUpsertAsset:
    @pytest.mark.parametrize(
        "first_size,first_mime,second_size,second_mime,expect_created,expect_updated,final_size,final_mime",
        [
            # New asset creation
            (None, None, 1024, "application/octet-stream", True, False, 1024, "application/octet-stream"),
            # Existing asset, same values - no update
            (500, "text/plain", 500, "text/plain", False, False, 500, "text/plain"),
            # Existing asset with size 0, update with new values
            (0, None, 2048, "image/png", False, True, 2048, "image/png"),
            # Existing asset, second call with size 0 - no update
            (1000, None, 0, None, False, False, 1000, None),
        ],
        ids=["new_asset", "existing_no_change", "update_from_zero", "zero_size_no_update"],
    )
    def test_upsert_scenarios(
        self,
        session: Session,
        first_size,
        first_mime,
        second_size,
        second_mime,
        expect_created,
        expect_updated,
        final_size,
        final_mime,
    ):
        asset_hash = f"blake3:test_{first_size}_{second_size}"

        # First upsert (if first_size is not None, we're testing the second call)
        if first_size is not None:
            upsert_asset(
                session,
                asset_hash=asset_hash,
                size_bytes=first_size,
                mime_type=first_mime,
            )
            session.commit()

        # The upsert call we're testing
        asset, created, updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=second_size,
            mime_type=second_mime,
        )
        session.commit()

        assert created is expect_created
        assert updated is expect_updated
        assert asset.size_bytes == final_size
        assert asset.mime_type == final_mime


class TestBulkInsertAssets:
    def test_inserts_multiple_assets(self, session: Session):
        now = get_utc_now()
        rows = [
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk1", "size_bytes": 100, "mime_type": "text/plain", "created_at": now},
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk2", "size_bytes": 200, "mime_type": "image/png", "created_at": now},
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk3", "size_bytes": 300, "mime_type": None, "created_at": now},
        ]
        bulk_insert_assets(session, rows)
        session.commit()

        assets = session.query(Asset).all()
        assert len(assets) == 3
        hashes = {a.hash for a in assets}
        assert hashes == {"blake3:bulk1", "blake3:bulk2", "blake3:bulk3"}

    def test_empty_list_is_noop(self, session: Session):
        bulk_insert_assets(session, [])
        session.commit()
        assert session.query(Asset).count() == 0

    def test_handles_large_batch(self, session: Session):
        """Test chunking logic with more rows than MAX_BIND_PARAMS allows."""
        now = get_utc_now()
        rows = [
            {"id": str(uuid.uuid4()), "hash": f"blake3:large{i}", "size_bytes": i, "mime_type": None, "created_at": now}
            for i in range(200)
        ]
        bulk_insert_assets(session, rows)
        session.commit()

        assert session.query(Asset).count() == 200


class TestMimeTypeImmutability:
    """mime_type on Asset is write-once: set on first ingest, never overwritten."""

    @pytest.mark.parametrize(
        "initial_mime,second_mime,expected_mime",
        [
            ("image/png", "image/jpeg", "image/png"),
            (None, "image/png", "image/png"),
        ],
        ids=["preserves_existing", "fills_null"],
    )
    def test_upsert_mime_immutability(self, session: Session, initial_mime, second_mime, expected_mime):
        h = f"blake3:upsert_{initial_mime}_{second_mime}"
        upsert_asset(session, asset_hash=h, size_bytes=100, mime_type=initial_mime)
        session.commit()

        asset, created, _ = upsert_asset(session, asset_hash=h, size_bytes=100, mime_type=second_mime)
        assert created is False
        assert asset.mime_type == expected_mime

    @pytest.mark.parametrize(
        "initial_mime,update_mime,update_hash,expected_mime,expected_hash",
        [
            (None, "image/png", None, "image/png", "blake3:upd0"),
            ("image/png", "image/jpeg", None, "image/png", "blake3:upd1"),
            ("image/png", "image/jpeg", "blake3:upd2_new", "image/png", "blake3:upd2_new"),
        ],
        ids=["fills_null", "preserves_existing", "hash_updates_mime_locked"],
    )
    def test_update_asset_hash_and_mime_immutability(
        self, session: Session, initial_mime, update_mime, update_hash, expected_mime, expected_hash,
    ):
        h = expected_hash.removesuffix("_new")
        asset = Asset(hash=h, size_bytes=100, mime_type=initial_mime)
        session.add(asset)
        session.flush()

        update_asset_hash_and_mime(session, asset_id=asset.id, mime_type=update_mime, asset_hash=update_hash)
        assert asset.mime_type == expected_mime
        assert asset.hash == expected_hash
