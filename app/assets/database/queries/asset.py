import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries.common import MAX_BIND_PARAMS, calculate_rows_per_statement, iter_chunks


def asset_exists_by_hash(
    session: Session,
    asset_hash: str,
) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    row = (
        session.execute(
            select(sa.literal(True))
            .select_from(Asset)
            .where(Asset.hash == asset_hash)
            .limit(1)
        )
    ).first()
    return row is not None


def get_asset_by_hash(
    session: Session,
    asset_hash: str,
) -> Asset | None:
    return (
        (session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1)))
        .scalars()
        .first()
    )


def upsert_asset(
    session: Session,
    asset_hash: str,
    size_bytes: int,
    mime_type: str | None = None,
) -> tuple[Asset, bool, bool]:
    """Upsert an Asset by hash. Returns (asset, created, updated)."""
    vals = {"hash": asset_hash, "size_bytes": int(size_bytes)}
    if mime_type:
        vals["mime_type"] = mime_type

    ins = (
        sqlite.insert(Asset)
        .values(**vals)
        .on_conflict_do_nothing(index_elements=[Asset.hash])
    )
    res = session.execute(ins)
    created = int(res.rowcount or 0) > 0

    asset = (
        session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
        .scalars()
        .first()
    )
    if not asset:
        raise RuntimeError("Asset row not found after upsert.")

    updated = False
    if not created:
        changed = False
        if asset.size_bytes != int(size_bytes) and int(size_bytes) > 0:
            asset.size_bytes = int(size_bytes)
            changed = True
        if mime_type and not asset.mime_type:
            asset.mime_type = mime_type
            changed = True
        if changed:
            updated = True

    return asset, created, updated


def create_stub_asset(
    session: Session,
    size_bytes: int,
    mime_type: str | None = None,
) -> Asset:
    """Create a new asset with no hash (stub for later enrichment)."""
    asset = Asset(size_bytes=size_bytes, mime_type=mime_type, hash=None)
    session.add(asset)
    session.flush()
    return asset


def bulk_insert_assets(
    session: Session,
    rows: list[dict],
) -> None:
    """Bulk insert Asset rows with ON CONFLICT DO NOTHING on hash."""
    if not rows:
        return
    ins = sqlite.insert(Asset).on_conflict_do_nothing(index_elements=[Asset.hash])
    for chunk in iter_chunks(rows, calculate_rows_per_statement(5)):
        session.execute(ins, chunk)


def get_existing_asset_ids(
    session: Session,
    asset_ids: list[str],
) -> set[str]:
    """Return the subset of asset_ids that exist in the database."""
    if not asset_ids:
        return set()
    found: set[str] = set()
    for chunk in iter_chunks(asset_ids, MAX_BIND_PARAMS):
        rows = session.execute(
            select(Asset.id).where(Asset.id.in_(chunk))
        ).fetchall()
        found.update(row[0] for row in rows)
    return found


def update_asset_hash_and_mime(
    session: Session,
    asset_id: str,
    asset_hash: str | None = None,
    mime_type: str | None = None,
) -> bool:
    """Update asset hash and/or mime_type. Returns True if asset was found."""
    asset = session.get(Asset, asset_id)
    if not asset:
        return False
    if asset_hash is not None:
        asset.hash = asset_hash
    if mime_type is not None and not asset.mime_type:
        asset.mime_type = mime_type
    return True


def reassign_asset_references(
    session: Session,
    from_asset_id: str,
    to_asset_id: str,
    reference_id: str,
) -> None:
    """Reassign a reference from one asset to another.

    Used when merging a stub asset into an existing asset with the same hash.
    """
    ref = session.get(AssetReference, reference_id)
    if ref and ref.asset_id == from_asset_id:
        ref.asset_id = to_asset_id

    session.flush()
