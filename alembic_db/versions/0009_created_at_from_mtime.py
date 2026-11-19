"""
Date each scanned record by its file's mtime instead of the scan time.

Revision ID: 0009_created_at_from_mtime
Revises: 0008_drop_asset_meta
Create Date: 2026-10-05
"""

from datetime import datetime, timedelta

from alembic import op
import sqlalchemy as sa

revision = "0009_created_at_from_mtime"
down_revision = "0008_drop_asset_meta"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The scanner writes a jobless record in the same transaction as its content; a record
    # added to existing content later (a re-upload, from-hash) is not the scan's, and uploads,
    # which also create both together, carry the "uploaded" tag.
    rows = op.get_bind().execute(sa.text(
        "SELECT a.id, a.created_at, c.mtime_ns FROM assets AS a JOIN asset_contents AS c"
        " ON c.id = a.content_id WHERE a.job_id IS NULL AND c.mtime_ns IS NOT NULL"
        " AND abs(julianday(a.created_at) - julianday(c.created_at)) * 86400 < 1"
        " AND NOT EXISTS (SELECT 1 FROM asset_tags AS t WHERE t.asset_id = a.id AND t.tag_name = 'uploaded')"
    ).columns(created_at=sa.DateTime)).all()
    # Only ever earlier: an mtime changed after cataloguing must not move a record above newer work.
    # The conversion is inlined, not app.assets.helpers', so this migration can't change if that helper does.
    params = [
        {"id": record_id, "created": min(datetime(1970, 1, 1) + timedelta(microseconds=mtime_ns // 1000), created_at)}
        for record_id, created_at, mtime_ns in rows
    ]
    if params:
        op.get_bind().execute(sa.text("UPDATE assets SET created_at = :created WHERE id = :id").bindparams(
            sa.bindparam("created", type_=sa.DateTime)), params)


def downgrade() -> None:
    pass
