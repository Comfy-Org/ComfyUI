"""
Add asset_contents.missing_since.

Set when a scan marks a content row missing. A returning file revives its row in
bulk only within a window of it. Rows already missing stay NULL and keep the
per-file revive.

Revision ID: 0009_add_missing_since
Revises: 0008_drop_asset_meta
Create Date: 2026-10-07
"""

from alembic import op
import sqlalchemy as sa

revision = "0009_add_missing_since"
down_revision = "0008_drop_asset_meta"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("asset_contents") as batch_op:
        batch_op.add_column(sa.Column("missing_since", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_asset_contents_missing_since", "asset_contents", ["missing_since"],
        sqlite_where=sa.text("missing_since IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_asset_contents_missing_since", table_name="asset_contents")
    with op.batch_alter_table("asset_contents") as batch_op:
        batch_op.drop_column("missing_since")
