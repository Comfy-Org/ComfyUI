"""
Drop the unused asset_meta table.

Asset metadata lives in the user_metadata and system_metadata JSON columns on
assets; no code path reads or writes asset_meta.

Revision ID: 0008_drop_asset_meta
Revises: 0007_record_content_split
Create Date: 2026-09-22
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_drop_asset_meta"
down_revision = "0007_record_content_split"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS asset_meta")


def downgrade() -> None:
    op.create_table(
        "asset_meta",
        sa.Column("asset_id", sa.String(36), sa.ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("key", sa.String(256), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("val_str", sa.String(2048)), sa.Column("val_num", sa.Numeric(38, 10)),
        sa.Column("val_bool", sa.Boolean()), sa.Column("val_json", sa.JSON()),
        sa.CheckConstraint("val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL", name="ck_asset_meta_has_value"),
    )
    op.create_index("ix_asset_meta_key", "asset_meta", ["key"])
    op.create_index("ix_asset_meta_key_val_str", "asset_meta", ["key", "val_str"])
    op.create_index("ix_asset_meta_key_val_num", "asset_meta", ["key", "val_num"])
    op.create_index("ix_asset_meta_key_val_bool", "asset_meta", ["key", "val_bool"])
