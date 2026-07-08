"""
Merge AssetInfo and AssetCacheState into unified asset_references table.

This migration drops old tables and creates the new unified schema.
All existing data is discarded.

Revision ID: 0002_merge_to_asset_references
Revises: 0001_assets
Create Date: 2025-02-11
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_merge_to_asset_references"
down_revision = "0001_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old tables (order matters due to FK constraints)
    op.drop_index("ix_asset_info_meta_key_val_bool", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_num", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_str", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key", table_name="asset_info_meta")
    op.drop_table("asset_info_meta")

    op.drop_index("ix_asset_info_tags_asset_info_id", table_name="asset_info_tags")
    op.drop_index("ix_asset_info_tags_tag_name", table_name="asset_info_tags")
    op.drop_table("asset_info_tags")

    op.drop_index("ix_asset_cache_state_asset_id", table_name="asset_cache_state")
    op.drop_index("ix_asset_cache_state_file_path", table_name="asset_cache_state")
    op.drop_table("asset_cache_state")

    op.drop_index("ix_assets_info_owner_name", table_name="assets_info")
    op.drop_index("ix_assets_info_last_access_time", table_name="assets_info")
    op.drop_index("ix_assets_info_created_at", table_name="assets_info")
    op.drop_index("ix_assets_info_name", table_name="assets_info")
    op.drop_index("ix_assets_info_asset_id", table_name="assets_info")
    op.drop_index("ix_assets_info_owner_id", table_name="assets_info")
    op.drop_table("assets_info")

    # Truncate assets table (cascades handled by dropping dependent tables first)
    op.execute("DELETE FROM assets")

    # Create asset_references table
    op.create_table(
        "asset_references",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(length=36),
            sa.ForeignKey("assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column(
            "needs_verify",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_missing", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("enrichment_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("owner_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column(
            "preview_id",
            sa.String(length=36),
            sa.ForeignKey("assets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("last_access_time", sa.DateTime(timezone=False), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=False), nullable=True),
        sa.CheckConstraint(
            "(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_ar_mtime_nonneg"
        ),
        sa.CheckConstraint(
            "enrichment_level >= 0 AND enrichment_level <= 2",
            name="ck_ar_enrichment_level_range",
        ),
    )
    op.create_index(
        "uq_asset_references_file_path", "asset_references", ["file_path"], unique=True
    )
    op.create_index("ix_asset_references_asset_id", "asset_references", ["asset_id"])
    op.create_index("ix_asset_references_owner_id", "asset_references", ["owner_id"])
    op.create_index("ix_asset_references_name", "asset_references", ["name"])
    op.create_index("ix_asset_references_is_missing", "asset_references", ["is_missing"])
    op.create_index(
        "ix_asset_references_enrichment_level", "asset_references", ["enrichment_level"]
    )
    op.create_index("ix_asset_references_created_at", "asset_references", ["created_at"])
    op.create_index(
        "ix_asset_references_last_access_time", "asset_references", ["last_access_time"]
    )
    op.create_index(
        "ix_asset_references_owner_name", "asset_references", ["owner_id", "name"]
    )
    op.create_index("ix_asset_references_deleted_at", "asset_references", ["deleted_at"])

    # Create asset_reference_tags table
    op.create_table(
        "asset_reference_tags",
        sa.Column(
            "asset_reference_id",
            sa.String(length=36),
            sa.ForeignKey("asset_references.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tag_name",
            sa.String(length=512),
            sa.ForeignKey("tags.name", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "origin", sa.String(length=32), nullable=False, server_default="manual"
        ),
        sa.Column("added_at", sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint(
            "asset_reference_id", "tag_name", name="pk_asset_reference_tags"
        ),
    )
    op.create_index(
        "ix_asset_reference_tags_tag_name", "asset_reference_tags", ["tag_name"]
    )
    op.create_index(
        "ix_asset_reference_tags_asset_reference_id",
        "asset_reference_tags",
        ["asset_reference_id"],
    )

    # Create asset_reference_meta table
    op.create_table(
        "asset_reference_meta",
        sa.Column(
            "asset_reference_id",
            sa.String(length=36),
            sa.ForeignKey("asset_references.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("val_str", sa.String(length=2048), nullable=True),
        sa.Column("val_num", sa.Numeric(38, 10), nullable=True),
        sa.Column("val_bool", sa.Boolean(), nullable=True),
        sa.Column("val_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint(
            "asset_reference_id", "key", "ordinal", name="pk_asset_reference_meta"
        ),
    )
    op.create_index("ix_asset_reference_meta_key", "asset_reference_meta", ["key"])
    op.create_index(
        "ix_asset_reference_meta_key_val_str", "asset_reference_meta", ["key", "val_str"]
    )
    op.create_index(
        "ix_asset_reference_meta_key_val_num", "asset_reference_meta", ["key", "val_num"]
    )
    op.create_index(
        "ix_asset_reference_meta_key_val_bool",
        "asset_reference_meta",
        ["key", "val_bool"],
    )


def downgrade() -> None:
    """Reverse 0002_merge_to_asset_references: drop new tables, recreate old schema.

    NOTE: Data is not recoverable. The upgrade discards all rows from the old
    tables and truncates assets. After downgrade the old schema will be empty.
    A filesystem rescan will repopulate data once the older code is running.
    """
    # Drop new tables (order matters due to FK constraints)
    op.drop_index("ix_asset_reference_meta_key_val_bool", table_name="asset_reference_meta")
    op.drop_index("ix_asset_reference_meta_key_val_num", table_name="asset_reference_meta")
    op.drop_index("ix_asset_reference_meta_key_val_str", table_name="asset_reference_meta")
    op.drop_index("ix_asset_reference_meta_key", table_name="asset_reference_meta")
    op.drop_table("asset_reference_meta")

    op.drop_index("ix_asset_reference_tags_asset_reference_id", table_name="asset_reference_tags")
    op.drop_index("ix_asset_reference_tags_tag_name", table_name="asset_reference_tags")
    op.drop_table("asset_reference_tags")

    op.drop_index("ix_asset_references_deleted_at", table_name="asset_references")
    op.drop_index("ix_asset_references_owner_name", table_name="asset_references")
    op.drop_index("ix_asset_references_last_access_time", table_name="asset_references")
    op.drop_index("ix_asset_references_created_at", table_name="asset_references")
    op.drop_index("ix_asset_references_enrichment_level", table_name="asset_references")
    op.drop_index("ix_asset_references_is_missing", table_name="asset_references")
    op.drop_index("ix_asset_references_name", table_name="asset_references")
    op.drop_index("ix_asset_references_owner_id", table_name="asset_references")
    op.drop_index("ix_asset_references_asset_id", table_name="asset_references")
    op.drop_index("uq_asset_references_file_path", table_name="asset_references")
    op.drop_table("asset_references")

    # Truncate assets (upgrade deleted all rows; downgrade starts fresh too)
    op.execute("DELETE FROM assets")

    # Recreate old tables from 0001_assets schema
    op.create_table(
        "assets_info",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("asset_id", sa.String(length=36), sa.ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("preview_id", sa.String(length=36), sa.ForeignKey("assets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("last_access_time", sa.DateTime(timezone=False), nullable=False),
        sa.UniqueConstraint("asset_id", "owner_id", "name", name="uq_assets_info_asset_owner_name"),
    )
    op.create_index("ix_assets_info_owner_id", "assets_info", ["owner_id"])
    op.create_index("ix_assets_info_asset_id", "assets_info", ["asset_id"])
    op.create_index("ix_assets_info_name", "assets_info", ["name"])
    op.create_index("ix_assets_info_created_at", "assets_info", ["created_at"])
    op.create_index("ix_assets_info_last_access_time", "assets_info", ["last_access_time"])
    op.create_index("ix_assets_info_owner_name", "assets_info", ["owner_id", "name"])

    op.create_table(
        "asset_cache_state",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("asset_id", sa.String(length=36), sa.ForeignKey("assets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column("needs_verify", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.CheckConstraint("(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_acs_mtime_nonneg"),
        sa.UniqueConstraint("file_path", name="uq_asset_cache_state_file_path"),
    )
    op.create_index("ix_asset_cache_state_file_path", "asset_cache_state", ["file_path"])
    op.create_index("ix_asset_cache_state_asset_id", "asset_cache_state", ["asset_id"])

    op.create_table(
        "asset_info_tags",
        sa.Column("asset_info_id", sa.String(length=36), sa.ForeignKey("assets_info.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tag_name", sa.String(length=512), sa.ForeignKey("tags.name", ondelete="RESTRICT"), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="manual"),
        sa.Column("added_at", sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint("asset_info_id", "tag_name", name="pk_asset_info_tags"),
    )
    op.create_index("ix_asset_info_tags_tag_name", "asset_info_tags", ["tag_name"])
    op.create_index("ix_asset_info_tags_asset_info_id", "asset_info_tags", ["asset_info_id"])

    op.create_table(
        "asset_info_meta",
        sa.Column("asset_info_id", sa.String(length=36), sa.ForeignKey("assets_info.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("val_str", sa.String(length=2048), nullable=True),
        sa.Column("val_num", sa.Numeric(38, 10), nullable=True),
        sa.Column("val_bool", sa.Boolean(), nullable=True),
        sa.Column("val_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("asset_info_id", "key", "ordinal", name="pk_asset_info_meta"),
    )
    op.create_index("ix_asset_info_meta_key", "asset_info_meta", ["key"])
    op.create_index("ix_asset_info_meta_key_val_str", "asset_info_meta", ["key", "val_str"])
    op.create_index("ix_asset_info_meta_key_val_num", "asset_info_meta", ["key", "val_num"])
    op.create_index("ix_asset_info_meta_key_val_bool", "asset_info_meta", ["key", "val_bool"])
