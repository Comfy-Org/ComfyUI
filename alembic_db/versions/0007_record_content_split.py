"""
Record/content split.

This migration intentionally discards the existing asset database. The
asset_reference_meta, asset_reference_tags, asset_references, and assets
tables are dropped, and DELETE FROM tags removes all existing tag rows. No
data migration is performed.

A filesystem rescan after this migration will not restore user_metadata,
manually-applied tags, preview_id nominations, name renames, or job_id.
build_asset_specs derives names and tags from the path alone.

Revision ID: 0007_record_content_split
Revises: 0006_add_loader_path
Create Date: 2026-08-26
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_record_content_split"
down_revision = "0006_add_loader_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("asset_reference_meta")
    op.drop_table("asset_reference_tags")
    op.drop_table("asset_references")
    op.drop_table("assets")
    op.execute("DELETE FROM tags")
    op.create_table(
        "asset_contents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("hash", sa.String(256)),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("mtime_ns", sa.BigInteger()),
        sa.Column("is_missing", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("size_bytes >= 0", name="ck_asset_contents_size_nonneg"),
        sa.CheckConstraint("mtime_ns >= 0", name="ck_asset_contents_mtime_nonneg"),
    )
    op.create_index("ix_asset_contents_hash", "asset_contents", ["hash"])
    op.create_index(
        "uq_asset_contents_path_live", "asset_contents", ["path"], unique=True,
        sqlite_where=sa.text("is_missing = 0"),
    )
    op.create_table(
        "assets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("content_id", sa.String(36), sa.ForeignKey("asset_contents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(255)),
        sa.Column("system_metadata", sa.JSON()),
        sa.Column("job_id", sa.String(36)),
        sa.Column("user_metadata", sa.JSON()),
        sa.Column("loader_path", sa.Text()),
        sa.Column("preview_id", sa.String(36), sa.ForeignKey("assets.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_access_time", sa.DateTime()),
    )
    op.create_index("ix_assets_content_id", "assets", ["content_id"])
    op.create_index("ix_assets_name", "assets", ["name"])
    op.create_index("ix_assets_created_at", "assets", ["created_at"])
    op.create_index("ix_assets_preview_id", "assets", ["preview_id"])
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
    op.create_table("asset_tags", sa.Column("asset_id", sa.String(36), sa.ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True), sa.Column("tag_name", sa.String(512), sa.ForeignKey("tags.name", ondelete="RESTRICT"), primary_key=True), sa.Column("origin", sa.String(32), nullable=False), sa.Column("added_at", sa.DateTime(), nullable=False))
    op.create_index("ix_asset_tags_tag_name", "asset_tags", ["tag_name"])
    op.create_index("ix_asset_tags_asset_id", "asset_tags", ["asset_id"])
    op.create_table("asset_system_state", sa.Column("key", sa.String(256), primary_key=True), sa.Column("value", sa.Text(), nullable=False))


def downgrade() -> None:
    op.drop_table("asset_system_state")
    op.drop_table("asset_tags")
    op.drop_table("asset_meta")
    op.drop_table("assets")
    op.drop_table("asset_contents")
    op.create_table(
        "assets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("hash", sa.String(length=256), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonneg"),
    )
    op.create_index("uq_assets_hash", "assets", ["hash"], unique=True)
    op.create_index("ix_assets_mime_type", "assets", ["mime_type"])
    op.create_table("asset_references", sa.Column("id", sa.String(36), primary_key=True), sa.Column("asset_id", sa.String(36), sa.ForeignKey("assets.id", ondelete="CASCADE"), nullable=False), sa.Column("file_path", sa.Text()), sa.Column("loader_path", sa.Text()), sa.Column("mtime_ns", sa.BigInteger()), sa.Column("needs_verify", sa.Boolean(), nullable=False), sa.Column("is_missing", sa.Boolean(), nullable=False), sa.Column("enrichment_level", sa.Integer(), nullable=False), sa.Column("owner_id", sa.String(128), nullable=False), sa.Column("name", sa.String(512), nullable=False), sa.Column("preview_id", sa.String(36), sa.ForeignKey("asset_references.id", ondelete="SET NULL")), sa.Column("user_metadata", sa.JSON()), sa.Column("system_metadata", sa.JSON()), sa.Column("job_id", sa.String(36)), sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False), sa.Column("last_access_time", sa.DateTime(), nullable=False), sa.Column("deleted_at", sa.DateTime()))
    op.create_index("uq_asset_references_file_path", "asset_references", ["file_path"], unique=True)
    op.create_index("ix_asset_references_asset_id", "asset_references", ["asset_id"])
    op.create_index("ix_asset_references_owner_id", "asset_references", ["owner_id"])
    op.create_index("ix_asset_references_name", "asset_references", ["name"])
    op.create_index("ix_asset_references_is_missing", "asset_references", ["is_missing"])
    op.create_index("ix_asset_references_enrichment_level", "asset_references", ["enrichment_level"])
    op.create_index("ix_asset_references_created_at", "asset_references", ["created_at"])
    op.create_index("ix_asset_references_last_access_time", "asset_references", ["last_access_time"])
    op.create_index("ix_asset_references_owner_name", "asset_references", ["owner_id", "name"])
    op.create_index("ix_asset_references_deleted_at", "asset_references", ["deleted_at"])
    op.create_index("ix_asset_references_preview_id", "asset_references", ["preview_id"])
    op.create_table("asset_reference_meta", sa.Column("asset_reference_id", sa.String(36), sa.ForeignKey("asset_references.id", ondelete="CASCADE"), primary_key=True), sa.Column("key", sa.String(256), primary_key=True), sa.Column("ordinal", sa.Integer(), primary_key=True), sa.Column("val_str", sa.String(2048)), sa.Column("val_num", sa.Numeric(38, 10)), sa.Column("val_bool", sa.Boolean()), sa.Column("val_json", sa.JSON()), sa.CheckConstraint("val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL", name="ck_asset_reference_meta_has_value"))
    op.create_index("ix_asset_reference_meta_key", "asset_reference_meta", ["key"])
    op.create_index("ix_asset_reference_meta_key_val_str", "asset_reference_meta", ["key", "val_str"])
    op.create_index("ix_asset_reference_meta_key_val_num", "asset_reference_meta", ["key", "val_num"])
    op.create_index("ix_asset_reference_meta_key_val_bool", "asset_reference_meta", ["key", "val_bool"])
    op.create_table("asset_reference_tags", sa.Column("asset_reference_id", sa.String(36), sa.ForeignKey("asset_references.id", ondelete="CASCADE"), primary_key=True), sa.Column("tag_name", sa.String(512), sa.ForeignKey("tags.name", ondelete="RESTRICT"), primary_key=True), sa.Column("origin", sa.String(32), nullable=False), sa.Column("added_at", sa.DateTime(), nullable=False))
    op.create_index("ix_asset_reference_tags_tag_name", "asset_reference_tags", ["tag_name"])
    op.create_index("ix_asset_reference_tags_asset_reference_id", "asset_reference_tags", ["asset_reference_id"])
