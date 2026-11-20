"""
Index asset_contents.path for prefix queries that include missing rows.

Revision ID: 0010_index_content_path
Revises: 0009_created_at_from_mtime
Create Date: 2026-11-20
"""

from alembic import op

revision = "0010_index_content_path"
down_revision = "0009_created_at_from_mtime"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_asset_contents_path", "asset_contents", ["path"])


def downgrade() -> None:
    op.drop_index("ix_asset_contents_path", table_name="asset_contents")
