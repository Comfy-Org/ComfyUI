"""
Add loader_path column to asset_references.

Stores the in-root loader path (path relative to the storage root with the
top-level model category dropped) derived from file_path at scan/ingest time,
so the assets API can return it without re-resolving against every registered
model-folder base on every request.

Revision ID: 0006_add_loader_path
Revises: 0005_allow_case_sensitive_tags
Create Date: 2026-07-02
"""

from alembic import op
import sqlalchemy as sa

revision = "0006_add_loader_path"
down_revision = "0005_allow_case_sensitive_tags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.add_column(sa.Column("loader_path", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.drop_column("loader_path")
