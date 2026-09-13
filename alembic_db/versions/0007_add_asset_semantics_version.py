"""
Add asset_semantics_version table.

Revision ID: 0007_add_asset_semantics_version
Revises: 0006_add_loader_path
Create Date: 2026-08-18
"""

from alembic import op
import sqlalchemy as sa

revision = "0007_add_asset_semantics_version"
down_revision = "0006_add_loader_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_semantics_version",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_asset_semantics_version"),
    )


def downgrade() -> None:
    op.drop_table("asset_semantics_version")
