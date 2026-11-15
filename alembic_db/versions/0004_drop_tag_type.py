"""
Drop the vestigial tags.tag_type column.

tag_type was always "user" in practice — no code path ever set it to anything
else (no system/seeded classification was ever wired up) and nothing queried it.
The column, its index (ix_tags_tag_type), and the corresponding API field were
dead weight, so they are removed.

Revision ID: 0004_drop_tag_type
Revises: 0003_add_metadata_job_id
Create Date: 2026-06-03
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_drop_tag_type"
down_revision = "0003_add_metadata_job_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tags") as batch_op:
        batch_op.drop_index("ix_tags_tag_type")
        batch_op.drop_column("tag_type")


def downgrade() -> None:
    with op.batch_alter_table("tags") as batch_op:
        batch_op.add_column(
            sa.Column(
                "tag_type",
                sa.String(length=32),
                nullable=False,
                server_default="user",
            )
        )
        batch_op.create_index("ix_tags_tag_type", ["tag_type"])
