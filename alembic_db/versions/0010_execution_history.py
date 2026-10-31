"""Add opt-in terminal execution history."""

from alembic import op
import sqlalchemy as sa

revision = "0010_execution_history"
down_revision = "0009_created_at_from_mtime"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "execution_history",
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("prompt_id", sa.Text(), nullable=False, unique=True),
        sa.Column("format_version", sa.Integer(), nullable=False),
        sa.Column("record_json", sa.Text(), nullable=False),
    )


def downgrade():
    op.drop_table("execution_history")
