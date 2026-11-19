"""
Add system_metadata and job_id columns to asset_references.
Change preview_id FK from assets.id to asset_references.id.

Revision ID: 0003_add_metadata_job_id
Revises: 0002_merge_to_asset_references
Create Date: 2026-03-09
"""

from alembic import op
import sqlalchemy as sa

from app.database.models import NAMING_CONVENTION

revision = "0003_add_metadata_job_id"
down_revision = "0002_merge_to_asset_references"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.add_column(
            sa.Column("system_metadata", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("job_id", sa.String(length=36), nullable=True)
        )

    # Change preview_id FK from assets.id to asset_references.id (self-ref).
    # Existing values are asset-content IDs that won't match reference IDs,
    # so null them out first.
    op.execute("UPDATE asset_references SET preview_id = NULL WHERE preview_id IS NOT NULL")
    with op.batch_alter_table(
        "asset_references", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_asset_references_preview_id_assets", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_asset_references_preview_id_asset_references",
            "asset_references",
            ["preview_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_asset_references_preview_id", ["preview_id"]
        )

    # Purge any all-null meta rows before adding the constraint
    op.execute(
        "DELETE FROM asset_reference_meta"
        " WHERE val_str IS NULL AND val_num IS NULL AND val_bool IS NULL AND val_json IS NULL"
    )
    with op.batch_alter_table("asset_reference_meta") as batch_op:
        batch_op.create_check_constraint(
            "ck_asset_reference_meta_has_value",
            "val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL",
        )


def downgrade() -> None:
    # SQLite doesn't reflect CHECK constraints, so we must declare it
    # explicitly via table_args for the batch recreate to find it.
    # Use the fully-rendered constraint name to avoid the naming convention
    # doubling the prefix.
    with op.batch_alter_table(
        "asset_reference_meta",
        table_args=[
            sa.CheckConstraint(
                "val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL",
                name="ck_asset_reference_meta_has_value",
            ),
        ],
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_asset_reference_meta_has_value", type_="check"
        )

    with op.batch_alter_table(
        "asset_references", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_index("ix_asset_references_preview_id")
        batch_op.drop_constraint(
            "fk_asset_references_preview_id_asset_references", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_asset_references_preview_id_assets",
            "assets",
            ["preview_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("asset_references") as batch_op:
        batch_op.drop_column("job_id")
        batch_op.drop_column("system_metadata")
