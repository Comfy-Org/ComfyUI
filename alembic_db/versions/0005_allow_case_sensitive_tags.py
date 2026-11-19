"""
Allow case-sensitive tag names.

Revision ID: 0005_allow_case_sensitive_tags
Revises: 0004_drop_tag_type
Create Date: 2026-06-16
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_allow_case_sensitive_tags"
down_revision = "0004_drop_tag_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite cannot ALTER/DROP CHECK constraints. Recreate the small tag
        # vocabulary table without the lowercase constraint while preserving
        # existing tag names.
        op.execute("PRAGMA foreign_keys=OFF")
        try:
            op.execute(
                "CREATE TABLE tags_new ("
                "name VARCHAR(512) NOT NULL, "
                "CONSTRAINT pk_tags PRIMARY KEY (name)"
                ")"
            )
            op.execute("INSERT INTO tags_new(name) SELECT name FROM tags")
            op.execute("DROP TABLE tags")
            op.execute("ALTER TABLE tags_new RENAME TO tags")
        finally:
            op.execute("PRAGMA foreign_keys=ON")
        return

    op.drop_constraint("ck_tags_ck_tags_lowercase", "tags", type_="check")


def downgrade() -> None:
    # Existing mixed-case tags cannot satisfy the old constraint. Lowercase them
    # before restoring it, merging duplicate vocabulary/link rows that collide.
    bind = op.get_bind()

    tag_names = [row[0] for row in bind.execute(sa.text("SELECT name FROM tags"))]
    existing_names = set(tag_names)
    lowercase_names = sorted({name.lower() for name in tag_names})
    missing_lowercase_rows = [
        {"name": name} for name in lowercase_names if name not in existing_names
    ]
    if missing_lowercase_rows:
        bind.execute(sa.text("INSERT INTO tags(name) VALUES (:name)"), missing_lowercase_rows)

    link_rows = bind.execute(
        sa.text(
            "SELECT asset_reference_id, tag_name, origin, added_at "
            "FROM asset_reference_tags "
            "ORDER BY asset_reference_id, tag_name"
        )
    ).mappings()
    deduped_links = {}
    for row in link_rows:
        key = (row["asset_reference_id"], row["tag_name"].lower())
        deduped_links.setdefault(
            key,
            {
                "asset_reference_id": row["asset_reference_id"],
                "tag_name": row["tag_name"].lower(),
                "origin": row["origin"],
                "added_at": row["added_at"],
            },
        )

    op.execute("DELETE FROM asset_reference_tags")
    if deduped_links:
        bind.execute(
            sa.text(
                "INSERT INTO asset_reference_tags "
                "(asset_reference_id, tag_name, origin, added_at) "
                "VALUES (:asset_reference_id, :tag_name, :origin, :added_at)"
            ),
            list(deduped_links.values()),
        )
    op.execute("DELETE FROM tags WHERE name != lower(name)")

    if bind.dialect.name == "sqlite":
        op.execute("PRAGMA foreign_keys=OFF")
        try:
            op.execute(
                "CREATE TABLE tags_new ("
                "name VARCHAR(512) NOT NULL, "
                "CONSTRAINT pk_tags PRIMARY KEY (name), "
                "CONSTRAINT ck_tags_lowercase CHECK (name = lower(name))"
                ")"
            )
            op.execute("INSERT INTO tags_new(name) SELECT name FROM tags")
            op.execute("DROP TABLE tags")
            op.execute("ALTER TABLE tags_new RENAME TO tags")
        finally:
            op.execute("PRAGMA foreign_keys=ON")
        return

    op.create_check_constraint(
        "ck_tags_ck_tags_lowercase", "tags", "name = lower(name)"
    )
