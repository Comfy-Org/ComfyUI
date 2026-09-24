"""Declares the asset schema: content rows describing bytes on disk, records
describing what a user sees, and the tag and metadata tables hanging off them.
The split is the point — many records can name one content row, and retiring
content by marking it missing rather than deleting it is what keeps a path's
history intact. Constraints declared here, not application code, are what make
a negative size or a second live row at one path impossible.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.assets.helpers import get_utc_now
from app.database.models import Base


class AssetContent(Base):
    __tablename__ = "asset_contents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    hash: Mapped[str | None] = mapped_column(String(256), index=True)
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        CheckConstraint("size_bytes >= 0", name="ck_asset_contents_size_nonneg"),
        nullable=False,
        default=0,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    mtime_ns: Mapped[int | None] = mapped_column(
        BigInteger,
        CheckConstraint("mtime_ns >= 0", name="ck_asset_contents_mtime_nonneg"),
    )
    is_missing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )

    records: Mapped[list[Asset]] = relationship(back_populates="content")

    __table_args__ = (
        Index(
            "uq_asset_contents_path_live",
            "path",
            unique=True,
            sqlite_where=text("is_missing = 0"),
        ),
    )


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    content_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("asset_contents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    system_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    job_id: Mapped[str | None] = mapped_column(String(36))
    user_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    loader_path: Mapped[str | None] = mapped_column(Text)
    preview_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("assets.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )
    # Explicit user/API edit time; never use onupdate because row writes are not always edits.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )
    last_access_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))

    content: Mapped[AssetContent] = relationship(back_populates="records", lazy="selectin")
    preview: Mapped[Asset | None] = relationship(
        "Asset", foreign_keys=[preview_id], remote_side=lambda: [Asset.id]
    )
    metadata_entries: Mapped[list[AssetMeta]] = relationship(
        back_populates="asset", cascade="all,delete-orphan", passive_deletes=True
    )
    tag_links: Mapped[list[AssetTag]] = relationship(
        back_populates="asset", cascade="all,delete-orphan", passive_deletes=True
    )
    tags: Mapped[list[Tag]] = relationship(
        secondary="asset_tags", back_populates="assets", viewonly=True, lazy="selectin"
    )

    __table_args__ = (
        Index("ix_assets_content_id", "content_id"),
        Index("ix_assets_name", "name"),
        Index("ix_assets_created_at", "created_at"),
        Index("ix_assets_preview_id", "preview_id"),
    )


class AssetMeta(Base):
    __tablename__ = "asset_meta"

    asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)
    val_str: Mapped[str | None] = mapped_column(String(2048))
    val_num: Mapped[Decimal | None] = mapped_column(Numeric(38, 10))
    val_bool: Mapped[bool | None] = mapped_column(Boolean)
    val_json: Mapped[Any | None] = mapped_column(JSON)

    asset: Mapped[Asset] = relationship(back_populates="metadata_entries")

    __table_args__ = (
        Index("ix_asset_meta_key", "key"),
        Index("ix_asset_meta_key_val_str", "key", "val_str"),
        Index("ix_asset_meta_key_val_num", "key", "val_num"),
        Index("ix_asset_meta_key_val_bool", "key", "val_bool"),
        CheckConstraint(
            "val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL",
            name="ck_asset_meta_has_value",
        ),
    )


class AssetTag(Base):
    __tablename__ = "asset_tags"

    asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    tag_name: Mapped[str] = mapped_column(
        String(512), ForeignKey("tags.name", ondelete="RESTRICT"), primary_key=True
    )
    origin: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )

    asset: Mapped[Asset] = relationship(back_populates="tag_links")
    tag: Mapped[Tag] = relationship(back_populates="asset_links")

    __table_args__ = (
        Index("ix_asset_tags_tag_name", "tag_name"),
        Index("ix_asset_tags_asset_id", "asset_id"),
    )


class Tag(Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(512), primary_key=True)
    asset_links: Mapped[list[AssetTag]] = relationship(back_populates="tag")
    assets: Mapped[list[Asset]] = relationship(
        secondary="asset_tags", back_populates="tags", viewonly=True
    )


class AssetSystemState(Base):
    __tablename__ = "asset_system_state"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
