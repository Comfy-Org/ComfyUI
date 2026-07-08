from __future__ import annotations

import uuid
from datetime import datetime
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
)
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from app.assets.helpers import get_utc_now
from app.database.models import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )

    references: Mapped[list[AssetReference]] = relationship(
        "AssetReference",
        back_populates="asset",
        primaryjoin=lambda: Asset.id == foreign(AssetReference.asset_id),
        foreign_keys=lambda: [AssetReference.asset_id],
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    # preview_id on AssetReference is a self-referential FK to asset_references.id

    __table_args__ = (
        Index("uq_assets_hash", "hash", unique=True),
        Index("ix_assets_mime_type", "mime_type"),
        CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonneg"),
    )

    def __repr__(self) -> str:
        return f"<Asset id={self.id} hash={(self.hash or '')[:12]}>"


class AssetReference(Base):
    """Unified model combining file cache state and user-facing metadata.

    Each row represents either:
    - A filesystem reference (file_path is set) with cache state
    - An API-created reference (file_path is NULL) without cache state
    """

    __tablename__ = "asset_references"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )

    # Cache state fields (from former AssetCacheState)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    mtime_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    needs_verify: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enrichment_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Info fields (from former AssetInfo)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    preview_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("asset_references.id", ondelete="SET NULL")
    )
    user_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True)
    )
    system_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True, default=None
    )
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )
    last_access_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True, default=None
    )

    asset: Mapped[Asset] = relationship(
        "Asset",
        back_populates="references",
        foreign_keys=[asset_id],
        lazy="selectin",
    )
    preview_ref: Mapped[AssetReference | None] = relationship(
        "AssetReference",
        foreign_keys=[preview_id],
        remote_side=lambda: [AssetReference.id],
    )

    metadata_entries: Mapped[list[AssetReferenceMeta]] = relationship(
        back_populates="asset_reference",
        cascade="all,delete-orphan",
        passive_deletes=True,
    )

    tag_links: Mapped[list[AssetReferenceTag]] = relationship(
        back_populates="asset_reference",
        cascade="all,delete-orphan",
        passive_deletes=True,
        overlaps="tags,asset_references",
    )

    tags: Mapped[list[Tag]] = relationship(
        secondary="asset_reference_tags",
        back_populates="asset_references",
        lazy="selectin",
        viewonly=True,
        overlaps="tag_links,asset_reference_links,asset_references,tag",
    )

    __table_args__ = (
        Index("uq_asset_references_file_path", "file_path", unique=True),
        Index("ix_asset_references_asset_id", "asset_id"),
        Index("ix_asset_references_owner_id", "owner_id"),
        Index("ix_asset_references_name", "name"),
        Index("ix_asset_references_is_missing", "is_missing"),
        Index("ix_asset_references_enrichment_level", "enrichment_level"),
        Index("ix_asset_references_created_at", "created_at"),
        Index("ix_asset_references_last_access_time", "last_access_time"),
        Index("ix_asset_references_deleted_at", "deleted_at"),
        Index("ix_asset_references_preview_id", "preview_id"),
        Index("ix_asset_references_owner_name", "owner_id", "name"),
        CheckConstraint(
            "(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_ar_mtime_nonneg"
        ),
        CheckConstraint(
            "enrichment_level >= 0 AND enrichment_level <= 2",
            name="ck_ar_enrichment_level_range",
        ),
    )

    def __repr__(self) -> str:
        path_part = f" path={self.file_path!r}" if self.file_path else ""
        return f"<AssetReference id={self.id} name={self.name!r}{path_part}>"


class AssetReferenceMeta(Base):
    __tablename__ = "asset_reference_meta"

    asset_reference_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("asset_references.id", ondelete="CASCADE"),
        primary_key=True,
    )
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)

    val_str: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    val_num: Mapped[float | None] = mapped_column(Numeric(38, 10), nullable=True)
    val_bool: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    val_json: Mapped[Any | None] = mapped_column(JSON(none_as_null=True), nullable=True)

    asset_reference: Mapped[AssetReference] = relationship(
        back_populates="metadata_entries"
    )

    __table_args__ = (
        Index("ix_asset_reference_meta_key", "key"),
        Index("ix_asset_reference_meta_key_val_str", "key", "val_str"),
        Index("ix_asset_reference_meta_key_val_num", "key", "val_num"),
        Index("ix_asset_reference_meta_key_val_bool", "key", "val_bool"),
        CheckConstraint(
            "val_str IS NOT NULL OR val_num IS NOT NULL OR val_bool IS NOT NULL OR val_json IS NOT NULL",
            name="has_value",
        ),
    )


class AssetReferenceTag(Base):
    __tablename__ = "asset_reference_tags"

    asset_reference_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("asset_references.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_name: Mapped[str] = mapped_column(
        String(512), ForeignKey("tags.name", ondelete="RESTRICT"), primary_key=True
    )
    origin: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=get_utc_now
    )

    asset_reference: Mapped[AssetReference] = relationship(back_populates="tag_links")
    tag: Mapped[Tag] = relationship(back_populates="asset_reference_links")

    __table_args__ = (
        Index("ix_asset_reference_tags_tag_name", "tag_name"),
        Index("ix_asset_reference_tags_asset_reference_id", "asset_reference_id"),
    )


class Tag(Base):
    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(512), primary_key=True)

    asset_reference_links: Mapped[list[AssetReferenceTag]] = relationship(
        back_populates="tag",
        overlaps="asset_references,tags",
    )
    asset_references: Mapped[list[AssetReference]] = relationship(
        secondary="asset_reference_tags",
        back_populates="tags",
        viewonly=True,
        overlaps="asset_reference_links,tag_links,tags,asset_reference",
    )

    def __repr__(self) -> str:
        return f"<Tag {self.name}>"
