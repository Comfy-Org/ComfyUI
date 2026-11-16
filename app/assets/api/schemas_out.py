"""Defines the response bodies the asset API returns, fixing the field names,
types and serialization its HTTP contract promises. Routes dump these models
with null fields omitted, so whether a field is absent and whether it is empty
stay distinguishable to a client.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Asset(BaseModel):
    id: str
    name: str = Field(
        ...,
        description="Record label, usually derived from the source filename.",
    )
    hash: str | None = None
    loader_path: str | None = Field(
        default=None,
        description="The value a loader consumes to load this asset. `None` when no loader can resolve the file.",
    )
    display_name: str | None = Field(
        default=None,
        description="Human-facing label for the asset. Not unique.",
    )
    is_immutable: bool = False
    size: int | None = None
    mime_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    preview_url: str | None = None
    preview_id: str | None = None  # references an asset_reference id, not an asset id
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None
    job_id: str | None = None
    prompt_id: str | None = None  # deprecated: use job_id
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at", "last_access_time")
    def _serialize_datetime(self, v: datetime | None, _info):
        return v.isoformat() if v else None


class AssetCreated(Asset):
    created_new: bool


class AssetsList(BaseModel):
    assets: list[Asset]
    total: int
    has_more: bool
    # Opaque cursor for the next page. Omitted when there are no more results.
    next_cursor: str | None = None


class TagUsage(BaseModel):
    name: str
    count: int


class TagsList(BaseModel):
    tags: list[TagUsage] = Field(default_factory=list)
    total: int
    has_more: bool


class TagsAdd(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    added: list[str] = Field(default_factory=list)
    already_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)


class TagsRemove(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    removed: list[str] = Field(default_factory=list)
    not_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)
    protected: list[str] = Field(default_factory=list)


class TagHistogram(BaseModel):
    tag_counts: dict[str, int]
