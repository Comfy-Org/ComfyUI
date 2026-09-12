"""Defines the frozen result types the service layer hands back to routes and to
the seeder, keeping those callers off ORM objects and the sessions that own
them. Each type carries only what its caller renders, so nothing can be lazily
loaded from a row after its session has closed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

UserMetadata = dict[str, Any] | None


@dataclass(frozen=True)
class AssetData:
    hash: str | None
    size_bytes: int | None
    mime_type: str | None
    is_missing: bool = False


@dataclass(frozen=True)
class ReferenceData:
    """Data transfer object for AssetReference."""

    id: str
    name: str
    file_path: str | None
    user_metadata: UserMetadata
    preview_id: str | None
    created_at: datetime
    updated_at: datetime
    loader_path: str | None = None
    system_metadata: dict[str, Any] | None = None
    job_id: str | None = None
    last_access_time: datetime | None = None


@dataclass(frozen=True)
class AssetDetailResult:
    ref: ReferenceData
    asset: AssetData | None
    tags: list[str]


@dataclass(frozen=True)
class RegisterAssetResult:
    ref: ReferenceData
    asset: AssetData
    tags: list[str]
    created: bool


@dataclass(frozen=True)
class IngestResult:
    asset_created: bool
    asset_updated: bool
    ref_created: bool
    ref_updated: bool
    reference_id: str | None


class TagUsage(NamedTuple):
    name: str
    count: int


@dataclass(frozen=True)
class AssetSummaryData:
    ref: ReferenceData
    asset: AssetData | None
    tags: list[str]


@dataclass(frozen=True)
class ListAssetsResult:
    items: list[AssetSummaryData]
    total: int
    next_cursor: str | None = None


@dataclass(frozen=True)
class DownloadResolutionResult:
    abs_path: str
    content_type: str
    download_name: str


@dataclass(frozen=True)
class RegisteredAsset:
    id: str
    content_id: str
    job_id: str | None
    name: str


@dataclass(frozen=True)
class UploadAssetView:
    asset: RegisteredAsset
    asset_hash: str | None
    size: int | None
    mime_type: str | None
    tags: list[str]


@dataclass(frozen=True)
class UploadResult:
    ref: ReferenceData
    content_id: str
    asset: AssetData
    tags: list[str]
    created_new: bool
