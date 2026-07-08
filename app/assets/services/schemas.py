from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

from app.assets.database.models import Asset, AssetReference

UserMetadata = dict[str, Any] | None


@dataclass(frozen=True)
class AssetData:
    hash: str | None
    size_bytes: int | None
    mime_type: str | None


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
class UploadResult:
    ref: ReferenceData
    asset: AssetData
    tags: list[str]
    created_new: bool


def extract_reference_data(ref: AssetReference) -> ReferenceData:
    return ReferenceData(
        id=ref.id,
        name=ref.name,
        file_path=ref.file_path,
        user_metadata=ref.user_metadata,
        preview_id=ref.preview_id,
        system_metadata=ref.system_metadata,
        job_id=ref.job_id,
        created_at=ref.created_at,
        updated_at=ref.updated_at,
        last_access_time=ref.last_access_time,
    )


def extract_asset_data(asset: Asset | None) -> AssetData | None:
    if asset is None:
        return None
    return AssetData(
        hash=asset.hash,
        size_bytes=asset.size_bytes,
        mime_type=asset.mime_type,
    )
