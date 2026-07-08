import json
from dataclasses import dataclass
from typing import Any, Literal

from app.assets.helpers import validate_blake3_hash
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conint,
    field_validator,
    model_validator,
)


class UploadError(Exception):
    """Error during upload parsing with HTTP status and code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class AssetValidationError(Exception):
    """Validation error in asset processing (invalid tags, metadata, etc.)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ParsedUpload:
    """Result of parsing a multipart upload request."""

    file_present: bool
    file_written: int
    file_client_name: str | None
    tmp_path: str | None
    tags_raw: list[str]
    provided_name: str | None
    user_metadata_raw: str | None
    provided_hash: str | None
    provided_hash_exists: bool | None
    provided_mime_type: str | None = None
    provided_preview_id: str | None = None


class ListAssetsQuery(BaseModel):
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    name_contains: str | None = None

    # Accept either a JSON string (query param) or a dict
    metadata_filter: dict[str, Any] | None = None

    limit: conint(ge=1, le=500) = 20
    offset: conint(ge=0) = 0
    # Opaque keyset cursor. When supplied, `offset` is ignored. Cursor pagination
    # is supported for sort values `created_at`, `updated_at`, `name`, `size`.
    # Supplying `after` together with `sort=last_access_time` returns
    # 400 INVALID_CURSOR; that sort only supports offset/limit.
    after: str | None = None

    sort: Literal["name", "created_at", "updated_at", "size", "last_access_time"] = (
        "created_at"
    )
    order: Literal["asc", "desc"] = "desc"

    @field_validator("include_tags", "exclude_tags", mode="before")
    @classmethod
    def _split_csv_tags(cls, v):
        # Accept "a,b,c" or ["a","b"] (we are liberal in what we accept)
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.extend([t.strip() for t in item.split(",") if t.strip()])
            return out
        return v

    @field_validator("metadata_filter", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v):
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
            except Exception as e:
                raise ValueError(f"metadata_filter must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter must be a JSON object")
            return parsed
        return None


class UpdateAssetBody(BaseModel):
    name: str | None = None
    user_metadata: dict[str, Any] | None = None
    preview_id: str | None = None  # references an asset_reference id, not an asset id

    @model_validator(mode="after")
    def _validate_at_least_one_field(self):
        if all(
            v is None
            for v in (self.name, self.user_metadata, self.preview_id)
        ):
            raise ValueError(
                "Provide at least one of: name, user_metadata, preview_id."
            )
        return self


class CreateFromHashBody(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    hash: str
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    mime_type: str | None = None
    preview_id: str | None = None  # references an asset_reference id, not an asset id

    @field_validator("hash")
    @classmethod
    def _require_blake3(cls, v):
        return validate_blake3_hash(v or "")

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags_field(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            out = [str(t).strip().lower() for t in v if str(t).strip()]
            seen = set()
            dedup = []
            for t in out:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            return dedup
        if isinstance(v, str):
            return [t.strip().lower() for t in v.split(",") if t.strip()]
        return []


class TagsRefineQuery(BaseModel):
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    name_contains: str | None = None
    metadata_filter: dict[str, Any] | None = None
    limit: conint(ge=1, le=1000) = 100

    @field_validator("include_tags", "exclude_tags", mode="before")
    @classmethod
    def _split_csv_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.extend([t.strip() for t in item.split(",") if t.strip()])
            return out
        return v

    @field_validator("metadata_filter", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v):
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
            except Exception as e:
                raise ValueError(f"metadata_filter must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter must be a JSON object")
            return parsed
        return None


class TagsListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    prefix: str | None = Field(None, min_length=1, max_length=256)
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0, le=10_000_000)
    order: Literal["count_desc", "name_asc"] = "count_desc"
    include_zero: bool = True

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v.lower() or None


class TagsAdd(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tags: list[str] = Field(..., min_length=1)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        out = []
        for t in v:
            if not isinstance(t, str):
                raise TypeError("tags must be strings")
            tnorm = t.strip().lower()
            if tnorm:
                out.append(tnorm)
        seen = set()
        deduplicated = []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduplicated.append(x)
        return deduplicated


class TagsRemove(TagsAdd):
    pass


class UploadAssetSpec(BaseModel):
    """Upload Asset operation.

    - tags: optional list; if provided, first is root ('models'|'input'|'output');
            if root == 'models', second must be a valid category
    - name: display name
    - user_metadata: arbitrary JSON object (optional)
    - hash: optional canonical 'blake3:<hex>' for validation / fast-path
    - mime_type: optional MIME type override
    - preview_id: optional asset_reference ID for preview

    Files are stored using the content hash as filename stem.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    tags: list[str] = Field(default_factory=list)
    name: str | None = Field(default=None, max_length=512, description="Display Name")
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    preview_id: str | None = Field(default=None)  # references an asset_reference id

    @field_validator("hash", mode="before")
    @classmethod
    def _parse_hash(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return validate_blake3_hash(s)

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(cls, v):
        """
        Accepts a list of strings (possibly multiple form fields),
        where each string can be:
          - JSON array (e.g., '["models","loras","foo"]')
          - comma-separated ('models, loras, foo')
          - single token ('models')
        Returns a normalized, deduplicated, ordered list.
        """
        items: list[str] = []
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]

        if isinstance(v, list):
            for item in v:
                if item is None:
                    continue
                s = str(item).strip()
                if not s:
                    continue
                if s.startswith("["):
                    try:
                        arr = json.loads(s)
                        if isinstance(arr, list):
                            items.extend(str(x) for x in arr)
                            continue
                    except Exception:
                        pass  # fallback to CSV parse below
                items.extend([p for p in s.split(",") if p.strip()])
        else:
            return []

        # normalize + dedupe
        norm = []
        seen = set()
        for t in items:
            tnorm = str(t).strip().lower()
            if tnorm and tnorm not in seen:
                seen.add(tnorm)
                norm.append(tnorm)
        return norm

    @field_validator("user_metadata", mode="before")
    @classmethod
    def _parse_metadata_json(cls, v):
        if v is None or isinstance(v, dict):
            return v or {}
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
            except Exception as e:
                raise ValueError(f"user_metadata must be JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError("user_metadata must be a JSON object")
            return parsed
        return {}

    @model_validator(mode="after")
    def _validate_order(self):
        if not self.tags:
            raise ValueError("at least one tag is required for uploads")
        root = self.tags[0]
        if root not in {"models", "input", "output"}:
            raise ValueError("first tag must be one of: models, input, output")
        if root == "models":
            if len(self.tags) < 2:
                raise ValueError(
                    "models uploads require a category tag as the second tag"
                )
        return self
