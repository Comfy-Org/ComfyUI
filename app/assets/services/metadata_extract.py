"""Metadata extraction for asset scanning.

Tier 1: Filesystem metadata (zero parsing)
Tier 2: Safetensors header metadata (fast JSON read only)
"""


import json
import logging
import mimetypes
import os
import struct
from dataclasses import dataclass
from typing import Any

from utils.mime_types import init_mime_types

init_mime_types()

# Supported safetensors extensions
SAFETENSORS_EXTENSIONS = frozenset({".safetensors", ".sft"})

# Maximum safetensors header size to read (8MB)
MAX_SAFETENSORS_HEADER_SIZE = 8 * 1024 * 1024


@dataclass
class ExtractedMetadata:
    """Metadata extracted from a file during scanning."""

    # Tier 1: Filesystem (always available)
    filename: str = ""
    file_path: str = ""  # Full absolute path to the file
    content_length: int = 0
    content_type: str | None = None
    format: str = ""  # file extension without dot

    # Tier 2: Safetensors header (if available)
    base_model: str | None = None
    trained_words: list[str] | None = None
    air: str | None = None  # CivitAI AIR identifier
    has_preview_images: bool = False

    # Source provenance (populated if embedded in safetensors)
    source_url: str | None = None
    source_arn: str | None = None
    repo_url: str | None = None
    preview_url: str | None = None
    source_hash: str | None = None

    # HuggingFace specific
    repo_id: str | None = None
    revision: str | None = None
    filepath: str | None = None
    resolve_url: str | None = None

    def to_user_metadata(self) -> dict[str, Any]:
        """Convert to user_metadata dict for AssetReference.user_metadata JSON field."""
        data: dict[str, Any] = {
            "filename": self.filename,
            "content_length": self.content_length,
            "format": self.format,
        }
        if self.file_path:
            data["file_path"] = self.file_path
        if self.content_type:
            data["content_type"] = self.content_type

        # Tier 2 fields
        if self.base_model:
            data["base_model"] = self.base_model
        if self.trained_words:
            data["trained_words"] = self.trained_words
        if self.air:
            data["air"] = self.air
        if self.has_preview_images:
            data["has_preview_images"] = True

        # Source provenance
        if self.source_url:
            data["source_url"] = self.source_url
        if self.source_arn:
            data["source_arn"] = self.source_arn
        if self.repo_url:
            data["repo_url"] = self.repo_url
        if self.preview_url:
            data["preview_url"] = self.preview_url
        if self.source_hash:
            data["source_hash"] = self.source_hash

        # HuggingFace
        if self.repo_id:
            data["repo_id"] = self.repo_id
        if self.revision:
            data["revision"] = self.revision
        if self.filepath:
            data["filepath"] = self.filepath
        if self.resolve_url:
            data["resolve_url"] = self.resolve_url

        return data

    def to_meta_rows(self, reference_id: str) -> list[dict]:
        """Convert to asset_reference_meta rows for typed/indexed querying."""
        rows: list[dict] = []

        def add_str(key: str, val: str | None, ordinal: int = 0) -> None:
            if val:
                rows.append({
                    "asset_reference_id": reference_id,
                    "key": key,
                    "ordinal": ordinal,
                    "val_str": val[:2048] if len(val) > 2048 else val,
                    "val_num": None,
                    "val_bool": None,
                    "val_json": None,
                })

        def add_num(key: str, val: int | float | None) -> None:
            if val is not None:
                rows.append({
                    "asset_reference_id": reference_id,
                    "key": key,
                    "ordinal": 0,
                    "val_str": None,
                    "val_num": val,
                    "val_bool": None,
                    "val_json": None,
                })

        def add_bool(key: str, val: bool | None) -> None:
            if val is not None:
                rows.append({
                    "asset_reference_id": reference_id,
                    "key": key,
                    "ordinal": 0,
                    "val_str": None,
                    "val_num": None,
                    "val_bool": val,
                    "val_json": None,
                })

        # Tier 1
        add_str("filename", self.filename)
        add_num("content_length", self.content_length)
        add_str("content_type", self.content_type)
        add_str("format", self.format)

        # Tier 2
        add_str("base_model", self.base_model)
        add_str("air", self.air)
        has_previews = self.has_preview_images if self.has_preview_images else None
        add_bool("has_preview_images", has_previews)

        # trained_words as multiple rows with ordinals
        if self.trained_words:
            for i, word in enumerate(self.trained_words[:100]):  # limit to 100 words
                add_str("trained_words", word, ordinal=i)

        # Source provenance
        add_str("source_url", self.source_url)
        add_str("source_arn", self.source_arn)
        add_str("repo_url", self.repo_url)
        add_str("preview_url", self.preview_url)
        add_str("source_hash", self.source_hash)

        # HuggingFace
        add_str("repo_id", self.repo_id)
        add_str("revision", self.revision)
        add_str("filepath", self.filepath)
        add_str("resolve_url", self.resolve_url)

        return rows


def _read_safetensors_header(
    path: str, max_size: int = MAX_SAFETENSORS_HEADER_SIZE
) -> dict[str, Any] | None:
    """Read only the JSON header from a safetensors file.

    This is very fast - reads 8 bytes for header length, then the JSON header.
    No tensor data is loaded.

    Args:
        path: Absolute path to safetensors file
        max_size: Maximum header size to read (default 8MB)

    Returns:
        Parsed header dict or None if failed
    """
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return None
            length_of_header = struct.unpack("<Q", header_bytes)[0]
            if length_of_header > max_size:
                return None
            header_data = f.read(length_of_header)
            if len(header_data) < length_of_header:
                return None
            return json.loads(header_data.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error):
        return None


def _extract_safetensors_metadata(
    header: dict[str, Any], meta: ExtractedMetadata
) -> None:
    """Extract metadata from safetensors header __metadata__ section.

    Modifies meta in-place.
    """
    st_meta = header.get("__metadata__", {})
    if not isinstance(st_meta, dict):
        return

    # Common model metadata
    meta.base_model = (
        st_meta.get("ss_base_model_version")
        or st_meta.get("modelspec.base_model")
        or st_meta.get("base_model")
    )

    # Trained words / trigger words
    trained_words = st_meta.get("ss_tag_frequency")
    if trained_words and isinstance(trained_words, str):
        try:
            tag_freq = json.loads(trained_words)
            # Extract unique tags from all datasets
            all_tags: set[str] = set()
            for dataset_tags in tag_freq.values():
                if isinstance(dataset_tags, dict):
                    all_tags.update(dataset_tags.keys())
            if all_tags:
                meta.trained_words = sorted(all_tags)[:100]
        except json.JSONDecodeError:
            pass

    # Direct trained_words field (some formats)
    if not meta.trained_words:
        tw = st_meta.get("trained_words")
        if isinstance(tw, str):
            try:
                parsed = json.loads(tw)
                if isinstance(parsed, list):
                    meta.trained_words = [str(x) for x in parsed]
                else:
                    meta.trained_words = [w.strip() for w in tw.split(",") if w.strip()]
            except json.JSONDecodeError:
                meta.trained_words = [w.strip() for w in tw.split(",") if w.strip()]
        elif isinstance(tw, list):
            meta.trained_words = [str(x) for x in tw]

    # CivitAI AIR
    meta.air = st_meta.get("air") or st_meta.get("modelspec.air")

    # Preview images (ssmd_cover_images)
    cover_images = st_meta.get("ssmd_cover_images")
    if cover_images:
        meta.has_preview_images = True

    # Source provenance fields
    meta.source_url = st_meta.get("source_url")
    meta.source_arn = st_meta.get("source_arn")
    meta.repo_url = st_meta.get("repo_url")
    meta.preview_url = st_meta.get("preview_url")
    meta.source_hash = st_meta.get("source_hash") or st_meta.get("sshs_model_hash")

    # HuggingFace fields
    meta.repo_id = st_meta.get("repo_id") or st_meta.get("hf_repo_id")
    meta.revision = st_meta.get("revision") or st_meta.get("hf_revision")
    meta.filepath = st_meta.get("filepath") or st_meta.get("hf_filepath")
    meta.resolve_url = st_meta.get("resolve_url") or st_meta.get("hf_url")


def extract_file_metadata(
    abs_path: str,
    stat_result: os.stat_result | None = None,
    relative_filename: str | None = None,
) -> ExtractedMetadata:
    """Extract metadata from a file using tier 1 and tier 2 methods.

    Tier 1: Filesystem metadata from path and stat
    Tier 2: Safetensors header parsing if applicable

    Args:
        abs_path: Absolute path to the file
        stat_result: Optional pre-fetched stat result (saves a syscall)
        relative_filename: Optional relative filename to use instead of basename
            (e.g., "flux/123/model.safetensors" for model paths)

    Returns:
        ExtractedMetadata with all available fields populated
    """
    meta = ExtractedMetadata()

    # Tier 1: Filesystem metadata
    meta.filename = relative_filename or os.path.basename(abs_path)
    meta.file_path = abs_path
    _, ext = os.path.splitext(abs_path)
    meta.format = ext.lstrip(".").lower() if ext else ""

    mime_type, _ = mimetypes.guess_type(abs_path)
    meta.content_type = mime_type

    # Size from stat
    if stat_result is None:
        try:
            stat_result = os.stat(abs_path, follow_symlinks=True)
        except OSError:
            pass

    if stat_result:
        meta.content_length = stat_result.st_size

    # Tier 2: Safetensors header (if applicable and enabled)
    if ext.lower() in SAFETENSORS_EXTENSIONS:
        header = _read_safetensors_header(abs_path)
        if header:
            try:
                _extract_safetensors_metadata(header, meta)
            except Exception as e:
                logging.debug("Safetensors meta extract failed %s: %s", abs_path, e)

    return meta
