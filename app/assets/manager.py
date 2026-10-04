"""Selects and implements the enabled and disabled asset managers.

``default_asset_manager`` checks database dependencies before enabling assets
and chooses ``NoAssets`` when the requested mode cannot run.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

from aiohttp import web

from app.assets import mode
from app.assets.event_log import error_kind
from app.assets.lifecycle import record_hash_mode_transition_intent, run_shutdown, run_startup
from app.database.db import dependencies_available, missing_dependencies
from app.user_manager import UserManager
from comfy.cli_args import args
from utils.install_util import get_missing_requirements_message

# These need the database packages. Without them only NoAssets is used, and it
# does not touch these names.
if dependencies_available():
    from app.assets.api.routes import register_assets_routes
    from app.assets.seeder import ScanPhase, asset_seeder
    from app.assets.services.ingest import (
        register_cached_output as ingest_register_cached_output,
        register_executed_output as ingest_register_executed_output,
        register_file_in_place,
    )
    from app.assets.services.path_utils import get_known_subfolder_tags
    from app.assets.services.schemas import RegisteredAsset, UploadAssetView

# SQLite already waits its 5 s busy timeout at each blocked write, so three attempts ride out a
# write lock held for ~15 s.
_LOCKED_ATTEMPTS = 3
_LOCKED_RETRY_PAUSE_SECONDS = 0.2


class AssetRegistrationError(Exception):
    """An uploaded file was saved but could not be registered as an asset."""

    def __init__(self, locked: bool):
        super().__init__("database is locked" if locked else "asset registration failed")
        self.locked = locked


def _retry_while_locked(register: Callable[[], Any]) -> Any:
    for _ in range(_LOCKED_ATTEMPTS - 1):
        try:
            return register()
        except Exception as exc:
            if error_kind(exc) != "database_locked":
                raise
        time.sleep(_LOCKED_RETRY_PAUSE_SECONDS)
    return register()


class AssetManager(Protocol):
    @property
    def enabled(self) -> bool: ...

    def startup(self) -> None: ...

    def shutdown(self) -> None: ...

    def register_routes(
        self, app: web.Application, user_manager: UserManager | None
    ) -> None: ...

    def ensure_scan_started(self) -> None: ...

    def pause_background_scan(self) -> None: ...

    def queue_output_scan(self) -> None: ...

    def resume_background_scan(self) -> None: ...

    def register_upload(
        self,
        abs_path: str,
        name: str,
        upload_type: str,
        subfolder: str,
        *,
        content_written: bool,
    ) -> UploadAssetView | None:
        """None when assets are disabled; raises AssetRegistrationError if registration fails."""
        ...

    def register_executed_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None: ...

    def register_cached_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None: ...

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None: ...


class _ArgsLike(Protocol):
    enable_assets: bool
    enable_asset_hashing: bool


def _shutdown_assets() -> None:
    if dependencies_available():
        asset_seeder.shutdown()
    run_shutdown()


class NoAssets:
    def __init__(self, args: _ArgsLike) -> None:
        self._args = args

    @property
    def enabled(self) -> bool:
        return False

    def startup(self) -> None:
        mode.init(self._args)
        run_startup(enable_assets=False)

    def shutdown(self) -> None:
        _shutdown_assets()

    def register_routes(
        self, app: web.Application, user_manager: UserManager | None
    ) -> None:
        if not dependencies_available():
            return
        register_assets_routes(app)
        asset_seeder.disable()

    def ensure_scan_started(self) -> None:
        return None

    def pause_background_scan(self) -> None:
        return None

    def queue_output_scan(self) -> None:
        return None

    def resume_background_scan(self) -> None:
        return None

    def register_upload(
        self,
        abs_path: str,
        name: str,
        upload_type: str,
        subfolder: str,
        *,
        content_written: bool,
    ) -> UploadAssetView | None:
        return None

    def register_executed_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None:
        return None

    def register_cached_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None:
        return None

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        return None


class AssetsEnabled:
    def __init__(self, args: _ArgsLike) -> None:
        self._args = args

    @property
    def enabled(self) -> bool:
        return True

    def startup(self) -> None:
        mode.init(self._args)
        record_hash_mode_transition_intent()
        run_startup(enable_assets=True)

    def shutdown(self) -> None:
        _shutdown_assets()

    def register_routes(
        self, app: web.Application, user_manager: UserManager | None
    ) -> None:
        register_assets_routes(app, user_manager)

    def ensure_scan_started(self) -> None:
        asset_seeder.start(roots=("models", "input", "output"))

    def pause_background_scan(self) -> None:
        asset_seeder.pause()

    def queue_output_scan(self) -> None:
        if not asset_seeder.is_disabled():
            # FULL, not ENRICH: only a walk finds outputs a node never declared. Do not downgrade without re-weighing the cost.
            asset_seeder.enqueue_scan(
                roots=("output",),
                phase=ScanPhase.FULL,
                compute_hashes=self._args.enable_asset_hashing,
            )

    def resume_background_scan(self) -> None:
        asset_seeder.resume()

    def register_upload(
        self,
        abs_path: str,
        name: str,
        upload_type: str,
        subfolder: str,
        *,
        content_written: bool,
    ) -> UploadAssetView | None:
        try:
            tag = upload_type if upload_type in ("input", "output") else "input"
            tags = [tag] + get_known_subfolder_tags(subfolder)
            result = _retry_while_locked(lambda: register_file_in_place(
                abs_path=abs_path,
                name=name,
                tags=tags,
                content_written=content_written,
            ))
            asset = RegisteredAsset(
                id=result.ref.id,
                content_id=result.content_id,
                job_id=result.ref.job_id,
                name=result.ref.name,
            )
            return UploadAssetView(
                asset=asset,
                asset_hash=result.asset.hash,
                size=result.asset.size_bytes,
                mime_type=result.asset.mime_type,
                tags=result.tags,
            )
        except Exception as exc:
            logging.warning("Failed to register uploaded image as asset", exc_info=True)
            raise AssetRegistrationError(error_kind(exc) == "database_locked") from exc

    def register_executed_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None:
        return ingest_register_executed_output(abs_path, job_id)

    def register_cached_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None:
        return ingest_register_cached_output(abs_path, job_id)

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        asset_seeder.set_event_sink(sink)


def default_asset_manager() -> AssetManager:
    if args.enable_assets and not dependencies_available():
        missing = ", ".join(missing_dependencies()) or "see the import error above"
        logging.error(
            f"--enable-assets requires packages that could not be imported: {missing}. "
            f"Assets are disabled.\n{get_missing_requirements_message()}"
        )
        return NoAssets(args)
    return AssetsEnabled(args) if args.enable_assets else NoAssets(args)
