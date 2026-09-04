import logging
from typing import Any, Callable, Protocol

from aiohttp import web

from app.assets import mode
from app.assets.api.routes import register_assets_routes
from app.assets.lifecycle import record_hash_mode_transition_intent, run_shutdown, run_startup
from app.assets.seeder import ScanPhase, asset_seeder
from app.assets.services.ingest import (
    register_cached_output as ingest_register_cached_output,
    register_executed_output as ingest_register_executed_output,
    register_file_in_place,
)
from app.assets.services.path_utils import get_known_subfolder_tags
from app.assets.services.schemas import RegisteredAsset, UploadAssetView
from app.database.db import dependencies_available
from app.user_manager import UserManager
from comfy.cli_args import args


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
    ) -> UploadAssetView | None: ...

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
        record_hash_mode_transition_intent()
        run_startup(enable_assets=False)

    def shutdown(self) -> None:
        _shutdown_assets()

    def register_routes(
        self, app: web.Application, user_manager: UserManager | None
    ) -> None:
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
            result = register_file_in_place(
                abs_path=abs_path,
                name=name,
                tags=tags,
                content_written=content_written,
            )
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
        except Exception:
            logging.warning("Failed to register uploaded image as asset", exc_info=True)
            return None

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
        logging.warning(
            "Assets requested but database dependencies unavailable; asset endpoints "
            "will answer 503. Please install the updated requirements.txt file."
        )
        return NoAssets(args)
    return AssetsEnabled(args) if args.enable_assets else NoAssets(args)
