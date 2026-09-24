from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

from aiohttp import web

from app.assets.manager import AssetManager
from app.assets.services.schemas import RegisteredAsset, UploadAssetView

if TYPE_CHECKING:
    from app.user_manager import UserManager


CallArgument: TypeAlias = str | bool | None
EventSink: TypeAlias = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class AssetCall:
    method: str
    arguments: tuple[CallArgument, ...]


@dataclass(frozen=True, slots=True)
class Delivery:
    abs_path: str
    asset: RegisteredAsset
    superseded: bool = False


class InMemoryAssets:

    def __init__(self) -> None:
        self.calls: list[AssetCall] = []
        self.deliveries_by_path: dict[str, list[Delivery]] = {}
        self._live_output_by_path: dict[str, RegisteredAsset] = {}
        self._counter: int = 0

    @property
    def enabled(self) -> bool:
        return True

    def startup(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def register_routes(
        self, app: web.Application, user_manager: UserManager | None
    ) -> None:
        return None

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
        self._record("register_executed_output", abs_path, job_id)
        deliveries = self.deliveries_by_path.setdefault(abs_path, [])
        for index, delivery in enumerate(deliveries):
            deliveries[index] = replace(delivery, superseded=True)

        asset_id = self._next_asset_id()
        asset = RegisteredAsset(
            id=asset_id,
            content_id=f"content-{asset_id}",
            job_id=job_id,
            name=os.path.basename(abs_path),
        )
        deliveries.append(Delivery(abs_path=abs_path, asset=asset))
        self._live_output_by_path[abs_path] = asset
        return asset

    def register_cached_output(
        self, abs_path: str, job_id: str | None
    ) -> RegisteredAsset | None:
        self._record("register_cached_output", abs_path, job_id)
        source = self._live_output_by_path.get(abs_path)
        if source is None:
            return None

        asset = RegisteredAsset(
            id=self._next_asset_id(),
            content_id=source.content_id,
            job_id=job_id,
            name=os.path.basename(abs_path),
        )
        self.deliveries_by_path[abs_path].append(
            Delivery(abs_path=abs_path, asset=asset)
        )
        return asset

    def set_event_sink(self, sink: EventSink) -> None:
        return None

    def _record(self, method: str, *arguments: CallArgument) -> None:
        self.calls.append(AssetCall(method, arguments))

    def _next_asset_id(self) -> str:
        self._counter += 1
        return f"asset-{self._counter}"


def test_conforms() -> None:
    manager: AssetManager = InMemoryAssets()
    assert manager.enabled, (
        "structural smoke test: AssetManager is a non-runtime-checkable Protocol, so isinstance is unavailable"
    )
    manager.startup()
    manager.shutdown()
    manager.register_routes(web.Application(), None)
    manager.ensure_scan_started()
    manager.pause_background_scan()
    manager.queue_output_scan()
    manager.resume_background_scan()
    assert (
        manager.register_upload(
            "/output/upload.png",
            "upload.png",
            "output",
            "",
            content_written=True,
        )
        is None
    )
    executed = manager.register_executed_output("/output/executed.png", "job-1")
    assert executed is not None
    cached = manager.register_cached_output("/output/executed.png", "job-2")
    assert cached is not None

    def sink(event: str, payload: dict[str, Any]) -> None:
        return None

    manager.set_event_sink(sink)


def test_register_executed_output_supersedes_prior_deliveries() -> None:
    manager = InMemoryAssets()
    abs_path = "/output/image.png"

    first = manager.register_executed_output(abs_path, "job-1")
    assert first is not None
    cached = manager.register_cached_output(abs_path, "job-2")
    assert cached is not None
    replacement = manager.register_executed_output(abs_path, "job-3")
    assert replacement is not None

    deliveries = manager.deliveries_by_path[abs_path]
    assert [delivery.superseded for delivery in deliveries] == [True, True, False]
    assert cached.content_id == first.content_id
    assert replacement.id != first.id
    assert replacement.content_id != first.content_id


def test_register_cached_output_returns_none_without_live_content() -> None:
    manager = InMemoryAssets()

    cached = manager.register_cached_output("/output/missing.png", "job-1")

    assert cached is None
    assert manager.calls == [
        AssetCall("register_cached_output", ("/output/missing.png", "job-1"))
    ]
