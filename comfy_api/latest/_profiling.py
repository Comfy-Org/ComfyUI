from __future__ import annotations

import datetime
import os
import pickle
import threading
import uuid
from urllib.parse import urlencode


_ENABLED = frozenset({"all", "state", "None"})
_CONTEXT = frozenset({"all", "state", "alloc", "None"})
_STACKS = frozenset({"python", "all"})


class _SnapshotUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = {
        "builtins": {
            "dict", "list", "tuple", "set", "frozenset", "bytearray",
            "complex",
        },
        "collections": {"OrderedDict"},
    }

    def find_class(self, module, name):
        if name in self._ALLOWED_GLOBALS.get(module, frozenset()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"CUDA memory snapshot contains disallowed global "
            f"{module}.{name}")


class CudaMemoryHistoryCoordinator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_owner: str | None = None

    @staticmethod
    def _settings(enabled, context, stacks, max_entries):
        enabled = str(enabled)
        context = str(context)
        stacks = str(stacks)
        if enabled not in _ENABLED:
            raise ValueError(f"unsupported CUDA memory history mode {enabled!r}")
        if context not in _CONTEXT:
            raise ValueError(
                f"unsupported CUDA memory history context {context!r}")
        if stacks not in _STACKS:
            raise ValueError(
                f"unsupported CUDA memory history stack mode {stacks!r}")
        if isinstance(max_entries, bool):
            raise TypeError("CUDA memory history max_entries must be an integer")
        max_entries = int(max_entries)
        if not 1000 <= max_entries <= 10_000_000:
            raise ValueError(
                "CUDA memory history max_entries must be in [1000, 10000000]")
        return (
            None if enabled == "None" else enabled,
            None if context == "None" else context,
            stacks,
            max_entries,
        )

    @staticmethod
    def _history_root() -> tuple[str, str]:
        import folder_paths

        output_root = os.path.realpath(folder_paths.get_output_directory())
        history_root = os.path.join(output_root, "memory_history")
        os.makedirs(history_root, exist_ok=True)
        history_root = os.path.realpath(history_root)
        if os.path.commonpath((output_root, history_root)) != output_root:
            raise ValueError("CUDA memory history output escapes the output directory")
        return output_root, history_root

    @staticmethod
    def _prefix(value: str) -> str:
        if type(value) is not str:
            raise TypeError("CUDA memory history filename prefix must be a string")
        if not value or len(value) > 255:
            raise ValueError(
                "CUDA memory history filename prefix must contain 1..255 characters")
        if (value in {".", ".."} or "/" in value or "\\" in value
                or any(ord(character) < 32 for character in value)):
            raise ValueError(
                "CUDA memory history filename prefix must be a logical output name")
        return value

    @classmethod
    def _snapshot_path(cls, logical_name: str) -> str:
        output_root, history_root = cls._history_root()
        if type(logical_name) is not str:
            raise TypeError("CUDA memory snapshot must be a logical output name")
        normalized = logical_name.replace("\\", "/")
        parts = normalized.split("/")
        if (len(parts) != 2 or parts[0] != "memory_history"
                or not parts[1].endswith(".pt")
                or parts[1] in {".", ".."}):
            raise ValueError(
                "CUDA memory snapshot must be a logical output name under "
                "memory_history")
        target = os.path.realpath(os.path.join(output_root, *parts))
        if os.path.commonpath((history_root, target)) != history_root:
            raise ValueError("CUDA memory snapshot escapes the output directory")
        return target

    @staticmethod
    def _require_cuda():
        import torch
        import comfy.model_management as model_management

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory history requires CUDA")
        device = model_management.get_torch_device()
        if torch.device(device).type != "cuda":
            raise RuntimeError("CUDA memory history requires the CUDA device")
        return torch, model_management, device

    def start(
        self, owner: str, *, enabled="all", context="all", stacks="all",
        max_entries=100000,
    ) -> None:
        checked = self._settings(enabled, context, stacks, max_entries)
        torch, model_management, device = self._require_cuda()
        with self._lock:
            if self._active_owner not in (None, owner):
                raise RuntimeError(
                    "CUDA memory history is already recording for another owner")
            model_management.soft_empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(
                max_entries=checked[3], enabled=checked[0],
                context=checked[1], stacks=checked[2])
            self._active_owner = None if checked[0] is None else owner

    def end(self, owner: str, filename_prefix: str) -> str:
        torch, model_management, _device = self._require_cuda()
        prefix = self._prefix(filename_prefix)
        with self._lock:
            if self._active_owner != owner:
                raise RuntimeError(
                    "CUDA memory history is not recording for this owner")
            _output_root, history_root = self._history_root()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"{prefix}{timestamp}"
            target = os.path.join(history_root, f"{stem}.pt")
            counter = 1
            while os.path.exists(target):
                target = os.path.join(history_root, f"{stem}_{counter}.pt")
                counter += 1

            model_management.soft_empty_cache()
            dump_error = None
            try:
                torch.cuda.memory._dump_snapshot(target)
            except BaseException as error:
                dump_error = error
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except BaseException as error:
                raise RuntimeError(
                    "CUDA memory history could not stop recording") from error
            self._active_owner = None
            if dump_error is not None:
                raise dump_error
            return f"memory_history/{os.path.basename(target)}"

    def release(self, owner: str) -> None:
        import torch

        with self._lock:
            if self._active_owner != owner:
                return
            torch.cuda.memory._record_memory_history(enabled=None)
            self._active_owner = None

    def visualize(self, logical_name: str, node_id: str = "") -> str:
        import torch

        target = self._snapshot_path(logical_name)
        max_bytes = int(os.environ.get(
            "COMFY_SECURE_CUDA_SNAPSHOT_MAX", str(2 * 1024**3)))
        if max_bytes < 1 or os.path.getsize(target) > max_bytes:
            raise ValueError(
                f"CUDA memory snapshot exceeds the {max_bytes}-byte limit")
        with open(target, "rb") as stream:
            snapshot = _SnapshotUnpickler(stream).load()
        html = torch.cuda._memory_viz.trace_plot(snapshot)
        if not isinstance(html, str):
            raise TypeError("CUDA memory visualizer did not return HTML")

        _output_root, history_root = self._history_root()
        html_name = f"cuda_memory_history_{uuid.uuid4().hex}.html"
        html_path = os.path.join(history_root, html_name)
        try:
            with open(html_path, "x", encoding="utf-8") as stream:
                stream.write(html)
        except BaseException:
            try:
                os.unlink(html_path)
            except FileNotFoundError:
                pass
            raise
        url = "/api/view?" + urlencode({
            "type": "output",
            "filename": html_name,
            "subfolder": "memory_history",
        })
        if node_id:
            try:
                from server import PromptServer

                instance = getattr(PromptServer, "instance", None)
                if instance is not None:
                    instance.send_progress_text(url, node_id)
            except Exception:
                pass
        return url


COORDINATOR = CudaMemoryHistoryCoordinator()


class InProcessProfiling:
    def __init__(
        self, owner: str, node_id: str = "", *,
        coordinator: CudaMemoryHistoryCoordinator = COORDINATOR,
    ) -> None:
        self._owner = owner
        self._node_id = node_id
        self._coordinator = coordinator

    def for_owner(self, owner: str, node_id: str | None = None):
        return type(self)(
            owner, self._node_id if node_id is None else node_id,
            coordinator=self._coordinator)

    async def cuda_memory_start(
        self, *, enabled="all", context="all", stacks="all",
        max_entries=100000,
    ) -> None:
        self._coordinator.start(
            self._owner, enabled=enabled, context=context,
            stacks=stacks, max_entries=max_entries)

    async def cuda_memory_end(
        self, filename_prefix="comfy_cuda_memory_history",
    ) -> str:
        return self._coordinator.end(self._owner, filename_prefix)

    async def cuda_memory_visualize(self, snapshot: str) -> str:
        return self._coordinator.visualize(snapshot, self._node_id)

    async def close(self) -> None:
        self._coordinator.release(self._owner)
