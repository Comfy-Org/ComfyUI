"""The seam between the task API and whatever actually moves the bytes.

Everything above this module speaks :class:`DownloadSnapshot`; below it a
backend speaks its own wire format. Swapping the transport means writing another
:class:`DownloadBackend` -- not touching the task projection, the websocket
events, or the routes, and not being visible to the browser at all.

A backend owns persistence. There is no save/restore here because :meth:`list`
is what rebuilds every caller-side view after a ComfyUI restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol


class DownloadPhase(str, Enum):
    """Transfer progress in transport-neutral terms.

    ``TRANSFERRED`` is not called "completed" on purpose: it only means the
    bytes stopped moving successfully. A model is completed once ComfyUI has
    proved the file landed where its own loaders enumerate it (BE-10028), and
    that proof belongs above this seam.
    """

    PENDING = "pending"
    TRANSFERRING = "transferring"
    TRANSFERRED = "transferred"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_PHASES = frozenset({DownloadPhase.TRANSFERRED, DownloadPhase.FAILED, DownloadPhase.CANCELLED})


class CancelOutcome(str, Enum):
    """Result of a cancel request. The three cases are kept apart because each
    implies a different next step: wait for a terminal phase, settle locally
    because none is coming, or re-read the download's real phase."""

    CANCELLING = "cancelling"
    MISSING = "missing"
    NOT_CANCELLABLE = "not_cancellable"


@dataclass(frozen=True)
class DownloadRequest:
    """A transfer to perform, with the directory already decided.

    ``directory`` is absolute and non-negotiable: ComfyUI is the only process
    that knows its own ``folder_paths``, and a backend choosing its own is how a
    download ends up somewhere no loader lists.

    ``filename`` is None when only the transport can name the file -- a CivitAI
    download URL carries the real name in its API response, not in its path.
    The directory still wins, so the file still lands somewhere loadable.
    """

    url: str
    directory: str
    filename: str | None = None


@dataclass(frozen=True)
class DownloadSnapshot:
    """One reading of a transfer's state.

    ``handle`` is opaque above the seam; callers only hand it back.
    ``bytes_total`` is None until the transport learns the size, so callers must
    not turn it into a zero denominator. Timestamps are timezone-aware UTC.
    """

    handle: str
    phase: DownloadPhase
    destination: str
    bytes_completed: int
    bytes_total: int | None
    started_at: datetime
    updated_at: datetime
    error: str | None = None
    cancellable: bool = True

    @property
    def is_terminal(self) -> bool:
        """Whether this transfer will never change again."""
        return self.phase in TERMINAL_PHASES


class DownloadBackendUnavailable(RuntimeError):
    """The transport is unusable in this installation. Unlike a rejected
    request, nothing the caller changes makes the next attempt work."""


class DownloadRejected(Exception):
    """The transport refused this request. ``code`` is the backend's own stable
    reason, passed through to the client without the route interpreting it."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


class DownloadBackend(Protocol):
    @property
    def key(self) -> str:
        """Stable transport identifier, used to namespace derived task ids."""
        ...

    async def available(self) -> bool:
        """Whether this transport can run at all in this installation."""
        ...

    def change_hint(self) -> object | None:
        """Cheap synchronous value that differs whenever :meth:`list` might.

        Lets a caller poll frequently without paying for an enumeration while
        nothing is happening. Returning None means no cheap probe exists, and
        the caller must enumerate on its own cadence.
        """
        ...

    async def start(self, request: DownloadRequest) -> DownloadSnapshot:
        """Begin a transfer and return its first snapshot.

        Raises :class:`DownloadRejected` if the transport refuses the request
        and :class:`DownloadBackendUnavailable` if it cannot run at all.
        """
        ...

    async def list(self) -> list[DownloadSnapshot]:
        """Every transfer the transport knows about, including other clients'."""
        ...

    async def get(self, handle: str) -> DownloadSnapshot | None:
        """One transfer's current state, or None if the transport forgot it."""
        ...

    async def cancel(self, handle: str) -> CancelOutcome:
        """Ask the transport to stop a transfer and reclaim what it wrote."""
        ...
