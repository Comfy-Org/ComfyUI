"""How the operation vocabulary describes itself.

Operations are addressed by name rather than by method, so the vocabulary can
grow without the API changing shape. That only works if a caller can find out
what exists and what a given operation expects, which is what a capability
descriptor carries.

A descriptor reuses the ``{attrs, inputs, outputs}`` schema shape that node
manifests already use, so one vocabulary describes both nodes and operations
rather than two that drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Capability:
    """One named, versioned operation the runtime can perform.

    ``version`` increments only when the operation's contract changes in a way
    that would break a caller written against the previous one. Adding an
    optional parameter is not such a change; removing one, or changing what an
    existing parameter means, is. Both versions may be registered at once, so a
    node written against the older contract keeps working while a newer one
    adopts the change.
    """

    name: str
    version: int = 1
    schema: Optional[Mapping[str, Any]] = None

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def split_request(requested: str) -> tuple[str, Optional[int]]:
    """Split ``"name@2"`` into ``("name", 2)`` and ``"name"`` into ``("name", None)``.

    A bare name means "whatever you have", which is what a node wants when it
    does not care; an explicit version means "exactly this contract".
    """
    name, separator, version = requested.partition("@")
    if not separator:
        return requested, None
    if not version.isdigit():
        raise ValueError(f"capability version must be a number: {requested!r}")
    return name, int(version)


class CapabilityRegistry:
    """The operation vocabulary: what exists, at which versions, and how to run it.

    Lookup answers "can you do this"; enumeration answers "what can you do".
    The second is what lets a caller adapt to an unfamiliar runtime instead of
    guessing, and it is why registration takes a descriptor rather than a bare
    function.
    """

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, int], Any] = {}
        self._descriptors: dict[tuple[str, int], Capability] = {}

    def register(self, capability: Capability, handler: Any) -> None:
        key = (capability.name, capability.version)
        self._handlers[key] = handler
        self._descriptors[key] = capability

    def resolve(self, requested: str) -> Optional[Any]:
        """The handler for a request, or None when nothing satisfies it."""
        key = self._match(requested)
        return self._handlers.get(key) if key else None

    def describe(self, requested: str) -> Optional[Capability]:
        key = self._match(requested)
        return self._descriptors.get(key) if key else None

    def supports(self, requested: str) -> bool:
        """Whether a request can be satisfied. Never raises for an unknown name:
        not knowing an operation is an ordinary answer, not a failure."""
        try:
            return self._match(requested) is not None
        except ValueError:
            return False

    def capabilities(self) -> tuple[Capability, ...]:
        """Every registered capability, ordered by name then version."""
        return tuple(
            self._descriptors[key]
            for key in sorted(self._descriptors)
        )

    def versions(self, name: str) -> tuple[int, ...]:
        return tuple(sorted(v for (n, v) in self._descriptors if n == name))

    def _match(self, requested: str) -> Optional[tuple[str, int]]:
        name, version = split_request(requested)
        if version is not None:
            key = (name, version)
            return key if key in self._handlers else None
        available = self.versions(name)
        # A bare name resolves to the newest contract: a caller that did not
        # pin a version is asking for current behaviour.
        return (name, available[-1]) if available else None
