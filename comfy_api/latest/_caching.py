from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


def allows_external_cache(value: object) -> bool:
    """Respect a value's optional non-consuming external-storage refusal.

    Managed values can declare ``_comfy_external_cacheable = False``. Neither
    this predicate nor local memory sizing may realize such a value. Existing
    values and providers retain their default storage behavior.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if getattr(item, "_comfy_external_cacheable", True) is False:
            return False
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return True


@dataclass
class CacheContext:
    node_id: str
    class_type: str
    cache_key_hash: str  # SHA256 hex digest


@dataclass
class CacheValue:
    outputs: list
    ui: dict = None


class CacheProvider(ABC):
    """Abstract base class for external cache providers.
    Exceptions from provider methods are caught by the caller and never break execution.
    """

    @abstractmethod
    async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
        """Called on local cache miss. Return CacheValue if found, None otherwise."""
        pass

    @abstractmethod
    async def on_store(self, context: CacheContext, value: CacheValue) -> None:
        """Called after local store. Dispatched via asyncio.create_task."""
        pass

    def should_cache(self, context: CacheContext, value: Optional[CacheValue] = None) -> bool:
        """Return False to skip external caching for this node. Default: True."""
        return True

    def on_prompt_start(self, prompt_id: str) -> None:
        pass

    def on_prompt_end(self, prompt_id: str) -> None:
        pass
