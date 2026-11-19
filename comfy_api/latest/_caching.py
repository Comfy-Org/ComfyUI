from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


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
