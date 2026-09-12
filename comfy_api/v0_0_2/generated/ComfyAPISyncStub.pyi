from typing import Any, Dict, List, Optional, Tuple, Union, Set, Sequence, cast, NamedTuple
from comfy_api.v0_0_2 import ComfyAPIAdapter_v0_0_2
from PIL.Image import Image
from comfy_api.latest._caching import CacheProvider
from comfy_api.latest._execution_lifecycle import LifecycleHandler
from comfy_api.latest._io import NodeReplace
from torch import Tensor
class ComfyAPISyncStub:
    def __init__(self) -> None: ...

    class CachingSync:
        """
        External cache provider API for sharing cached node outputs
        across ComfyUI instances.

        Example::

            from comfy_api.latest import Caching

            class MyCacheProvider(Caching.CacheProvider):
                async def on_lookup(self, context):
                    ...  # check external storage

                async def on_store(self, context, value):
                    ...  # store to external storage

            Caching.register_provider(MyCacheProvider())
        """
        def __init__(self) -> None: ...
        """
        Register an external cache provider. Providers are called in registration order.
        """
        def register_provider(self, provider: CacheProvider) -> None: ...
        """
        Unregister a previously registered cache provider.
        """
        def unregister_provider(self, provider: CacheProvider) -> None: ...

    class ExecutionSync:
        def __init__(self) -> None: ...
        """
        Update the progress bar displayed in the ComfyUI interface.

        This function allows custom nodes and API calls to report their progress
        back to the user interface, providing visual feedback during long operations.

        Migration from previous API: comfy.utils.PROGRESS_BAR_HOOK
        """
        def set_progress(self, value: float, max_value: float, node_id: Union[str, None] = None, preview_image: Union[Image, Tensor, None] = None, ignore_size_limit: bool = False) -> None: ...

    class ExecutionLifecycleSync:
        def __init__(self) -> None: ...
        """
        Register a task lifecycle handler during extension loading.
        """
        def register(self, handler: LifecycleHandler) -> None: ...

    class NodeReplacementSync:
        def __init__(self) -> None: ...
        """
        Register a node replacement mapping.
        """
        def register(self, node_replace: NodeReplace) -> None: ...

    caching: CachingSync
    execution: ExecutionSync
    execution_lifecycle: ExecutionLifecycleSync
    node_replacement: NodeReplacementSync
