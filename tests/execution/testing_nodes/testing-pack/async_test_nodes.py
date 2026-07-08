import torch
import asyncio
from typing import Dict
from comfy.utils import ProgressBar
from comfy_execution.graph_utils import GraphBuilder
from comfy.comfy_types.node_typing import ComfyNodeABC
from comfy.comfy_types import IO


class TestAsyncValidation(ComfyNodeABC):
    """Test node with async VALIDATE_INPUTS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 5.0}),
                "threshold": ("FLOAT", {"default": 10.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "experimental/async"

    @classmethod
    async def VALIDATE_INPUTS(cls, value, threshold):
        # Simulate async validation (e.g., checking remote service)
        await asyncio.sleep(0.05)

        if value > threshold:
            return f"Value {value} exceeds threshold {threshold}"
        return True

    def process(self, value, threshold):
        # Create image based on value
        intensity = value / 10.0
        image = torch.ones([1, 512, 512, 3]) * intensity
        return (image,)


class TestAsyncError(ComfyNodeABC):
    """Test node that errors during async execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
                "error_after": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "error_execution"
    CATEGORY = "experimental/async"

    async def error_execution(self, value, error_after):
        await asyncio.sleep(error_after)
        raise RuntimeError("Intentional async execution error for testing")


class TestAsyncValidationError(ComfyNodeABC):
    """Test node with async validation that always fails."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 5.0}),
                "max_value": ("FLOAT", {"default": 10.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "experimental/async"

    @classmethod
    async def VALIDATE_INPUTS(cls, value, max_value):
        await asyncio.sleep(0.05)
        # Always fail validation for values > max_value
        if value > max_value:
            return f"Async validation failed: {value} > {max_value}"
        return True

    def process(self, value, max_value):
        # This won't be reached if validation fails
        image = torch.ones([1, 512, 512, 3]) * (value / max_value)
        return (image,)


class TestAsyncTimeout(ComfyNodeABC):
    """Test node that simulates timeout scenarios."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
                "timeout": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "operation_time": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "timeout_execution"
    CATEGORY = "experimental/async"

    async def timeout_execution(self, value, timeout, operation_time):
        try:
            # This will timeout if operation_time > timeout
            await asyncio.wait_for(asyncio.sleep(operation_time), timeout=timeout)
            return (value,)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Operation timed out after {timeout} seconds")


class TestSyncError(ComfyNodeABC):
    """Test node that errors synchronously (for mixed sync/async testing)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "sync_error"
    CATEGORY = "experimental/async"

    def sync_error(self, value):
        raise RuntimeError("Intentional sync execution error for testing")


class TestAsyncLazyCheck(ComfyNodeABC):
    """Test node with async check_lazy_status."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": (IO.ANY, {"lazy": True}),
                "input2": (IO.ANY, {"lazy": True}),
                "condition": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "experimental/async"

    async def check_lazy_status(self, condition, input1, input2):
        # Simulate async checking (e.g., querying remote service)
        await asyncio.sleep(0.05)

        needed = []
        if condition and input1 is None:
            needed.append("input1")
        if not condition and input2 is None:
            needed.append("input2")
        return needed

    def process(self, input1, input2, condition):
        # Return a simple image
        return (torch.ones([1, 512, 512, 3]),)


class TestDynamicAsyncGeneration(ComfyNodeABC):
    """Test node that dynamically generates async nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "num_async_nodes": ("INT", {"default": 3, "min": 1, "max": 10}),
                "sleep_duration": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_async_workflow"
    CATEGORY = "experimental/async"

    def generate_async_workflow(self, image1, image2, num_async_nodes, sleep_duration):
        g = GraphBuilder()

        # Create multiple async sleep nodes
        sleep_nodes = []
        for i in range(num_async_nodes):
            image = image1 if i % 2 == 0 else image2
            sleep_node = g.node("TestSleep", value=image, seconds=sleep_duration)
            sleep_nodes.append(sleep_node)

        # Average all results
        if len(sleep_nodes) == 1:
            final_node = sleep_nodes[0]
        else:
            avg_inputs = {"input1": sleep_nodes[0].out(0)}
            for i, node in enumerate(sleep_nodes[1:], 2):
                avg_inputs[f"input{i}"] = node.out(0)
            final_node = g.node("TestVariadicAverage", **avg_inputs)

        return {
            "result": (final_node.out(0),),
            "expand": g.finalize(),
        }


class TestAsyncResourceUser(ComfyNodeABC):
    """Test node that uses resources during async execution."""

    # Class-level resource tracking for testing
    _active_resources: Dict[str, bool] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
                "resource_id": ("STRING", {"default": "resource_0"}),
                "duration": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "use_resource"
    CATEGORY = "experimental/async"

    async def use_resource(self, value, resource_id, duration):
        # Check if resource is already in use
        if self._active_resources.get(resource_id, False):
            raise RuntimeError(f"Resource {resource_id} is already in use!")

        # Mark resource as in use
        self._active_resources[resource_id] = True

        try:
            # Simulate resource usage
            await asyncio.sleep(duration)
            return (value,)
        finally:
            # Always clean up resource
            self._active_resources[resource_id] = False


class TestAsyncBatchProcessing(ComfyNodeABC):
    """Test async processing of batched inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "process_time_per_item": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_batch"
    CATEGORY = "experimental/async"

    async def process_batch(self, images, process_time_per_item, unique_id):
        batch_size = images.shape[0]
        pbar = ProgressBar(batch_size, node_id=unique_id)

        # Process each image in the batch
        processed = []
        for i in range(batch_size):
            # Simulate async processing
            await asyncio.sleep(process_time_per_item)

            # Simple processing: invert the image
            processed_image = 1.0 - images[i:i+1]
            processed.append(processed_image)

            pbar.update(1)

        # Stack processed images
        result = torch.cat(processed, dim=0)
        return (result,)


class TestAsyncConcurrentLimit(ComfyNodeABC):
    """Test concurrent execution limits for async nodes."""

    _semaphore = asyncio.Semaphore(2)  # Only allow 2 concurrent executions

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.ANY, {}),
                "duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0}),
                "node_id": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "limited_execution"
    CATEGORY = "experimental/async"

    async def limited_execution(self, value, duration, node_id):
        async with self._semaphore:
            # Node {node_id} acquired semaphore
            await asyncio.sleep(duration)
            # Node {node_id} releasing semaphore
            return (value,)


# Add node mappings
ASYNC_TEST_NODE_CLASS_MAPPINGS = {
    "TestAsyncValidation": TestAsyncValidation,
    "TestAsyncError": TestAsyncError,
    "TestAsyncValidationError": TestAsyncValidationError,
    "TestAsyncTimeout": TestAsyncTimeout,
    "TestSyncError": TestSyncError,
    "TestAsyncLazyCheck": TestAsyncLazyCheck,
    "TestDynamicAsyncGeneration": TestDynamicAsyncGeneration,
    "TestAsyncResourceUser": TestAsyncResourceUser,
    "TestAsyncBatchProcessing": TestAsyncBatchProcessing,
    "TestAsyncConcurrentLimit": TestAsyncConcurrentLimit,
}

ASYNC_TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestAsyncValidation": "Test Async Validation",
    "TestAsyncError": "Test Async Error",
    "TestAsyncValidationError": "Test Async Validation Error",
    "TestAsyncTimeout": "Test Async Timeout",
    "TestSyncError": "Test Sync Error",
    "TestAsyncLazyCheck": "Test Async Lazy Check",
    "TestDynamicAsyncGeneration": "Test Dynamic Async Generation",
    "TestAsyncResourceUser": "Test Async Resource User",
    "TestAsyncBatchProcessing": "Test Async Batch Processing",
    "TestAsyncConcurrentLimit": "Test Async Concurrent Limit",
}
