from .latency_profiler import ComfyLatencyProfiler

NODE_CLASS_MAPPINGS = {
    "ComfyLatencyProfiler": ComfyLatencyProfiler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLatencyProfiler": "⚡ Latency Profiler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']