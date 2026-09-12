import time
import tracemalloc
import logging
from typing import Any, Tuple, Dict, Optional, NamedTuple

logger = logging.getLogger("ComfyUI-Latency-Profiler")

class PerformanceMetrics(NamedTuple):
    """Container for recorded execution latency and memory metrics."""
    step_label: str
    timestamp_ms: float
    execution_time_ms: float
    memory_peak_mb: Optional[float]


class TerminalLatencyProfiler:
    """ComfyUI custom node that monitors workflow execution timing and memory overhead."""
    
    def __init__(self) -> None:
        """Initialize per-instance execution state to prevent cross-workflow contamination."""
        self._last_timestamp: Optional[float] = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define node parameters and inputs for the ComfyUI frontend interface."""
        return {
            "required": {
                "passthrough_data": ("*",),
                "step_label": ("STRING", {"default": "Pipeline Checkpoint"}),
                "reset_timer": ("BOOLEAN", {"default": False}),
                "track_memory": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough_data",)
    FUNCTION = "profile_latency"
    CATEGORY = "profiling"

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        """Force re-execution on every prompt queue submission by returning NaN."""
        return float("nan")

    def profile_latency(
        self,
        passthrough_data: Any,
        step_label: str = "Pipeline Checkpoint",
        reset_timer: bool = False,
        track_memory: bool = False
    ) -> Tuple[Any]:
        """
        Record monotonic execution timing and memory utilization before returning passthrough data.
        
        Guarantees workflow safety by catching unexpected failures during metric collection.
        """
        current_time = time.perf_counter()
        memory_peak_mb: Optional[float] = None
        
        # 1. Scoped Memory Measurement
        if track_memory:
            try:
                # Enable memory tracing locally to prevent global leaks
                tracemalloc.start()
            except Exception as err:
                logger.warning(f"Failed to initialize memory tracking: {err}")
        
        try:
            # 2. Timing Measurement
            if reset_timer or self._last_timestamp is None:
                execution_time_ms = 0.0
            else:
                execution_time_ms = (current_time - self._last_timestamp) * 1000.0

            self._last_timestamp = current_time

            # 3. Peak Memory Capture & Clean Teardown
            if track_memory and tracemalloc.is_tracing():
                try:
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    memory_peak_mb = peak_bytes / (1024 * 1024)
                finally:
                    tracemalloc.stop()

            # 4. Formatted Terminal Logging
            metrics = PerformanceMetrics(
                step_label=step_label,
                timestamp_ms=current_time * 1000.0,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=memory_peak_mb
            )
            self._output_formatted_log(metrics)

        except Exception as err:
            # Catch-all exception handling to ensure pipeline safety
            logger.error(f"Error during latency profiler execution: {err}", exc_info=True)
            if track_memory and tracemalloc.is_tracing():
                tracemalloc.stop()

        # Always safely return the input data unchanged
        return (passthrough_data,)

    def _output_formatted_log(self, metrics: PerformanceMetrics) -> None:
        """Output metric results as a single line to standard output."""
        mem_str = f" | Peak Mem: {metrics.memory_peak_mb:.2f} MB" if metrics.memory_peak_mb is not None else ""
        print(f"[LATENCY LOG] Label: {metrics.step_label} | Exec: {metrics.execution_time_ms:.2f} ms{mem_str}")