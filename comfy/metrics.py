import logging
import sys

from comfy.cli_args import args


_enabled = False
_prometheus_client = None
queue_length_gauge = None
queue_wait_histogram = None
job_duration_histogram = None
jobs_counter = None
vram_gauge = None
loaded_models_gauge = None
model_swaps_counter = None
node_execution_histogram = None
cache_requests_counter = None


DURATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0,
    15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0,
)


def init_metrics():
    global _enabled, _prometheus_client
    global queue_length_gauge, queue_wait_histogram, job_duration_histogram, jobs_counter
    global vram_gauge, loaded_models_gauge, model_swaps_counter, node_execution_histogram, cache_requests_counter

    if _enabled:
        return True
    if not args.enable_prometheus:
        return False

    try:
        import prometheus_client
    except ModuleNotFoundError:
        logging.error("Prometheus metrics require the prometheus_client package.")
        return False

    queue_length_gauge = prometheus_client.Gauge(
        "comfyui_queue_length",
        "Current number of pending workflows in the prompt queue",
    )
    queue_wait_histogram = prometheus_client.Histogram(
        "comfyui_queue_wait_seconds",
        "Time from workflow submission to processing",
        buckets=DURATION_BUCKETS,
    )
    job_duration_histogram = prometheus_client.Histogram(
        "comfyui_job_duration_seconds",
        "Execution time for completed workflows",
        buckets=DURATION_BUCKETS,
    )
    jobs_counter = prometheus_client.Counter(
        "comfyui_jobs",
        "Completed, failed, and interrupted workflow executions",
        ["status"],
    )
    vram_gauge = prometheus_client.Gauge(
        "comfyui_vram_bytes",
        "GPU memory usage reported by the PyTorch allocator and device driver",
        ["device", "type"],
    )
    loaded_models_gauge = prometheus_client.Gauge(
        "comfyui_loaded_models_count",
        "Models currently held in memory",
    )
    model_swaps_counter = prometheus_client.Counter(
        "comfyui_model_swaps",
        "Model load and unload transfers",
    )
    node_execution_histogram = prometheus_client.Histogram(
        "comfyui_node_execution_seconds",
        "Execution time for individual nodes",
        ["node_type"],
        buckets=DURATION_BUCKETS,
    )
    cache_requests_counter = prometheus_client.Counter(
        "comfyui_cache_requests",
        "Node execution cache lookups",
        ["result"],
    )

    if args.prometheus_port is not None:
        prometheus_client.start_http_server(args.prometheus_port, addr="127.0.0.1")
        logging.info("Prometheus metrics server started on port %s", args.prometheus_port)

    _prometheus_client = prometheus_client
    _enabled = True
    return True


def is_enabled():
    return _enabled


def generate_latest():
    return _prometheus_client.generate_latest()


def update_queue_length(length):
    if queue_length_gauge is not None:
        queue_length_gauge.set(length)


def record_queue_wait(seconds):
    if queue_wait_histogram is not None:
        queue_wait_histogram.observe(seconds)


def record_job_duration(seconds):
    if job_duration_histogram is not None:
        job_duration_histogram.observe(seconds)


def increment_jobs_total(status):
    if jobs_counter is not None:
        jobs_counter.labels(status=status).inc()


def update_vram_metrics(device):
    if vram_gauge is None or device.type == "cpu":
        return

    torch = sys.modules["torch"]
    device_name = f"{device.type}:{device.index if device.index is not None else 0}"
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory()
        reserved = torch.mps.driver_allocated_memory()
    elif device.type == "xpu":
        allocated = torch.xpu.memory_allocated(device)
        reserved = torch.xpu.memory_reserved(device)
    elif device.type == "npu":
        allocated = torch.npu.memory_allocated(device)
        reserved = torch.npu.memory_reserved(device)
    else:
        return

    vram_gauge.labels(device=device_name, type="allocated").set(allocated)
    vram_gauge.labels(device=device_name, type="reserved").set(reserved)
    if device.type == "cuda":
        vram_gauge.labels(device=device_name, type="device_used").set(total - free)


def update_loaded_models_count(count):
    if loaded_models_gauge is not None:
        loaded_models_gauge.set(count)


def increment_model_swaps():
    if model_swaps_counter is not None:
        model_swaps_counter.inc()


def record_node_execution(node_type, seconds):
    if node_execution_histogram is not None:
        node_execution_histogram.labels(node_type=node_type).observe(seconds)


def increment_cache_requests(result):
    if cache_requests_counter is not None:
        cache_requests_counter.labels(result=result).inc()
