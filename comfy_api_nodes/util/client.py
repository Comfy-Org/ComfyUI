import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Literal, TypeVar
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp.client_exceptions import ClientError, ContentTypeError
from pydantic import BaseModel

from comfy import utils
from comfy_api.latest import IO
from server import PromptServer

from . import request_logger
from ._helpers import (
    _retry_after_wait,
    default_base_url,
    get_comfy_api_headers,
    get_node_id,
    is_processing_interrupted,
    sleep_with_interrupt,
)
from .common_exceptions import ApiServerError, LocalNetworkError, ProcessingInterrupted

M = TypeVar("M", bound=BaseModel)


class ApiEndpoint:
    def __init__(
        self,
        path: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        *,
        query_params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.path = path
        self.method = method
        self.query_params = query_params or {}
        self.headers = headers or {}


@dataclass
class _RequestConfig:
    node_cls: type[IO.ComfyNode]
    endpoint: ApiEndpoint
    timeout: float
    content_type: str
    data: dict[str, Any] | None
    files: dict[str, Any] | list[tuple[str, Any]] | None
    multipart_parser: Callable | None
    max_retries: int
    max_retries_on_rate_limit: int
    retry_delay: float
    retry_backoff: float
    wait_label: str = "Waiting"
    monitor_progress: bool = True
    estimated_total: int | None = None
    final_label_on_success: str | None = "Completed"
    progress_origin_ts: float | None = None
    price_extractor: Callable[[dict[str, Any]], float | None] | None = None
    is_rate_limited: Callable[[int, Any], bool] | None = None
    response_header_validator: Callable[[dict[str, str]], None] | None = None


@dataclass
class _PollUIState:
    started: float
    status_label: str = "Queued"
    is_queued: bool = True
    price: float | None = None
    estimated_duration: int | None = None
    base_processing_elapsed: float = 0.0  # sum of completed active intervals
    active_since: float | None = None  # start time of current active interval (None if queued)


_RETRY_STATUS = {408, 500, 502, 503, 504}  # status 429 is handled separately
_MAX_RETRY_AFTER_WAIT = 150.0  # Cap a server Retry-After at this many seconds so a large hint can't block execution
COMPLETED_STATUSES = ["succeeded", "succeed", "success", "completed", "finished", "done", "complete"]
FAILED_STATUSES = ["cancelled", "canceled", "canceling", "fail", "failed", "error"]
QUEUED_STATUSES = ["created", "queued", "queueing", "submitted", "initializing", "wait", "in_queue"]


async def sync_op(
    cls: type[IO.ComfyNode],
    endpoint: ApiEndpoint,
    *,
    response_model: type[M],
    price_extractor: Callable[[M | Any], float | None] | None = None,
    data: BaseModel | None = None,
    files: dict[str, Any] | list[tuple[str, Any]] | None = None,
    content_type: str = "application/json",
    timeout: float = 3600.0,
    multipart_parser: Callable | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    wait_label: str = "Waiting for server",
    estimated_duration: int | None = None,
    final_label_on_success: str | None = "Completed",
    progress_origin_ts: float | None = None,
    monitor_progress: bool = True,
    max_retries_on_rate_limit: int = 16,
    is_rate_limited: Callable[[int, Any], bool] | None = None,
) -> M:
    raw = await sync_op_raw(
        cls,
        endpoint,
        price_extractor=_wrap_model_extractor(response_model, price_extractor),
        data=data,
        files=files,
        content_type=content_type,
        timeout=timeout,
        multipart_parser=multipart_parser,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        wait_label=wait_label,
        estimated_duration=estimated_duration,
        as_binary=False,
        final_label_on_success=final_label_on_success,
        progress_origin_ts=progress_origin_ts,
        monitor_progress=monitor_progress,
        max_retries_on_rate_limit=max_retries_on_rate_limit,
        is_rate_limited=is_rate_limited,
    )
    if not isinstance(raw, dict):
        raise Exception("Expected JSON response to validate into a Pydantic model, got non-JSON (binary or text).")
    return _validate_or_raise(response_model, raw)


async def poll_op(
    cls: type[IO.ComfyNode],
    poll_endpoint: ApiEndpoint,
    *,
    response_model: type[M],
    status_extractor: Callable[[M | Any], str | int | None],
    progress_extractor: Callable[[M | Any], int | None] | None = None,
    price_extractor: Callable[[M | Any], float | None] | None = None,
    completed_statuses: list[str | int] | None = None,
    failed_statuses: list[str | int] | None = None,
    queued_statuses: list[str | int] | None = None,
    data: BaseModel | None = None,
    poll_interval: float = 5.0,
    max_poll_attempts: int = 480,
    timeout_per_poll: float = 120.0,
    max_retries_per_poll: int = 10,
    retry_delay_per_poll: float = 1.0,
    retry_backoff_per_poll: float = 1.4,
    estimated_duration: int | None = None,
    cancel_endpoint: ApiEndpoint | None = None,
    cancel_timeout: float = 10.0,
    extra_text: str | None = None,
) -> M:
    raw = await poll_op_raw(
        cls,
        poll_endpoint=poll_endpoint,
        status_extractor=_wrap_model_extractor(response_model, status_extractor),
        progress_extractor=_wrap_model_extractor(response_model, progress_extractor),
        price_extractor=_wrap_model_extractor(response_model, price_extractor),
        completed_statuses=completed_statuses,
        failed_statuses=failed_statuses,
        queued_statuses=queued_statuses,
        data=data,
        poll_interval=poll_interval,
        max_poll_attempts=max_poll_attempts,
        timeout_per_poll=timeout_per_poll,
        max_retries_per_poll=max_retries_per_poll,
        retry_delay_per_poll=retry_delay_per_poll,
        retry_backoff_per_poll=retry_backoff_per_poll,
        estimated_duration=estimated_duration,
        cancel_endpoint=cancel_endpoint,
        cancel_timeout=cancel_timeout,
        extra_text=extra_text,
    )
    if not isinstance(raw, dict):
        raise Exception("Expected JSON response to validate into a Pydantic model, got non-JSON (binary or text).")
    return _validate_or_raise(response_model, raw)


async def sync_op_raw(
    cls: type[IO.ComfyNode],
    endpoint: ApiEndpoint,
    *,
    price_extractor: Callable[[dict[str, Any]], float | None] | None = None,
    data: dict[str, Any] | BaseModel | None = None,
    files: dict[str, Any] | list[tuple[str, Any]] | None = None,
    content_type: str = "application/json",
    timeout: float = 3600.0,
    multipart_parser: Callable | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    wait_label: str = "Waiting for server",
    estimated_duration: int | None = None,
    as_binary: bool = False,
    final_label_on_success: str | None = "Completed",
    progress_origin_ts: float | None = None,
    monitor_progress: bool = True,
    max_retries_on_rate_limit: int = 16,
    is_rate_limited: Callable[[int, Any], bool] | None = None,
    response_header_validator: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, Any] | bytes:
    """
    Make a single network request.
      - If as_binary=False (default): returns JSON dict (or {'_raw': '<text>'} if non-JSON).
      - If as_binary=True: returns bytes.
      - response_header_validator: optional callback receiving response headers dict
    """
    if isinstance(data, BaseModel):
        data = data.model_dump(exclude_none=True)
        for k, v in list(data.items()):
            if isinstance(v, Enum):
                data[k] = v.value
    cfg = _RequestConfig(
        node_cls=cls,
        endpoint=endpoint,
        timeout=timeout,
        content_type=content_type,
        data=data,
        files=files,
        multipart_parser=multipart_parser,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        wait_label=wait_label,
        monitor_progress=monitor_progress,
        estimated_total=estimated_duration,
        final_label_on_success=final_label_on_success,
        progress_origin_ts=progress_origin_ts,
        price_extractor=price_extractor,
        max_retries_on_rate_limit=max_retries_on_rate_limit,
        is_rate_limited=is_rate_limited,
        response_header_validator=response_header_validator,
    )
    return await _request_base(cfg, expect_binary=as_binary)


async def poll_op_raw(
    cls: type[IO.ComfyNode],
    poll_endpoint: ApiEndpoint,
    *,
    status_extractor: Callable[[dict[str, Any]], str | int | None],
    progress_extractor: Callable[[dict[str, Any]], int | None] | None = None,
    price_extractor: Callable[[dict[str, Any]], float | None] | None = None,
    completed_statuses: list[str | int] | None = None,
    failed_statuses: list[str | int] | None = None,
    queued_statuses: list[str | int] | None = None,
    data: dict[str, Any] | BaseModel | None = None,
    poll_interval: float = 5.0,
    max_poll_attempts: int = 480,
    timeout_per_poll: float = 120.0,
    max_retries_per_poll: int = 10,
    retry_delay_per_poll: float = 1.0,
    retry_backoff_per_poll: float = 1.4,
    estimated_duration: int | None = None,
    cancel_endpoint: ApiEndpoint | None = None,
    cancel_timeout: float = 10.0,
    extra_text: str | None = None,
) -> dict[str, Any]:
    """
    Polls an endpoint until the task reaches a terminal state. Displays time while queued/processing,
    checks interruption every second, and calls Cancel endpoint (if provided) on interruption.

    Uses default complete, failed and queued states assumption.

    Returns the final JSON response from the poll endpoint.
    """
    completed_states = _normalize_statuses(COMPLETED_STATUSES if completed_statuses is None else completed_statuses)
    failed_states = _normalize_statuses(FAILED_STATUSES if failed_statuses is None else failed_statuses)
    queued_states = _normalize_statuses(QUEUED_STATUSES if queued_statuses is None else queued_statuses)
    started = time.monotonic()
    consumed_attempts = 0  # counts only non-queued polls

    progress_bar = utils.ProgressBar(100) if progress_extractor else None
    last_progress: int | None = None

    state = _PollUIState(started=started, estimated_duration=estimated_duration)
    stop_ticker = asyncio.Event()

    async def _ticker():
        """Emit a UI update every second while polling is in progress."""
        try:
            while not stop_ticker.is_set():
                if is_processing_interrupted():
                    break
                now = time.monotonic()
                proc_elapsed = state.base_processing_elapsed + (
                    (now - state.active_since) if state.active_since is not None else 0.0
                )
                _display_time_progress(
                    cls,
                    status=state.status_label,
                    elapsed_seconds=int(now - state.started),
                    estimated_total=state.estimated_duration,
                    price=state.price,
                    is_queued=state.is_queued,
                    processing_elapsed_seconds=int(proc_elapsed),
                    extra_text=extra_text,
                )
                await asyncio.sleep(1.0)
        except Exception as exc:
            logging.debug("Polling ticker exited: %s", exc)

    ticker_task = asyncio.create_task(_ticker())
    try:
        while consumed_attempts < max_poll_attempts:
            try:
                resp_json = await sync_op_raw(
                    cls,
                    poll_endpoint,
                    data=data,
                    timeout=timeout_per_poll,
                    max_retries=max_retries_per_poll,
                    retry_delay=retry_delay_per_poll,
                    retry_backoff=retry_backoff_per_poll,
                    wait_label="Checking",
                    estimated_duration=None,
                    as_binary=False,
                    final_label_on_success=None,
                    monitor_progress=False,
                )
                if not isinstance(resp_json, dict):
                    raise Exception("Polling endpoint returned non-JSON response.")
            except ProcessingInterrupted:
                if cancel_endpoint:
                    with contextlib.suppress(Exception):
                        await sync_op_raw(
                            cls,
                            cancel_endpoint,
                            timeout=cancel_timeout,
                            max_retries=0,
                            wait_label="Cancelling task",
                            estimated_duration=None,
                            as_binary=False,
                            final_label_on_success=None,
                            monitor_progress=False,
                        )
                raise

            try:
                status = _normalize_status_value(status_extractor(resp_json))
            except Exception as e:
                logging.error("Status extraction failed: %s", e)
                status = None

            if price_extractor:
                new_price = price_extractor(resp_json)
                if new_price is not None:
                    state.price = new_price

            if progress_extractor:
                new_progress = progress_extractor(resp_json)
                if new_progress is not None and last_progress != new_progress:
                    progress_bar.update_absolute(new_progress, total=100)
                    last_progress = new_progress

            now_ts = time.monotonic()
            is_queued = status in queued_states

            if is_queued:
                if state.active_since is not None:  # If we just moved from active -> queued, close the active interval
                    state.base_processing_elapsed += now_ts - state.active_since
                    state.active_since = None
            else:
                if state.active_since is None:  # If we just moved from queued -> active, open a new active interval
                    state.active_since = now_ts

            state.is_queued = is_queued
            state.status_label = status or ("Queued" if is_queued else "Processing")
            if status in completed_states:
                if state.active_since is not None:
                    state.base_processing_elapsed += now_ts - state.active_since
                    state.active_since = None
                stop_ticker.set()
                with contextlib.suppress(Exception):
                    await ticker_task

                if progress_bar and last_progress != 100:
                    progress_bar.update_absolute(100, total=100)

                _display_time_progress(
                    cls,
                    status=status if status else "Completed",
                    elapsed_seconds=int(now_ts - started),
                    estimated_total=estimated_duration,
                    price=state.price,
                    is_queued=False,
                    processing_elapsed_seconds=int(state.base_processing_elapsed),
                    extra_text=extra_text,
                )
                return resp_json

            if status in failed_states:
                msg = f"Task failed: {json.dumps(resp_json)}"
                logging.error(msg)
                raise Exception(msg)

            try:
                await sleep_with_interrupt(poll_interval, cls, None, None, None)
            except ProcessingInterrupted:
                if cancel_endpoint:
                    with contextlib.suppress(Exception):
                        await sync_op_raw(
                            cls,
                            cancel_endpoint,
                            timeout=cancel_timeout,
                            max_retries=0,
                            wait_label="Cancelling task",
                            estimated_duration=None,
                            as_binary=False,
                            final_label_on_success=None,
                            monitor_progress=False,
                        )
                raise
            if not is_queued:
                consumed_attempts += 1

        raise Exception(
            f"Polling timed out after {max_poll_attempts} non-queued attempts "
            f"(~{int(max_poll_attempts * poll_interval)}s of active polling)."
        )
    except ProcessingInterrupted:
        raise
    except (LocalNetworkError, ApiServerError):
        raise
    except Exception as e:
        raise Exception(f"Polling aborted due to error: {e}") from e
    finally:
        stop_ticker.set()
        with contextlib.suppress(Exception):
            await ticker_task


def _display_text(
    node_cls: type[IO.ComfyNode],
    text: str | None,
    *,
    status: str | int | None = None,
    price: float | None = None,
) -> None:
    display_lines: list[str] = []
    if status:
        display_lines.append(f"Status: {status.capitalize() if isinstance(status, str) else status}")
    if price is not None:
        p = f"{float(price) * 211:,.1f}".rstrip("0").rstrip(".")
        if p != "0":
            display_lines.append(f"Price: {p} credits")
    if text is not None:
        display_lines.append(text)
    if display_lines:
        PromptServer.instance.send_progress_text("\n".join(display_lines), get_node_id(node_cls))


def _display_time_progress(
    node_cls: type[IO.ComfyNode],
    status: str | int | None,
    elapsed_seconds: int,
    estimated_total: int | None = None,
    *,
    price: float | None = None,
    is_queued: bool | None = None,
    processing_elapsed_seconds: int | None = None,
    extra_text: str | None = None,
) -> None:
    if estimated_total is not None and estimated_total > 0 and is_queued is False:
        pe = processing_elapsed_seconds if processing_elapsed_seconds is not None else elapsed_seconds
        remaining = max(0, int(estimated_total) - int(pe))
        time_line = f"Time elapsed: {int(elapsed_seconds)}s (~{remaining}s remaining)"
    else:
        time_line = f"Time elapsed: {int(elapsed_seconds)}s"
    text = f"{time_line}\n\n{extra_text}" if extra_text else time_line
    _display_text(node_cls, text, status=status, price=price)


async def _diagnose_connectivity() -> dict[str, bool]:
    """Best-effort connectivity diagnostics to distinguish local vs. server issues."""
    results = {
        "internet_accessible": False,
        "api_accessible": False,
    }
    timeout = aiohttp.ClientTimeout(total=5.0)

    # Probe Google and Baidu in parallel: Google is blocked by the GFW in mainland China, so a Baidu probe is required
    # to correctly detect that Chinese users with working internet do have working internet.
    internet_probe_urls = ("https://www.google.com", "https://www.baidu.com")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _probe(url: str) -> bool:
            try:
                async with session.get(url) as resp:
                    return resp.status < 500
            except (ClientError, OSError, asyncio.TimeoutError):
                return False

        probe_tasks = [asyncio.create_task(_probe(u)) for u in internet_probe_urls]
        try:
            for fut in asyncio.as_completed(probe_tasks):
                if await fut:
                    results["internet_accessible"] = True
                    break
        finally:
            for t in probe_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*probe_tasks, return_exceptions=True)
        if not results["internet_accessible"]:
            return results

        parsed = urlparse(default_base_url())
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        with contextlib.suppress(ClientError, OSError):
            async with session.get(health_url) as resp:
                results["api_accessible"] = resp.status < 500
    return results


def _unpack_tuple(t: tuple) -> tuple[str, Any, str]:
    """Normalize (filename, value, content_type)."""
    if len(t) == 2:
        return t[0], t[1], "application/octet-stream"
    if len(t) == 3:
        return t[0], t[1], t[2]
    raise ValueError("files tuple must be (filename, file[, content_type])")


def _merge_params(endpoint_params: dict[str, Any], method: str, data: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(endpoint_params or {})
    if method.upper() == "GET" and data:
        for k, v in data.items():
            if v is not None:
                params[k] = v
    return params


def _friendly_http_message(status: int, body: Any) -> str:
    if status == 401:
        return "Unauthorized: Please login first to use this node."
    if status == 402:
        return "Payment Required: Please add credits to your account to use this node."
    if status == 409:
        return "There is a problem with your account. Please contact support@comfy.org."
    if status == 429:
        return "Rate Limit Exceeded: The server returned 429 after all retry attempts. Please wait and try again."
    try:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                typ = err.get("type")
                if msg and typ:
                    return f"API Error: {msg} (Type: {typ})"
                if msg:
                    return f"API Error: {msg}"
            return f"API Error: {json.dumps(body)}"
        else:
            txt = str(body)
            if len(txt) <= 200:
                return f"API Error (raw): {txt}"
            return f"API Error (status {status})"
    except Exception:
        return f"HTTP {status}: Unknown error"


def _generate_operation_id(method: str, path: str, attempt: int) -> str:
    slug = path.strip("/").replace("/", "_") or "op"
    return f"{method}_{slug}_try{attempt}_{uuid.uuid4().hex[:8]}"


def _snapshot_request_body_for_logging(
    content_type: str,
    method: str,
    data: dict[str, Any] | None,
    files: dict[str, Any] | list[tuple[str, Any]] | None,
) -> dict[str, Any] | str | None:
    if method.upper() == "GET":
        return None
    if content_type == "multipart/form-data":
        form_fields = sorted([k for k, v in (data or {}).items() if v is not None])
        file_fields: list[dict[str, str]] = []
        if files:
            file_iter = files if isinstance(files, list) else list(files.items())
            for field_name, file_obj in file_iter:
                if file_obj is None:
                    continue
                if isinstance(file_obj, tuple):
                    filename = file_obj[0]
                else:
                    filename = getattr(file_obj, "name", field_name)
                file_fields.append({"field": field_name, "filename": str(filename or "")})
        return {"_multipart": True, "form_fields": form_fields, "file_fields": file_fields}
    if content_type == "application/x-www-form-urlencoded":
        return data or {}
    return data or {}


async def _request_base(cfg: _RequestConfig, expect_binary: bool):
    """Core request with retries, per-second interruption monitoring, true cancellation, and friendly errors."""
    url = cfg.endpoint.path
    parsed_url = urlparse(url)
    if not parsed_url.scheme and not parsed_url.netloc:  # is URL relative?
        url = urljoin(default_base_url().rstrip("/") + "/", url.lstrip("/"))

    method = cfg.endpoint.method
    params = _merge_params(cfg.endpoint.query_params, method, cfg.data if method == "GET" else None)

    async def _monitor(stop_evt: asyncio.Event, start_ts: float):
        """Every second: update elapsed time and signal interruption."""
        try:
            while not stop_evt.is_set():
                if is_processing_interrupted():
                    return
                if cfg.monitor_progress:
                    _display_time_progress(
                        cfg.node_cls, cfg.wait_label, int(time.monotonic() - start_ts), cfg.estimated_total
                    )
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return  # normal shutdown

    start_time = cfg.progress_origin_ts if cfg.progress_origin_ts is not None else time.monotonic()
    attempt = 0
    delay = cfg.retry_delay
    rate_limit_attempts = 0
    rate_limit_delay = cfg.retry_delay
    operation_succeeded: bool = False
    final_elapsed_seconds: int | None = None
    extracted_price: float | None = None
    while True:
        attempt += 1
        stop_event = asyncio.Event()
        monitor_task: asyncio.Task | None = None
        sess: aiohttp.ClientSession | None = None

        operation_id = _generate_operation_id(method, cfg.endpoint.path, attempt)
        logging.debug("[DEBUG] HTTP %s %s (attempt %d)", method, url, attempt)

        payload_headers = {"Accept": "*/*"} if expect_binary else {"Accept": "application/json"}
        if not parsed_url.scheme and not parsed_url.netloc:  # is URL relative?
            payload_headers.update(get_comfy_api_headers(cfg.node_cls))
        if cfg.endpoint.headers:
            payload_headers.update(cfg.endpoint.headers)

        payload_kw: dict[str, Any] = {"headers": payload_headers}
        if method == "GET":
            payload_headers.pop("Content-Type", None)
        request_body_log = _snapshot_request_body_for_logging(cfg.content_type, method, cfg.data, cfg.files)
        try:
            if cfg.monitor_progress:
                monitor_task = asyncio.create_task(_monitor(stop_event, start_time))

            timeout = aiohttp.ClientTimeout(total=cfg.timeout)
            sess = aiohttp.ClientSession(timeout=timeout)

            if cfg.content_type == "multipart/form-data" and method != "GET":
                # aiohttp will set Content-Type boundary; remove any fixed Content-Type
                payload_headers.pop("Content-Type", None)
                if cfg.multipart_parser and cfg.data:
                    form = cfg.multipart_parser(cfg.data)
                    if not isinstance(form, aiohttp.FormData):
                        raise ValueError("multipart_parser must return aiohttp.FormData")
                else:
                    form = aiohttp.FormData(default_to_multipart=True)
                    if cfg.data:
                        for k, v in cfg.data.items():
                            if v is None:
                                continue
                            form.add_field(k, str(v) if not isinstance(v, (bytes, bytearray)) else v)
                if cfg.files:
                    file_iter = cfg.files if isinstance(cfg.files, list) else cfg.files.items()
                    for field_name, file_obj in file_iter:
                        if file_obj is None:
                            continue
                        if isinstance(file_obj, tuple):
                            filename, file_value, content_type = _unpack_tuple(file_obj)
                        else:
                            filename = getattr(file_obj, "name", field_name)
                            file_value = file_obj
                            content_type = "application/octet-stream"
                        # Attempt to rewind BytesIO for retries
                        if isinstance(file_value, BytesIO):
                            with contextlib.suppress(Exception):
                                file_value.seek(0)
                        form.add_field(field_name, file_value, filename=filename, content_type=content_type)
                payload_kw["data"] = form
            elif cfg.content_type == "application/x-www-form-urlencoded" and method != "GET":
                payload_headers["Content-Type"] = "application/x-www-form-urlencoded"
                payload_kw["data"] = cfg.data or {}
            elif method != "GET":
                payload_headers["Content-Type"] = "application/json"
                payload_kw["json"] = cfg.data or {}

            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method,
                request_url=url,
                request_headers=dict(payload_headers) if payload_headers else None,
                request_params=dict(params) if params else None,
                request_data=request_body_log,
            )

            req_coro = sess.request(method, url, params=params, **payload_kw)
            req_task = asyncio.create_task(req_coro)

            # Race: request vs. monitor (interruption)
            tasks = {req_task}
            if monitor_task:
                tasks.add(monitor_task)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            if monitor_task and monitor_task in done:
                # Interrupted – cancel the request and abort
                if req_task in pending:
                    req_task.cancel()
                raise ProcessingInterrupted("Task cancelled")

            # Otherwise, request finished
            resp = await req_task
            async with resp:
                if resp.status >= 400:
                    try:
                        body = await resp.json()
                    except (ContentTypeError, json.JSONDecodeError):
                        body = await resp.text()
                    should_retry = False
                    wait_time = 0.0
                    retry_label = ""
                    is_rl = resp.status == 429 or (
                        cfg.is_rate_limited is not None and cfg.is_rate_limited(resp.status, body)
                    )
                    if is_rl and rate_limit_attempts < cfg.max_retries_on_rate_limit:
                        rate_limit_attempts += 1
                        wait_time = min(rate_limit_delay, 30.0)
                        rate_limit_delay *= cfg.retry_backoff
                        retry_label = f"rate-limit retry {rate_limit_attempts} of {cfg.max_retries_on_rate_limit}"
                        should_retry = True
                    elif resp.status in _RETRY_STATUS and (attempt - rate_limit_attempts) <= cfg.max_retries:
                        wait_time = delay
                        delay *= cfg.retry_backoff
                        retry_label = f"retry {attempt - rate_limit_attempts} of {cfg.max_retries}"
                        should_retry = True

                    if should_retry:
                        wait_time = _retry_after_wait(resp.headers.get("Retry-After"), wait_time, _MAX_RETRY_AFTER_WAIT)
                        logging.warning(
                            "HTTP %s %s -> %s. Waiting %.2fs (%s).",
                            method,
                            url,
                            resp.status,
                            wait_time,
                            retry_label,
                        )
                        request_logger.log_request_response(
                            operation_id=operation_id,
                            request_method=method,
                            request_url=url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content=body,
                            error_message=f"HTTP {resp.status} ({retry_label}, will retry in {wait_time:.1f}s)",
                        )
                        await sleep_with_interrupt(
                            wait_time,
                            cfg.node_cls,
                            cfg.wait_label if cfg.monitor_progress else None,
                            start_time if cfg.monitor_progress else None,
                            cfg.estimated_total,
                            display_callback=_display_time_progress if cfg.monitor_progress else None,
                        )
                        continue
                    msg = _friendly_http_message(resp.status, body)
                    request_logger.log_request_response(
                        operation_id=operation_id,
                        request_method=method,
                        request_url=url,
                        response_status_code=resp.status,
                        response_headers=dict(resp.headers),
                        response_content=body,
                        error_message=msg,
                    )
                    raise Exception(msg)

                if expect_binary:
                    buff = bytearray()
                    last_tick = time.monotonic()
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        buff.extend(chunk)
                        now = time.monotonic()
                        if now - last_tick >= 1.0:
                            last_tick = now
                            if is_processing_interrupted():
                                raise ProcessingInterrupted("Task cancelled")
                            if cfg.monitor_progress:
                                _display_time_progress(
                                    cfg.node_cls, cfg.wait_label, int(now - start_time), cfg.estimated_total
                                )
                    bytes_payload = bytes(buff)
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    if cfg.price_extractor:
                        with contextlib.suppress(Exception):
                            extracted_price = cfg.price_extractor(resp_headers)
                    if cfg.response_header_validator:
                        cfg.response_header_validator(resp_headers)
                    operation_succeeded = True
                    final_elapsed_seconds = int(time.monotonic() - start_time)
                    request_logger.log_request_response(
                        operation_id=operation_id,
                        request_method=method,
                        request_url=url,
                        response_status_code=resp.status,
                        response_headers=resp_headers,
                        response_content=bytes_payload,
                    )
                    return bytes_payload
                else:
                    try:
                        payload = await resp.json()
                        response_content_to_log: Any = payload
                    except (ContentTypeError, json.JSONDecodeError):
                        text = await resp.text()
                        try:
                            payload = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            payload = {"_raw": text}
                        response_content_to_log = payload if isinstance(payload, dict) else text
                    with contextlib.suppress(Exception):
                        extracted_price = cfg.price_extractor(payload) if cfg.price_extractor else None
                    operation_succeeded = True
                    final_elapsed_seconds = int(time.monotonic() - start_time)
                    request_logger.log_request_response(
                        operation_id=operation_id,
                        request_method=method,
                        request_url=url,
                        response_status_code=resp.status,
                        response_headers=dict(resp.headers),
                        response_content=response_content_to_log,
                    )
                    return payload

        except ProcessingInterrupted:
            logging.debug("Polling was interrupted by user")
            raise
        except (ClientError, OSError) as e:
            if (attempt - rate_limit_attempts) <= cfg.max_retries:
                logging.warning(
                    "Connection error calling %s %s. Retrying in %.2fs (%d/%d): %s",
                    method,
                    url,
                    delay,
                    attempt - rate_limit_attempts,
                    cfg.max_retries,
                    str(e),
                )
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method=method,
                    request_url=url,
                    request_headers=dict(payload_headers) if payload_headers else None,
                    request_params=dict(params) if params else None,
                    request_data=request_body_log,
                    error_message=f"{type(e).__name__}: {str(e)} (will retry)",
                )
                await sleep_with_interrupt(
                    delay,
                    cfg.node_cls,
                    cfg.wait_label if cfg.monitor_progress else None,
                    start_time if cfg.monitor_progress else None,
                    cfg.estimated_total,
                    display_callback=_display_time_progress if cfg.monitor_progress else None,
                )
                delay *= cfg.retry_backoff
                continue
            diag = await _diagnose_connectivity()
            if not diag["internet_accessible"]:
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method=method,
                    request_url=url,
                    request_headers=dict(payload_headers) if payload_headers else None,
                    request_params=dict(params) if params else None,
                    request_data=request_body_log,
                    error_message=f"LocalNetworkError: {str(e)}",
                )
                raise LocalNetworkError(
                    "Unable to connect to the API server due to local network issues. "
                    "Please check your internet connection and try again."
                ) from e
            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method,
                request_url=url,
                request_headers=dict(payload_headers) if payload_headers else None,
                request_params=dict(params) if params else None,
                request_data=request_body_log,
                error_message=f"ApiServerError: {str(e)}",
            )
            raise ApiServerError(
                f"The API server at {default_base_url()} is currently unreachable. "
                f"The service may be experiencing issues."
            ) from e
        finally:
            stop_event.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await monitor_task
            if sess:
                with contextlib.suppress(Exception):
                    await sess.close()
            if operation_succeeded and cfg.monitor_progress and cfg.final_label_on_success:
                _display_time_progress(
                    cfg.node_cls,
                    status=cfg.final_label_on_success,
                    elapsed_seconds=(
                        final_elapsed_seconds
                        if final_elapsed_seconds is not None
                        else int(time.monotonic() - start_time)
                    ),
                    estimated_total=cfg.estimated_total,
                    price=extracted_price,
                    is_queued=False,
                    processing_elapsed_seconds=final_elapsed_seconds,
                )


def _validate_or_raise(response_model: type[M], payload: Any) -> M:
    try:
        return response_model.model_validate(payload)
    except Exception as e:
        logging.error(
            "Response validation failed for %s: %s",
            getattr(response_model, "__name__", response_model),
            e,
        )
        raise Exception(
            f"Response validation failed for {getattr(response_model, '__name__', response_model)}: {e}"
        ) from e


def _wrap_model_extractor(
    response_model: type[M],
    extractor: Callable[[M], Any] | None,
) -> Callable[[dict[str, Any]], Any] | None:
    """Wrap a typed extractor so it can be used by the dict-based poller.
    Validates the dict into `response_model` before invoking `extractor`.
    Uses a small per-wrapper cache keyed by `id(dict)` to avoid re-validating
    the same response for multiple extractors in a single poll attempt.
    """
    if extractor is None:
        return None
    _cache: dict[int, M] = {}

    def _wrapped(d: dict[str, Any]) -> Any:
        try:
            key = id(d)
            model = _cache.get(key)
            if model is None:
                model = response_model.model_validate(d)
                _cache[key] = model
            return extractor(model)
        except Exception as e:
            logging.error("Extractor failed (typed -> dict wrapper): %s", e)
            raise

    return _wrapped


def _normalize_statuses(values: Iterable[str | int] | None) -> set[str | int]:
    if not values:
        return set()
    out: set[str | int] = set()
    for v in values:
        nv = _normalize_status_value(v)
        if nv is not None:
            out.add(nv)
    return out


def _normalize_status_value(val: str | int | None) -> str | int | None:
    if isinstance(val, str):
        return val.strip().lower()
    return val
