import copy
import heapq
import inspect
import logging
import sys
import threading
import time
import traceback
from enum import Enum
from typing import List, Literal, NamedTuple, Optional, Union
import asyncio

import torch

from comfy.cli_args import args, get_console_log_level
import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.system_memory
import comfy_aimdo.model_vbar
from comfy.internal_logging import detail

from latent_preview import set_preview_method
import nodes
from comfy_execution.caching import (
    BasicCache,
    CacheKeySetID,
    CacheKeySetInputSignature,
    NullCache,
    get_lazy_input_keys,
    HierarchicalCache,
    LRUCache,
    RAMPressureCache,
    RAM_CACHE_LARGE_INTERMEDIATE,
)
from comfy_execution.graph import (
    DynamicPrompt,
    ExecutionBlocker,
    ExecutionList,
    get_input_info,
)
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.validation import validate_node_input
from comfy_execution.progress import get_progress_state, reset_progress_state, add_progress_handler, WebUIProgressHandler
from comfy_execution.utils import CurrentNodeContext
from comfy_execution.asset_enrichment import enrich_output_with_assets
from comfy_api.internal import _ComfyNodeInternal, _NodeOutputInternal, first_real_override, is_class, make_locked_method_func
from comfy_api.latest import io, _io
from comfy_execution.cache_provider import _has_cache_providers, _get_cache_providers, _logger as _cache_logger


class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2

class DuplicateNodeError(Exception):
    pass

class IsChangedCache:
    def __init__(self, prompt_id: str, dynprompt: DynamicPrompt, outputs_cache: BasicCache):
        self.prompt_id = prompt_id
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    async def get(self, node_id):
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        has_is_changed = False
        is_changed_name = None
        if issubclass(class_def, _ComfyNodeInternal) and first_real_override(class_def, "fingerprint_inputs") is not None:
            has_is_changed = True
            is_changed_name = "fingerprint_inputs"
        elif hasattr(class_def, "IS_CHANGED"):
            has_is_changed = True
            is_changed_name = "IS_CHANGED"
        if not has_is_changed:
            self.is_changed[node_id] = False
            return self.is_changed[node_id]

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return self.is_changed[node_id]

        # Intentionally do not use cached outputs here. We only want constants in IS_CHANGED
        input_data_all, _, v3_data = get_input_data(node["inputs"], class_def, node_id, None)
        try:
            is_changed = await _async_map_node_over_list(self.prompt_id, node_id, class_def, input_data_all, is_changed_name, v3_data=v3_data)
            is_changed = await resolve_map_node_over_list_results(is_changed)
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except Exception as e:
            logging.warning("WARNING: {}".format(e))
            node["is_changed"] = float("NaN")
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]


class CacheEntry(NamedTuple):
    ui: dict
    outputs: list


class CacheType(Enum):
    CLASSIC = 0
    LRU = 1
    NONE = 2
    RAM_PRESSURE = 3


class CacheSet:
    def __init__(self, cache_type=None, cache_args={}):
        if cache_type == CacheType.NONE:
            self.init_null_cache()
            logging.info("Disabling intermediate node cache.")
        elif cache_type == CacheType.RAM_PRESSURE:
            cache_ram = cache_args.get("ram", 16.0)
            self.init_ram_cache(cache_ram)
            logging.info("Using RAM pressure cache.")
        elif cache_type == CacheType.LRU:
            cache_size = cache_args.get("lru", 0)
            self.init_lru_cache(cache_size)
            logging.info("Using LRU cache")
        else:
            self.init_classic_cache()

        self.all = [self.outputs, self.objects]

        # Executor-owned record of which lazy inputs were actually evaluated, shared with the
        # cache key builder so unconsumed lazy inputs stop poisoning downstream cache keys.
        self.lazy_evaluated: dict = {}
        if hasattr(self.outputs, "key_class") and self.outputs.key_class is CacheKeySetInputSignature:
            self.outputs.key_class_kwargs = {"lazy_evaluated": self.lazy_evaluated}

    # Performs like the old cache -- dump data ASAP
    def init_classic_cache(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature, enable_providers=True)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_lru_cache(self, cache_size):
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size, enable_providers=True)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_ram_cache(self, min_headroom):
        self.outputs = RAMPressureCache(CacheKeySetInputSignature, enable_providers=True)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_null_cache(self):
        self.outputs = NullCache()
        self.objects = NullCache()

    def recursive_debug_dump(self):
        result = {
            "outputs": self.outputs.recursive_debug_dump(),
        }
        return result

SENSITIVE_EXTRA_DATA_KEYS = ("auth_token_comfy_org", "api_key_comfy_org")

def get_input_data(inputs, class_def, unique_id, execution_list=None, dynprompt=None, extra_data={}):
    is_v3 = issubclass(class_def, _ComfyNodeInternal)
    v3_data: io.V3Data = {}
    hidden_inputs_v3 = {}
    valid_inputs = class_def.INPUT_TYPES()
    if is_v3:
        valid_inputs, hidden, v3_data = _io.get_finalized_class_inputs(valid_inputs, inputs)
    input_data_all = {}
    missing_keys = {}
    for x in inputs:
        input_data = inputs[x]
        _, input_category, input_info = get_input_info(class_def, x, valid_inputs)
        def mark_missing():
            missing_keys[x] = True
            input_data_all[x] = (None,)
        if is_link(input_data) and (not input_info or not input_info.get("rawLink", False)):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if execution_list is None:
                mark_missing()
                continue # This might be a lazily-evaluated input
            cached = execution_list.get_cache(input_unique_id, unique_id)
            if cached is None or cached.outputs is None:
                mark_missing()
                continue
            if output_index >= len(cached.outputs):
                mark_missing()
                continue
            obj = cached.outputs[output_index]
            input_data_all[x] = obj
        elif input_category is not None or (is_v3 and class_def.ACCEPT_ALL_INPUTS):
            input_data_all[x] = [input_data]

    if is_v3:
        if hidden is not None:
            if io.Hidden.prompt.name in hidden:
                hidden_inputs_v3[io.Hidden.prompt] = dynprompt.get_original_prompt() if dynprompt is not None else {}
            if io.Hidden.dynprompt.name in hidden:
                hidden_inputs_v3[io.Hidden.dynprompt] = dynprompt
            if io.Hidden.extra_pnginfo.name in hidden:
                hidden_inputs_v3[io.Hidden.extra_pnginfo] = extra_data.get('extra_pnginfo', None)
            if io.Hidden.unique_id.name in hidden:
                hidden_inputs_v3[io.Hidden.unique_id] = unique_id
            if io.Hidden.auth_token_comfy_org.name in hidden:
                hidden_inputs_v3[io.Hidden.auth_token_comfy_org] = extra_data.get("auth_token_comfy_org", None)
            if io.Hidden.api_key_comfy_org.name in hidden:
                hidden_inputs_v3[io.Hidden.api_key_comfy_org] = extra_data.get("api_key_comfy_org", None)
            if io.Hidden.comfy_usage_source.name in hidden:
                hidden_inputs_v3[io.Hidden.comfy_usage_source] = extra_data.get("comfy_usage_source", None)
    else:
        if "hidden" in valid_inputs:
            h = valid_inputs["hidden"]
            for x in h:
                if h[x] == "PROMPT":
                    input_data_all[x] = [dynprompt.get_original_prompt() if dynprompt is not None else {}]
                if h[x] == "DYNPROMPT":
                    input_data_all[x] = [dynprompt]
                if h[x] == "EXTRA_PNGINFO":
                    input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
                if h[x] == "UNIQUE_ID":
                    input_data_all[x] = [unique_id]
                if h[x] == "AUTH_TOKEN_COMFY_ORG":
                    input_data_all[x] = [extra_data.get("auth_token_comfy_org", None)]
                if h[x] == "API_KEY_COMFY_ORG":
                    input_data_all[x] = [extra_data.get("api_key_comfy_org", None)]
                if h[x] == "COMFY_USAGE_SOURCE":
                    input_data_all[x] = [extra_data.get("comfy_usage_source", None)]
    v3_data["hidden_inputs"] = hidden_inputs_v3
    return input_data_all, missing_keys, v3_data

def record_lazy_evidence(caches, unique_id, class_type, lazy_requested):
    """Persist which lazy inputs check_lazy_status requested for this node run."""
    if lazy_requested is None:
        return
    for lazy_key in get_lazy_input_keys(class_type):
        key = (unique_id, lazy_key)
        # Once consumed, an input keeps invalidating: later runs see it cached and
        # check_lazy_status stops naming it, which must not flip the recorded value.
        caches.lazy_evaluated[key] = caches.lazy_evaluated.get(key, False) or lazy_key in lazy_requested

map_node_over_list = None #Don't hook this please

async def resolve_map_node_over_list_results(results):
    remaining = [x for x in results if isinstance(x, asyncio.Task) and not x.done()]
    if len(remaining) == 0:
        return [x.result() if isinstance(x, asyncio.Task) else x for x in results]
    else:
        done, pending = await asyncio.wait(remaining)
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
        return [x.result() if isinstance(x, asyncio.Task) else x for x in results]

async def _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None, v3_data=None):
    # check if node wants the lists
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max(len(x) for x in input_data_all.values())

    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        return {k: v[i if len(v) > i else -1] for k, v in d.items()}

    results = []
    async def process_inputs(inputs, index=None, input_is_list=False):
        if allow_interrupt:
            nodes.before_node_execution()
        execution_block = None
        for k, v in inputs.items():
            if input_is_list:
                for e in v:
                    if isinstance(e, ExecutionBlocker):
                        v = e
                        break
            if isinstance(v, ExecutionBlocker):
                execution_block = execution_block_cb(v) if execution_block_cb else v
                break
        if execution_block is None:
            if pre_execute_cb is not None and index is not None:
                pre_execute_cb(index)
            # V3
            if isinstance(obj, _ComfyNodeInternal) or (is_class(obj) and issubclass(obj, _ComfyNodeInternal)):
                # if is just a class, then assign no state, just create clone
                if is_class(obj):
                    type_obj = obj
                    obj.VALIDATE_CLASS()
                    class_clone = obj.PREPARE_CLASS_CLONE(v3_data)
                # otherwise, use class instance to populate/reuse some fields
                else:
                    type_obj = type(obj)
                    type_obj.VALIDATE_CLASS()
                    class_clone = type_obj.PREPARE_CLASS_CLONE(v3_data)
                f = make_locked_method_func(type_obj, func, class_clone)
                # in case of dynamic inputs, restructure inputs to expected nested dict
                if v3_data is not None:
                    inputs = _io.build_nested_inputs(inputs, v3_data)
            # V1
            else:
                f = getattr(obj, func)
            if inspect.iscoroutinefunction(f):
                async def async_wrapper(f, prompt_id, unique_id, list_index, args):
                    with CurrentNodeContext(prompt_id, unique_id, list_index):
                        return await f(**args)
                task = asyncio.create_task(async_wrapper(f, prompt_id, unique_id, index, args=inputs))
                # Give the task a chance to execute without yielding
                await asyncio.sleep(0)
                if task.done():
                    result = task.result()
                    results.append(result)
                else:
                    results.append(task)
            else:
                with CurrentNodeContext(prompt_id, unique_id, index):
                    result = f(**inputs)
                results.append(result)
        else:
            results.append(execution_block)

    if input_is_list:
        await process_inputs(input_data_all, 0, input_is_list=input_is_list)
    elif max_len_input == 0:
        await process_inputs({})
    else:
        for i in range(max_len_input):
            input_dict = slice_dict(input_data_all, i)
            await process_inputs(input_dict, i)
    return results


def merge_result_data(results, obj):
    # check which outputs need concatenating
    output = []
    output_is_list = [False] * len(results[0])
    if hasattr(obj, "OUTPUT_IS_LIST"):
        output_is_list = obj.OUTPUT_IS_LIST

    # merge node execution results
    for i, is_list in zip(range(len(results[0])), output_is_list):
        if is_list:
            value = []
            for o in results:
                if isinstance(o[i], ExecutionBlocker):
                    value.append(o[i])
                else:
                    value.extend(o[i])
            output.append(value)
        else:
            output.append([o[i] for o in results])
    return output

async def get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=None, pre_execute_cb=None, v3_data=None):
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
    has_pending_task = any(isinstance(r, asyncio.Task) and not r.done() for r in return_values)
    if has_pending_task:
        return return_values, {}, False, has_pending_task
    output, ui, has_subgraph = get_output_from_returns(return_values, obj)
    return output, ui, has_subgraph, False

def get_output_from_returns(return_values, obj):
    results = []
    uis = []
    subgraph_results = []
    has_subgraph = False
    for i in range(len(return_values)):
        r = return_values[i]
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'expand' in r:
                # Perform an expansion, but do not append results
                has_subgraph = True
                new_graph = r['expand']
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif 'result' in r:
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        elif isinstance(r, _NodeOutputInternal):
            # V3
            if r.ui is not None:
                if isinstance(r.ui, dict):
                    uis.append(r.ui)
                else:
                    uis.append(r.ui.as_dict())
            if r.expand is not None:
                has_subgraph = True
                new_graph = r.expand
                result = r.result
                if r.block_execution is not None:
                    result = tuple([ExecutionBlocker(r.block_execution)] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif r.result is not None:
                result = r.result
                if r.block_execution is not None:
                    result = tuple([ExecutionBlocker(r.block_execution)] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        else:
            if isinstance(r, ExecutionBlocker):
                r = tuple([r] * len(obj.RETURN_TYPES))
            results.append(r)
            subgraph_results.append((None, r))

    if has_subgraph:
        output = subgraph_results
    elif len(results) > 0:
        output = merge_result_data(results, obj)
    else:
        output = []
    ui = dict()
    # TODO: Think there's an existing bug here
    # If we're performing a subgraph expansion, we probably shouldn't be returning UI values yet.
    # They'll get cached without the completed subgraphs. It's an edge case and I'm not aware of
    # any nodes that use both subgraph expansion and custom UI outputs, but might be a problem in the future.
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def _is_intermediate_output(dynprompt, node_id):
    class_type = dynprompt.get_node(node_id)["class_type"]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    return getattr(class_def, 'HAS_INTERMEDIATE_OUTPUT', False)


def _send_cached_ui(server, node_id, display_node_id, cached, prompt_id, ui_outputs):
    if cached.ui is not None:
        ui_outputs[node_id] = cached.ui
    if server.client_id is None:
        return
    cached_ui = cached.ui or {}
    server.send_sync("executed", { "node": node_id, "display_node": display_node_id, "output": cached_ui.get("output", None), "prompt_id": prompt_id }, server.client_id)

async def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes, ui_outputs):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    parent_node_id = dynprompt.get_parent_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    cached = await caches.outputs.get(unique_id)
    if cached is not None:
        _send_cached_ui(server, unique_id, display_node_id, cached, prompt_id, ui_outputs)
        get_progress_state().finish_progress(unique_id)
        execution_list.cache_update(unique_id, cached)
        return (ExecutionResult.SUCCESS, None, None)

    input_data_all = None
    lazy_requested = None
    try:
        if unique_id in pending_async_nodes:
            results = []
            for r in pending_async_nodes[unique_id]:
                if isinstance(r, asyncio.Task):
                    try:
                        results.append(r.result())
                    except Exception as ex:
                        # An async task failed - propagate the exception up
                        del pending_async_nodes[unique_id]
                        raise ex
                else:
                    results.append(r)
            del pending_async_nodes[unique_id]
            output_data, output_ui, has_subgraph = get_output_from_returns(results, class_def)
        elif unique_id in pending_subgraph_results:
            cached_results = pending_subgraph_results[unique_id]
            resolved_outputs = []
            for is_subgraph, result in cached_results:
                if not is_subgraph:
                    resolved_outputs.append(result)
                else:
                    resolved_output = []
                    for r in result:
                        if is_link(r):
                            source_node, source_output = r[0], r[1]
                            node_cached = execution_list.get_cache(source_node, unique_id)
                            for o in node_cached.outputs[source_output]:
                                resolved_output.append(o)

                        else:
                            resolved_output.append(r)
                    resolved_outputs.append(tuple(resolved_output))
            output_data = merge_result_data(resolved_outputs, class_def)
            output_ui = []
            del pending_subgraph_results[unique_id]
            has_subgraph = False
        else:
            get_progress_state().start_progress(unique_id)
            input_data_all, missing_keys, v3_data = get_input_data(inputs, class_def, unique_id, execution_list, dynprompt, extra_data)
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": unique_id, "display_node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            obj = await caches.objects.get(unique_id)
            if obj is None:
                obj = class_def()
                await caches.objects.set(unique_id, obj)

            if issubclass(class_def, _ComfyNodeInternal):
                lazy_status_present = first_real_override(class_def, "check_lazy_status") is not None
            else:
                lazy_status_present = getattr(obj, "check_lazy_status", None) is not None
            if lazy_status_present:
                # for check_lazy_status, the returned data should include the original key of the input
                v3_data_lazy = v3_data.copy()
                v3_data_lazy["create_dynamic_tuple"] = True
                required_inputs = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, "check_lazy_status", allow_interrupt=True, v3_data=v3_data_lazy)
                required_inputs = await resolve_map_node_over_list_results(required_inputs)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                lazy_requested = {x for x in required_inputs if isinstance(x, str)}
                required_inputs = [x for x in required_inputs if isinstance(x,str) and (
                    x not in input_data_all or x in missing_keys
                )]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    # Inputs requested here get consumed when the pending pass resumes,
                    # so they must count as evaluated even though the tail is skipped.
                    record_lazy_evidence(caches, unique_id, class_type, lazy_requested)
                    return (ExecutionResult.PENDING, None, None)

            def execution_block_cb(block):
                if block.message is not None:
                    mes = {
                        "prompt_id": prompt_id,
                        "node_id": unique_id,
                        "node_type": class_type,
                        "executed": list(executed),

                        "exception_message": f"Execution Blocked: {block.message}",
                        "exception_type": "ExecutionBlocked",
                        "traceback": [],
                        "current_inputs": [],
                        "current_outputs": [],
                    }
                    server.send_sync("execution_error", mes, server.client_id)
                    return ExecutionBlocker(None)
                else:
                    return block
            def pre_execute_cb(call_index):
                # TODO - How to handle this with async functions without contextvars (which requires Python 3.12)?
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)

            try:
                output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
            finally:
                if comfy.memory_management.aimdo_enabled:
                    if get_console_log_level(args.verbose) == "DEBUG":
                        comfy_aimdo.control.analyze()
                    comfy.model_prefetch.cleanup_prefetch_queues()
                    comfy.model_management.reset_cast_buffers()
                    comfy_aimdo.model_vbar.vbars_reset_watermark_limits()

            if has_pending_tasks:
                pending_async_nodes[unique_id] = output_data
                unblock = execution_list.add_external_block(unique_id)
                async def await_completion():
                    tasks = [x for x in output_data if isinstance(x, asyncio.Task)]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    unblock()
                asyncio.create_task(await_completion())
                return (ExecutionResult.PENDING, None, None)
        if len(output_ui) > 0:
            # Enrich at output-processing time (not in the send path) so assets
            # are registered even when no client is connected, and the asset id
            # flows into ui_outputs and the cache alongside the raw entries.
            output_ui = enrich_output_with_assets(output_ui)
            ui_outputs[unique_id] = {
                "meta": {
                    "node_id": unique_id,
                    "display_node": display_node_id,
                    "parent_node": parent_node_id,
                    "real_node_id": real_node_id,
                },
                "output": output_ui
            }
            if server.client_id is not None:
                server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)
        if has_subgraph:
            cached_outputs = []
            new_node_ids = []
            new_output_ids = []
            new_output_links = []
            for i in range(len(output_data)):
                new_graph, node_outputs = output_data[i]
                if new_graph is None:
                    cached_outputs.append((False, node_outputs))
                else:
                    for node_id, node_info in new_graph.items():
                        new_node_ids.append(node_id)
                        display_id = node_info.get("override_display_id", unique_id)
                        dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
                        # Figure out if the newly created node is an output node
                        class_type = node_info["class_type"]
                        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            new_output_ids.append(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            new_output_links.append((from_node_id, from_socket))
                    cached_outputs.append((True, node_outputs))
            new_node_ids = set(new_node_ids)
            for cache in caches.all:
                subcache = await cache.ensure_subcache_for(unique_id, new_node_ids)
                subcache.clean_unused()
            for node_id in new_output_ids:
                execution_list.add_node(node_id)
                execution_list.cache_link(node_id, unique_id)
            for link in new_output_links:
                execution_list.add_strong_link(link[0], link[1], unique_id)
            pending_subgraph_results[unique_id] = cached_outputs
            return (ExecutionResult.PENDING, None, None)

        cache_entry = CacheEntry(ui=ui_outputs.get(unique_id), outputs=output_data)
        execution_list.cache_update(unique_id, cache_entry)
        await caches.outputs.set(unique_id, cache_entry)

    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": real_node_id,
        }

        return (ExecutionResult.FAILURE, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        logging.error(f"!!! Exception during processing !!! {ex}")
        logging.error(traceback.format_exc())
        tips = ""

        if comfy.model_management.is_oom(ex):
            tips = "This error means you ran out of memory on your GPU.\n\nTIPS: If the workflow worked before you might have accidentally set the batch_size to a large number."
            logging.info("Memory summary:\n{}".format(comfy.model_management.debug_memory_summary()))
            logging.error("Got an OOM, unloading all loaded models.")
            comfy.model_management.unload_all_models()
        elif isinstance(ex, RuntimeError) and ("mat1 and mat2 shapes" in str(ex)) and "Sampler" in class_type:
            tips = "\n\nTIPS: If you have any \"Load CLIP\" or \"*CLIP Loader\" nodes in your workflow connected to this sampler node make sure the correct file(s) and type is selected."

        error_details = {
            "node_id": real_node_id,
            "exception_message": "{}\n{}".format(ex, tips),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted
        }

        return (ExecutionResult.FAILURE, error_details, ex)

    get_progress_state().finish_progress(unique_id)
    executed.add(unique_id)

    if lazy_requested is not None:
        # Evidence comes from what check_lazy_status actually requested this pass; inputs
        # resolved opportunistically from cache don't count as consumed.
        record_lazy_evidence(caches, unique_id, class_type, lazy_requested)

    return (ExecutionResult.SUCCESS, None, None)

class PromptExecutor:
    def __init__(self, server, cache_type=False, cache_args=None):
        self.cache_args = cache_args
        self.cache_type = cache_type
        self.server = server
        self.prompt_model_tracker = comfy.model_patcher.PromptModelTracker()
        self.reset()

    def reset(self):
        self.caches = CacheSet(cache_type=self.cache_type, cache_args=self.cache_args)
        self.status_messages = []
        self.success = True

    def add_message(self, event, data: dict, broadcast: bool):
        data = {
            **data,
            "timestamp": int(time.time() * 1000),
        }
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": list(current_outputs),
            }
            self.add_message("execution_error", mes, broadcast=False)

    def _notify_prompt_lifecycle(self, event: str, prompt_id: str):
        if not _has_cache_providers():
            return

        for provider in _get_cache_providers():
            try:
                if event == "start":
                    provider.on_prompt_start(prompt_id)
                elif event == "end":
                    provider.on_prompt_end(prompt_id)
            except Exception as e:
                _cache_logger.warning(f"Cache provider {provider.__class__.__name__} error on {event}: {e}")

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        asyncio.run(self.execute_async(prompt, prompt_id, extra_data, execute_outputs))

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        set_preview_method(extra_data.get("preview_method"))

        nodes.interrupt_processing(False)
        self.prompt_model_tracker.start()

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)

        self._notify_prompt_lifecycle("start", prompt_id)
        ram_headroom = int(self.cache_args["ram"] * (1024 ** 3))
        ram_inactive_headroom = int(self.cache_args["ram_inactive"] * (1024 ** 3))
        ram_release_callback = self.caches.outputs.ram_release if self.cache_type == CacheType.RAM_PRESSURE else None
        comfy.memory_management.set_ram_cache_release_state(ram_release_callback, ram_headroom)

        try:
            with torch.inference_mode():
                dynamic_prompt = DynamicPrompt(prompt)
                reset_progress_state(prompt_id, dynamic_prompt)
                add_progress_handler(WebUIProgressHandler(self.server))
                is_changed_cache = IsChangedCache(prompt_id, dynamic_prompt, self.caches.outputs)
                for cache in self.caches.all:
                    await cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
                    cache.clean_unused()

                node_ids = list(prompt.keys())
                cache_results = await asyncio.gather(
                    *(self.caches.outputs.get(node_id) for node_id in node_ids)
                )
                cached_nodes = [
                    node_id for node_id, result in zip(node_ids, cache_results)
                    if result is not None
                ]

                comfy.model_management.cleanup_models_gc()
                self.add_message("execution_cached",
                              { "nodes": cached_nodes, "prompt_id": prompt_id},
                              broadcast=False)
                pending_subgraph_results = {}
                pending_async_nodes = {} # TODO - Unify this with pending_subgraph_results
                ui_node_outputs = {}
                executed = set()
                execution_list = ExecutionList(dynamic_prompt, self.caches.outputs, self.prompt_model_tracker.add)
                current_outputs = self.caches.outputs.all_node_ids()
                for node_id in list(execute_outputs):
                    execution_list.add_node(node_id)

                while not execution_list.is_empty():
                    node_id, error, ex = await execution_list.stage_node_execution()
                    if error is not None:
                        self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                        break

                    assert node_id is not None, "Node ID should not be None at this point"
                    result, error, ex = await execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes, ui_node_outputs)
                    self.success = result != ExecutionResult.FAILURE
                    if result == ExecutionResult.FAILURE:
                        self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                        break
                    elif result == ExecutionResult.PENDING:
                        execution_list.unstage_node_execution()
                    else: # result == ExecutionResult.SUCCESS:
                        execution_list.complete_node_execution()

                    if self.cache_type == CacheType.RAM_PRESSURE:
                        ram_release_callback(ram_inactive_headroom)
                        ram_shortfall = ram_headroom - comfy.system_memory.virtual_memory_available()
                        if ram_shortfall > 0:
                            freed = ram_release_callback(ram_headroom, free_active=True, min_entry_size=RAM_CACHE_LARGE_INTERMEDIATE)
                            ram_shortfall -= freed
                        if comfy.model_management.should_free_pins_for_ram_pressure(ram_shortfall):
                            freed = comfy.model_management.free_pins(ram_shortfall + 512 * (1024 ** 2))
                            if freed < ram_shortfall:
                                if freed > 64 * (1024 ** 2):
                                    # AIMDO MEM_DECOMMIT can outrun psutil.available catching up.
                                    time.sleep(0.05)
                                ram_release_callback(ram_headroom, free_active=True)
                else:
                    # Only execute when the while-loop ends without break
                    # Send cached UI for intermediate output nodes that weren't executed
                    for node_id in dynamic_prompt.all_node_ids():
                        if node_id in executed:
                            continue
                        if not _is_intermediate_output(dynamic_prompt, node_id):
                            continue
                        cached = await self.caches.outputs.get(node_id)
                        if cached is not None:
                            display_node_id = dynamic_prompt.get_display_node_id(node_id)
                            _send_cached_ui(self.server, node_id, display_node_id, cached, prompt_id, ui_node_outputs)
                    self.add_message("execution_success", { "prompt_id": prompt_id }, broadcast=False)

                ui_outputs = {}
                meta_outputs = {}
                for node_id, ui_info in ui_node_outputs.items():
                    ui_outputs[node_id] = ui_info["output"]
                    meta_outputs[node_id] = ui_info["meta"]
                self.history_result = {
                    "outputs": ui_outputs,
                    "meta": meta_outputs,
                }
                self.server.last_node_id = None
                if comfy.model_management.DISABLE_SMART_MEMORY:
                    comfy.model_management.unload_all_models()
        finally:
            if self.cache_type == CacheType.RAM_PRESSURE:
                detail("RAM cache evictions: prompt=%s active=%s full=%s", prompt_id, self.caches.outputs.active_evictions, self.caches.outputs.full_evictions)
            comfy.memory_management.set_ram_cache_release_state(None, 0)
            self.prompt_model_tracker.end()
            self._notify_prompt_lifecycle("end", prompt_id)


async def validate_inputs(prompt_id, prompt, item, validated, visiting=None):
    if visiting is None:
        visiting = []

    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    if unique_id in visiting:
        cycle_path_nodes = visiting[visiting.index(unique_id):] + [unique_id]
        cycle_nodes = list(dict.fromkeys(cycle_path_nodes))
        cycle_path = " -> ".join(f"{node_id} ({prompt[node_id]['class_type']})" for node_id in cycle_path_nodes)
        for node_id in cycle_nodes:
            validated[node_id] = (False, [{
                "type": "dependency_cycle",
                "message": "Dependency cycle detected",
                "details": cycle_path,
                "extra_info": {
                    "node_id": node_id,
                    "cycle_nodes": cycle_nodes,
                }
            }], node_id)
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    errors = []
    valid = True

    v3_data = None
    validate_function_inputs = []
    validate_has_kwargs = False
    if issubclass(obj_class, _ComfyNodeInternal):
        obj_class: _io._ComfyNodeBaseInternal
        class_inputs = obj_class.INPUT_TYPES()
        class_inputs, _, v3_data = _io.get_finalized_class_inputs(class_inputs, inputs)
        validate_function_name = "validate_inputs"
        validate_function = first_real_override(obj_class, validate_function_name)
    else:
        class_inputs = obj_class.INPUT_TYPES()
        validate_function_name = "VALIDATE_INPUTS"
        validate_function = getattr(obj_class, validate_function_name, None)
    if validate_function is not None:
        argspec = inspect.getfullargspec(validate_function)
        validate_function_inputs = argspec.args
        validate_has_kwargs = argspec.varkw is not None
    received_types = {}

    valid_inputs = set(class_inputs.get('required',{})).union(set(class_inputs.get('optional',{})))

    for x in valid_inputs:
        input_type, input_category, extra_info = get_input_info(obj_class, x, class_inputs)
        assert extra_info is not None
        if x not in inputs:
            if input_category == "required":
                details = f"{x}" if not v3_data else x.split(".")[-1]
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": details,
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)
            continue

        val = inputs[x]
        info = (input_type, extra_info)
        if isinstance(val, list):
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            received_type = r[val[1]]
            received_types[x] = received_type
            if 'input_types' not in validate_function_inputs and not validate_node_input(received_type, input_type):
                details = f"{x}, received_type({received_type}) mismatch input_type({input_type})"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)
                continue
            try:
                visiting.append(unique_id)
                try:
                    r = await validate_inputs(prompt_id, prompt, o_id, validated, visiting)
                finally:
                    visiting.pop()
                if r[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = (False, reasons, o_id)
                continue
        else:
            try:
                # Unwraps values wrapped in __value__ key or typed wrapper.
                # This is used to pass list widget values to execution,
                # as by default list value is reserved to represent the
                # connection between nodes.
                if isinstance(val, dict):
                    if "__value__" in val:
                        val = val["__value__"]
                        inputs[x] = val

                if input_type == "INT":
                    val = int(val)
                    inputs[x] = val
                if input_type == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if input_type == "STRING":
                    val = str(val)
                    inputs[x] = val
                if input_type == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {input_type} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if x not in validate_function_inputs and not validate_has_kwargs:
                if "min" in extra_info and val < extra_info["min"]:
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, extra_info["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in extra_info and val > extra_info["max"]:
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, extra_info["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

                if isinstance(input_type, list) or input_type == io.Combo.io_type:
                    if input_type == io.Combo.io_type:
                        combo_options = extra_info.get("options", [])
                    else:
                        combo_options = input_type
                    is_multiselect = extra_info.get("multiselect", False)
                    if is_multiselect and isinstance(val, list):
                        invalid_vals = [v for v in val if v not in combo_options]
                    else:
                        invalid_vals = [val] if val not in combo_options else []
                    if invalid_vals:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(combo_options) > 20:
                            list_info = f"(list of length {len(combo_options)})"
                            input_config = None
                        else:
                            list_info = str(combo_options)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: {', '.join(repr(v) for v in invalid_vals)} not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(validate_function_inputs) > 0 or validate_has_kwargs:
        input_data_all, _, v3_data = get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs or validate_has_kwargs:
                input_filtered[x] = input_data_all[x]
        if 'input_types' in validate_function_inputs:
            input_filtered['input_types'] = [received_types]

        ret = await _async_map_node_over_list(prompt_id, unique_id, obj_class, input_filtered, validate_function_name, v3_data=v3_data)
        ret = await resolve_map_node_over_list_results(ret)
        for x in input_filtered:
            for i, r in enumerate(ret):
                if r is not True and not isinstance(r, ExecutionBlocker):
                    details = f"{x}"
                    if r is not False:
                        details += f" - {str(r)}"

                    error = {
                        "type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                        }
                    }
                    errors.append(error)
                    continue

    ret = validated.get(unique_id, (True, [], unique_id))
    # Recursive cycle detection may have already populated an error on us. Join it.
    ret = (
        ret[0] and valid is True and not errors,
        ret[1] + [error for error in errors if error not in ret[1]],
        unique_id,
    )

    validated[unique_id] = ret
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

async def validate_prompt(prompt_id, prompt, partial_execution_list: Union[list[str], None]):
    outputs = set()
    for x in prompt:
        if 'class_type' not in prompt[x]:
            node_data = prompt[x]
            node_title = node_data.get('_meta', {}).get('title')
            error = {
                "type": "missing_node_type",
                "message": f"Node '{node_title or f'ID #{x}'}' has no class_type. The workflow may be corrupted or a custom node is missing.",
                "details": f"Node ID '#{x}'",
                "extra_info": {
                    "node_id": x,
                    "class_type": None,
                    "node_title": node_title
                }
            }
            return (False, error, [], {})

        class_type = prompt[x]['class_type']
        class_ = nodes.NODE_CLASS_MAPPINGS.get(class_type, None)
        if class_ is None:
            node_data = prompt[x]
            node_title = node_data.get('_meta', {}).get('title', class_type)
            error = {
                "type": "missing_node_type",
                "message": f"Node '{node_title}' not found. The custom node may not be installed.",
                "details": f"Node ID '#{x}'",
                "extra_info": {
                    "node_id": x,
                    "class_type": class_type,
                    "node_title": node_title
                }
            }
            return (False, error, [], {})

        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE is True:
            if partial_execution_list is None or x in partial_execution_list:
                outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], {})

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = await validate_inputs(prompt_id, prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            logging.error(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        logging.error(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            logging.error(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logging.error("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)

MAXIMUM_HISTORY_SIZE = 10000

class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        self.flags = {}

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    class ExecutionStatus(NamedTuple):
        status_str: Literal['success', 'error']
        completed: bool
        messages: List[str]

    def task_done(self, item_id, history_result,
                  status: Optional['PromptQueue.ExecutionStatus'], process_item=None):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())

            if process_item is not None:
                prompt = process_item(prompt)

            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": {},
                'status': status_dict,
            }
            self.history[prompt[1]].update(history_result)
            self.server.queue_updated()

    # Note: slow
    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    # read-safe as long as queue items are immutable
    def get_current_queue_volatile(self):
        with self.mutex:
            running = [x for x in self.currently_running.values()]
            queued = copy.copy(self.queue)
            return (running, queued)

    def interrupt_if_running(self, prompt_id):
        """Interrupt the running prompt with this id, atomically.

        Checks the live running set and signals the interrupt under the queue
        mutex, so the worker cannot move the job to done (and start the next
        prompt) in between. Returns True if a matching job was running and an
        interrupt was signalled, False otherwise. The atomicity is what keeps a
        cancel from landing on an unrelated prompt that started after a separate
        is-running check: the global interrupt flag is reset at the start of
        every prompt (execute_async), so a job that finishes before consuming
        the flag cannot leak the interrupt onto its successor.
        """
        with self.mutex:
            for item in self.currently_running.values():
                if item[1] == prompt_id:
                    nodes.interrupt_processing()
                    return True
        return False

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1, map_function=None):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        p = self.history[k]
                        if map_function is not None:
                            p = map_function(p)
                        out[k] = p
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                p = self.history[prompt_id]
                if map_function is None:
                    p = copy.deepcopy(p)
                else:
                    p = map_function(p)
                return {prompt_id: p}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret = self.flags
                self.flags = {}
                return ret
            else:
                return self.flags.copy()
