"""Job-owned serial execution of admitted upstream variants.

The graph RPC only creates native continuation controls. Node bodies run from
the ordinary executor after the requesting guest dispatch has returned.
"""
import asyncio
import contextvars
import copy
import time
import uuid
from dataclasses import dataclass

_active_owner = contextvars.ContextVar('comfy_upstream_variant_owner', default=None)
_active_cell = contextvars.ContextVar('comfy_upstream_variant_cell', default=None)
CONTROL_KEY = '_core_upstream_variants'
CONTINUATION_KEY = '_core_upstream_variant_continuation'


class VariantProfilePending(RuntimeError):
    pass


def current(prompt_id):
    owner = _active_owner.get()
    if owner is None or owner.closed or owner.prompt_id != str(prompt_id):
        return None
    return owner


def active_dynamic_prompt(owner):
    cell = _active_cell.get()
    return cell.dynamic_prompt if cell is not None and cell.owner is owner else owner.dynamic_prompt


class VariantExecutionFailure(Exception):
    def __init__(self, details, error):
        self.details = details
        self.error = error
        super().__init__(str(error))


class _NodeCache:
    def __init__(self, objects=False):
        self.entries = {}
        self.dynamic_prompt = None
        self.objects = objects

    def _key(self, node_id):
        if self.objects:
            return node_id, self.dynamic_prompt.get_node(node_id)['class_type']
        return node_id

    def get_local(self, node_id):
        return self.entries.get(self._key(node_id))

    async def get(self, node_id):
        return self.get_local(node_id)

    def set_local(self, node_id, value):
        self.entries[self._key(node_id)] = value

    async def set(self, node_id, value):
        self.set_local(node_id, value)

    def remove(self,node_id):
        self.entries.pop(self._key(node_id),None)

    def all_node_ids(self):
        return {key[0] if self.objects else key for key in self.entries}

    async def ensure_subcache_for(self, node_id, children_ids):
        # Dynamic targets require a separately supported original semantic
        # profile. Do not silently replace the reference executor with a DAG.
        raise VariantProfilePending('nested variant subgraphs require target-profile qualification')


class _CellCaches:
    def __init__(self):
        self.outputs = _NodeCache()
        self.objects = _NodeCache(objects=True)
        self.all = [self.outputs, self.objects]
        self.old_prompt = {}
        self.old_fingerprints = {}

    def bind(self, dynamic_prompt):
        for cache in self.all:
            cache.dynamic_prompt = dynamic_prompt

    def clear(self):
        for cache in self.all:
            cache.entries.clear()
            cache.dynamic_prompt = None
        self.old_prompt.clear()
        self.old_fingerprints.clear()


@dataclass(frozen=True)
class NativeVariantControl:
    state: object
    index: int


@dataclass(frozen=True)
class NativeVariantContinuation:
    state: object


def native_control(node_info):
    value = node_info.get(CONTROL_KEY)
    return value if type(value) is NativeVariantControl else None


def _binding(node_info):
    control = native_control(node_info)
    if control is not None:
        return control
    value = node_info.get(CONTINUATION_KEY)
    return value if type(value) is NativeVariantContinuation else None


def assert_node_current(dynamic_prompt,node_id,cache=None):
    binding = _binding(dynamic_prompt.get_node(node_id))
    if binding is None:
        return
    state = binding.state
    state.assert_current()
    if dynamic_prompt is not state.owner.dynamic_prompt:
        raise PermissionError('variant continuation belongs to another dynamic prompt')
    if cache is not None:
        state.owner.record_output(cache,node_id)


def invocation_guard(prompt_id,node_id):
    owner = current(prompt_id)
    if owner is None:
        return None
    cell = _active_cell.get()
    if cell is not None and cell.owner is owner:
        if node_id not in cell.plan.upstream:
            raise PermissionError('variant invocation is outside the admitted upstream graph')
        return cell.assert_current
    if not owner.dynamic_prompt.has_node(node_id):
        return None
    binding = _binding(owner.dynamic_prompt.get_node(node_id))
    if binding is None:
        return None
    if binding.state.owner is not owner:
        raise PermissionError('variant invocation belongs to another execution owner')
    return binding.state.assert_current


def cache_identity(node_info):
    binding = _binding(node_info)
    if binding is None:
        return None
    slot = binding.index if type(binding) is NativeVariantControl else 'continuation'
    return binding.state.cache_token,slot


def track_completion(task):
    cell = _active_cell.get()
    if cell is not None:
        cell.tasks.add(task)


class CoreVariantExecutionOwner:
    def __init__(self, prompt_id, dynamic_prompt, server, extra_data, asset_manager):
        self.prompt_id = str(prompt_id)
        self.dynamic_prompt = dynamic_prompt
        self.server = server
        self.extra_data = extra_data
        self.asset_manager = asset_manager
        self.states = []
        self.output_entries = {}
        self.closed = False
        self._token = None

    def __enter__(self):
        self._token = _active_owner.set(self)
        return self

    async def close(self):
        self.closed = True
        try:
            try:
                for cache,node_id in self.output_entries.values():
                    cache.remove(node_id)
            finally:
                for state in self.states:
                    await state.close()
        finally:
            self.states.clear()
            self.output_entries.clear()
            self.dynamic_prompt = None
            self.server = None
            self.extra_data = None
            self.asset_manager = None
            if self._token is not None:
                _active_owner.reset(self._token)
                self._token = None

    def record_output(self,cache,node_id):
        self.output_entries[(id(cache),node_id)] = cache,node_id

    async def materialize(self, plan, policy):
        policy.current(policy.graph_owner, 'materialize-deferred-variants')
        if self.closed or policy.graph_owner._execution_owner is not self:
            raise PermissionError('variant plan does not belong to this execution owner')
        if active_dynamic_prompt(self) is not self.dynamic_prompt:
            raise VariantProfilePending('nested upstream-variant controllers are not admitted')
        from comfy_execution.graph_utils import GraphBuilder

        # All node metadata is parsed/accounted before copying the job view.
        node_ids = self.dynamic_prompt.all_node_ids()
        if len(node_ids) > policy.budget.limits.max_items:
            raise ValueError('variant job prompt exceeds node-accounting budget')
        source = {node_id:{key:value for key,value in self.dynamic_prompt.get_node(node_id).items()
                           if key != 'is_changed'} for node_id in node_ids
                  if _binding(self.dynamic_prompt.get_node(node_id)) is None}
        policy.measure_source(source)
        barriers = _continuation_barriers(plan)
        policy.measure_source(barriers)
        state = _VariantSeries(self, plan, policy, copy.deepcopy(source))
        self.states.append(state)
        graph = GraphBuilder(prefix=plan.continuation_prefix)
        variant_nodes = {}
        previous = None
        for index, variant in enumerate(plan.variants):
            target = graph.node('CreateList', '#variant-' + str(index))
            if previous is not None:
                target.set_input('input0',previous.out(0))
            variant_nodes[variant['id']] = target
            previous = target
        continuation = {}
        for spec in plan.continuation:
            class_type = (plan.direct_clones[spec['id']][1]['class_type']
                          if 'clone_input' in spec else spec['class_type'])
            continuation[spec['id']] = graph.node(class_type,spec['id'])

        def resolve(value):
            if type(value) is dict and set(value) == {'variant'}:
                return variant_nodes[value['variant']].out(0)
            if type(value) is dict and set(value) == {'node','output'}:
                return continuation[value['node']].out(value['output'])
            if type(value) is dict:
                return {key:resolve(child) for key,child in value.items()}
            if type(value) is list:
                return [resolve(child) for child in value]
            return value

        for spec in plan.continuation:
            target = continuation[spec['id']]
            if 'clone_input' in spec:
                source_id,source_node = plan.direct_clones[spec['id']]
                for name,value in source_node.get('inputs',{}).items():
                    target.set_input(name,copy.deepcopy(value))
                target.set_override_display_id(str(self.dynamic_prompt.get_display_node_id(source_id)))
            else:
                for name,value in spec.get('inputs',{}).items():
                    target.set_input(name,resolve(value))
                target.set_override_display_id(str(self.dynamic_prompt.get_display_node_id(plan.caller_id)))
        for index,node_ids in barriers.items():
            target = variant_nodes[plan.variants[int(index)]['id']]
            for number,node_id in enumerate(node_ids,1):
                target.set_input('input'+str(number),continuation[node_id].out(0))
        expansion = graph.finalize()
        for index,variant in enumerate(plan.variants):
            expansion[variant_nodes[variant['id']].id][CONTROL_KEY] = NativeVariantControl(state,index)
        for target in continuation.values():
            expansion[target.id][CONTINUATION_KEY] = NativeVariantContinuation(state)
        result = [continuation[item['node']].out(item['output']) for item in plan.outputs]
        return {'result':result,'expand':expansion}


def _continuation_barriers(plan):
    """Order output-reachable continuations before the next variant they precede."""
    indices = {variant['id']:index for index,variant in enumerate(plan.variants)}
    specs = {spec['id']:spec for spec in plan.continuation}
    active = set()
    latest = {}
    def visit(node_id):
        if node_id in latest:
            return latest[node_id]
        if node_id in active:
            raise ValueError('continuation barrier cycle')
        active.add(node_id)
        last = -1
        def scan(value):
            nonlocal last
            if type(value) is dict:
                if set(value) == {'variant'}:
                    last = max(last,indices[value['variant']])
                elif set(value) == {'node','output'}:
                    last = max(last,visit(value['node']))
                else:
                    for child in value.values():
                        scan(child)
            elif type(value) is list:
                for child in value:
                    scan(child)
        scan(specs[node_id].get('inputs',{}))
        active.remove(node_id)
        latest[node_id] = last
        return last
    for output in plan.outputs:
        visit(output['node'])
    barriers = {}
    for node_id,last in latest.items():
        if 0 <= last < len(plan.variants)-1:
            barriers.setdefault(str(last+1),[]).append(node_id)
    # String keys keep accounting in the same bounded JSON domain as the plan.
    return barriers


class _VariantSeries:
    def __init__(self, owner, plan, policy, source):
        self.owner = owner
        self.cache_token = uuid.uuid4().hex
        self.plan = plan
        self.policy = policy
        self.source = source
        self.caches = _CellCaches()
        self.dynamic_prompt = None
        self.results = []
        self.tasks = set()
        self.completed = False
        self.next_index = 0
        self.closed = False

    def assert_current(self):
        if self.closed or self.owner.closed or current(self.owner.prompt_id) is not self.owner:
            raise PermissionError('variant execution lifetime is closed')
        self.policy.current(self.policy.graph_owner,'execute-native-variant')

    async def _invalidate(self, execution, nodes_module):
        # Last-cell invalidation deliberately does not resurrect older A/B/A
        # output signatures. Original-ID/class object instances remain intact.
        fingerprints = {}
        for node_id in self.plan.upstream:
            self.assert_current()
            node = self.dynamic_prompt.get_node(node_id)
            class_def = nodes_module.NODE_CLASS_MAPPINGS[node['class_type']]
            node.pop('is_changed',None)
            method = None
            if (issubclass(class_def,execution._ComfyNodeInternal)
                    and execution.first_real_override(class_def,'fingerprint_inputs') is not None):
                method = 'fingerprint_inputs'
            elif hasattr(class_def,'IS_CHANGED'):
                method = 'IS_CHANGED'
            if method is None:
                fingerprints[node_id] = ''
                continue
            inputs,_,v3_data = execution.get_input_data(node['inputs'],class_def,node_id,
                _CachedInputs(self.caches.outputs),self.dynamic_prompt,self.owner.extra_data)
            try:
                values = await execution._async_map_node_over_list(self.owner.prompt_id,node_id,
                    class_def,inputs,method,v3_data=v3_data)
                fingerprints[node_id] = await execution.resolve_map_node_over_list_results(values)
            except asyncio.CancelledError:
                raise
            except Exception:
                fingerprints[node_id] = float('nan')

        invalid = {}
        def changed(node_id):
            if node_id in invalid:
                return invalid[node_id]
            node = self.dynamic_prompt.get_node(node_id)
            old = self.caches.old_prompt.get(node_id)
            bad = (self.caches.outputs.get_local(node_id) is None or old is None
                   or node['inputs'] != old['inputs'] or node['class_type'] != old['class_type']
                   or fingerprints[node_id] != self.caches.old_fingerprints.get(node_id,''))
            if not bad:
                for value in node['inputs'].values():
                    if execution.is_link(value) and changed(value[0]):
                        bad = True
                        break
            invalid[node_id] = bad
            if bad:
                self.caches.outputs.entries.pop(node_id,None)
            return bad
        for node_id in self.plan.upstream:
            changed(node_id)
        return fingerprints

    async def _run_cell(self, variant):
        import execution
        import nodes
        from comfy_execution.graph import DynamicPrompt

        self.assert_current()
        prompt = copy.deepcopy(self.source)
        for node_id,node in self.plan.upstream.items():
            prompt[node_id] = copy.deepcopy(node)
        for override in variant['overrides']:
            prompt[override['node_id']]['inputs'][override['input_name']] = copy.deepcopy(override['value'])
        self.dynamic_prompt = DynamicPrompt(prompt)
        self.caches.bind(self.dynamic_prompt)
        token = _active_cell.set(self)
        pending = {}
        try:
            valid, errors, invalid_id = await execution.validate_inputs(
                self.owner.prompt_id,prompt,self.plan.producer_id,{},_ignore_numeric_hints=True)
            if not valid:
                raise VariantExecutionFailure(dict(node_id=invalid_id,validation_errors=errors,
                    exception_message='variant input validation failed',exception_type='ValueError',
                    traceback=[],current_inputs=[]),
                                              ValueError('variant input validation failed'))
            fingerprints = await self._invalidate(execution,nodes)
            queue = _ordered_execution_list(self.dynamic_prompt,self.caches.outputs,self.plan.upstream)
            pending_subgraphs = {}
            executed = set()
            ui = {}
            while not queue.is_empty():
                self.assert_current()
                node_id,error,exception = await queue.stage_node_execution()
                self.assert_current()
                if error is not None:
                    raise VariantExecutionFailure(error,exception)
                if node_id not in self.plan.upstream:
                    raise VariantProfilePending('variant scheduler encountered an unadmitted dynamic target')
                if node_id in self.plan.configuration_nodes:
                    # stage_node_execution has already waited for every linked
                    # ancestor. Keep original IDs/dependency order and cache the
                    # native ttN_Plotting result, without calling the config body.
                    if self.policy.validate_configuration(node_id,
                            self.dynamic_prompt.get_node(node_id)['class_type']) is not True:
                        raise PermissionError('configuration substitution was not reauthorized')
                    self.assert_current()
                    entry = execution.CacheEntry(ui=None,outputs=[[None]])
                    queue.cache_update(node_id,entry)
                    await self.caches.outputs.set(node_id,entry)
                    self.assert_current()
                    executed.add(node_id)
                    queue.complete_node_execution()
                    continue
                result,error,exception = await execution.execute(self.owner.server,self.dynamic_prompt,
                    self.caches,node_id,self.owner.extra_data,executed,self.owner.prompt_id,
                    queue,pending_subgraphs,pending,ui,self.owner.asset_manager)
                if result is execution.ExecutionResult.FAILURE:
                    raise VariantExecutionFailure(error,exception)
                if result is execution.ExecutionResult.PENDING:
                    queue.unstage_node_execution()
                else:
                    queue.complete_node_execution()
            self.assert_current()
            cached = self.caches.outputs.get_local(self.plan.producer_id)
            if cached is None or not 0 <= self.plan.output_index < len(cached.outputs):
                raise RuntimeError('the admitted upstream producer did not return its wired output')
            # A snapshot at cell completion preserves an image even if a later
            # stateful target mutates an older output tensor in-place.
            self.results.append(_snapshot_values(cached.outputs[self.plan.output_index],self.policy))
            for node_id in executed:
                self.caches.old_prompt[node_id] = copy.deepcopy(prompt[node_id])
                self.caches.old_fingerprints[node_id] = fingerprints[node_id]
        finally:
            for values in pending.values():
                self.tasks.update(value for value in values if isinstance(value,asyncio.Task))
            if self.tasks:
                tasks = tuple(self.tasks)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                self.tasks.clear()
            _active_cell.reset(token)

    async def run_cell(self,index):
        try:
            self.assert_current()
            if index != self.next_index or self.completed:
                raise PermissionError('variant cells must execute once in their original order')
            if self.policy.deadline_monotonic is None:
                raise PermissionError('variant execution has no original job deadline')
            variant = self.plan.variants[index]
            async with asyncio.timeout(max(0,self.policy.deadline_monotonic-time.monotonic())):
                await self._run_cell(variant)
            self.next_index += 1
            self.completed = self.next_index == len(self.plan.variants)
        except VariantExecutionFailure as failure:
            failure.details = {**failure.details,'upstream_variant_id':variant['id']}
            await self.close()
            raise
        except BaseException:
            await self.close()
            raise

    async def close(self):
        self.closed = True
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks,return_exceptions=True)
        self.tasks.clear()
        self.results.clear()
        self.caches.clear()
        self.source.clear()
        self.dynamic_prompt = None
        self.plan = None
        self.policy = None


class _CachedInputs:
    def __init__(self, outputs):self.outputs=outputs
    def get_cache(self,node_id,_consumer):return self.outputs.get_local(node_id)


def _ordered_execution_list(dynamic_prompt, outputs, upstream):
    from comfy_execution.graph import ExecutionList

    queue = ExecutionList(dynamic_prompt,outputs)
    previous = None
    for node_id in upstream:
        # The reference walks every linked ancestor in input encounter order,
        # including lazy links. Explicit edges prevent ready async/output nodes
        # from changing that order inside a cell.
        queue.add_node(node_id,include_lazy=True)
        if previous is not None:
            queue.add_strong_link(previous,0,node_id)
        previous = node_id
    return queue


def _snapshot_values(values, policy):
    import torch

    # This first execution profile supports IMAGE-producing branches. Other
    # wired output semantics need an explicit target profile, not a raw unwrap.
    if type(values) is not list or any(not isinstance(value,torch.Tensor) for value in values):
        raise VariantProfilePending('the current variant output profile requires IMAGE tensors')
    size = sum(value.nelement()*value.element_size() for value in values)
    policy.budget.charge_images(size,policy.max_retained_image_bytes)
    return [value.clone() for value in values]


async def execute_control(prompt_id,dynamic_prompt,node_id,caches,queue):
    import execution

    control = native_control(dynamic_prompt.get_node(node_id))
    owner = current(prompt_id)
    if (control is None or owner is None or control.state.owner is not owner
            or dynamic_prompt is not owner.dynamic_prompt):
        raise PermissionError('native variant control belongs to another execution owner')
    state = control.state
    state.assert_current()
    if control.index == state.next_index:
        await state.run_cell(control.index)
    elif not 0 <= control.index < state.next_index:
        raise PermissionError('variant output is not complete')
    state.assert_current()
    outputs = [state.results[control.index]]
    entry = execution.CacheEntry(ui=None,outputs=outputs)
    queue.cache_update(node_id,entry)
    owner.record_output(caches.outputs,node_id)
    await caches.outputs.set(node_id,entry)
