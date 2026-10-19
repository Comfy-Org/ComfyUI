"""Host-only planning for ordered variants of an already linked branch.

The scheduler and target admission belong to the execution owner. This module
does not select them, infer purity, or run legacy node validators.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re
import uuid
import time


class UpstreamVariantsPending(RuntimeError):
    pass


@dataclass(frozen=True)
class VariantLimits:
    max_request_bytes: int
    max_items: int
    max_depth: int
    max_upstream_nodes: int
    max_variants: int
    max_cloned_nodes: int
    max_overrides: int
    max_literal_bytes: int
    max_retained_tickets: int
    max_retained_nodes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("variant limits require positive host-selected integers")


class VariantBudget:
    """One mutable budget shared by every variant request in a dispatch."""

    def __init__(self, limits):
        if type(limits) is not VariantLimits:
            raise TypeError("variant limits must be host-owned")
        self.limits = limits
        self.used = dict(variants=0, cloned_nodes=0, overrides=0, literal_bytes=0,
                         retained_tickets=0, retained_nodes=0)
        self.retained_image_bytes = 0

    def charge_images(self, cost, limit):
        if type(cost) is not int or cost < 0 or type(limit) is not int or limit <= 0:
            raise PermissionError('variant IMAGE retention needs an explicit host byte budget')
        if self.retained_image_bytes + cost > limit:
            raise ValueError('job variant IMAGE retention budget exhausted')
        self.retained_image_bytes += cost

    def charge(self, **cost):
        for name, value in cost.items():
            if type(value) is not int or value < 0:
                raise ValueError("invalid variant budget charge")
            if self.used[name] + value > getattr(self.limits, 'max_' + name):
                raise ValueError(f"dispatch variant budget exhausted: {name}")
        for name, value in cost.items():
            self.used[name] += value


@dataclass(frozen=True)
class VariantPlan:
    caller_id: str
    source_sha256: str
    producer_id: str
    output_index: int
    upstream: dict
    variants: tuple
    continuation: tuple
    direct_clones: dict
    outputs: tuple
    retained_nodes: int
    continuation_prefix: str
    configuration_nodes: tuple[str, ...] = ()


class UpstreamVariantPolicy:
    """A private host construction boundary, never an RPC argument.

    ``admit_target`` must select Core or an actual V2 proxy BEFORE consulting
    custom schemas. ``validate_override`` uses the admitted schema and actual
    input/resource authority, not UI numeric recommendations. ``schedule`` is
    the canonical owner's ordered/cache-preserving implementation. A missing
    implementation remains pending rather than silently using independent DAGs.
    """

    def __init__(self, graph_owner, budget, *, assert_current, admit_target,
                 validate_override, schedule, external_node_types=frozenset(),
                 scheduler_overhead_nodes=0, deadline_monotonic=None,
                 max_retained_image_bytes=None, validate_configuration=None):
        if type(budget) is not VariantBudget:
            raise TypeError("variant budget must be host-owned")
        if any(not callable(fn) for fn in (assert_current, admit_target, validate_override)):
            raise TypeError("variant policy requires current admission functions")
        if schedule is not None and not callable(schedule):
            raise TypeError("invalid variant scheduler")
        if (type(external_node_types) is not frozenset or len(external_node_types) > 64
                or any(type(name) is not str or not 1 <= len(name) <= 256
                       or '\x00' in name for name in external_node_types)):
            raise ValueError("invalid external expansion policy")
        self.graph_owner = graph_owner
        self.budget = budget
        self.assert_current = assert_current
        self.admit_target = admit_target
        self.validate_override = validate_override
        if validate_configuration is not None and not callable(validate_configuration):
            raise TypeError("configuration validation must be host-owned")
        self.validate_configuration = validate_configuration
        self.schedule = schedule
        self.external_node_types = external_node_types
        if type(scheduler_overhead_nodes) is not int or scheduler_overhead_nodes < 0:
            raise ValueError("invalid scheduler node accounting")
        self.scheduler_overhead_nodes = scheduler_overhead_nodes
        self.deadline_monotonic = deadline_monotonic
        self.max_retained_image_bytes = max_retained_image_bytes
        self._sources = []

    def current(self, graph_owner, stage):
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise PermissionError('upstream variant original deadline expired')
        if graph_owner is not self.graph_owner or self.assert_current(stage) is not True:
            raise PermissionError("upstream variant selection is no longer current")
        for dynprompt, caller_id, node_ids, expected in self._sources:
            if _source_fingerprint(dynprompt, caller_id, node_ids, self.budget.limits) != expected:
                raise PermissionError("upstream graph source changed after planning")

    def measure_source(self, source):
        return _measure(source,self.budget.limits)


def _link(value):
    return (type(value) is list and len(value) == 2 and type(value[0]) is str
            and type(value[1]) is int and value[1] >= 0)


def _name(value, label, *, local=False):
    if type(value) is not str or not 1 <= len(value) <= (128 if local else 256) or '\x00' in value:
        raise ValueError(f"invalid {label}")
    if local and not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', value):
        raise ValueError(f"invalid {label}")
    return value


def _node_id(value, limits):
    if (type(value) is not str or not value or '\x00' in value
            or len(value) > limits.max_request_bytes):
        raise ValueError("invalid upstream node id")
    return value


def _measure(value, limits, *, override=False, ref_type=()):
    # Count nested input before copying, serializing or invoking a target policy.
    pending = [(value, 0)]
    count = size = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > limits.max_items or depth > limits.max_depth:
            raise ValueError("variant metadata exceeds item/depth budget")
        kind = type(item)
        if item is None or kind is bool:
            size += 5
        elif kind in (int, float):
            if kind is float and not math.isfinite(item):
                raise ValueError("variant literals require finite numbers")
            if kind is int and item.bit_length() > 1024:
                raise ValueError("variant integer exceeds metadata budget")
            size += len(str(item)) + 1
        elif kind is str:
            if len(item) > limits.max_request_bytes:
                raise ValueError("variant string exceeds metadata budget")
            size += 2
            for character in item:
                point = ord(character)
                if 0xD800 <= point <= 0xDFFF:
                    raise ValueError("variant strings require valid Unicode")
                size += (6 if point < 32 else 2 if character in {'"','\\'}
                         else 1 if point < 128 else 2 if point < 2048
                         else 3 if point < 65536 else 4)
                if size > limits.max_request_bytes:
                    raise ValueError("variant request exceeds metadata byte budget")
        elif kind in (dict, list):
            size += 2 + len(item) * (2 if kind is dict else 1)
            if count + len(pending) + len(item) > limits.max_items:
                raise ValueError("variant metadata exceeds item budget")
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise TypeError("variant object keys must be strings")
                if any(key.startswith('__') for key in item):
                    raise ValueError("wire authority is not variant literal data")
                if override and (set(item) in ({'node', 'output'}, {'variant'})
                                 or any(key in {'class_type', 'clone_input'} for key in item)):
                    raise ValueError("expansion tokens are not override literal data")
                for key, child in item.items():
                    pending.append((key, depth + 1))
                    pending.append((child, depth + 1))
            else:
                if override and _link(item):
                    raise ValueError("an override cannot introduce a graph link")
                pending.extend((child, depth + 1) for child in item)
        elif not override and ref_type and isinstance(item, ref_type):
            # Existing continuation refs retain the normal scoped resolver path.
            size += 256
        else:
            raise TypeError("variant metadata must contain bounded JSON literals")
        if size > limits.max_request_bytes:
            raise ValueError("variant request exceeds metadata byte budget")
    return size


def _fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=False, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source_node(node):
    # IS_CHANGED is execution-owned cached state, not submitted graph source.
    return {key:value for key,value in node.items() if key != 'is_changed'}


def _source_fingerprint(dynprompt, caller_id, node_ids, limits):
    snapshot = dict(caller=_source_node(dynprompt.get_node(caller_id)),
                    nodes={key: _source_node(dynprompt.get_node(key)) for key in node_ids})
    _measure(snapshot, limits)
    return _fingerprint(snapshot)


def plan_upstream_variants(graph_owner, dynprompt, caller_id, input_name, variants,
                           nodes, outputs, policy, *, ref_type=(), configuration_nodes=None):
    if type(policy) is not UpstreamVariantPolicy:
        raise UpstreamVariantsPending("no host upstream-variant policy is selected")
    policy.current(graph_owner, 'before-parse')
    limits = policy.budget.limits
    _name(input_name, 'linked input name')
    if type(variants) is not list or not 1 <= len(variants) <= limits.max_variants:
        raise ValueError("invalid variant count")
    if type(nodes) is not list or not 1 <= len(nodes) <= 128:
        raise ValueError("graph expansion must contain 1..128 continuation nodes")
    if type(outputs) is not list or len(outputs) > 128:
        raise ValueError("graph expansion has too many outputs")
    if configuration_nodes is None:
        configuration_nodes = []
    if (type(configuration_nodes) is not list
            or len(configuration_nodes) > limits.max_upstream_nodes
            or any(type(item) is not str for item in configuration_nodes)
            or len(set(configuration_nodes)) != len(configuration_nodes)):
        raise ValueError('configuration nodes must be bounded unique upstream IDs')
    _measure([input_name, variants, nodes, outputs, configuration_nodes], limits, ref_type=ref_type)
    caller = _source_node(dynprompt.get_node(caller_id))
    source = caller.get('inputs', {}).get(input_name)
    if not _link(source):
        raise ValueError("selected current input must be a direct graph link")
    producer_id, output_index = source
    snapshot = {}
    visiting = set()
    ordered = []
    source_bytes = 0

    def visit(node_id, depth):
        nonlocal source_bytes
        _node_id(node_id, limits)
        if depth > limits.max_depth or node_id == caller_id or node_id in visiting:
            raise ValueError("upstream branch is cyclic, recursive or too deep")
        if node_id in snapshot:
            return
        if len(snapshot) + len(visiting) >= limits.max_upstream_nodes:
            raise ValueError("upstream branch exceeds node budget")
        visiting.add(node_id)
        node = _source_node(dynprompt.get_node(node_id))
        source_bytes += _measure(node, limits)
        if source_bytes > limits.max_request_bytes:
            raise ValueError("upstream snapshot exceeds metadata byte budget")
        class_type = _name(node.get('class_type'), 'upstream class')
        inputs = node.get('inputs', {})
        if type(inputs) is not dict or len(inputs) > 256:
            raise ValueError("upstream inputs must be a bounded mapping")
        # Admission may not fall through to arbitrary legacy INPUT_TYPES code.
        if policy.admit_target(node_id, class_type, 'upstream') is not True:
            raise UpstreamVariantsPending(f"upstream target is not admitted: {node_id}")
        for name, value in inputs.items():
            _name(name, 'upstream input')
            if _link(value):
                visit(value[0], depth + 1)
        snapshot[node_id] = copy.deepcopy(node)
        visiting.remove(node_id)
        ordered.append(node_id)

    visit(producer_id, 0)
    for identifier in configuration_nodes:
        _node_id(identifier, limits)
        if identifier not in snapshot or identifier == producer_id:
            raise PermissionError('configuration substitution requires a non-producer upstream node')
        if (policy.validate_configuration is None or
                policy.validate_configuration(identifier, snapshot[identifier]['class_type']) is not True):
            raise PermissionError('configuration substitution has no host selection')
    caller_type = _name(caller.get('class_type'), 'current class')
    namespace = caller_type.split(' ', 1)[0]
    variant_ids = set()
    override_count = literal_bytes = 0
    for variant in variants:
        if type(variant) is not dict or set(variant) != {'id', 'overrides'}:
            raise ValueError("variants accept exactly id and overrides")
        identifier = _name(variant['id'], 'variant id', local=True)
        if identifier in variant_ids:
            raise ValueError("duplicate variant id")
        variant_ids.add(identifier)
        overrides = variant['overrides']
        if type(overrides) is not list:
            raise TypeError("variant overrides must be a list")
        pairs = set()
        for override in overrides:
            override_count += 1
            if override_count > limits.max_overrides:
                raise ValueError("too many variant overrides")
            if type(override) is not dict or set(override) != {'node_id', 'input_name', 'value'}:
                raise ValueError("override fields must be node_id, input_name and value")
            node_id = _node_id(override['node_id'], limits)
            name = _name(override['input_name'], 'override input')
            if node_id not in snapshot or (node_id, name) in pairs:
                raise ValueError("unrelated target or duplicate override")
            pairs.add((node_id, name))
            inputs = snapshot[node_id].get('inputs', {})
            if name not in inputs or _link(inputs[name]):
                raise ValueError("only existing literal widget inputs may be overridden")
            literal_bytes += _measure(override['value'], limits, override=True)
            if literal_bytes > limits.max_literal_bytes:
                raise ValueError("variant literals exceed byte budget")

    continuation_ids = set()
    direct_clones = {}
    direct_clone_sources = dict(snapshot)
    for spec in nodes:
        if type(spec) is not dict:
            raise TypeError("continuation node specs must be dictionaries")
        identifier = _name(spec.get('id'), 'continuation id', local=True)
        if identifier in continuation_ids:
            raise ValueError("duplicate continuation id")
        continuation_ids.add(identifier)
        if 'clone_input' in spec:
            if set(spec) != {'id', 'clone_input'}:
                raise ValueError("clone specs accept only id and clone_input")
            link = caller.get('inputs', {}).get(_name(spec['clone_input'], 'cloned input'))
            if not _link(link) or link[0] == caller_id:
                raise ValueError("clone input must be a direct nonrecursive current link")
            target = _source_node(dynprompt.get_node(link[0]))
            if policy.admit_target(link[0], target.get('class_type'), 'direct-clone') is not True:
                raise UpstreamVariantsPending("direct cloned producer is not admitted")
            if link[0] not in direct_clone_sources:
                source_bytes += _measure(target, limits)
                if source_bytes > limits.max_request_bytes:
                    raise ValueError("direct-clone snapshot exceeds aggregate byte budget")
                direct_clone_sources[link[0]] = copy.deepcopy(target)
            direct_clones[identifier] = (link[0],direct_clone_sources[link[0]])
        else:
            if set(spec) - {'id', 'class_type', 'inputs'}:
                raise ValueError("unknown continuation fields")
            class_type = _name(spec.get('class_type'), 'continuation class')
            if (not class_type.startswith(namespace + ' ') and class_type != 'CreateList'
                    and class_type not in policy.external_node_types):
                raise PermissionError("continuation is outside the admitted pack namespace")
            if policy.admit_target(identifier, class_type, 'continuation') is not True:
                raise UpstreamVariantsPending("continuation target is not admitted")
            if type(spec.get('inputs', {})) is not dict or len(spec.get('inputs', {})) > 256:
                raise ValueError("invalid continuation inputs")

    dependencies = {name:set() for name in continuation_ids}

    def links(value, consumer=None):
        if type(value) is dict:
            if 'variant' in value:
                if set(value) != {'variant'} or value['variant'] not in variant_ids:
                    raise ValueError("malformed or unknown variant token")
                return
            if 'node' in value or 'output' in value:
                if (set(value) != {'node', 'output'} or value['node'] not in continuation_ids
                        or type(value['output']) is not int or not 0 <= value['output'] <= 1024):
                    raise ValueError("invalid continuation output token")
                if consumer is not None:
                    dependencies[consumer].add(value['node'])
                return
            for child in value.values():
                links(child, consumer)
        elif type(value) is list:
            if _link(value):
                raise ValueError("raw graph links are not continuation authority")
            for child in value:
                links(child, consumer)

    for spec in nodes:
        if 'clone_input' not in spec:
            for name, value in spec.get('inputs', {}).items():
                _name(name, 'continuation input')
                links(value, spec['id'])
    for output in outputs:
        if type(output) is not dict or set(output) != {'node', 'output'}:
            raise ValueError("results must link to continuation outputs")
        links(output)

    visited = set()
    active = set()
    def check_order(name):
        if name in active:
            raise ValueError("continuation graph is cyclic")
        if name in visited:
            return
        active.add(name)
        for dependency in dependencies[name]:
            check_order(dependency)
        active.remove(name)
        visited.add(name)
    for name in dependencies:
        check_order(name)

    retained_nodes = len(snapshot)*len(variants) + len(nodes) + policy.scheduler_overhead_nodes
    if retained_nodes > 512:
        raise ValueError("variant expansion exceeds the existing ticket graph ceiling")
    cost = dict(variants=len(variants), cloned_nodes=len(snapshot)*len(variants),
                overrides=override_count, literal_bytes=literal_bytes,
                retained_tickets=1, retained_nodes=retained_nodes)
    policy.budget.charge(**cost)
    for variant in variants:
        for override in variant['overrides']:
            if policy.validate_override(override['node_id'], override['input_name'],
                                        override['value']) is not True:
                raise PermissionError("override is not permitted by current input authority")
    policy.current(graph_owner, 'after-plan')
    watched_nodes = {**snapshot,**{source_id:node for source_id,node in direct_clones.values()}}
    bound_source = dict(caller=copy.deepcopy(caller), nodes=watched_nodes)
    _measure(bound_source, limits)
    fingerprint = _fingerprint(bound_source)
    if _source_fingerprint(dynprompt, caller_id, watched_nodes, limits) != fingerprint:
        raise PermissionError("upstream source changed during planning")
    policy._sources.append((dynprompt, caller_id, tuple(watched_nodes), fingerprint))
    plan = VariantPlan(caller_id, fingerprint, producer_id, output_index,
                       {key: snapshot[key] for key in ordered}, tuple(copy.deepcopy(variants)),
                       tuple(copy.deepcopy(nodes)), direct_clones, tuple(copy.deepcopy(outputs)), retained_nodes,
                       uuid.uuid4().hex + '.', tuple(configuration_nodes))
    return plan


async def expand_upstream_variants(graph_owner, dynprompt, caller_id, input_name,
                                   variants, nodes, outputs, policy, *, ref_type=(), configuration_nodes=None):
    plan = plan_upstream_variants(graph_owner, dynprompt, caller_id, input_name,
                                  variants, nodes, outputs, policy, ref_type=ref_type,
                                  configuration_nodes=configuration_nodes)
    if policy.schedule is None:
        raise UpstreamVariantsPending("ordered cache/object-preserving scheduler is not selected")
    policy.current(graph_owner, 'before-schedule')
    result = await policy.schedule(plan)
    policy.current(graph_owner, 'after-schedule')
    if (type(result) is not dict or set(result) != {'result', 'expand'}
            or type(result['expand']) is not dict or len(result['expand']) > plan.retained_nodes
            or type(result['result']) not in (tuple, list) or len(result['result']) != len(plan.outputs)):
        raise ValueError("variant scheduler exceeded the reserved expansion shape")
    for link in result['result']:
        if (not _link(link) or link[0] not in result['expand'] or link[1] > 1024):
            raise ValueError("variant scheduler must return only its own continuation output links")
    expected = [[plan.continuation_prefix + output['node'],output['output']] for output in plan.outputs]
    if list(result['result']) != expected:
        raise ValueError("variant scheduler changed declared continuation output mapping or order")
    for spec in plan.continuation:
        target = result['expand'].get(plan.continuation_prefix + spec['id'])
        expected_type = (plan.direct_clones[spec['id']][1]['class_type']
                         if 'clone_input' in spec else spec['class_type'])
        if type(target) is not dict or target.get('class_type') != expected_type:
            raise ValueError("variant scheduler did not materialize the declared continuation")
    return result
