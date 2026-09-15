from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder, is_link
from server import PromptServer


def _cache_enabled(value):
    return value[0] if isinstance(value, list) else value


def _expand_loop(dynprompt, opener_id, body, close_id, values, list_items, initial_value, reuse_cache):
    graph = GraphBuilder()
    loop_metadata = {}
    close_inputs = dynprompt.get_node(close_id)["inputs"]
    output_source = close_inputs.get("output_value")
    next_source = close_inputs.get("next_iteration_value")
    terminations = [value for name, value in close_inputs.items() if name.startswith("termination") and is_link(value)]
    accumulate = bool(close_inputs.get("accumulate", False))
    previous_carry = initial_value
    previous_dependencies = []
    previous_progress = None
    result_inputs = {"close_id": close_id}

    for position, value in enumerate(values):
        item = list_items[position] if list_items is not None else None
        iteration_inputs = {
            "iteration_index": value,
            "is_first": position == 0,
            "is_last": position == len(values) - 1,
            "list_item": item,
            "current_iteration_value": previous_carry,
            "reuse_cache": reuse_cache,
            **{f"dependency{index}": dependency for index, dependency in enumerate(previous_dependencies)},
        }
        iteration = graph.node(
            "LoopIteration",
            f"iteration_{position}",
            **iteration_inputs,
        )
        iteration.set_override_display_id(opener_id)
        copies = {}
        for node_id in body:
            original = dynprompt.get_node(node_id)
            copy = graph.node(original["class_type"], f"{position}_{node_id}")
            copy.set_override_display_id(node_id)
            copies[node_id] = copy

        def copied_link(source):
            if not is_link(source):
                return source
            if source[0] == opener_id:
                return iteration.out(source[1])
            if source[0] in copies:
                return copies[source[0]].out(source[1])
            return source

        for node_id, copy in copies.items():
            original = dynprompt.get_node(node_id)
            for name, input_value in original.get("inputs", {}).items():
                copy.set_input(name, copied_link(input_value))
            if "_loop_end" in original:
                loop_metadata[copy.id] = {
                    "_loop_body": [copies[body_id].id for body_id in original["_loop_body"]],
                    "_loop_end": copies[original["_loop_end"]].id,
                }

        if is_link(output_source):
            copied_output = copied_link(output_source)
            if accumulate:
                result_inputs[f"output{position}"] = copied_output
            elif position == len(values) - 1:
                result_inputs["output0"] = copied_output
            dependencies = [copied_output]
        else:
            dependencies = []
        if is_link(next_source):
            previous_carry = copied_link(next_source)
            dependencies.append(previous_carry)
        dependencies.extend(copied_link(source) for source in terminations)
        previous_dependencies = dependencies
        progress_inputs = {
            "start_id": opener_id,
            "position": position + 1,
            "total": len(values),
            **{f"dependency{index}": dependency for index, dependency in enumerate(dependencies)},
        }
        if previous_progress is not None:
            progress_inputs["previous_progress"] = previous_progress
        progress = graph.node(
            "LoopProgress",
            f"progress_{position}",
            **progress_inputs,
        )
        previous_progress = progress.out(0)

    if previous_progress is not None:
        result_inputs["progress"] = previous_progress
    result_inputs.update({f"dependency{index}": dependency for index, dependency in enumerate(previous_dependencies)})
    graph.node("LoopResult", "result", **result_inputs)
    expanded = graph.finalize()
    for node_id, metadata in loop_metadata.items():
        expanded[node_id].update(metadata)
    return expanded


class StartLoop(io.ComfyNode):
    LOOP_BOUNDARY = "start"

    @classmethod
    def define_schema(cls):
        list_item_type = io.MatchType.Template("list_item")
        carried_type = io.MatchType.Template("carried_value")
        return io.Schema(
            node_id="StartLoop",
            display_name="Start Loop",
            category="utilities/looping",
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option("simple", [
                        io.Int.Input(
                            "num_iterations",
                            default=4,
                            min=0,
                            tooltip="Number of times to execute the loop body.",
                        ),
                    ]),
                    io.DynamicCombo.Option("For", [
                        io.Int.Input(
                            "start_iteration_index",
                            default=0,
                            tooltip="Index of the first iteration when using For loop mode.",
                        ),
                        io.Int.Input(
                            "max_iteration",
                            default=4,
                            max=0xffffffffffffffff,
                            tooltip="The exclusive stopping value for iteration_index in For mode.",
                        ),
                        io.Int.Input(
                            "step",
                            default=1,
                            min=1,
                            tooltip="The index step size between each iteration when using For loop mode.",
                        ),
                    ]),
                    io.DynamicCombo.Option("List", [
                        io.MatchType.Input(
                            "list",
                            list_item_type,
                            tooltip="List of items the loop iterates on. The loop body executes once per item.",
                        ),
                    ]),
                ], tooltip="The loop iteration mode."),
                io.Boolean.Input(
                    "cache_iterations",
                    default=False,
                    advanced=True,
                    tooltip="Reuse unchanged iteration results from previous executions. Disable to execute every iteration again.",
                ),
                io.Int.Input(
                    "parent_iteration",
                    optional=True,
                    force_input=True,
                    tooltip="Connect iteration_index from an outer Start Loop to nest this loop.",
                ),
                io.MatchType.Input(
                    "initial_iteration_value",
                    carried_type,
                    optional=True,
                    tooltip="Value exposed as current_iteration_value on the first iteration.",
                ),
            ],
            outputs=[
                io.Int.Output("iteration_index", tooltip="Index of the current loop iteration."),
                io.Boolean.Output("is_first", tooltip="True during the first iteration of the loop."),
                io.Boolean.Output("is_last", tooltip="True during the last iteration of the loop."),
                io.MatchType.Output(
                    list_item_type,
                    id="list_item",
                    tooltip="Current item from the list when using List mode. None in Simple and For modes.",
                ),
                io.MatchType.Output(
                    carried_type,
                    id="current_iteration_value",
                    tooltip="Loop-carried value for the current iteration: initial_iteration_value on the first iteration, then next_iteration_value from End Loop on each subsequent iteration.",
                ),
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.execution_list, io.Hidden.unique_id],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, mode, cache_iterations=False, parent_iteration=None, initial_iteration_value=None):
        selected_mode = mode.get("mode", ["simple"])[0]
        if selected_mode == "simple":
            values = list(range(mode.get("num_iterations", [4])[0]))
            list_items = None
        elif selected_mode == "For":
            step = mode.get("step", [1])[0]
            if step == 0:
                raise ValueError("Start Loop step must not be 0")
            values = list(range(mode.get("start_iteration_index", [0])[0], mode.get("max_iteration", [4])[0], step))
            list_items = None
        else:
            list_items = mode["list"]
            values = list(range(len(list_items)))

        dynprompt = cls.hidden.dynprompt
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        loop = dynprompt.get_node(unique_id)
        body = set(loop["_loop_body"])
        close_id = loop["_loop_end"]
        graph = _expand_loop(
            dynprompt,
            unique_id,
            body,
            close_id,
            values,
            list_items,
            initial_iteration_value[0] if initial_iteration_value else None,
            _cache_enabled(cache_iterations),
        )
        close = dynprompt.get_node(close_id)
        close_inputs = close["inputs"].copy()
        for name in tuple(close_inputs):
            if name in ("output_value", "next_iteration_value") or name.startswith("termination"):
                del close_inputs[name]
        execution_list.add_node(close_id)
        execution_list.add_external_block(close_id)
        execution_list.inhibit_nodes(body)
        dynprompt.override_node(close_id, {"class_type": close["class_type"], "inputs": close_inputs})
        PromptServer.instance.send_progress_text(f"Iteration 0 / {len(values)}", unique_id)
        return io.NodeOutput(None, False, not values, None, None, expand=graph)

    @classmethod
    def fingerprint_inputs(cls, cache_iterations=False, **kwargs):
        return float("NaN")


class LoopIteration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopIteration",
            is_input_list=True,
            inputs=[
                io.Int.Input("iteration_index"),
                io.Boolean.Input("is_first"),
                io.Boolean.Input("is_last"),
                io.AnyType.Input("list_item", optional=True),
                io.AnyType.Input("current_iteration_value", optional=True),
                io.Boolean.Input("reuse_cache"),
            ],
            outputs=[io.Int.Output(), io.Boolean.Output(), io.Boolean.Output(), io.AnyType.Output(), io.AnyType.Output()],
            is_dev_only=True,
            accept_all_inputs=True,
        )

    @classmethod
    def execute(
        cls,
        iteration_index,
        is_first,
        is_last,
        reuse_cache,
        list_item=None,
        current_iteration_value=None,
        **kwargs,
    ):
        return io.NodeOutput(
            iteration_index[0],
            is_first[0],
            is_last[0],
            list_item[0] if list_item else None,
            current_iteration_value[0] if current_iteration_value else None,
        )

    @classmethod
    def fingerprint_inputs(cls, reuse_cache, **kwargs):
        return None if _cache_enabled(reuse_cache) else float("NaN")


class LoopProgress(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopProgress",
            is_input_list=True,
            inputs=[io.String.Input("start_id"), io.Int.Input("position"), io.Int.Input("total")],
            outputs=[io.Int.Output()],
            is_output_node=True,
            is_dev_only=True,
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, start_id, position, total, **kwargs):
        PromptServer.instance.send_progress_text(f"Iteration {position[0]} / {total[0]}", start_id[0])
        return io.NodeOutput(position[0])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class LoopResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopResult",
            is_input_list=True,
            inputs=[io.String.Input("close_id")],
            outputs=[],
            is_output_node=True,
            is_dev_only=True,
            accept_all_inputs=True,
            hidden=[io.Hidden.execution_list],
        )

    @classmethod
    def execute(cls, close_id, **kwargs):
        outputs = []
        while f"output{len(outputs)}" in kwargs:
            outputs.append(kwargs[f"output{len(outputs)}"])
        cls.hidden.execution_list.release_external_block(close_id[0], outputs)
        return io.NodeOutput()

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class EndLoop(io.ComfyNode):
    LOOP_BOUNDARY = "end"

    @classmethod
    def define_schema(cls):
        output_type = io.MatchType.Template("output_value")
        carried_type = io.MatchType.Template("carried_value")
        terminations = io.Autogrow.TemplatePrefix(
            io.AnyType.Input(
                "termination",
                tooltip="Connect a preview or side-effect output that must execute on every iteration. Its value is not returned.",
            ),
            prefix="termination",
            min=0,
            max=50,
        )
        return io.Schema(
            node_id="EndLoop",
            display_name="End Loop",
            category="utilities/looping",
            is_input_list=True,
            inputs=[
                io.MatchType.Input(
                    "output_value",
                    output_type,
                    optional=True,
                    tooltip="Value returned by End Loop. It returns the final iteration or all iterations according to accumulate.",
                ),
                io.MatchType.Input(
                    "next_iteration_value",
                    carried_type,
                    optional=True,
                    tooltip="Value sent from End Loop back to Start Loop for the next iteration.",
                ),
                io.Boolean.Input(
                    "accumulate",
                    default=False,
                    tooltip="Return output_value from every iteration when enabled; otherwise return only the final iteration.",
                ),
                io.Autogrow.Input(
                    "terminations",
                    template=terminations,
                    optional=True,
                    tooltip="Connect outputs that must execute on every iteration. Their values are not returned.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    output_type,
                    id="outputs",
                    is_output_list=True,
                    tooltip="The final iteration's output_value, or values accumulated across iterations when accumulate is enabled.",
                ),
            ],
            hidden=[io.Hidden.execution_list, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, accumulate, **kwargs):
        outputs = cls.hidden.execution_list.get_external_block_result(cls.hidden.unique_id)
        return io.NodeOutput([value for output in outputs for value in output])


NODE_CLASS_MAPPINGS = {
    "StartLoop": StartLoop,
    "EndLoop": EndLoop,
    "LoopIteration": LoopIteration,  # Dev-only; instantiated by loop expansion.
    "LoopProgress": LoopProgress,  # Dev-only; instantiated by loop expansion.
    "LoopResult": LoopResult,  # Dev-only; instantiated by loop expansion.
}

NODE_DISPLAY_NAME_MAPPINGS = {"StartLoop": "Start Loop", "EndLoop": "End Loop"}
