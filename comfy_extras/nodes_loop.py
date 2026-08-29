from comfy_api.latest import io
from comfy_extras.graph_traversal import loop_projection
from comfy_execution.graph_utils import is_link
from server import PromptServer


class Loop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        list_item_type = io.MatchType.Template("list_item")
        return io.Schema(
            node_id="Loop",
            display_name="Loop",
            category="looping",
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option("simple", [
                        io.Int.Input("num_iterations", default=4, min=0),
                    ]),
                    io.DynamicCombo.Option("For", [
                        io.Int.Input("start_iteration", default=0),
                        io.Int.Input("max_iteration", default=4, max=0xffffffffffffffff),
                        io.Int.Input("step", default=1),
                    ]),
                    io.DynamicCombo.Option("List", [
                        io.MatchType.Input("list", list_item_type),
                    ]),
                ]),
                io.Int.Input("iteration_outer", optional=True, force_input=True),
            ],
            outputs=[
                io.Int.Output("iteration"),
                io.Boolean.Output("is_first"),
                io.Boolean.Output("is_last"),
                io.MatchType.Output(list_item_type, id="list_item"),
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.execution_list, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mode, iteration_outer=None):
        selected_mode = mode.get("mode", ["simple"])[0]
        if selected_mode == "simple":
            values = range(mode.get("num_iterations", [4])[0])
            list_items = None
        elif selected_mode == "For":
            step = mode.get("step", [1])[0]
            if step == 0:
                raise ValueError("Loop step must not be 0")
            values = range(mode.get("start_iteration", [0])[0], mode.get("max_iteration", [4])[0], step)
            list_items = None
        else:
            list_items = mode["list"]
            values = range(len(list_items))

        dynprompt = cls.hidden.dynprompt
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id

        state = execution_list.get_projection_state(unique_id)
        if state is None:
            projected_nodes, close_nodes, variable_nodes = loop_projection(dynprompt, unique_id)
            projected_nodes.difference_update(close_nodes)
            nested_openers = {
                node_id for node_id in projected_nodes
                if dynprompt.get_node(node_id)["class_type"] == "Loop"
            }
            nested_links = []
            for node_id in projected_nodes.union(close_nodes):
                for input_name, value in dynprompt.get_node(node_id)["inputs"].items():
                    if not is_link(value) or value[0] not in nested_openers:
                        continue
                    _, _, input_info = execution_list.get_input_info(node_id, input_name)
                    if input_info is None or not input_info.get("lazy", False):
                        nested_links.append((value[0], value[1], node_id))
            projected_nodes.difference_update(nested_openers)
            close_sources = {
                node_id: tuple(dynprompt.get_node(node_id)["inputs"]["value"])
                for node_id in close_nodes
            }
            internal_close_source_nodes = {
                source_id for source_id, _ in close_sources.values()
                if source_id != unique_id and source_id in projected_nodes
            }
            variable_sources = {
                node_id: tuple(dynprompt.get_node(node_id)["inputs"]["next_value"])
                for node_id in variable_nodes
            }
            internal_variable_sources = {
                node_id: source
                for node_id, source in variable_sources.items()
                if source[0] == unique_id or source[0] in projected_nodes
            }
            external_variable_sources = variable_sources.keys() - internal_variable_sources.keys()
            internal_variable_source_nodes = {
                source_id for source_id, _ in internal_variable_sources.values()
                if source_id != unique_id
            }
            state = {
                "values": values,
                "list_items": list_items,
                "index": -1,
                "projected_nodes": projected_nodes,
                "scheduled_nodes": projected_nodes.intersection(execution_list.pendingNodes).union(
                    variable_nodes, internal_variable_source_nodes, internal_close_source_nodes
                ),
                "nested_openers": nested_openers.intersection(execution_list.pendingNodes),
                "nested_links": nested_links,
                "invalidated_nodes": projected_nodes.union(nested_openers),
                "close_sources": close_sources,
                "variable_sources": variable_sources,
                "internal_variable_sources": internal_variable_sources,
            }
            execution_list.set_projection_state(unique_id, state)
            state["projected_nodes"], state["scheduled_nodes"] = execution_list.project_nodes(
                state["projected_nodes"], state["scheduled_nodes"]
            )
            execution_list.requeue_nodes(close_nodes)
            for node_id, (source_id, source_socket) in close_sources.items():
                node_state = {}
                execution_list.set_projection_state(node_id, node_state)
                node_state["values"] = []
                node_state["unblock"] = execution_list.add_external_block(node_id)
                if source_id != unique_id:
                    execution_list.cache_link(source_id, unique_id, source_socket)
            for node_id, (source_id, source_socket) in variable_sources.items():
                execution_list.set_projection_state(node_id, {})
                if node_id in external_variable_sources:
                    execution_list.add_strong_link(source_id, source_socket, unique_id)
                elif source_id != unique_id:
                    execution_list.cache_link(source_id, unique_id, source_socket)
        elif state["index"] >= 0:
            for node_id, (source_id, source_socket) in state["close_sources"].items():
                values = execution_list.get_projection_state(node_id)["values"]
                if source_id == unique_id:
                    values.append(state["opener_outputs"][source_socket])
                else:
                    source = execution_list.get_cache(source_id, unique_id)
                    if source is None:
                        raise RuntimeError(f"Close Loop {node_id} input was not produced during the iteration")
                    values.extend(source.outputs[source_socket])
            for node_id, (source_id, source_socket) in state["variable_sources"].items():
                if source_id == unique_id:
                    value = state["opener_outputs"][source_socket]
                else:
                    source = execution_list.get_cache(source_id, unique_id)
                    if source is None:
                        raise RuntimeError(f"Loop Variable {node_id} next iteration value was not produced")
                    value = source.outputs[source_socket][0]
                execution_list.get_projection_state(node_id)["value"] = value

        state["index"] += 1
        if state["index"] >= len(state["values"]):
            if not state["values"]:
                PromptServer.instance.send_progress_text("Iteration 0 / 0", unique_id)
            for node_id in state["close_sources"]:
                execution_list.get_projection_state(node_id)["unblock"]()
            execution_list.release_projected_nodes()
            for node_id in state["variable_sources"]:
                execution_list.clear_projection_state(node_id)
            execution_list.clear_projection_state(unique_id)
            return io.NodeOutput(None, False, True, None)

        execution_list.requeue_nodes(
            state["scheduled_nodes"].union(state["nested_openers"]),
            state["invalidated_nodes"],
        )
        for link in state["nested_links"]:
            execution_list.add_strong_link(*link)
        for source_id, source_socket in state["close_sources"].values():
            if source_id != unique_id:
                execution_list.cache_link(source_id, unique_id, source_socket)
        for source_id, source_socket in state["internal_variable_sources"].values():
            if source_id != unique_id:
                execution_list.cache_link(source_id, unique_id, source_socket)
        execution_list.defer_staged_node()
        value = state["values"][state["index"]]
        list_item = state["list_items"][state["index"]] if state["list_items"] is not None else None
        outputs = (value, state["index"] == 0, state["index"] == len(state["values"]) - 1, list_item)
        state["opener_outputs"] = outputs
        PromptServer.instance.send_progress_text(
            f"Iteration {state['index'] + 1} / {len(state['values'])}",
            unique_id,
        )
        return io.NodeOutput(*outputs)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class CloseLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        matchtype = io.MatchType.Template("value")
        return io.Schema(
            node_id="CloseLoop",
            display_name="Close Loop",
            category="looping",
            is_input_list=True,
            inputs=[
                io.MatchType.Input("value", matchtype, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(matchtype, id="all_outputs", is_output_list=True),
                io.MatchType.Output(matchtype, id="last_output"),
            ],
            hidden=[io.Hidden.execution_list, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, value):
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        if execution_list.get_projection_state(unique_id) is None:
            return ["value"]
        return []

    @classmethod
    def execute(cls, value):
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        state = execution_list.get_projection_state(unique_id)
        if state is None or "values" not in state:
            raise ValueError(f"Close Loop {unique_id} does not belong to a Loop")
        execution_list.clear_projection_state(unique_id)
        values = state["values"]
        return io.NodeOutput(values, values[-1] if values else None)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class LoopVariable:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iteration": ("INT", {"lazy": True, "forceInput": True}),
                "next_value": ("*", {"lazy": True, "nonNavigable": True}),
            },
            "optional": {
                "initial_value": ("*", {"lazy": True}),
            },
            "hidden": {
                "execution_list": "EXECUTION_LIST",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("current_value",)
    FUNCTION = "current"
    CATEGORY = "looping"

    def check_lazy_status(self, iteration, next_value, initial_value=None, execution_list=None, unique_id=None):
        state = execution_list.get_projection_state(unique_id)
        if state is None or "value" not in state:
            required = ["iteration"]
            if "initial_value" in execution_list.dynprompt.get_node(unique_id)["inputs"]:
                required.append("initial_value")
            return required
        return ["iteration"]

    def current(self, iteration, next_value, initial_value=None, execution_list=None, unique_id=None):
        state = execution_list.get_projection_state(unique_id)
        if state is None:
            raise ValueError(f"Loop Variable {unique_id} does not belong to a Loop")
        return (state.get("value", initial_value),)

    @classmethod
    def VALIDATE_INPUTS(cls, iteration):
        if iteration is not None:
            return "Loop Variable iteration must be connected"
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "Loop": Loop,
    "CloseLoop": CloseLoop,
    "LoopVariable": LoopVariable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Loop": "Loop",
    "CloseLoop": "Close Loop",
    "LoopVariable": "Loop Variable",
}
