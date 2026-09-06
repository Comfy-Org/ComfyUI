from unittest.mock import patch

import nodes
from comfy_api.latest import io
from comfy_execution.graph import DynamicPrompt, TopologicalSort


class SourceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class LazyAutogrowNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        route_type = io.MatchType.Template("route_type")
        route_inputs = io.Autogrow.TemplateNames(
            input=io.MatchType.Input("route", template=route_type, lazy=True, optional=True),
            names=["route_a", "route_b"],
            min=0,
        )

        return io.Schema(
            node_id="LazyAutogrowNode",
            inputs=[
                io.String.Input("selected_route"),
                io.Autogrow.Input("routes", template=route_inputs, optional=True),
            ],
            outputs=[io.MatchType.Output(template=route_type)],
        )

    @classmethod
    def execute(cls, selected_route, routes=None):
        return io.NodeOutput(routes[selected_route])

    @classmethod
    def check_lazy_status(cls, selected_route, routes=None):
        route, input_name = routes[selected_route]
        return [input_name] if route is None else []


def test_autogrow_lazy_inputs_remain_weak_until_requested():
    prompt = {
        "source_a": {"class_type": "SourceNode", "inputs": {}},
        "source_b": {"class_type": "SourceNode", "inputs": {}},
        "switch": {
            "class_type": "LazyAutogrowNode",
            "inputs": {
                "selected_route": "route_a",
                "routes.route_a": ["source_a", 0],
                "routes.route_b": ["source_b", 0],
            },
        },
    }

    with patch.dict(
        nodes.NODE_CLASS_MAPPINGS,
        {"SourceNode": SourceNode, "LazyAutogrowNode": LazyAutogrowNode},
    ):
        sort = TopologicalSort(DynamicPrompt(prompt))
        sort.add_node("switch")

        assert set(sort.pendingNodes) == {"switch"}

        sort.make_input_strong_link("switch", "routes.route_a")

        assert set(sort.pendingNodes) == {"switch", "source_a"}
        assert "source_b" not in sort.pendingNodes
