import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("prediction", ["Albedo", "Depth", "Normal"])
def test_marigold_blueprint_sampling_values(prediction):
    """Keep velocity sampling in both serialized widget formats of each blueprint."""
    path = Path(__file__).resolve().parents[1] / "blueprints" / f"Image {prediction} Estimation (Marigold V2).json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    sampling_nodes = [
        node
        for subgraph in workflow["definitions"]["subgraphs"]
        for node in subgraph["nodes"]
        if node["type"] == "ModelSamplingAuraFlow"
    ]

    assert len(sampling_nodes) == 1
    node = sampling_nodes[0]
    assert node["widgets_values"][0] == 1.73
    assert node["widgets_values"][1] == "img_to_img_velocity"
    assert node["widgets_values_named"]["shift"] == 1.73
    assert node["widgets_values_named"]["sampling"] == "img_to_img_velocity"
