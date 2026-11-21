import asyncio
import json
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[2] / "script_examples" / "tensorfold"


def test_image_api_workflow_passes_real_prompt_validation(comfy_environment, monkeypatch):
    import execution
    import folder_paths
    import nodes
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_tensorfold import TensorFoldMOSSImage

    prompt = json.loads((EXAMPLES / "moss_image.api.json").read_text())
    selections = {"text_encoders": [prompt["1"]["inputs"]["clip_name"]],
                  "diffusion_models": [prompt["4"]["inputs"]["checkpoint_name"]],
                  "vae": [prompt["5"]["inputs"]["vae_name"]]}
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda kind: selections.get(kind, []))
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TensorFoldMOSSImage", TensorFoldMOSSImage)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PreviewAny", PreviewAny)
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("tensorfold-workflow-test", prompt, None))
    assert valid, (error, node_errors)
    assert set(outputs) == {"7", "8"}
    assert not node_errors


def test_editor_workflows_have_consistent_typed_links(comfy_environment):
    import nodes
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_tensorfold import TensorFoldMOSSImage

    classes = nodes.NODE_CLASS_MAPPINGS | {"TensorFoldMOSSImage": TensorFoldMOSSImage, "PreviewAny": PreviewAny}
    for name in ("moss_image.workflow.json",):
        workflow = json.loads((EXAMPLES / name).read_text())
        by_id = {node["id"]: node for node in workflow["nodes"]}
        assert len(by_id) == len(workflow["nodes"])
        for link_id, source, output_index, target, input_index, kind in workflow["links"]:
            output = by_id[source]["outputs"][output_index]
            input_ = by_id[target]["inputs"][input_index]
            assert link_id in output["links"] and input_["link"] == link_id
            assert output["type"] == kind
            assert input_["type"] in (kind, "*")
            assert classes[by_id[source]["type"]].RETURN_TYPES[output_index] == kind
            required = classes[by_id[target]["type"]].INPUT_TYPES()["required"]
            assert required[input_["name"]][0] in (kind, "*")


def test_image_editor_values_match_api_prompt():
    prompt = json.loads((EXAMPLES / "moss_image.api.json").read_text())
    workflow = json.loads((EXAMPLES / "moss_image.workflow.json").read_text())
    by_id = {str(node["id"]): node for node in workflow["nodes"]}
    assert by_id["1"]["widgets_values"][:2] == [prompt["1"]["inputs"]["clip_name"], "moss_image"]
    assert by_id["2"]["widgets_values"] == [prompt["2"]["inputs"]["text"]]
    assert by_id["3"]["widgets_values"] == [prompt["3"]["inputs"]["text"]]
    fields = ["checkpoint_name", "width", "height", "steps", "seed"]
    assert by_id["4"]["widgets_values"][:5] == [prompt["4"]["inputs"][name] for name in fields]
    assert by_id["5"]["widgets_values"] == [prompt["5"]["inputs"]["vae_name"]]
