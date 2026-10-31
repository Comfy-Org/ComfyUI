from comfy_execution.filename_prefix import format_filename_prefix


def test_format_filename_prefix_uses_resolved_node_inputs():
    prompt = {
        "1": {
            "class_type": "KSampler",
            "inputs": {"steps": ["2", 0], "cfg": ["3", 0]},
        }
    }
    outputs = {("2", 0): 12, ("3", 0): 2.5}

    assert (
        format_filename_prefix(
            "Steps%KSampler.steps%-Cfg%KSampler.cfg%",
            prompt,
            lambda node_id, output_index: outputs.get((node_id, output_index)),
        )
        == "Steps12-Cfg2.5"
    )


def test_format_filename_prefix_preserves_unknown_or_unavailable_values():
    prompt = {
        "1": {
            "class_type": "KSampler",
            "inputs": {"steps": ["2", 0]},
        }
    }

    assert (
        format_filename_prefix("Steps%KSampler.steps%-%Unknown.value%", prompt)
        == "Steps%KSampler.steps%-%Unknown.value%"
    )
