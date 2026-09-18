import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "node_input_compat.py"
ALLOWLIST = SCRIPT.parent / "node_optional_input_allowlist.json"

spec = importlib.util.spec_from_file_location("node_input_compat", SCRIPT)
node_input_compat = importlib.util.module_from_spec(spec)
sys.modules["node_input_compat"] = node_input_compat
spec.loader.exec_module(node_input_compat)


def node(inputs):
    return {"module": "comfy_extras.nodes_test", "inputs": inputs}


def widget(category, **extra):
    return {"category": category, "type": "INT", "omittable": True, "default": 0, **extra}


def test_new_required_input_is_breaking():
    base = {"Sampler": node({"steps": widget("required")})}
    head = {"Sampler": node({"steps": widget("required"), "shift": widget("required")})}
    assert node_input_compat.diff_snapshots(base, head) == [
        "Sampler.shift: new required input on an existing node"
    ]


def test_new_optional_input_is_fine():
    base = {"Sampler": node({"steps": widget("required")})}
    head = {"Sampler": node({"steps": widget("required"), "shift": widget("optional")})}
    assert node_input_compat.diff_snapshots(base, head) == []


def test_new_node_is_fine():
    head = {"Sampler": node({"steps": widget("required")})}
    assert node_input_compat.diff_snapshots({}, head) == []


def test_optional_becoming_required_is_breaking():
    base = {"Sampler": node({"shift": widget("optional")})}
    head = {"Sampler": node({"shift": widget("required")})}
    assert node_input_compat.diff_snapshots(base, head) == [
        "Sampler.shift: optional input became required"
    ]


def test_required_becoming_optional_is_fine():
    base = {"Sampler": node({"shift": widget("required")})}
    head = {"Sampler": node({"shift": widget("optional")})}
    assert node_input_compat.diff_snapshots(base, head) == []


def test_dropping_an_accepted_type_is_breaking():
    base = {"Save": node({"mesh": widget("required", type="MESH,FILE_3D_GLB")})}
    head = {"Save": node({"mesh": widget("required", type="MESH")})}
    assert node_input_compat.diff_snapshots(base, head) == [
        "Save.mesh: no longer accepts FILE_3D_GLB"
    ]


def test_widening_accepted_types_is_fine():
    base = {"Latent": node({"frame_rate": widget("required", type="INT")})}
    head = {"Latent": node({"frame_rate": widget("required", type="FLOAT,INT")})}
    assert node_input_compat.diff_snapshots(base, head) == []


def test_combo_becoming_dynamic_combo_is_fine():
    base = {"SaveVideo": node({"codec": widget("required", type="COMBO", combo_options=["h264"])})}
    head = {
        "SaveVideo": node(
            {"codec": widget("required", type="COMFY_DYNAMICCOMBO_V3", combo_options=["h264"])}
        )
    }
    assert node_input_compat.diff_snapshots(base, head) == []


def test_removing_a_combo_option_is_breaking():
    base = {"Save": node({"codec": widget("required", type="COMBO", combo_options=["h264", "av1"])})}
    head = {"Save": node({"codec": widget("required", type="COMBO", combo_options=["h264"])})}
    assert node_input_compat.diff_snapshots(base, head) == [
        "Save.codec: combo options removed: av1"
    ]


def test_nullable_input_without_a_python_default_is_unsafe():
    """No default in the schema means nullable, so the function has to carry one."""
    snapshot = {"Wan": node({"images": {"category": "optional", "type": "IMAGE", "omittable": False}})}
    assert node_input_compat.unsafe_optionals(snapshot) == [
        "Wan.images: nullable input with no default in the comfy_extras.nodes_test entry function"
    ]


def test_nullable_input_with_a_python_default_is_fine():
    snapshot = {"Wan": node({"images": {"category": "optional", "type": "IMAGE", "omittable": True}})}
    assert node_input_compat.unsafe_optionals(snapshot) == []


def test_schema_default_without_a_python_default_is_fine():
    """Execution fills these in, so the function does not need its own default."""
    snapshot = {"Sampler": node({"seed": widget("optional", omittable=False)})}
    assert node_input_compat.unsafe_optionals(snapshot) == []


def test_required_inputs_are_not_checked_for_defaults():
    snapshot = {"Sampler": node({"seed": widget("required", omittable=False)})}
    assert node_input_compat.unsafe_optionals(snapshot) == []


def test_allowlist_is_sorted_and_unique():
    allowlist = json.loads(ALLOWLIST.read_text())
    assert allowlist == sorted(set(allowlist))
