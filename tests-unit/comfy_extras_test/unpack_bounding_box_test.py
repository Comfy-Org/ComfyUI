import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_images import BoundingBox, UnpackBoundingBox  # noqa: E402


def test_unpacks_fields_in_order():
    box = {"x": 10, "y": 20, "width": 300, "height": 400}

    assert UnpackBoundingBox.execute(box).result == (10, 20, 300, 400)


def test_round_trips_primitive_bounding_box():
    box = BoundingBox.execute(1, 2, 3, 4).result[0]

    assert UnpackBoundingBox.execute(box).result == (1, 2, 3, 4)


def test_schema_has_four_int_outputs():
    schema = UnpackBoundingBox.define_schema()

    assert [o.display_name for o in schema.outputs] == ["x", "y", "width", "height"]
    assert {o.io_type for o in schema.outputs} == {"INT"}
