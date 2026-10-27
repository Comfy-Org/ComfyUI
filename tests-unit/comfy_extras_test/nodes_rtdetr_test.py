import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_rtdetr import DrawBBoxes  # noqa: E402


def draw(label):
    bbox = {"x": 4, "y": 4, "width": 40, "height": 40, "score": 0.9}
    if label is not None:
        bbox["label"] = label
    return DrawBBoxes.execute([bbox], torch.zeros(1, 64, 64, 3)).args[0][0]


def painted(image):
    return (image > 0).any(-1).sum().item()


def test_draws_labels_outside_coco_classes():
    assert painted(draw("screwdriver")) > painted(draw(None))


def test_coco_labels_keep_their_class_colours():
    # COCO classes 0 and 1 use the first two palette colours
    assert draw("person")[4, 4].tolist() == [1.0, 0.0, 0.0]
    assert torch.allclose(draw("bicycle")[4, 4], torch.tensor([0.0, 200 / 255, 0.0]))
