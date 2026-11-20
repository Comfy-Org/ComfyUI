import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_mask import MaskComposite  # noqa: E402

OPERATIONS = ["multiply", "add", "subtract", "and", "or", "xor"]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("x,y", [(6, 0), (0, 5), (7, 6), (8, 0), (0, 100)])
def test_source_starting_outside_destination_leaves_it_unchanged(operation, x, y):
    destination = torch.full((1, 4, 6), 0.5)
    source = torch.ones((1, 3, 4))

    output = MaskComposite.execute(destination, source, x, y, operation).result[0]

    torch.testing.assert_close(output, destination)
