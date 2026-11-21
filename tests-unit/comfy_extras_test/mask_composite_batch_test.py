import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_mask import MaskComposite


@pytest.mark.parametrize("operation", ["multiply", "add", "subtract", "and", "or", "xor"])
@pytest.mark.parametrize("destination_batch,source_batch", [(1, 3), (2, 3), (3, 2), (3, 1), (3, 3)])
@pytest.mark.parametrize("x,y", [(1, 1), (4, 3)])
def test_mask_batch_matches_individual_composites(destination_batch, source_batch, operation, x, y):
    destination = torch.linspace(0, 1, destination_batch * 4 * 5).reshape(destination_batch, 4, 5)
    source = torch.linspace(1, 0, source_batch * 2 * 3).reshape(source_batch, 2, 3)
    original_destination, original_source = destination.clone(), source.clone()
    batch_size = max(destination_batch, source_batch)
    expected = torch.cat([
        MaskComposite.execute(
            destination[i % destination_batch:i % destination_batch + 1],
            source[i % source_batch:i % source_batch + 1],
            x, y, operation,
        )[0]
        for i in range(batch_size)
    ])

    output = MaskComposite.execute(destination, source, x, y, operation)[0]

    torch.testing.assert_close(output, expected)
    assert output.dtype == destination.dtype
    assert output.device == destination.device
    assert torch.equal(destination, original_destination)
    assert torch.equal(source, original_source)


def test_mask_composite_keeps_2d_inputs():
    destination = torch.ones((4, 5))
    source = torch.ones((2, 3))
    expected = torch.ones((1, 4, 5))
    expected[:, 1:3, 1:4] = 0

    output = MaskComposite.execute(destination, source, 1, 1, "subtract")[0]

    torch.testing.assert_close(output, expected)
    assert torch.all(destination == 1)
    assert torch.all(source == 1)
