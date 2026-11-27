import torch
import torch.nn as nn

from comfy.ldm.modules.diffusionmodules.util import checkpoint


def test_checkpoint_backward_with_frozen_parameter():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.trainable = nn.Parameter(torch.tensor(3.0))

        def forward(self, x):
            return x * self.frozen + x * self.trainable

    block = Block()
    x = torch.tensor(4.0, requires_grad=True)

    checkpoint(block, (x,), tuple(block.parameters()), True).backward()

    torch.testing.assert_close(x.grad, torch.tensor(5.0))
    torch.testing.assert_close(block.trainable.grad, torch.tensor(4.0))
    assert block.frozen.grad is None
