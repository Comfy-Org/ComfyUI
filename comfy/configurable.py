from torch import nn


class ConfigurableModule(nn.Module):
    def with_config(self, encoded_config):
        """Return a new module configured from a uint8 JSON tensor, without modifying this module."""
        raise NotImplementedError
