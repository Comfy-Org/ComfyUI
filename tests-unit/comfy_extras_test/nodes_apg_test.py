import uuid

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.conds  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy_extras.nodes_apg import APG  # noqa: E402

# The three settings from the cfg 1.0 report: defaults, the norm_threshold=0 case that
# doubled the prediction, and a momentum case.
APG_SETTINGS = [
    (1.0, 5.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 5.0, 0.5),
]


class FakePatcher:
    def prepare_state(self, *args, **kwargs):
        pass

    def get_free_memory(self, device):
        return 1 << 40

    def apply_hooks(self, hooks):
        return {}


class FakeModel:
    """Denoised prediction depends only on the text embedding: x0 = embedding broadcast over the latent."""

    current_patcher = FakePatcher()

    def memory_required(self, input_shape, cond_shapes=None):
        return 0

    def apply_model(self, x, t, c_crossattn=None, **kwargs):
        return x * 0 + c_crossattn.mean(dim=(1, 2)).view(-1, 1, 1, 1)


def conditioning(value):
    return [
        {
            "model_conds": {"c_crossattn": comfy.conds.CONDCrossAttn(torch.full((1, 1, 4), value))},
            "uuid": uuid.uuid4(),
        }
    ]


@pytest.fixture
def sampling_setup():
    model = FakeModel()
    patcher = comfy.model_patcher.ModelPatcher(torch.nn.Identity(), torch.device("cpu"), torch.device("cpu"))
    latent = torch.zeros(1, 4, 8, 8)
    timestep = torch.tensor([1.0])
    return model, patcher, latent, timestep


def sample(model, latent, timestep, positive, negative, cfg, model_options=None):
    return comfy.samplers.sampling_function(
        model, latent, timestep, negative, positive, cfg, model_options=model_options or {}
    )


@pytest.mark.parametrize(("eta", "norm_threshold", "momentum"), APG_SETTINGS)
def test_cfg_1_output_matches_plain_sampling(sampling_setup, eta, norm_threshold, momentum):
    """At cfg 1.0 the negative prompt is never evaluated, so APG must leave the prediction unchanged."""
    model, patcher, latent, timestep = sampling_setup
    positive, negative = conditioning(1.0), conditioning(0.25)

    plain = sample(model, latent, timestep, positive, negative, 1.0)

    apg = APG.execute(patcher, eta=eta, norm_threshold=norm_threshold, momentum=momentum).result[0]
    out = sample(model, latent, timestep, positive, negative, 1.0, model_options=apg.model_options)

    assert torch.equal(out, plain)


def test_cfg_above_1_apg_still_applies_guidance(sampling_setup):
    """The skip guard must only trigger when a guidance branch is actually missing."""
    model, patcher, latent, timestep = sampling_setup
    positive, negative = conditioning(1.0), conditioning(0.25)

    plain = sample(model, latent, timestep, positive, negative, 3.5)

    apg = APG.execute(patcher, eta=1.0, norm_threshold=5.0, momentum=0.0).result[0]
    out = sample(model, latent, timestep, positive, negative, 3.5, model_options=apg.model_options)

    assert not torch.equal(out, plain)
