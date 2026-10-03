"""Regression test for https://github.com/Comfy-Org/ComfyUI/issues/16573

The corrector coefficients in UniPC's multistep bh-update are a recurrence of
near-cancelling terms for small hh (high-shift flow schedules). Computed in
float32 they drift from their exact value; on MPS torch.expm1's extra
imprecision (pytorch/pytorch#198708) blows the drift up enough to corrupt the
sampler output. Computing the recurrence in float64 fixes both.
"""

import math

import pytest
import torch

from comfy.extra_samplers.uni_pc import UniPC


class _FakeNoiseSchedule:
    def __init__(self, lambdas, log_alphas, sigmas):
        self.lambdas = lambdas
        self.log_alphas = log_alphas
        self.sigmas = sigmas

    def marginal_lambda(self, t):
        return self.lambdas[float(t[0])]

    def marginal_log_mean_coeff(self, t):
        return self.log_alphas[float(t[0])]

    def marginal_alpha(self, t):
        return torch.exp(self.log_alphas[float(t[0])])

    def marginal_std(self, t):
        return self.sigmas[float(t[0])]


def test_order3_bh_coefficient_matches_float64_reference(monkeypatch):
    # hh from the issue: one early step of a shift-8 Wan 2.2 schedule, where
    # the exact (float64) b3 coefficient is 0.2496 but float32 gives 0.7585.
    hh = -math.log(0.98705 / 0.97973)
    lambda_prev0 = 0.0
    lambda_t = lambda_prev0 - hh

    ns = _FakeNoiseSchedule(
        lambdas={
            0.0: torch.tensor([lambda_t]),
            1.0: torch.tensor([lambda_prev0]),
            2.0: torch.tensor([0.4]),
            3.0: torch.tensor([0.9]),
        },
        log_alphas={t: torch.tensor([-0.001 * (t + 1)]) for t in (0.0, 1.0, 2.0, 3.0)},
        sigmas={
            0.0: torch.tensor([0.97973]),
            1.0: torch.tensor([0.98705]),
            2.0: torch.tensor([0.99]),
            3.0: torch.tensor([0.995]),
        },
    )
    uni_pc = UniPC(lambda x, t: torch.ones_like(x) * 0.3, ns, predict_x0=True, variant='bh1')

    captured_b = {}
    orig_solve = torch.linalg.solve

    def spy_solve(R, b):
        captured_b['b'] = b
        return orig_solve(R, b)

    monkeypatch.setattr(torch.linalg, 'solve', spy_solve)

    x = torch.full((1, 1, 1, 1), 0.5)
    model_prev_list = [torch.full((1, 1, 1, 1), 0.1), torch.full((1, 1, 1, 1), 0.9), torch.full((1, 1, 1, 1), -0.4)]
    t_prev_list = [torch.tensor([3.0]), torch.tensor([2.0]), torch.tensor([1.0])]
    t = torch.tensor([0.0])

    uni_pc.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order=3, use_corrector=True)

    b3 = captured_b['b'][-1].item()
    assert b3 == pytest.approx(0.2496, abs=1e-3)
