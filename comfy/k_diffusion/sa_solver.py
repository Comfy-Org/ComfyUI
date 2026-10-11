# SA-Solver: Stochastic Adams Solver (NeurIPS 2023, arXiv:2309.05019)
# Conference: https://proceedings.neurips.cc/paper_files/paper/2023/file/f4a6806490d31216a3ba667eb240c897-Paper-Conference.pdf
# Codebase ref: https://github.com/scxue/SA-Solver

import math
from typing import Union, Callable
import torch


def compute_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float) -> torch.Tensor:
    """Compute (1 + tau^2) * integral of exp((1 + tau^2) * x) * x^p dx from s to t with exp((1 + tau^2) * t) factored out, using integration by parts.

    Integral of exp((1 + tau^2) * x) * x^p dx
        = product_terms[p] - (p / (1 + tau^2)) * integral of exp((1 + tau^2) * x) * x^(p-1) dx,
    with base case p=0 where integral equals product_terms[0].

    where
        product_terms[p] = x^p * exp((1 + tau^2) * x) / (1 + tau^2).

    Construct a recursive coefficient matrix following the above recursive relation to compute all integral terms up to p = (solver_order - 1).
    Return coefficients used by the SA-Solver in data prediction mode.

    Args:
        s: Start time s.
        t: End time t.
        solver_order: Current order of the solver.
        tau_t: Stochastic strength parameter in the SDE.

    Returns:
        Exponential coefficients used in data prediction, with exp((1 + tau^2) * t) factored out, ordered from p=0 to p=solver_order−1, shape (solver_order,).
    """
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)

    # product_terms after factoring out exp((1 + tau^2) * t)
    # Includes (1 + tau^2) factor from outside the integral
    product_terms_factored = (t ** p - s ** p * (-tau_mul * h).exp())

    # Lower triangular recursive coefficient matrix
    # Accumulates recursive coefficients based on p / (1 + tau^2)
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()

    return recursive_coeff_mat @ product_terms_factored


def compute_simple_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, is_corrector_step: bool = False) -> torch.Tensor:
    """Compute simple order-2 b coefficients from SA-Solver paper (Appendix D. Implementation Details)."""
    tau_mul = 1 + tau_t ** 2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        # Simplified 1-step (order-2) corrector
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        # Simplified 2-step predictor
        b_2 = alpha_t * (0.5 * tau_mul * h ** 2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])


def compute_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, simple_order_2: bool = False, is_corrector_step: bool = False) -> torch.Tensor:
    """Compute b_i coefficients for the SA-Solver (see eqs. 15 and 18).

    The solver order corresponds to the number of input lambdas (half-logSNR points).

    Args:
        sigma_next: Sigma at end time t.
        curr_lambdas: Lambda time points used to construct the Lagrange basis, shape (N,).
        lambda_s: Lambda at start time s.
        lambda_t: Lambda at end time t.
        tau_t: Stochastic strength parameter in the SDE.
        simple_order_2: Whether to enable the simple order-2 scheme.
        is_corrector_step: Flag for corrector step in simple order-2 mode.

    Returns:
        b_i coefficients for the SA-Solver, shape (N,), where N is the solver order.
    """
    num_timesteps = curr_lambdas.shape[0]

    if simple_order_2 and num_timesteps == 2:
        return compute_simple_stochastic_adams_b_coeffs(sigma_next, curr_lambdas, lambda_s, lambda_t, tau_t, is_corrector_step)

    # Compute coefficients by solving a linear system from Lagrange basis interpolation
    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_T = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_T, exp_integral_coeffs)

    # (sigma_t * exp(-tau^2 * lambda_t)) * exp((1 + tau^2) * lambda_t)
    # = sigma_t * exp(lambda_t) = alpha_t
    # exp((1 + tau^2) * lambda_t) is extracted from the integral
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


def get_tau_interval_func(start_sigma: float, end_sigma: float, eta: float = 1.0) -> Callable[[Union[torch.Tensor, float]], float]:
    """Return a function that controls the stochasticity of SA-Solver.

    When eta = 0, SA-Solver runs as ODE. The official approach uses
    time t to determine the SDE interval, while here we use sigma instead.

    See:
        https://github.com/scxue/SA-Solver/blob/main/README.md
    """

    def tau_func(sigma: Union[torch.Tensor, float]) -> float:
        if eta <= 0:
            return 0.0  # ODE

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return eta if start_sigma >= sigma > end_sigma else 0.0

    return tau_func
