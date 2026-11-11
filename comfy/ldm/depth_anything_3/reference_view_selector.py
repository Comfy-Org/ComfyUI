"""Reference-view selection for the multi-view path of Depth Anything 3."""

from __future__ import annotations

from typing import Literal

import torch


RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


# Per the upstream constants module: ``THRESH_FOR_REF_SELECTION = 3``.
# Reference selection only runs when there are at least this many views.
THRESH_FOR_REF_SELECTION: int = 3


def select_reference_view(x: torch.Tensor, strategy: RefViewStrategy = "saddle_balanced") -> torch.Tensor:
    """Pick a reference view index per batch element."""
    B, S, _, _ = x.shape
    if S <= 1:
        return torch.zeros(B, dtype=torch.long, device=x.device)
    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    if strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)

    # Feature-based strategies: normalised cls/cam token per view.
    img_class_feat = x[:, :, 0] / x[:, :, 0].norm(dim=-1, keepdim=True)  # (B,S,C)

    if strategy == "saddle_balanced":
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))  # (B,S,S)
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_score = sim_no_diag.sum(dim=-1) / (S - 1)               # (B,S)
        feat_norm = x[:, :, 0].norm(dim=-1)                          # (B,S)
        feat_var = img_class_feat.var(dim=-1)                        # (B,S)

        def _normalize(metric):
            mn = metric.min(dim=1, keepdim=True).values
            mx = metric.max(dim=1, keepdim=True).values
            return (metric - mn) / (mx - mn + 1e-8)

        sim_n, norm_n, var_n = _normalize(sim_score), _normalize(feat_norm), _normalize(feat_var)
        balance = (sim_n - 0.5).abs() + (norm_n - 0.5).abs() + (var_n - 0.5).abs()
        return balance.argmin(dim=1)

    if strategy == "saddle_sim_range":
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_max = sim_no_diag.max(dim=-1).values
        sim_min = sim_no_diag.min(dim=-1).values
        return (sim_max - sim_min).argmax(dim=1)

    raise ValueError(
        f"Unknown reference view selection strategy: {strategy!r}. "
        f"Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'"
    )


def reorder_by_reference(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Reorder x so the reference view is at position 0 in axis S."""
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x
    positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
    b_idx_exp = b_idx.unsqueeze(1)
    reorder = torch.where(
        (positions > 0) & (positions <= b_idx_exp),
        positions - 1,
        positions,
    )
    reorder[:, 0] = b_idx
    batch = torch.arange(B, device=x.device).unsqueeze(1)
    return x[batch, reorder]


def restore_original_order(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Inverse of reorder_by_reference."""
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x
    target_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
    b_idx_exp = b_idx.unsqueeze(1)
    restore = torch.where(target_positions < b_idx_exp, target_positions + 1, target_positions)
    restore = torch.scatter(restore, dim=1, index=b_idx_exp, src=torch.zeros_like(b_idx_exp))
    batch = torch.arange(B, device=x.device).unsqueeze(1)
    return x[batch, restore]
