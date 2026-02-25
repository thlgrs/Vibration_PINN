"""
Causal physics loss weighting (Wang et al., 2022).

Addresses the causality problem for long time sequences by weighting
physics residuals based on accumulated early-time errors.
"""

import torch


def causal_physics_loss(residuals_ordered, epsilon=1.0):
    """Compute causally-weighted physics loss.

    Time-ordered residuals are weighted so that early time steps receive
    weight ~1 and later time steps are down-weighted until early physics
    is well satisfied.

    Parameters
    ----------
    residuals_ordered : Tensor, shape (N_c, n_floors)
        Physics residuals ordered by time (ascending).
    epsilon : float
        Causal weighting strength. Larger values enforce stronger causality.

    Returns
    -------
    loss : Tensor (scalar)
    """
    r_sq = (residuals_ordered**2).mean(dim=1)  # (N_c,)
    cumulative = torch.cumsum(r_sq, dim=0)
    weights = torch.exp(-epsilon * torch.roll(cumulative, 1))
    weights[0] = 1.0
    return (weights * r_sq).mean()
