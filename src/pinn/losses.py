"""
Loss functions for PINN structural identification.

- data_loss: MSE between predicted and measured accelerations
- physics_loss: equation of motion residual
- ic_loss: initial condition enforcement
- reg_loss: physical constraint regularization
"""

import torch


def data_loss(u_pred, t, a_measured, floor_indices, struct_params, ground_acc_fn):
    """Data fidelity loss: MSE on absolute accelerations at instrumented floors.

    Parameters
    ----------
    u_pred : Tensor, shape (N_d, n_floors)
        Predicted relative displacements at data time points.
    t : Tensor, shape (N_d, 1)
        Time points (requires_grad=True).
    a_measured : Tensor, shape (N_d, n_sensors)
        Measured absolute accelerations at instrumented floors.
    floor_indices : list of int
        Indices of instrumented floors.
    struct_params : StructuralParameters
        Current structural parameters.
    ground_acc_fn : callable
        Returns ground acceleration at time t.

    Returns
    -------
    loss : Tensor (scalar)
    """
    # Compute predicted accelerations via autograd
    u_dot = torch.autograd.grad(
        u_pred, t,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True, retain_graph=True,
    )[0]
    u_ddot = torch.autograd.grad(
        u_dot, t,
        grad_outputs=torch.ones_like(u_dot),
        create_graph=True, retain_graph=True,
    )[0]

    # Absolute acceleration = relative + ground
    ddot_ug = ground_acc_fn(t)
    a_abs_pred = u_ddot + ddot_ug

    # Select instrumented floors
    a_pred_selected = a_abs_pred[:, floor_indices]

    return torch.mean((a_pred_selected - a_measured) ** 2)


def physics_loss(t, u_pred, struct_params, ground_acc_fn):
    """Physics residual loss: equation of motion residual.

    Computes: M*u_ddot + C*u_dot + K*u + M*iota*ddot_ug = 0

    Parameters
    ----------
    t : Tensor, shape (N_c, 1)
        Collocation time points (requires_grad=True).
    u_pred : Tensor, shape (N_c, n_floors)
        Predicted displacements at collocation points.
    struct_params : StructuralParameters
        Current structural parameters.
    ground_acc_fn : callable
        Returns ground acceleration at time t.

    Returns
    -------
    loss : Tensor (scalar)
    """
    # Velocities and accelerations via autograd
    u_dot = torch.autograd.grad(
        u_pred, t,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True, retain_graph=True,
    )[0]
    u_ddot = torch.autograd.grad(
        u_dot, t,
        grad_outputs=torch.ones_like(u_dot),
        create_graph=True,
    )[0]

    M = struct_params.build_M()
    K = struct_params.build_K()

    # Estimate omega1, omega3 for Rayleigh damping
    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(torch.linalg.solve(M, K))
        omega1 = eigvals[0].sqrt()
        omega3 = eigvals[min(2, len(eigvals) - 1)].sqrt()

    C = struct_params.build_C(omega1, omega3)

    # Ground excitation
    iota = torch.ones(struct_params.n_floors, 1, device=t.device, dtype=t.dtype)
    ddot_ug = ground_acc_fn(t)  # (N_c, 1)

    # EOM residual: M*u_ddot + C*u_dot + K*u + M*iota*ddot_ug = 0
    residual = (
        u_ddot @ M.T
        + u_dot @ C.T
        + u_pred @ K.T
        + ddot_ug * (M @ iota).T
    )

    return (residual**2).mean()


def ic_loss(model, struct_params):
    """Initial condition loss: u(0) = 0, u_dot(0) = 0.

    Parameters
    ----------
    model : SirenNet
        The displacement network.
    struct_params : StructuralParameters
        Current structural parameters (unused but kept for API consistency).

    Returns
    -------
    loss : Tensor (scalar)
    """
    t0 = torch.zeros(1, 1, requires_grad=True, device=next(model.parameters()).device)
    u0 = model(t0)

    u_dot_0 = torch.autograd.grad(
        u0, t0,
        grad_outputs=torch.ones_like(u0),
        create_graph=True,
    )[0]

    return torch.mean(u0**2) + torch.mean(u_dot_0**2)


def reg_loss(struct_params, m_prior, k_prior):
    """Regularization loss penalizing deviation from prior parameter estimates.

    Parameters
    ----------
    struct_params : StructuralParameters
        Current structural parameters.
    m_prior : Tensor
        Prior mass estimates.
    k_prior : Tensor
        Prior stiffness estimates.

    Returns
    -------
    loss : Tensor (scalar)
    """
    m_err = ((struct_params.m - m_prior) / m_prior) ** 2
    k_err = ((struct_params.k - k_prior) / k_prior) ** 2
    return m_err.mean() + k_err.mean()
