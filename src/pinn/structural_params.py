"""
Trainable structural parameters module.

Parameters are stored in unconstrained space (log/logit) to enforce
positivity and boundedness without explicit projection.
"""

import torch
import torch.nn as nn


class StructuralParameters(nn.Module):
    """Trainable structural parameters: masses, stiffnesses, damping ratio.

    Parameters are reparameterized:
    - masses and stiffnesses in log-space (always positive)
    - damping ratio in logit-space (bounded in (0, 0.3))

    Parameters
    ----------
    n_floors : int
        Number of floors/stories.
    m_prior : array_like
        Initial mass estimates (kg).
    k_prior : array_like
        Initial stiffness estimates (N/m).
    """

    def __init__(self, n_floors, m_prior, k_prior):
        super().__init__()
        self.n_floors = n_floors
        m0 = torch.as_tensor(m_prior, dtype=torch.float32).clone().detach()
        k0 = torch.as_tensor(k_prior, dtype=torch.float32).clone().detach()
        self.log_m = nn.Parameter(torch.log(m0))
        self.log_k = nn.Parameter(torch.log(k0))
        self.logit_xi = nn.Parameter(torch.tensor(0.0))  # maps to (0, 0.3)

    @property
    def m(self):
        """Floor masses (always positive)."""
        return torch.exp(self.log_m)

    @property
    def k(self):
        """Story stiffnesses (always positive)."""
        return torch.exp(self.log_k)

    @property
    def xi(self):
        """Damping ratio, bounded in (0, 0.3)."""
        return 0.3 * torch.sigmoid(self.logit_xi)

    def build_K(self):
        """Build tridiagonal stiffness matrix."""
        k = self.k
        n = self.n_floors
        K = torch.zeros(n, n, dtype=k.dtype, device=k.device)
        for i in range(n):
            K[i, i] += k[i]
            if i > 0:
                K[i, i] += k[i - 1]
                K[i, i - 1] -= k[i - 1]
                K[i - 1, i] -= k[i - 1]
        return K

    def build_M(self):
        """Build diagonal mass matrix."""
        return torch.diag(self.m)

    def build_C(self, omega1, omega3):
        """Build Rayleigh damping matrix C = alpha*M + beta*K.

        Parameters
        ----------
        omega1 : float or Tensor
            First target angular frequency.
        omega3 : float or Tensor
            Third target angular frequency.
        """
        alpha = 2 * self.xi * omega1 * omega3 / (omega1 + omega3)
        beta = 2 * self.xi / (omega1 + omega3)
        return alpha * self.build_M() + beta * self.build_K()
