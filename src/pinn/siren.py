"""
SIREN (Sinusoidal Representation Network) implementation.

Based on Sitzmann et al., 2020 — "Implicit Neural Representations
with Periodic Activation Functions".
"""

import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    """Single SIREN layer: linear + sin activation with specific initialization."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features
                )
            else:
                n = self.linear.weight.shape[1]
                self.linear.weight.uniform_(
                    -np.sqrt(6 / n) / self.omega_0,
                    np.sqrt(6 / n) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNet(nn.Module):
    """SIREN network mapping t -> u(t) for n_floors DOFs.

    Input:  t  (N, 1) — normalized time
    Output: u  (N, n_floors) — relative floor displacements

    Parameters
    ----------
    n_floors : int
        Number of output DOFs.
    hidden_features : int
        Width of hidden layers.
    hidden_layers : int
        Number of hidden layers.
    omega_0 : float
        Frequency scaling factor for sin activations.
    """

    def __init__(self, n_floors, hidden_features=128, hidden_layers=4, omega_0=30.0):
        super().__init__()
        self.n_floors = n_floors

        layers = [SirenLayer(1, hidden_features, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(
                SirenLayer(hidden_features, hidden_features, omega_0=omega_0)
            )
        # Final linear layer (no sin activation)
        layers.append(nn.Linear(hidden_features, n_floors))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)
