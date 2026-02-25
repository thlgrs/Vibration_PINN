"""
ShearBuilding class: assembly of M, K, C matrices for an n-story shear building.
"""

import numpy as np
from scipy.linalg import eigh


class ShearBuilding:
    """Lumped-parameter shear building model.

    Parameters
    ----------
    masses : array_like
        Floor masses [m1, m2, ..., mn] in kg.
    stiffnesses : array_like
        Story stiffnesses [k1, k2, ..., kn] in N/m.
    xi : float
        Damping ratio (assumed constant, Rayleigh damping).
    """

    def __init__(self, masses, stiffnesses, xi=0.05):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.stiffnesses = np.asarray(stiffnesses, dtype=np.float64)
        self.xi = xi
        self.n_floors = len(masses)
        assert len(stiffnesses) == self.n_floors

        self.M = self._build_M()
        self.K = self._build_K()
        self.C = self._build_C()

    def _build_M(self):
        """Diagonal mass matrix."""
        return np.diag(self.masses)

    def _build_K(self):
        """Tridiagonal stiffness matrix for a shear building."""
        n = self.n_floors
        k = self.stiffnesses
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] += k[i]
            if i + 1 < n:
                K[i, i] += k[i + 1]
                K[i, i + 1] -= k[i + 1]
                K[i + 1, i] -= k[i + 1]
        return K

    def _build_C(self):
        """Rayleigh damping matrix C = alpha*M + beta*K.

        Uses modes 1 and 3 (or last mode if n < 3) as target frequencies.
        """
        omegas, _ = self.modal_frequencies()
        omega1 = omegas[0]
        omega3 = omegas[min(2, self.n_floors - 1)]

        alpha = 2 * self.xi * omega1 * omega3 / (omega1 + omega3)
        beta = 2 * self.xi / (omega1 + omega3)
        return alpha * self.M + beta * self.K

    def modal_frequencies(self):
        """Solve the generalized eigenvalue problem K*phi = omega^2*M*phi.

        Returns
        -------
        omegas : ndarray
            Angular frequencies in rad/s, sorted ascending.
        phi : ndarray
            Mass-normalized mode shapes (columns).
        """
        eigvals, eigvecs = eigh(self.K, self.M)
        omegas = np.sqrt(np.maximum(eigvals, 0.0))
        return omegas, eigvecs

    def natural_frequencies_hz(self):
        """Return natural frequencies in Hz."""
        omegas, _ = self.modal_frequencies()
        return omegas / (2 * np.pi)

    def natural_periods(self):
        """Return natural periods in seconds."""
        freqs = self.natural_frequencies_hz()
        return 1.0 / freqs
