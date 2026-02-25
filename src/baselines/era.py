"""
Eigensystem Realization Algorithm (ERA) — Juang & Pappa, 1985.

Classical modal identification baseline for comparison with PINN-SID.
"""

import numpy as np
from scipy.linalg import svd, eig


class ERA:
    """Eigensystem Realization Algorithm for modal identification.

    Parameters
    ----------
    dt : float
        Sampling time step in seconds.
    n_modes : int
        Number of modes to identify.
    """

    def __init__(self, dt, n_modes):
        self.dt = dt
        self.n_modes = n_modes
        self.frequencies = None
        self.damping_ratios = None
        self.mode_shapes = None

    def _build_hankel(self, impulse_response, r, s):
        """Build block Hankel matrix from impulse response (Markov parameters).

        Parameters
        ----------
        impulse_response : ndarray, shape (n_steps, n_outputs)
            Free decay or impulse response data.
        r : int
            Number of block rows.
        s : int
            Number of block columns.

        Returns
        -------
        H : ndarray
            Block Hankel matrix.
        """
        n_out = impulse_response.shape[1]
        H = np.zeros((r * n_out, s * n_out))
        for i in range(r):
            for j in range(s):
                idx = i + j
                if idx < len(impulse_response):
                    H[i * n_out:(i + 1) * n_out,
                      j * n_out:(j + 1) * n_out] = np.diag(impulse_response[idx])
        return H

    def identify(self, impulse_response):
        """Run ERA identification.

        Parameters
        ----------
        impulse_response : ndarray, shape (n_steps, n_outputs)
            Free decay or impulse response data.

        Returns
        -------
        frequencies : ndarray, shape (n_modes,)
            Identified natural frequencies in Hz.
        damping_ratios : ndarray, shape (n_modes,)
            Identified damping ratios.
        mode_shapes : ndarray, shape (n_outputs, n_modes)
            Identified mode shapes.
        """
        n_steps = len(impulse_response)
        r = min(n_steps // 3, 100)
        s = r

        H0 = self._build_hankel(impulse_response, r, s)
        H1 = self._build_hankel(impulse_response[1:], r, s)

        U, S_vals, Vt = svd(H0, full_matrices=False)

        # Truncate to 2*n_modes
        n = 2 * self.n_modes
        n = min(n, len(S_vals))
        U_n = U[:, :n]
        S_n = np.diag(S_vals[:n])
        Vt_n = Vt[:n, :]

        S_n_sqrt = np.diag(np.sqrt(S_vals[:n]))
        S_n_sqrt_inv = np.diag(1.0 / np.sqrt(S_vals[:n]))

        # Discrete state matrix
        A_d = S_n_sqrt_inv @ U_n.T @ H1 @ Vt_n.T @ S_n_sqrt_inv

        # Eigenvalue decomposition
        eigvals, eigvecs = eig(A_d)

        # Convert discrete eigenvalues to continuous
        lambd = np.log(eigvals.astype(complex)) / self.dt

        # Extract frequencies and damping
        frequencies = np.abs(lambd) / (2 * np.pi)
        damping_ratios = -np.real(lambd) / np.abs(lambd)

        # Sort by frequency, keep positive-frequency poles
        mask = np.imag(lambd) > 0
        frequencies = frequencies[mask]
        damping_ratios = damping_ratios[mask]

        # Sort ascending
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx[:self.n_modes]]
        damping_ratios = damping_ratios[sort_idx[:self.n_modes]]

        self.frequencies = np.real(frequencies)
        self.damping_ratios = np.real(damping_ratios)

        return self.frequencies, self.damping_ratios
