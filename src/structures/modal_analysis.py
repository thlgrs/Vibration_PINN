"""
Modal analysis utilities: eigenvalue solver, natural frequencies, mode shapes.
"""

import numpy as np
from scipy.linalg import eigh


def modal_analysis(M, K):
    """Solve the generalized eigenvalue problem K*phi = omega^2*M*phi.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Mass matrix.
    K : ndarray, shape (n, n)
        Stiffness matrix.

    Returns
    -------
    omegas : ndarray, shape (n,)
        Angular frequencies in rad/s, sorted ascending.
    freqs_hz : ndarray, shape (n,)
        Natural frequencies in Hz.
    periods : ndarray, shape (n,)
        Natural periods in seconds.
    mode_shapes : ndarray, shape (n, n)
        Mass-normalized mode shapes (columns).
    """
    eigvals, eigvecs = eigh(K, M)
    omegas = np.sqrt(np.maximum(eigvals, 0.0))
    freqs_hz = omegas / (2 * np.pi)
    periods = np.where(freqs_hz > 0, 1.0 / freqs_hz, np.inf)

    return omegas, freqs_hz, periods, eigvecs
