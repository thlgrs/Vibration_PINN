"""
Gaussian noise addition and SNR computation.
"""

import numpy as np


def add_noise(signal, noise_level, seed=None):
    """Add Gaussian noise to a signal.

    Parameters
    ----------
    signal : ndarray
        Clean signal, shape (n_steps, n_channels).
    noise_level : float
        Noise standard deviation as a fraction of peak signal amplitude.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_signal : ndarray
        Signal with added noise.
    """
    rng = np.random.default_rng(seed)
    peak = np.max(np.abs(signal))
    noise = rng.standard_normal(signal.shape) * noise_level * peak
    return signal + noise


def compute_snr(clean, noisy):
    """Compute Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    clean : ndarray
        Clean signal.
    noisy : ndarray
        Noisy signal.

    Returns
    -------
    snr_db : float
        SNR in decibels.
    """
    noise = noisy - clean
    power_signal = np.mean(clean**2)
    power_noise = np.mean(noise**2)
    if power_noise == 0:
        return np.inf
    return 10 * np.log10(power_signal / power_noise)
