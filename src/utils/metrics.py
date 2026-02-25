"""
Evaluation metrics for structural identification.
"""

import numpy as np


def nrmse(predicted, true):
    """Normalized Root Mean Square Error.

    NRMSE = sqrt(mean((pred - true)^2)) / max(|true|)

    Parameters
    ----------
    predicted : ndarray
        Predicted values.
    true : ndarray
        Ground truth values.

    Returns
    -------
    float
        NRMSE value.
    """
    rmse = np.sqrt(np.mean((predicted - true) ** 2))
    return rmse / np.max(np.abs(true))


def relative_error(estimated, true):
    """Relative identification error for each parameter.

    e_i = |estimated_i - true_i| / |true_i| * 100%

    Parameters
    ----------
    estimated : ndarray
        Estimated parameter values.
    true : ndarray
        True parameter values.

    Returns
    -------
    ndarray
        Relative errors in percent.
    """
    return np.abs(estimated - true) / np.abs(true) * 100.0


def frequency_error(freq_identified, freq_true):
    """Relative frequency identification error.

    Parameters
    ----------
    freq_identified : ndarray
        Identified frequencies (Hz).
    freq_true : ndarray
        True frequencies (Hz).

    Returns
    -------
    ndarray
        Relative errors in percent.
    """
    return relative_error(freq_identified, freq_true)
