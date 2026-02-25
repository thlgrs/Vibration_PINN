"""
Synthetic data generation pipeline for PINN-SID.

Generates time histories using the Newmark solver and saves to .npz format.
"""

import numpy as np
from ..structures import ShearBuilding, NewmarkSolver


# Default 3-story reference structure (from PRD Section 5, Phase 1)
DEFAULT_MASSES = [5000.0, 5000.0, 5000.0]           # kg
DEFAULT_STIFFNESSES = [8e6, 6e6, 4e6]                # N/m (8000, 6000, 4000 kN/m)
DEFAULT_DAMPING = 0.05                                 # 5%


def el_centro_like(dt, duration, peak_g=0.35):
    """Generate a synthetic El Centro-like ground motion (modulated noise).

    Parameters
    ----------
    dt : float
        Time step in seconds.
    duration : float
        Total duration in seconds.
    peak_g : float
        Peak ground acceleration in g.

    Returns
    -------
    t : ndarray
        Time vector.
    ag : ndarray
        Ground acceleration in m/s^2.
    """
    t = np.arange(0, duration, dt)
    n = len(t)

    # Modulated filtered noise (envelope * bandpass noise)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)

    # Envelope: ramp up, plateau, exponential decay
    envelope = np.ones(n)
    t_ramp = 2.0
    t_decay_start = duration * 0.4
    ramp_idx = t < t_ramp
    decay_idx = t > t_decay_start
    envelope[ramp_idx] = t[ramp_idx] / t_ramp
    envelope[decay_idx] = np.exp(-2.0 * (t[decay_idx] - t_decay_start))

    ag = envelope * noise
    ag = ag / np.max(np.abs(ag)) * peak_g * 9.81  # scale to m/s^2

    return t, ag


def chirp_signal(dt, duration, f_start=0.5, f_end=20.0, amplitude=2.0):
    """Generate a chirp (frequency sweep) ground motion.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    duration : float
        Total duration in seconds.
    f_start : float
        Start frequency in Hz.
    f_end : float
        End frequency in Hz.
    amplitude : float
        Peak acceleration in m/s^2.

    Returns
    -------
    t : ndarray
        Time vector.
    ag : ndarray
        Ground acceleration in m/s^2.
    """
    t = np.arange(0, duration, dt)
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * duration) * t**2)
    ag = amplitude * np.sin(phase)
    return t, ag


def white_noise_excitation(dt, duration, amplitude=1.0, seed=0):
    """Generate band-limited white noise ground motion.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    duration : float
        Total duration in seconds.
    amplitude : float
        RMS acceleration in m/s^2.
    seed : int
        Random seed.

    Returns
    -------
    t : ndarray
        Time vector.
    ag : ndarray
        Ground acceleration in m/s^2.
    """
    t = np.arange(0, duration, dt)
    rng = np.random.default_rng(seed)
    ag = amplitude * rng.standard_normal(len(t))
    return t, ag


def generate_synthetic_data(
    masses=None,
    stiffnesses=None,
    xi=None,
    dt=0.01,
    duration=10.0,
    excitation="el_centro",
    noise_levels=(0.0, 0.05, 0.15),
    output_path=None,
):
    """Generate synthetic vibration response data.

    Parameters
    ----------
    masses : array_like, optional
        Floor masses. Defaults to 3-story reference.
    stiffnesses : array_like, optional
        Story stiffnesses. Defaults to 3-story reference.
    xi : float, optional
        Damping ratio. Default 0.05.
    dt : float
        Time step in seconds.
    duration : float
        Total duration in seconds.
    excitation : str
        One of 'el_centro', 'chirp', 'white_noise'.
    noise_levels : tuple of float
        Noise levels as fraction of peak acceleration.
    output_path : str, optional
        Path to save .npz file. If None, data is returned but not saved.

    Returns
    -------
    data : dict
        Dictionary with keys: t, ground_acc, u, v, a_rel, a_abs,
        a_abs_noisy_{level}, true_params.
    """
    masses = masses or DEFAULT_MASSES
    stiffnesses = stiffnesses or DEFAULT_STIFFNESSES
    xi = xi or DEFAULT_DAMPING

    building = ShearBuilding(masses, stiffnesses, xi)
    solver = NewmarkSolver(building, dt)

    # Generate ground motion
    if excitation == "el_centro":
        t, ag = el_centro_like(dt, duration)
    elif excitation == "chirp":
        t, ag = chirp_signal(dt, duration)
    elif excitation == "white_noise":
        t, ag = white_noise_excitation(dt, duration)
    else:
        raise ValueError(f"Unknown excitation type: {excitation}")

    # Solve
    u, v, a_rel, a_abs = solver.solve(ag)

    # Package results
    data = {
        "t": t,
        "ground_acc": ag,
        "u": u,
        "v": v,
        "a_rel": a_rel,
        "a_abs": a_abs,
        "true_masses": np.array(masses),
        "true_stiffnesses": np.array(stiffnesses),
        "true_xi": xi,
        "dt": dt,
    }

    # Add noisy versions
    from .add_noise import add_noise
    for level in noise_levels:
        if level > 0:
            key = f"a_abs_noisy_{int(level * 100):02d}"
            data[key] = add_noise(a_abs, level)

    if output_path is not None:
        np.savez(output_path, **data)

    return data
