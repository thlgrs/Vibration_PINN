"""
Plotting utilities for PINN-SID results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_history(t, signals, labels=None, title="Time History",
                      ylabel="Acceleration (m/s²)", save_path=None):
    """Plot time history signals.

    Parameters
    ----------
    t : ndarray
        Time vector.
    signals : list of ndarray
        Signals to plot, each shape (n_steps,).
    labels : list of str, optional
        Legend labels.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, sig in enumerate(signals):
        label = labels[i] if labels else None
        ax.plot(t, sig, label=label, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_parameter_convergence(history, true_params=None, param_name="Parameter",
                                save_path=None):
    """Plot parameter convergence during training.

    Parameters
    ----------
    history : list of ndarray
        Parameter values at each epoch.
    true_params : ndarray, optional
        True parameter values for reference lines.
    param_name : str
        Name for axis labels.
    save_path : str, optional
        Path to save figure.
    """
    history_arr = np.array(history)
    n_params = history_arr.shape[1] if history_arr.ndim > 1 else 1
    epochs = np.arange(len(history_arr))

    fig, ax = plt.subplots(figsize=(10, 5))
    if history_arr.ndim == 1:
        ax.plot(epochs, history_arr, label=param_name)
        if true_params is not None:
            ax.axhline(true_params, color="k", linestyle="--", label="True")
    else:
        for i in range(n_params):
            ax.plot(epochs, history_arr[:, i], label=f"{param_name}_{i+1}")
            if true_params is not None:
                ax.axhline(true_params[i], color=f"C{i}", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(param_name)
    ax.set_title(f"{param_name} Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_loss_history(history, save_path=None):
    """Plot training loss curves.

    Parameters
    ----------
    history : dict
        Training history from PINNTrainer.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for key in ["loss_total", "loss_data", "loss_physics", "loss_ic", "loss_reg"]:
        if key in history and any(v > 0 for v in history[key]):
            ax.semilogy(history[key], label=key.replace("loss_", ""), linewidth=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
