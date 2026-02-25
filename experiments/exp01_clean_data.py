"""
Experiment 01: Full sensor, no noise — baseline identification.

Runs PINN-SID on clean synthetic data with all floors instrumented.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "..")

from src.data import generate_synthetic_data
from src.pinn import SirenNet, StructuralParameters, PINNTrainer
from src.utils.plotting import plot_loss_history, plot_parameter_convergence


def main():
    # Generate clean data
    data = generate_synthetic_data(
        excitation="el_centro",
        noise_levels=[],
        dt=0.01,
        duration=10.0,
    )

    t = data["t"]
    a_abs = data["a_abs"]
    ag = data["ground_acc"]
    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    n_floors = len(true_m)

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data points: every 6th time step
    idx_data = np.arange(0, len(t), 6)
    t_data = torch.tensor(t[idx_data, None], dtype=torch.float32, device=device)
    a_measured = torch.tensor(a_abs[idx_data], dtype=torch.float32, device=device)

    # Collocation points: 2000 random points
    rng = np.random.default_rng(42)
    t_colloc_np = rng.uniform(0, t[-1], (2000, 1))
    t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=device)

    # Ground acceleration interpolation function
    t_np = t.copy()
    ag_np = ag.copy()

    def ground_acc_fn(t_query):
        t_cpu = t_query.detach().cpu().numpy().flatten()
        ag_interp = np.interp(t_cpu, t_np, ag_np)
        return torch.tensor(ag_interp[:, None], dtype=torch.float32, device=device)

    # Initial guess: +20% offset from true values
    m_prior = torch.tensor(true_m * 1.2, dtype=torch.float32, device=device)
    k_prior = torch.tensor(true_k * 1.2, dtype=torch.float32, device=device)

    # Build model
    model = SirenNet(n_floors, hidden_features=128, hidden_layers=4).to(device)
    struct_params = StructuralParameters(n_floors, m_prior.cpu(), k_prior.cpu()).to(device)

    # All floors instrumented
    floor_indices = list(range(n_floors))

    # Train
    trainer = PINNTrainer(
        model=model,
        struct_params=struct_params,
        ground_acc_fn=ground_acc_fn,
        t_data=t_data,
        a_measured=a_measured,
        floor_indices=floor_indices,
        t_colloc=t_colloc,
        m_prior=m_prior,
        k_prior=k_prior,
    )

    history = trainer.train()

    # Report results
    print("\n=== Identified Parameters ===")
    print(f"Masses:      {struct_params.m.detach().cpu().numpy()}")
    print(f"True masses: {true_m}")
    print(f"Stiffnesses: {struct_params.k.detach().cpu().numpy()}")
    print(f"True stiff.: {true_k}")
    print(f"Damping:     {struct_params.xi.detach().cpu().item():.4f}")
    print(f"True damp.:  0.05")

    # Save plots
    plot_loss_history(history, save_path="../results/parameter_convergence/exp01_loss.png")
    plot_parameter_convergence(
        history["stiffnesses"], true_k, "Stiffness (N/m)",
        save_path="../results/parameter_convergence/exp01_stiffness.png",
    )


if __name__ == "__main__":
    main()
