"""
Experiment 01: Full sensor, no noise — baseline identification.

Runs PINN-SID on clean synthetic data with all floors instrumented.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_synthetic_data
from src.pinn import PINNTrainer, SirenNet, StructuralParameters
from src.utils.plotting import plot_loss_history, plot_parameter_convergence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 01 (clean data baseline) for PINN-SID."
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--data-stride", type=int, default=6)
    parser.add_argument("--n-colloc", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=500)
    parser.add_argument("--phase2-epochs", type=int, default=3000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", default="results/parameter_convergence")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate clean data
    data = generate_synthetic_data(
        excitation="el_centro",
        noise_levels=[],
        dt=args.dt,
        duration=args.duration,
    )

    t = data["t"]
    a_abs = data["a_abs"]
    ag = data["ground_acc"]
    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    n_floors = len(true_m)

    # Convert to tensors
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Data points: every Nth time step
    idx_data = np.arange(0, len(t), args.data_stride)
    t_data = torch.tensor(t[idx_data, None], dtype=torch.float32, device=device)
    a_measured = torch.tensor(a_abs[idx_data], dtype=torch.float32, device=device)

    # Collocation points
    rng = np.random.default_rng(args.seed)
    t_colloc_np = rng.uniform(0, t[-1], (args.n_colloc, 1))
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
    model = SirenNet(
        n_floors,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
    ).to(device)
    struct_params = StructuralParameters(n_floors, m_prior.cpu(), k_prior.cpu()).to(
        device
    )

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
        config={
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
        },
    )

    history = trainer.train()

    # Report results
    print("\n=== Identified Parameters ===")
    print(f"Masses:      {struct_params.m.detach().cpu().numpy()}")
    print(f"True masses: {true_m}")
    print(f"Stiffnesses: {struct_params.k.detach().cpu().numpy()}")
    print(f"True stiff.: {true_k}")
    print(f"Damping:     {struct_params.xi.detach().cpu().item():.4f}")
    print("True damp.:  0.05")

    # Save plots
    if not args.skip_plots:
        project_root = Path(__file__).resolve().parents[1]
        results_dir = project_root / args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        loss_path = results_dir / "exp01_loss.png"
        stiff_path = results_dir / "exp01_stiffness.png"
        loss_fig, loss_ax = plot_loss_history(history, save_path=str(loss_path))
        param_fig, param_ax = plot_parameter_convergence(
            history["stiffnesses"],
            true_k,
            "Stiffness (N/m)",
            save_path=str(stiff_path),
        )
        print(f"Saved plots:\n- {loss_path}\n- {stiff_path}")


if __name__ == "__main__":
    main()
