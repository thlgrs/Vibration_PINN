"""
Experiment 02: Noise sensitivity study.

Runs PINN-SID with varying noise levels: 0%, 5%, 10%, 15%.
For each noise level, runs multiple trials with different random seeds
and reports mean +/- std of identified parameters.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_synthetic_data
from src.pinn import PINNTrainer, SirenNet, StructuralParameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 02 (noise sensitivity) for PINN-SID."
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--data-stride", type=int, default=6)
    parser.add_argument("--n-colloc", type=int, default=2000)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=500)
    parser.add_argument("--phase2-epochs", type=int, default=3000)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", default="results/noise_sensitivity")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_trial(data, noise_key, device, args, seed):
    """Run one PINN-SID trial and return identified parameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    t = data["t"]
    ag = data["ground_acc"]
    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    n_floors = len(true_m)

    # Select measurement data (clean or noisy)
    if noise_key is None:
        a_meas_np = data["a_abs"]
    else:
        a_meas_np = data[noise_key]

    # Data points: every Nth time step
    idx_data = np.arange(0, len(t), args.data_stride)
    t_data = torch.tensor(t[idx_data, None], dtype=torch.float32, device=device)
    a_measured = torch.tensor(a_meas_np[idx_data], dtype=torch.float32, device=device)

    # Collocation points
    rng = np.random.default_rng(seed)
    t_colloc_np = rng.uniform(0, t[-1], (args.n_colloc, 1))
    t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=device)

    # Ground acceleration interpolation
    t_np = t.copy()
    ag_np = ag.copy()

    def ground_acc_fn(t_query):
        t_cpu = t_query.detach().cpu().numpy().flatten()
        ag_interp = np.interp(t_cpu, t_np, ag_np)
        return torch.tensor(ag_interp[:, None], dtype=torch.float32, device=device)

    # Initial guess: +20% offset
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

    floor_indices = list(range(n_floors))

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

    trainer.train()

    return {
        "masses": struct_params.m.detach().cpu().numpy().copy(),
        "stiffnesses": struct_params.k.detach().cpu().numpy().copy(),
        "damping": struct_params.xi.detach().cpu().item(),
    }


def noise_key_for_level(level):
    """Return the data dict key for a given noise level, or None for 0%."""
    if level == 0.0:
        return None
    pct = int(round(level * 100))
    return f"a_abs_noisy_{pct:02d}"


def main():
    args = parse_args()

    noise_levels = args.noise_levels
    n_trials = args.n_trials

    # Generate data with all requested noise levels
    nonzero_levels = [nl for nl in noise_levels if nl > 0.0]
    data = generate_synthetic_data(
        excitation="el_centro",
        noise_levels=nonzero_levels,
        dt=args.dt,
        duration=args.duration,
    )

    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    true_xi = data["true_xi"]

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Noise levels: {noise_levels}")
    print(f"Trials per level: {n_trials}")
    print()

    # Results storage
    all_results = {}

    for nl in noise_levels:
        pct = int(round(nl * 100))
        key = noise_key_for_level(nl)
        label = f"{pct}%"
        print(f"=== Noise level: {label} ===")

        trial_masses = []
        trial_stiffnesses = []
        trial_damping = []

        for trial in range(n_trials):
            seed = 1000 * pct + trial
            print(f"  Trial {trial + 1}/{n_trials} (seed={seed})", end=" ... ")
            result = run_single_trial(data, key, device, args, seed)
            trial_masses.append(result["masses"])
            trial_stiffnesses.append(result["stiffnesses"])
            trial_damping.append(result["damping"])
            print("done")

        trial_masses = np.array(trial_masses)
        trial_stiffnesses = np.array(trial_stiffnesses)
        trial_damping = np.array(trial_damping)

        all_results[label] = {
            "masses": trial_masses,
            "stiffnesses": trial_stiffnesses,
            "damping": trial_damping,
        }

        # Per-level summary
        print(f"\n  Masses (true: {true_m}):")
        print(f"    mean: {trial_masses.mean(axis=0)}")
        print(f"    std:  {trial_masses.std(axis=0)}")
        print(f"  Stiffnesses (true: {true_k}):")
        print(f"    mean: {trial_stiffnesses.mean(axis=0)}")
        print(f"    std:  {trial_stiffnesses.std(axis=0)}")
        print(f"  Damping (true: {true_xi}):")
        print(f"    mean: {trial_damping.mean():.4f}")
        print(f"    std:  {trial_damping.std():.4f}")
        print()

    # Save results
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "noise_levels": np.array(noise_levels),
        "n_trials": n_trials,
        "true_masses": true_m,
        "true_stiffnesses": true_k,
        "true_xi": true_xi,
    }
    for label, res in all_results.items():
        safe = label.replace("%", "pct")
        save_dict[f"masses_{safe}"] = res["masses"]
        save_dict[f"stiffnesses_{safe}"] = res["stiffnesses"]
        save_dict[f"damping_{safe}"] = res["damping"]

    npz_path = results_dir / "exp02_results.npz"
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean relative error (%) per noise level")
    print("=" * 70)
    header = f"{'Noise':>8s}"
    for i in range(len(true_k)):
        header += f"  {'k' + str(i + 1) + ' err%':>10s}"
    header += f"  {'xi err%':>10s}"
    print(header)
    print("-" * 70)

    for label, res in all_results.items():
        k_mean = res["stiffnesses"].mean(axis=0)
        k_err = np.abs(k_mean - true_k) / true_k * 100
        xi_mean = res["damping"].mean()
        xi_err = abs(xi_mean - true_xi) / true_xi * 100
        row = f"{label:>8s}"
        for e in k_err:
            row += f"  {e:>10.2f}"
        row += f"  {xi_err:>10.2f}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
