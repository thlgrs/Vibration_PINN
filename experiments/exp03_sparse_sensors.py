"""
Experiment 03: Partial instrumentation study.

Runs PINN-SID with varying sensor configurations:
- full: floors [0, 1, 2] — all 3 sensors
- partial_2: floors [0, 2] — floors 1 & 3 (skip middle)
- partial_1: floors [2] — roof only

For each config, runs multiple trials with different random seeds
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

SENSOR_CONFIGS = {
    "full": [0, 1, 2],
    "partial_2": [0, 2],
    "partial_1": [2],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 03 (sparse sensors) for PINN-SID."
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
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", default="results/sparse_sensors")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_trial(data, noise_key, floor_indices, device, args, seed):
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

    # Data points: every Nth time step, only instrumented floors
    idx_data = np.arange(0, len(t), args.data_stride)
    t_data = torch.tensor(t[idx_data, None], dtype=torch.float32, device=device)
    a_measured = torch.tensor(
        a_meas_np[idx_data][:, floor_indices], dtype=torch.float32, device=device
    )

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

    n_trials = args.n_trials
    noise_level = args.noise_level

    # Generate data with the requested noise level
    nonzero_levels = [noise_level] if noise_level > 0.0 else []
    data = generate_synthetic_data(
        excitation="el_centro",
        noise_levels=nonzero_levels,
        dt=args.dt,
        duration=args.duration,
    )

    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    true_xi = data["true_xi"]

    noise_key = noise_key_for_level(noise_level)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Noise level: {int(round(noise_level * 100))}%")
    print(f"Sensor configs: {list(SENSOR_CONFIGS.keys())}")
    print(f"Trials per config: {n_trials}")
    print()

    # Results storage
    all_results = {}

    for config_name, floor_indices in SENSOR_CONFIGS.items():
        floor_labels = [f"F{i+1}" for i in floor_indices]
        print(f"=== Config: {config_name} (floors: {floor_labels}) ===")

        trial_masses = []
        trial_stiffnesses = []
        trial_damping = []

        for trial in range(n_trials):
            seed = hash(config_name) % 10000 + trial
            print(f"  Trial {trial + 1}/{n_trials} (seed={seed})", end=" ... ")
            result = run_single_trial(
                data, noise_key, floor_indices, device, args, seed
            )
            trial_masses.append(result["masses"])
            trial_stiffnesses.append(result["stiffnesses"])
            trial_damping.append(result["damping"])
            print("done")

        trial_masses = np.array(trial_masses)
        trial_stiffnesses = np.array(trial_stiffnesses)
        trial_damping = np.array(trial_damping)

        all_results[config_name] = {
            "masses": trial_masses,
            "stiffnesses": trial_stiffnesses,
            "damping": trial_damping,
        }

        # Per-config summary
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
        "noise_level": noise_level,
        "n_trials": n_trials,
        "true_masses": true_m,
        "true_stiffnesses": true_k,
        "true_xi": true_xi,
    }
    for config_name, res in all_results.items():
        save_dict[f"masses_{config_name}"] = res["masses"]
        save_dict[f"stiffnesses_{config_name}"] = res["stiffnesses"]
        save_dict[f"damping_{config_name}"] = res["damping"]

    npz_path = results_dir / "exp03_results.npz"
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean relative error (%) per sensor config")
    print("=" * 70)
    header = f"{'Config':>12s}"
    for i in range(len(true_k)):
        header += f"  {'k' + str(i + 1) + ' err%':>10s}"
    header += f"  {'xi err%':>10s}"
    print(header)
    print("-" * 70)

    for config_name, res in all_results.items():
        k_mean = res["stiffnesses"].mean(axis=0)
        k_err = np.abs(k_mean - true_k) / true_k * 100
        xi_mean = res["damping"].mean()
        xi_err = abs(xi_mean - true_xi) / true_xi * 100
        row = f"{config_name:>12s}"
        for e in k_err:
            row += f"  {e:>10.2f}"
        row += f"  {xi_err:>10.2f}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
