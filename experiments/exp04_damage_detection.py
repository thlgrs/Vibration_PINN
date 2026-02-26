"""
Experiment 04: Damage detection — pre/post stiffness reduction.

Simulate 30% stiffness reduction at story 2 and verify PINN-SID
can localize and quantify the damage.

For each scenario (pre-damage, post-damage), runs multiple trials
and reports stiffness reduction index: Dk_i = (k_before - k_after) / k_before.
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
        description="Run Experiment 04 (damage detection) for PINN-SID."
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
    parser.add_argument("--damage-floor", type=int, default=1,
                        help="0-indexed floor where damage occurs (default: 1 = story 2)")
    parser.add_argument("--damage-ratio", type=float, default=0.30,
                        help="Fractional stiffness reduction (default: 0.30 = 30%%)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", default="results/damage_detection")
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

    # Initial guess: +20% offset from nominal (pre-damage) stiffnesses
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


def run_scenario(label, data, noise_key, device, args, n_trials, seed_offset):
    """Run multiple trials for one scenario and return stacked results."""
    print(f"\n=== Scenario: {label} ===")

    trial_masses = []
    trial_stiffnesses = []
    trial_damping = []

    for trial in range(n_trials):
        seed = seed_offset + trial
        print(f"  Trial {trial + 1}/{n_trials} (seed={seed})", end=" ... ")
        result = run_single_trial(data, noise_key, device, args, seed)
        trial_masses.append(result["masses"])
        trial_stiffnesses.append(result["stiffnesses"])
        trial_damping.append(result["damping"])
        print("done")

    trial_masses = np.array(trial_masses)
    trial_stiffnesses = np.array(trial_stiffnesses)
    trial_damping = np.array(trial_damping)

    true_k = data["true_stiffnesses"]
    true_m = data["true_masses"]
    true_xi = data["true_xi"]

    print(f"\n  Masses (true: {true_m}):")
    print(f"    mean: {trial_masses.mean(axis=0)}")
    print(f"    std:  {trial_masses.std(axis=0)}")
    print(f"  Stiffnesses (true: {true_k}):")
    print(f"    mean: {trial_stiffnesses.mean(axis=0)}")
    print(f"    std:  {trial_stiffnesses.std(axis=0)}")
    print(f"  Damping (true: {true_xi}):")
    print(f"    mean: {trial_damping.mean():.4f}")
    print(f"    std:  {trial_damping.std():.4f}")

    return {
        "masses": trial_masses,
        "stiffnesses": trial_stiffnesses,
        "damping": trial_damping,
    }


def main():
    args = parse_args()

    from src.data.generate_synthetic import DEFAULT_MASSES, DEFAULT_STIFFNESSES, DEFAULT_DAMPING

    n_trials = args.n_trials
    noise_level = args.noise_level
    damage_floor = args.damage_floor
    damage_ratio = args.damage_ratio

    masses = list(DEFAULT_MASSES)
    k_nominal = list(DEFAULT_STIFFNESSES)
    xi = DEFAULT_DAMPING

    # Damaged stiffnesses
    k_damaged = list(k_nominal)
    k_damaged[damage_floor] = k_nominal[damage_floor] * (1.0 - damage_ratio)

    nonzero_levels = [noise_level] if noise_level > 0.0 else []
    noise_key = noise_key_for_level(noise_level)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Noise level: {int(round(noise_level * 100))}%")
    print(f"Damage: {int(round(damage_ratio * 100))}% stiffness reduction at story {damage_floor + 1}")
    print(f"  k_nominal = {k_nominal}")
    print(f"  k_damaged = {k_damaged}")
    print(f"Trials per scenario: {n_trials}")

    # --- Pre-damage scenario ---
    data_pre = generate_synthetic_data(
        masses=masses, stiffnesses=k_nominal, xi=xi,
        excitation="el_centro", noise_levels=nonzero_levels,
        dt=args.dt, duration=args.duration,
    )
    results_pre = run_scenario(
        "Pre-damage", data_pre, noise_key, device, args, n_trials, seed_offset=1000
    )

    # --- Post-damage scenario ---
    data_post = generate_synthetic_data(
        masses=masses, stiffnesses=k_damaged, xi=xi,
        excitation="el_centro", noise_levels=nonzero_levels,
        dt=args.dt, duration=args.duration,
    )
    results_post = run_scenario(
        "Post-damage", data_post, noise_key, device, args, n_trials, seed_offset=2000
    )

    # --- Stiffness reduction index ---
    k_pre_mean = results_pre["stiffnesses"].mean(axis=0)
    k_post_mean = results_post["stiffnesses"].mean(axis=0)
    k_pre_std = results_pre["stiffnesses"].std(axis=0)
    k_post_std = results_post["stiffnesses"].std(axis=0)

    # Dk_i = (k_before - k_after) / k_before
    dki = (k_pre_mean - k_post_mean) / k_pre_mean
    # True reduction index
    k_nom = np.array(k_nominal)
    k_dam = np.array(k_damaged)
    dki_true = (k_nom - k_dam) / k_nom

    # Save results
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "noise_level": noise_level,
        "damage_floor": damage_floor,
        "damage_ratio": damage_ratio,
        "n_trials": n_trials,
        "k_nominal": k_nom,
        "k_damaged": k_dam,
        "true_masses": np.array(masses),
        "true_xi": xi,
        "masses_pre": results_pre["masses"],
        "stiffnesses_pre": results_pre["stiffnesses"],
        "damping_pre": results_pre["damping"],
        "masses_post": results_post["masses"],
        "stiffnesses_post": results_post["stiffnesses"],
        "damping_post": results_post["damping"],
        "dki": dki,
        "dki_true": dki_true,
    }
    npz_path = results_dir / "exp04_results.npz"
    np.savez(npz_path, **save_dict)
    print(f"\nResults saved to {npz_path}")

    # --- Summary table ---
    n_floors = len(k_nominal)
    print("\n" + "=" * 78)
    print("SUMMARY: Identified stiffnesses and damage detection")
    print("=" * 78)

    # Stiffness comparison
    header = f"{'':>14s}"
    for i in range(n_floors):
        header += f"  {'k' + str(i + 1):>12s}"
    print(header)
    print("-" * 78)

    row = f"{'True (pre)':>14s}"
    for k in k_nom:
        row += f"  {k:>12.0f}"
    print(row)

    row = f"{'PINN (pre)':>14s}"
    for k, s in zip(k_pre_mean, k_pre_std):
        row += f"  {k:>7.0f}±{s:<4.0f}"
    print(row)

    row = f"{'True (post)':>14s}"
    for k in k_dam:
        row += f"  {k:>12.0f}"
    print(row)

    row = f"{'PINN (post)':>14s}"
    for k, s in zip(k_post_mean, k_post_std):
        row += f"  {k:>7.0f}±{s:<4.0f}"
    print(row)

    print("-" * 78)

    # Damage index
    print("\nStiffness Reduction Index  Dk_i = (k_pre - k_post) / k_pre")
    print("-" * 78)
    header = f"{'':>14s}"
    for i in range(n_floors):
        header += f"  {'Story ' + str(i + 1):>12s}"
    print(header)
    print("-" * 78)

    row = f"{'True Dk':>14s}"
    for d in dki_true:
        row += f"  {d * 100:>11.1f}%"
    print(row)

    row = f"{'PINN Dk':>14s}"
    for d in dki:
        row += f"  {d * 100:>11.1f}%"
    print(row)

    print("=" * 78)

    # Detection verdict
    print(f"\nDamage localization: ", end="")
    detected_floor = np.argmax(dki)
    if detected_floor == damage_floor:
        print(f"CORRECT — largest Dk at story {detected_floor + 1}")
    else:
        print(f"MISSED — largest Dk at story {detected_floor + 1}, "
              f"expected story {damage_floor + 1}")

    print(f"Damage quantification: "
          f"true = {dki_true[damage_floor]*100:.1f}%, "
          f"identified = {dki[damage_floor]*100:.1f}%")


if __name__ == "__main__":
    main()
