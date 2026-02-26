"""
Experiment 05: Classical vs PINN benchmark.

Compare PINN-SID identified frequencies and damping ratios against
ERA (Eigensystem Realization Algorithm) results, both under varying
noise levels.

ERA uses free-decay (impulse) response; PINN-SID uses forced (El Centro)
response. Both are compared to the analytical truth from the structural
model.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.era import ERA
from src.data import generate_synthetic_data
from src.pinn import PINNTrainer, SirenNet, StructuralParameters
from src.structures import ShearBuilding, NewmarkSolver
from src.structures.modal_analysis import modal_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 05 (ERA vs PINN-SID comparison)."
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--data-stride", type=int, default=6)
    parser.add_argument("--n-colloc", type=int, default=2000)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=500)
    parser.add_argument("--phase2-epochs", type=int, default=3000)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", default="results/era_comparison")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ERA identification
# ---------------------------------------------------------------------------

def generate_free_decay(masses, stiffnesses, xi, dt, duration):
    """Generate impulse (free-decay) response for ERA.

    Applies a single-step ground acceleration spike, then lets the
    structure ring down freely.
    """
    building = ShearBuilding(masses, stiffnesses, xi)
    solver = NewmarkSolver(building, dt)

    n_steps = int(duration / dt)
    ag = np.zeros(n_steps)
    ag[0] = 1.0  # unit impulse

    u, v, a_rel, a_abs = solver.solve(ag)
    t = np.arange(n_steps) * dt
    return t, a_abs


def run_era(free_decay_response, noise_level, dt, n_modes, seed=0):
    """Run ERA on (optionally noisy) free-decay data."""
    response = free_decay_response.copy()
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        peak = np.max(np.abs(response))
        response += rng.normal(0, noise_level * peak, response.shape)

    era = ERA(dt, n_modes)
    freqs, damping = era.identify(response)
    return freqs, damping


# ---------------------------------------------------------------------------
# PINN-SID identification
# ---------------------------------------------------------------------------

def noise_key_for_level(level):
    if level == 0.0:
        return None
    pct = int(round(level * 100))
    return f"a_abs_noisy_{pct:02d}"


def run_pinn_trial(data, noise_key, device, args, seed):
    """Run one PINN-SID trial and return identified parameters."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    t = data["t"]
    ag = data["ground_acc"]
    true_m = data["true_masses"]
    true_k = data["true_stiffnesses"]
    n_floors = len(true_m)

    if noise_key is None:
        a_meas_np = data["a_abs"]
    else:
        a_meas_np = data[noise_key]

    idx_data = np.arange(0, len(t), args.data_stride)
    t_data = torch.tensor(t[idx_data, None], dtype=torch.float32, device=device)
    a_measured = torch.tensor(a_meas_np[idx_data], dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    t_colloc_np = rng.uniform(0, t[-1], (args.n_colloc, 1))
    t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=device)

    t_np = t.copy()
    ag_np = ag.copy()

    def ground_acc_fn(t_query):
        t_cpu = t_query.detach().cpu().numpy().flatten()
        ag_interp = np.interp(t_cpu, t_np, ag_np)
        return torch.tensor(ag_interp[:, None], dtype=torch.float32, device=device)

    m_prior = torch.tensor(true_m * 1.2, dtype=torch.float32, device=device)
    k_prior = torch.tensor(true_k * 1.2, dtype=torch.float32, device=device)

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

    m_id = struct_params.m.detach().cpu().numpy().copy()
    k_id = struct_params.k.detach().cpu().numpy().copy()
    xi_id = struct_params.xi.detach().cpu().item()

    return {"masses": m_id, "stiffnesses": k_id, "damping": xi_id}


def freqs_from_params(masses, stiffnesses):
    """Compute natural frequencies (Hz) from identified M and K."""
    M = np.diag(masses)
    K = np.zeros((len(masses), len(masses)))
    k = stiffnesses
    for i in range(len(k)):
        K[i, i] += k[i]
        if i + 1 < len(k):
            K[i, i] += k[i + 1]
            K[i, i + 1] -= k[i + 1]
            K[i + 1, i] -= k[i + 1]
    _, freqs_hz, _, _ = modal_analysis(M, K)
    return freqs_hz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from src.data.generate_synthetic import DEFAULT_MASSES, DEFAULT_STIFFNESSES, DEFAULT_DAMPING

    masses = list(DEFAULT_MASSES)
    stiffnesses = list(DEFAULT_STIFFNESSES)
    xi = DEFAULT_DAMPING
    n_floors = len(masses)
    n_modes = n_floors

    noise_levels = args.noise_levels
    n_trials = args.n_trials

    # True frequencies
    building = ShearBuilding(masses, stiffnesses, xi)
    true_freqs = building.natural_frequencies_hz()
    print(f"True natural frequencies (Hz): {true_freqs}")
    print(f"True damping ratio: {xi}")
    print()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Noise levels: {noise_levels}")
    print(f"PINN trials per noise level: {n_trials}")
    print()

    # Generate free-decay data for ERA
    t_free, free_decay = generate_free_decay(
        masses, stiffnesses, xi, args.dt, args.duration
    )

    # Generate forced-response data for PINN-SID
    nonzero_levels = [nl for nl in noise_levels if nl > 0.0]
    data_pinn = generate_synthetic_data(
        masses=masses, stiffnesses=stiffnesses, xi=xi,
        excitation="el_centro", noise_levels=nonzero_levels,
        dt=args.dt, duration=args.duration,
    )

    # Storage
    era_results = {}
    pinn_results = {}

    for nl in noise_levels:
        pct = int(round(nl * 100))
        label = f"{pct}%"
        print(f"=== Noise level: {label} ===")

        # --- ERA (run n_trials with different noise seeds) ---
        era_freqs_all = []
        era_damp_all = []
        for trial in range(n_trials):
            seed = 3000 + pct * 100 + trial
            freqs_era, damp_era = run_era(
                free_decay, nl, args.dt, n_modes, seed=seed
            )
            era_freqs_all.append(freqs_era)
            era_damp_all.append(damp_era)

        era_freqs_all = np.array(era_freqs_all)
        era_damp_all = np.array(era_damp_all)

        era_results[label] = {
            "frequencies": era_freqs_all,
            "damping": era_damp_all,
        }

        print(f"  ERA frequencies (mean): {era_freqs_all.mean(axis=0)} Hz")
        print(f"  ERA damping     (mean): {era_damp_all.mean(axis=0)}")

        # --- PINN-SID ---
        noise_key = noise_key_for_level(nl)
        pinn_freqs_all = []
        pinn_damp_all = []
        pinn_k_all = []
        pinn_m_all = []

        for trial in range(n_trials):
            seed = 4000 + pct * 100 + trial
            print(f"  PINN trial {trial + 1}/{n_trials} (seed={seed})", end=" ... ")
            result = run_pinn_trial(data_pinn, noise_key, device, args, seed)
            pinn_m_all.append(result["masses"])
            pinn_k_all.append(result["stiffnesses"])
            pinn_damp_all.append(result["damping"])

            freqs_pinn = freqs_from_params(result["masses"], result["stiffnesses"])
            pinn_freqs_all.append(freqs_pinn)
            print("done")

        pinn_freqs_all = np.array(pinn_freqs_all)
        pinn_damp_all = np.array(pinn_damp_all)
        pinn_k_all = np.array(pinn_k_all)
        pinn_m_all = np.array(pinn_m_all)

        pinn_results[label] = {
            "frequencies": pinn_freqs_all,
            "damping": pinn_damp_all,
            "stiffnesses": pinn_k_all,
            "masses": pinn_m_all,
        }

        print(f"  PINN frequencies (mean): {pinn_freqs_all.mean(axis=0)} Hz")
        print(f"  PINN damping     (mean): {np.mean(pinn_damp_all):.4f}")
        print()

    # --- Save results ---
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "noise_levels": np.array(noise_levels),
        "n_trials": n_trials,
        "true_frequencies": true_freqs,
        "true_xi": xi,
        "true_masses": np.array(masses),
        "true_stiffnesses": np.array(stiffnesses),
    }
    for label, res in era_results.items():
        safe = label.replace("%", "pct")
        save_dict[f"era_freqs_{safe}"] = res["frequencies"]
        save_dict[f"era_damp_{safe}"] = res["damping"]
    for label, res in pinn_results.items():
        safe = label.replace("%", "pct")
        save_dict[f"pinn_freqs_{safe}"] = res["frequencies"]
        save_dict[f"pinn_damp_{safe}"] = res["damping"]
        save_dict[f"pinn_k_{safe}"] = res["stiffnesses"]
        save_dict[f"pinn_m_{safe}"] = res["masses"]

    npz_path = results_dir / "exp05_results.npz"
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    # --- Summary table ---
    print("\n" + "=" * 85)
    print("SUMMARY: Frequency identification — relative error (%)")
    print("=" * 85)

    header = f"{'Noise':>8s}  {'Method':>8s}"
    for i in range(n_modes):
        header += f"  {'f' + str(i + 1) + ' err%':>10s}"
    header += f"  {'xi err%':>10s}"
    print(header)
    print("-" * 85)

    for nl in noise_levels:
        pct = int(round(nl * 100))
        label = f"{pct}%"

        # ERA row
        era_f_mean = era_results[label]["frequencies"].mean(axis=0)
        era_f_err = np.abs(era_f_mean - true_freqs) / true_freqs * 100
        era_xi_mean = era_results[label]["damping"].mean(axis=0)
        # ERA gives per-mode damping; average for summary
        era_xi_avg = era_xi_mean.mean()
        era_xi_err = abs(era_xi_avg - xi) / xi * 100

        row = f"{label:>8s}  {'ERA':>8s}"
        for e in era_f_err:
            row += f"  {e:>10.2f}"
        row += f"  {era_xi_err:>10.2f}"
        print(row)

        # PINN row
        pinn_f_mean = pinn_results[label]["frequencies"].mean(axis=0)
        pinn_f_err = np.abs(pinn_f_mean - true_freqs) / true_freqs * 100
        pinn_xi_mean = np.mean(pinn_results[label]["damping"])
        pinn_xi_err = abs(pinn_xi_mean - xi) / xi * 100

        row = f"{'':>8s}  {'PINN':>8s}"
        for e in pinn_f_err:
            row += f"  {e:>10.2f}"
        row += f"  {pinn_xi_err:>10.2f}"
        print(row)

        print("-" * 85)

    print("=" * 85)

    # Stiffness accuracy (PINN only — ERA doesn't identify stiffnesses)
    print("\nPINN-SID stiffness relative error (%):")
    print("-" * 60)
    header = f"{'Noise':>8s}"
    true_k = np.array(stiffnesses)
    for i in range(n_floors):
        header += f"  {'k' + str(i + 1) + ' err%':>10s}"
    print(header)
    print("-" * 60)

    for nl in noise_levels:
        pct = int(round(nl * 100))
        label = f"{pct}%"
        k_mean = pinn_results[label]["stiffnesses"].mean(axis=0)
        k_err = np.abs(k_mean - true_k) / true_k * 100
        row = f"{label:>8s}"
        for e in k_err:
            row += f"  {e:>10.2f}"
        print(row)
    print("-" * 60)


if __name__ == "__main__":
    main()
