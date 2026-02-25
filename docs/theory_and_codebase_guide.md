# Theory and Codebase Guide

## 1. Scope

This document explains:

1. the theoretical model behind `Vibration_PINN`,
2. how each equation is implemented in code,
3. why many training parameters are heuristic (not universal constants),
4. how to modify those parameters to gain better control over learning behavior.

Primary implementation references:

- `src/structures/shear_building.py`
- `src/structures/newmark_solver.py`
- `src/data/generate_synthetic.py`
- `src/pinn/siren.py`
- `src/pinn/structural_params.py`
- `src/pinn/losses.py`
- `src/pinn/trainer.py`
- `experiments/exp01_clean_data.py`

---

## 2. End-to-End Pipeline

The baseline experiment (`experiments/exp01_clean_data.py`) runs this chain:

1. Build synthetic response from known physical parameters:
   - masses `m = [5000, 5000, 5000] kg`
   - stiffnesses `k = [8e6, 6e6, 4e6] N/m`
   - damping ratio `xi = 0.05`
2. Sample measured data points (`data_stride`) and collocation points (`n_colloc`).
3. Train a SIREN network `u_theta(t)` plus trainable structural parameters.
4. Minimize a two-phase loss:
   - phase 1: data + initial conditions
   - phase 2: data + physics + initial conditions + regularization
5. Save:
   - `exp01_loss.png`
   - `exp01_stiffness.png`

---

## 3. Theoretical Background

## 3.1 Shear Building Dynamics with Base Excitation

For an `n`-DOF shear building in relative coordinates `u(t)`:

`M u_ddot + C u_dot + K u + M iota ag(t) = 0`

where:

- `M` is diagonal mass matrix,
- `K` is tridiagonal stiffness matrix,
- `C` is Rayleigh damping matrix,
- `iota` is a vector of ones,
- `ag(t)` is ground acceleration.

This is exactly the residual enforced in `physics_loss` (`src/pinn/losses.py`).

## 3.2 Matrix Assembly

Implemented in `src/structures/shear_building.py`:

- `M = diag(m1,...,mn)`.
- `K` assembled from story stiffnesses with nearest-neighbor coupling.

The same structure is rebuilt in torch form inside `StructuralParameters` (`src/pinn/structural_params.py`).

## 3.3 Rayleigh Damping

Rayleigh form:

`C = alpha M + beta K`

with coefficients from target modal damping ratio `xi` at mode 1 and mode 3:

- `alpha = 2 xi omega1 omega3 / (omega1 + omega3)`
- `beta  = 2 xi / (omega1 + omega3)`

Used in both:

- data generation (`ShearBuilding._build_C`),
- training (`StructuralParameters.build_C`).

## 3.4 Time Integration (Reference Solver)

`src/structures/newmark_solver.py` uses Newmark constant-average-acceleration:

- `gamma = 0.5`
- `beta = 0.25`

This scheme is unconditionally stable for linear systems and generates the synthetic "ground truth" trajectories used in training.

---

## 4. PINN Formulation in This Codebase

## 4.1 Network Representation

`src/pinn/siren.py` defines:

- `SirenNet: t -> u(t)` (relative displacement for each floor),
- sinusoidal hidden layers (`sin(omega_0 * linear(...))`),
- final linear output layer.

Autograd then produces:

- `u_dot(t)`,
- `u_ddot(t)`,

inside `src/pinn/losses.py`.

## 4.2 Trainable Physical Parameters

`src/pinn/structural_params.py` stores parameters in unconstrained form:

- `log_m` -> `m = exp(log_m)` (positive),
- `log_k` -> `k = exp(log_k)` (positive),
- `logit_xi` -> `xi = 0.3 * sigmoid(logit_xi)` (bounded in `(0, 0.3)`).

Important implication:

- initial `logit_xi = 0` means initial `xi = 0.15`, which is far above true `0.05`.
- this is a design choice, not a physical prior.

---

## 5. Loss Functions and What They Mean

Defined in `src/pinn/losses.py`.

## 5.1 Data Loss

`L_data = mean((a_abs_pred_selected - a_measured)^2)`

with:

- `a_abs_pred = u_ddot + ag(t)`.

This enforces fit to sensor measurements on instrumented floors only.

## 5.2 Physics Loss

`L_phys = mean(|| M u_ddot + C u_dot + K u + M iota ag ||^2)`

This enforces the equation of motion at collocation points.

## 5.3 Initial Condition Loss

`L_ic = mean(||u(0)||^2) + mean(||u_dot(0)||^2)`

This anchors the trajectory at rest initial state.

## 5.4 Prior Regularization

`L_reg = mean(((m - m_prior)/m_prior)^2) + mean(((k - k_prior)/k_prior)^2)`

This keeps identified `m,k` near priors.

---

## 6. Two-Phase Training Schedule

Implemented in `src/pinn/trainer.py`.

Phase 1:

`L_total = lambda_data L_data + lambda_ic L_ic`

Phase 2:

`L_total = lambda_data L_data + lambda_phys(epoch) L_phys + lambda_ic L_ic + lambda_reg L_reg`

with:

- `lambda_phys(epoch)` log-ramped from `0.01` to `1.0`.
- defaults: `phase1=500`, `phase2=3000`.

Why this design exists:

- phase 1 preconditions the neural field,
- phase 2 gradually imposes physics to avoid immediate instability.

---

## 7. Why Hyperparameters Are "Arbitrary" (Heuristic)

They are not arbitrary in the sense of random, but they are heuristic:

1. There is no closed-form formula that gives globally optimal values for this PINN setup.
2. Values depend on scaling, architecture, excitation, and identifiability.
3. Defaults came from practical starting points in the PRD and implementation experience.

So these values are controllable knobs, not universal truths.

---

## 8. Hyperparameter Control Map

Main knobs exposed by `experiments/exp01_clean_data.py`:

| Parameter | Default | Role | If increased | If decreased | Typical reason to modify |
|---|---:|---|---|---|---|
| `hidden_features` | 128 | Network width | More capacity, slower, can overfit/oscillate | Less capacity, may underfit | Balance fit quality vs stability |
| `hidden_layers` | 4 | Network depth | Richer representation, harder optimization | Simpler optimization, may underfit | Control representational complexity |
| `phase1_epochs` | 500 | Warm-up length | Better pre-fit before physics, but longer no-physics training | Earlier physics enforcement | Control phase transition shock |
| `phase2_epochs` | 3000 | Physics-constrained refinement | More time to settle | Faster run, less convergence | Improve final parameter stability |
| `data_stride` | 6 | Number of measured points | Fewer points (coarser supervision) | More points (denser supervision) | Speed vs measurement fidelity |
| `n_colloc` | 2000 | Physics constraint coverage | Better physics coverage, slower | Faster but weaker physics enforcement | Control physics residual quality |
| `dt` | 0.01 | Simulation resolution | More time steps for fixed duration | Fewer steps, coarser dynamics | Numerical fidelity vs runtime |
| `duration` | 10.0 | Signal length | More information but harder long-horizon training | Less information | Control identifiability and runtime |
| `seed` | 42 | Randomness in sampling/init | Different optimization trajectory | Different optimization trajectory | Evaluate robustness |

Trainer defaults in `src/pinn/trainer.py`:

| Parameter | Default | Notes |
|---|---:|---|
| `lr_net` | `1e-4` | Conservative for SIREN + higher-order derivatives |
| `lr_params` | `1e-3` | Fast structural updates, can amplify oscillations |
| `lambda_data` | `1.0` | Data fit weight |
| `lambda_ic` | `1.0` | Initial condition weight |
| `lambda_phys_start` | `0.01` | Physics weight at phase-2 start |
| `lambda_phys_end` | `1.0` | Physics weight at phase-2 end |
| `lambda_reg` | `0.01` | Pull toward `m_prior, k_prior` |

---

## 9. Code-Level Caveats That Explain Current Behavior

These are important for interpreting results:

1. `data_loss` currently does not use structural parameters directly.
   - It depends on `u_ddot` and `ag`, not on `M,K,C`.
2. `ic_loss` also does not use structural parameters.
3. During phase 1, there is no `reg_loss` and no `physics_loss`.
   - Result: structural params get little or no direct gradient in phase 1.
4. `m_prior` and `k_prior` are initialized at `+20%` from truth in `exp01_clean_data.py`.
5. `lambda_reg` in phase 2 actively pulls toward those high priors.

Consequences:

- stiffness trajectories can remain above true values or cross true lines late,
- physics loss can be very large and spiky (unnormalized residual units + competing objectives),
- reducing loss does not guarantee exact parameter recovery.

---

## 10. How to Control Training Better

If you want more predictable behavior, apply changes in this order:

1. Reduce structural parameter step size first:
   - lower `lr_params` (for example to `3e-4` or `1e-4`).
2. Reduce prior bias if needed:
   - lower `lambda_reg`,
   - use less biased priors (closer to engineering estimates).
3. Keep warm-up moderate:
   - do not make phase 1 too long unless transition instability is severe.
4. Increase phase 2 only after steps 1-3:
   - longer training helps only if optimization is already stable.
5. If identification is the target, improve identifiability:
   - fix masses and identify stiffnesses,
   - or add constraints/priors from external engineering information.

---

## 11. Why You See Large/Spiky Physics Loss

This is expected in this implementation for three reasons:

1. Physics is turned on abruptly at phase-2 start.
2. Residual is in physical units with large scales (`k ~ 1e6`, `m ~ 1e3`), not normalized.
3. Data fit and physics constraints compete; parameter updates can improve one and worsen the other.

Interpretation rule:

- absolute magnitude of `L_phys` is less informative than trend and eventual parameter stability.

---

## 12. Practical Experiment Tuning Recipes

## 12.1 More Stable Baseline

Start from baseline and change only:

- `lr_params`: `1e-3 -> 3e-4`
- `lambda_reg`: `0.01 -> 0.003`

Expected effect:

- smaller physics spikes,
- less overshoot/crossing in stiffness curves,
- slower but smoother parameter evolution.

## 12.2 More Physics-Driven Identification

If data fit is good but physical consistency is poor:

- keep phase 1 near `300-500`,
- increase phase 2 moderately,
- increase `n_colloc` (for example `2000 -> 4000`) if compute allows.

## 12.3 Faster Debug Runs

Use small models and epochs to inspect behavior quickly:

- lower `hidden_features`,
- lower `hidden_layers`,
- lower phase epochs.

Do not judge final identification accuracy from smoke-run settings.

---

## 13. Current Implementation Status

Implemented and usable:

- data generation,
- structural model + Newmark solver,
- SIREN + structural parameters,
- losses + trainer,
- Experiment 01 baseline.

Not implemented yet (stubs):

- `experiments/exp02_noisy_data.py`
- `experiments/exp03_sparse_sensors.py`
- `experiments/exp04_damage_detection.py`
- `experiments/exp05_era_comparison.py`

---

## 14. Suggested Next Documentation Extensions

If you continue this project, add:

1. a dedicated "normalization strategy" section once residual scaling is introduced,
2. a "fixed-mass identification" variant guide,
3. benchmark tables for multiple seeds and noise levels.
