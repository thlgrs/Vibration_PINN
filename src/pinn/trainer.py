"""
Two-phase PINN training loop for structural identification.

Phase 1 (warm-up): Train on data + IC loss only.
Phase 2 (full):    Activate physics loss with curriculum scheduling.
"""

import numpy as np
import torch


class PINNTrainer:
    """Two-phase training loop for PINN structural identification.

    Parameters
    ----------
    model : SirenNet
        Displacement network.
    struct_params : StructuralParameters
        Trainable structural parameters.
    ground_acc_fn : callable
        Ground acceleration function: t -> ag(t).
    t_data : Tensor
        Data time points (N_d, 1).
    a_measured : Tensor
        Measured absolute accelerations (N_d, n_sensors).
    floor_indices : list of int
        Instrumented floor indices.
    t_colloc : Tensor
        Collocation time points (N_c, 1).
    m_prior : Tensor
        Prior mass estimates for regularization.
    k_prior : Tensor
        Prior stiffness estimates for regularization.
    config : dict, optional
        Training hyperparameters.
    """

    DEFAULT_CONFIG = {
        "lr_net": 1e-4,
        "lr_params": 1e-3,
        "phase1_epochs": 500,
        "phase2_epochs": 3000,
        "lambda_data": 1.0,
        "lambda_ic": 1.0,
        "lambda_phys_start": 0.01,
        "lambda_phys_end": 1.0,
        "lambda_reg": 0.01,
        "causal_epsilon": 1.0,
    }

    def __init__(
        self,
        model,
        struct_params,
        ground_acc_fn,
        t_data,
        a_measured,
        floor_indices,
        t_colloc,
        m_prior,
        k_prior,
        config=None,
    ):
        self.model = model
        self.struct_params = struct_params
        self.ground_acc_fn = ground_acc_fn
        self.t_data = t_data
        self.a_measured = a_measured
        self.floor_indices = floor_indices
        self.t_colloc = t_colloc
        self.m_prior = m_prior
        self.k_prior = k_prior

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self.optimizer_net = torch.optim.Adam(
            model.parameters(), lr=self.config["lr_net"]
        )
        self.optimizer_params = torch.optim.Adam(
            struct_params.parameters(), lr=self.config["lr_params"]
        )

        self.history = {
            "loss_total": [],
            "loss_data": [],
            "loss_physics": [],
            "loss_ic": [],
            "loss_reg": [],
            "masses": [],
            "stiffnesses": [],
            "xi": [],
        }

    def _log_params(self):
        """Record current parameter values."""
        with torch.no_grad():
            self.history["masses"].append(self.struct_params.m.detach().cpu().numpy().copy())
            self.history["stiffnesses"].append(self.struct_params.k.detach().cpu().numpy().copy())
            self.history["xi"].append(self.struct_params.xi.detach().cpu().item())

    def train_phase1(self):
        """Phase 1: warm-up on data + IC loss only."""
        from .losses import data_loss, ic_loss

        cfg = self.config
        for epoch in range(cfg["phase1_epochs"]):
            self.optimizer_net.zero_grad()
            self.optimizer_params.zero_grad()

            self.t_data.requires_grad_(True)
            u_pred = self.model(self.t_data)

            l_data = data_loss(
                u_pred, self.t_data, self.a_measured,
                self.floor_indices, self.struct_params, self.ground_acc_fn,
            )
            l_ic = ic_loss(self.model, self.struct_params)

            loss = cfg["lambda_data"] * l_data + cfg["lambda_ic"] * l_ic

            loss.backward()
            self.optimizer_net.step()
            self.optimizer_params.step()

            self.history["loss_total"].append(loss.item())
            self.history["loss_data"].append(l_data.item())
            self.history["loss_physics"].append(0.0)
            self.history["loss_ic"].append(l_ic.item())
            self.history["loss_reg"].append(0.0)
            self._log_params()

    def train_phase2(self):
        """Phase 2: full physics-constrained training with curriculum."""
        from .losses import data_loss, physics_loss, ic_loss, reg_loss

        cfg = self.config
        lambda_phys_schedule = np.logspace(
            np.log10(cfg["lambda_phys_start"]),
            np.log10(cfg["lambda_phys_end"]),
            cfg["phase2_epochs"],
        )

        for epoch in range(cfg["phase2_epochs"]):
            self.optimizer_net.zero_grad()
            self.optimizer_params.zero_grad()

            lp = float(lambda_phys_schedule[epoch])

            # Data loss
            self.t_data.requires_grad_(True)
            u_data = self.model(self.t_data)
            l_data = data_loss(
                u_data, self.t_data, self.a_measured,
                self.floor_indices, self.struct_params, self.ground_acc_fn,
            )

            # Physics loss
            self.t_colloc.requires_grad_(True)
            u_colloc = self.model(self.t_colloc)
            l_phys = physics_loss(
                self.t_colloc, u_colloc, self.struct_params, self.ground_acc_fn,
            )

            # IC and regularization
            l_ic = ic_loss(self.model, self.struct_params)
            l_reg = reg_loss(self.struct_params, self.m_prior, self.k_prior)

            loss = (
                cfg["lambda_data"] * l_data
                + lp * l_phys
                + cfg["lambda_ic"] * l_ic
                + cfg["lambda_reg"] * l_reg
            )

            loss.backward()
            self.optimizer_net.step()
            self.optimizer_params.step()

            self.history["loss_total"].append(loss.item())
            self.history["loss_data"].append(l_data.item())
            self.history["loss_physics"].append(l_phys.item())
            self.history["loss_ic"].append(l_ic.item())
            self.history["loss_reg"].append(l_reg.item())
            self._log_params()

    def train(self):
        """Run the full two-phase training protocol."""
        self.train_phase1()
        self.train_phase2()
        return self.history
