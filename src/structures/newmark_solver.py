"""
Newmark-beta time integration for linear MDOF systems.

Uses constant average acceleration (gamma=0.5, beta=0.25) which is
unconditionally stable for linear systems.
"""

import numpy as np
from scipy.linalg import solve


class NewmarkSolver:
    """Newmark constant average acceleration solver.

    Parameters
    ----------
    building : ShearBuilding
        Structural model providing M, K, C matrices.
    dt : float
        Time step in seconds.
    gamma : float
        Newmark gamma parameter (default 0.5).
    beta : float
        Newmark beta parameter (default 0.25).
    """

    def __init__(self, building, dt, gamma=0.5, beta=0.25):
        self.building = building
        self.dt = dt
        self.gamma = gamma
        self.beta = beta

        self.M = building.M
        self.K = building.K
        self.C = building.C
        self.n = building.n_floors

        # Precompute effective stiffness matrix
        self.K_eff = (
            self.K
            + self.gamma / (self.beta * dt) * self.C
            + 1.0 / (self.beta * dt**2) * self.M
        )

    def solve(self, ground_acc):
        """Integrate the equations of motion.

        Parameters
        ----------
        ground_acc : ndarray, shape (n_steps,)
            Ground acceleration time history.

        Returns
        -------
        u : ndarray, shape (n_steps, n_floors)
            Relative displacement time history.
        v : ndarray, shape (n_steps, n_floors)
            Relative velocity time history.
        a : ndarray, shape (n_steps, n_floors)
            Relative acceleration time history.
        a_abs : ndarray, shape (n_steps, n_floors)
            Absolute acceleration time history.
        """
        n_steps = len(ground_acc)
        n = self.n
        dt = self.dt
        gamma = self.gamma
        beta = self.beta

        u = np.zeros((n_steps, n))
        v = np.zeros((n_steps, n))
        a = np.zeros((n_steps, n))

        # Influence vector
        iota = np.ones(n)

        # Initial acceleration: M*a0 = -M*iota*ag0 - C*v0 - K*u0
        f0 = -self.M @ iota * ground_acc[0]
        a[0] = solve(self.M, f0 - self.C @ v[0] - self.K @ u[0])

        for i in range(n_steps - 1):
            # Effective force at step i+1
            f_ext = -self.M @ iota * ground_acc[i + 1]

            f_eff = (
                f_ext
                + self.M @ (
                    1.0 / (beta * dt**2) * u[i]
                    + 1.0 / (beta * dt) * v[i]
                    + (1.0 / (2 * beta) - 1.0) * a[i]
                )
                + self.C @ (
                    gamma / (beta * dt) * u[i]
                    + (gamma / beta - 1.0) * v[i]
                    + dt * (gamma / (2 * beta) - 1.0) * a[i]
                )
            )

            u[i + 1] = solve(self.K_eff, f_eff)

            # Update acceleration and velocity
            a[i + 1] = (
                1.0 / (beta * dt**2) * (u[i + 1] - u[i])
                - 1.0 / (beta * dt) * v[i]
                - (1.0 / (2 * beta) - 1.0) * a[i]
            )
            v[i + 1] = (
                v[i]
                + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1])
            )

        # Absolute acceleration = relative acc + ground acc
        a_abs = a + ground_acc[:, np.newaxis] * iota[np.newaxis, :]

        return u, v, a, a_abs
