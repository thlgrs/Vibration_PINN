"""
Experiment 04: Damage detection — pre/post stiffness reduction.

Simulate 30% stiffness reduction at story 2 and verify PINN-SID
can localize and quantify the damage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # TODO: Implement damage detection experiment
    # - Generate pre-damage response (nominal k2)
    # - Generate post-damage response (k2_damaged = 0.7 * k2)
    # - Identify parameters for both cases
    # - Compute stiffness reduction index: Dk_i = (k_before - k_after) / k_before
    raise NotImplementedError("Experiment 04 not yet implemented.")


if __name__ == "__main__":
    main()
