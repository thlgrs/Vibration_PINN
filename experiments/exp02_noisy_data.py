"""
Experiment 02: Noise sensitivity study.

Runs PINN-SID with varying noise levels: 0%, 5%, 10%, 15%.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # TODO: Implement noise sensitivity experiment
    # - Generate data with noise levels [0%, 5%, 10%, 15%]
    # - Run identification for each noise level (10 trials each)
    # - Report mean +/- std of identified parameters
    raise NotImplementedError("Experiment 02 not yet implemented.")


if __name__ == "__main__":
    main()
