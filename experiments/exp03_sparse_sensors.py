"""
Experiment 03: Partial instrumentation study.

Sensor configurations:
- Full: floors 1, 2, 3
- Partial-2: floors 1, 3
- Partial-1: floor 3 only (roof)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # TODO: Implement sparse sensor experiment
    # - Run identification for each sensor configuration
    # - Compare parameter accuracy across configurations
    raise NotImplementedError("Experiment 03 not yet implemented.")


if __name__ == "__main__":
    main()
