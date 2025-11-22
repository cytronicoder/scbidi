"""Example runner for BiDEA simulations.

This script reproduces the four scenarios described in the specification
and prints summary distances and asymmetry for each replicate.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from bidea.simulation import simulate_bidea_scenarios


SCENARIOS = ["null", "shared_driver", "a_to_b", "b_to_a"]


def summarise_result(record: Dict[str, object]) -> str:
    association = record["associations"][0]
    return (
        f"rep {record['replicate']:02d} | D_A|B={association.D_A_given_B:.3f} "
        f"p={association.p_A_given_B:.3f} | D_B|A={association.D_B_given_A:.3f} "
        f"p={association.p_B_given_A:.3f} | S={association.asymmetry:.3f}"
    )


def main() -> None:
    rng = np.random.default_rng(123)
    for scenario in SCENARIOS:
        print(f"\n--- Scenario: {scenario} ---")
        results = simulate_bidea_scenarios(
            scenario,
            n_replicates=5,
            random_state=int(rng.integers(0, 1e9)),
            n_permutations=200,
            q=0.3,
            n_min=80,
            min_high=30,
            weight_zero=0.5,
        )
        for record in results:
            print(summarise_result(record))


if __name__ == "__main__":
    main()
