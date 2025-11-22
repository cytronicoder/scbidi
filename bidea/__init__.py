"""BiDEA: bidirectional distributional association testing for scRNA-seq.

This package provides:
- Thresholding utilities for defining "high" vs "low" expression using
  quantile-based rules.
- A two-part distance that mixes zero-inflation and continuous
  Kolmogorov–Smirnov components.
- Permutation-based calibration and bidirectional association summaries.
- Simulation generators for negative-binomial single-cell scenarios.
"""

from .metrics import (
    compute_two_part_distance,
    directional_association,
    permutation_pvalue,
    threshold_high_low,
)
from .simulation import (
    generate_association_scenarios,
    simulate_bidea_scenarios,
)

__all__ = [
    "compute_two_part_distance",
    "directional_association",
    "permutation_pvalue",
    "threshold_high_low",
    "generate_association_scenarios",
    "simulate_bidea_scenarios",
]
