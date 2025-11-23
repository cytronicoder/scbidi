"""Utility functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ThresholdResult:
    """
    Result of splitting cells into "high" and "low" expressers for one gene.

    Attributes
    ----------
    labels : np.ndarray
        1 for "high" cells, 0 for "low" cells.
    threshold : float
        Expression cutoff used to call a cell "high".
    n_cells : int
        Number of cells in this cluster.
    n_high : int
        Number of cells labelled as "high".
    skipped : bool
        True if we decided not to analyze this cluster.
    reason : Optional[str]
        Human-readable explanation if skipped.
    """

    labels: np.ndarray
    threshold: float
    n_cells: int
    n_high: int
    skipped: bool
    reason: Optional[str] = None


def threshold_high_low(
    expression: Sequence[float],
    q: float = 0.3,
    n_min: int = 80,
    min_high: int = 30,
) -> ThresholdResult:
    """
    Split cells into "high" and "low" expressers for a given gene.

    Parameters
    ----------
    expression
        1D array of expression values for one gene within a cluster.
    q
        Fraction of cells to label as "high". q=0.3 means "top 30%".
    n_min
        Minimum number of cells required to trust this split at all.
    min_high
        Minimum number of "high" cells required.

    Returns
    -------
    ThresholdResult
        An object describing the split and whether it was usable.
    """

    values = np.asarray(expression, dtype=float)
    n_cells = values.size
    if n_cells < n_min:
        return ThresholdResult(
            labels=np.zeros_like(values, dtype=int),
            threshold=float("nan"),
            n_cells=n_cells,
            n_high=0,
            skipped=True,
            reason=f"insufficient cells (n={n_cells}, min={n_min})",
        )

    threshold = float(np.quantile(values, 1 - q))
    labels = (values > threshold).astype(int)
    n_high = int(labels.sum())
    if n_high < min_high:
        return ThresholdResult(
            labels=labels,
            threshold=threshold,
            n_cells=n_cells,
            n_high=n_high,
            skipped=True,
            reason=f"insufficient high cells (n_high={n_high}, min={min_high})",
        )

    return ThresholdResult(
        labels=labels,
        threshold=threshold,
        n_cells=n_cells,
        n_high=n_high,
        skipped=False,
        reason=None,
    )


def validate_inputs(
    expr_a: np.ndarray, expr_b: np.ndarray, clusters: np.ndarray
) -> None:
    """Validate input arrays for consistency and NaNs."""
    if not (expr_a.shape == expr_b.shape == clusters.shape):
        raise ValueError(
            "expression_a, expression_b, and clusters must have same length"
        )
    if np.isnan(expr_a).any() or np.isnan(expr_b).any():
        raise ValueError(
            "NaNs detected in expression; please handle missing values before analysis."
        )
