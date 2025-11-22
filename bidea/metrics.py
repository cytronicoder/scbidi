"""Core metrics and permutation calibration for BiDEA.

The implementation follows the specification from the prompt:
- Thresholding uses a fixed upper quantile within each cluster.
- A two-part distance mixes the zero-mass difference and KS distance on
  positive values.
- Permutation p-values shuffle the high/low labels within cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ThresholdResult:
    """Result of the high/low thresholding.

    Attributes
    ----------
    labels : np.ndarray
        Binary array (0/1) indicating high cells.
    threshold : float
        Quantile threshold used to define the high group.
    n_cells : int
        Number of cells used in the cluster.
    n_high : int
        Number of cells labelled as high.
    skipped : bool
        Whether the thresholding was skipped because of insufficient cells.
    reason : Optional[str]
        Explanation if skipped.
    """

    labels: np.ndarray
    threshold: float
    n_cells: int
    n_high: int
    skipped: bool
    reason: Optional[str] = None


@dataclass
class DirectionalAssociation:
    """Collection of metrics for a single cluster and gene pair."""

    cluster: int
    n_cells: int
    threshold_A: Optional[float]
    threshold_B: Optional[float]
    D_A_given_B: Optional[float]
    p_A_given_B: Optional[float]
    D_B_given_A: Optional[float]
    p_B_given_A: Optional[float]
    asymmetry: Optional[float]
    notes: Optional[str]


def threshold_high_low(
    expression: Sequence[float],
    q: float = 0.3,
    n_min: int = 80,
    min_high: int = 30,
) -> ThresholdResult:
    """Define high/low labels using a fixed upper quantile.

    The rule is cluster-specific and does not rely on any ad hoc tuning.
    The quantile is applied to the raw expression vector provided.
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


def _ks_distance_nonzero(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute the KS distance between two positive-valued samples.

    If either sample has no positive values, the continuous component is
    defined as zero so that the distance is driven solely by the zero-mass
    difference.
    """

    if values_a.size == 0 or values_b.size == 0:
        return 0.0

    a_sorted = np.sort(values_a)
    b_sorted = np.sort(values_b)
    combined = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))

    cdf_a = np.searchsorted(a_sorted, combined, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, combined, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compute_two_part_distance(
    expression: Sequence[float],
    high_labels: Sequence[int],
    weight_zero: float = 0.5,
) -> float:
    """Two-part distance D_{A|B} combining zero mass and KS on positives."""

    values = np.asarray(expression, dtype=float)
    labels = np.asarray(high_labels, dtype=int)
    if values.size != labels.size:
        raise ValueError("expression and high_labels must have the same length")

    foreground = labels == 1
    background = labels == 0
    if foreground.sum() == 0 or background.sum() == 0:
        return 0.0

    fg_values = values[foreground]
    bg_values = values[background]

    fg_zero = (fg_values == 0).mean()
    bg_zero = (bg_values == 0).mean()
    d_zero = abs(fg_zero - bg_zero)

    fg_positive = fg_values[fg_values > 0]
    bg_positive = bg_values[bg_values > 0]
    d_cont = _ks_distance_nonzero(fg_positive, bg_positive)

    weight_zero = float(np.clip(weight_zero, 0.0, 1.0))
    return weight_zero * d_zero + (1.0 - weight_zero) * d_cont


def permutation_pvalue(
    expression: Sequence[float],
    high_labels: Sequence[int],
    weight_zero: float = 0.5,
    n_permutations: int = 1000,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Compute observed distance and permutation-calibrated p-value."""

    rng = random_state or np.random.default_rng()
    observed = compute_two_part_distance(expression, high_labels, weight_zero)
    if observed == 0.0:
        return observed, 1.0

    labels = np.asarray(high_labels, dtype=int)
    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        permuted_distance = compute_two_part_distance(expression, permuted, weight_zero)
        if permuted_distance >= observed:
            exceed += 1
    p_value = (1 + exceed) / (1 + n_permutations)
    return observed, p_value


def _cluster_slices(cluster_labels: Sequence[int]) -> Dict[int, np.ndarray]:
    labels = np.asarray(cluster_labels)
    slices: Dict[int, np.ndarray] = {}
    for cluster in np.unique(labels):
        slices[int(cluster)] = np.where(labels == cluster)[0]
    return slices


def _directional_single_cluster(
    cluster_id: int,
    idx: np.ndarray,
    expr_a: np.ndarray,
    expr_b: np.ndarray,
    q: float,
    n_min: int,
    min_high: int,
    weight_zero: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> DirectionalAssociation:
    cluster_a = expr_a[idx]
    cluster_b = expr_b[idx]

    thres_b = threshold_high_low(cluster_b, q=q, n_min=n_min, min_high=min_high)
    thres_a = threshold_high_low(cluster_a, q=q, n_min=n_min, min_high=min_high)

    if thres_b.skipped or thres_a.skipped:
        note = "; ".join(
            filter(
                None,
                [
                    thres_b.reason if thres_b.skipped else None,
                    thres_a.reason if thres_a.skipped else None,
                ],
            )
        )
        return DirectionalAssociation(
            cluster=cluster_id,
            n_cells=idx.size,
            threshold_A=thres_a.threshold if not np.isnan(thres_a.threshold) else None,
            threshold_B=thres_b.threshold if not np.isnan(thres_b.threshold) else None,
            D_A_given_B=None,
            p_A_given_B=None,
            D_B_given_A=None,
            p_B_given_A=None,
            asymmetry=None,
            notes=note or None,
        )

    d_ab, p_ab = permutation_pvalue(
        cluster_a, thres_b.labels, weight_zero=weight_zero, n_permutations=n_permutations, random_state=rng
    )
    d_ba, p_ba = permutation_pvalue(
        cluster_b, thres_a.labels, weight_zero=weight_zero, n_permutations=n_permutations, random_state=rng
    )
    asym = d_ab - d_ba

    return DirectionalAssociation(
        cluster=cluster_id,
        n_cells=idx.size,
        threshold_A=thres_a.threshold,
        threshold_B=thres_b.threshold,
        D_A_given_B=d_ab,
        p_A_given_B=p_ab,
        D_B_given_A=d_ba,
        p_B_given_A=p_ba,
        asymmetry=asym,
        notes=None,
    )


def directional_association(
    expression_a: Sequence[float],
    expression_b: Sequence[float],
    clusters: Sequence[int],
    q: float = 0.3,
    n_min: int = 80,
    min_high: int = 30,
    weight_zero: float = 0.5,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> List[DirectionalAssociation]:
    """Compute BiDEA metrics for each cluster.

    Parameters are directly tied to the defaults proposed in the prompt.
    """

    expr_a = np.asarray(expression_a, dtype=float)
    expr_b = np.asarray(expression_b, dtype=float)
    if expr_a.shape != expr_b.shape:
        raise ValueError("expression_a and expression_b must have the same shape")

    cluster_labels = np.asarray(clusters)
    if cluster_labels.shape[0] != expr_a.shape[0]:
        raise ValueError("clusters must align with expression vectors")

    rng = np.random.default_rng(random_state)
    cluster_map = _cluster_slices(cluster_labels)

    results: List[DirectionalAssociation] = []
    for cluster_id, idx in cluster_map.items():
        results.append(
            _directional_single_cluster(
                cluster_id,
                idx,
                expr_a,
                expr_b,
                q=q,
                n_min=n_min,
                min_high=min_high,
                weight_zero=weight_zero,
                n_permutations=n_permutations,
                rng=rng,
            )
        )
    return results


def benjamini_hochberg(pvalues: Iterable[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR control for a sequence of p-values."""

    pvals = np.asarray(list(pvalues), dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([], dtype=bool)

    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros_like(pvals, dtype=bool)

    max_idx = np.where(passed)[0].max()
    cutoff = thresholds[max_idx]
    return pvals <= cutoff
