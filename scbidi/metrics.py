"""Core metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .utils import ThresholdResult, threshold_high_low, validate_inputs
from .null_models import build_local_groups, permute_indices_locally


@dataclass
class TwoPartDistance:
    """
    Distance components between two distributions.

    d_zero: Difference in fraction of zeros (binary on/off).
    d_continuous: KS distance between non-zero values.
    total: Weighted combination of d_zero and d_continuous.
    """

    d_zero: float
    d_continuous: float
    total: float


@dataclass
class PairwiseAssociationResult:
    """
    Summary of how two genes co-vary within one cell group (cluster).

    This object is meant to be readable by biologists. All distances lie
    in [0, 1], where 0 means 'no detectable difference' and values closer
    to 1 mean 'very different distributions'.

    Attributes
    ----------
    cluster : int
        Cluster identifier (e.g. 0, 1, 2) or annotation index.
    n_cells : int
        Number of cells in this cluster.
    n_high_a : int
        Number of A-high cells used in B|A test.
    n_high_b : int
        Number of B-high cells used in A|B test.

    D_A_given_B : Optional[float]
        Total two-part distance for "A when B is high versus B is low".
    D_A_given_B_zero : Optional[float]
        Difference in zero-fraction of A between B-high and B-low.
    D_A_given_B_cont : Optional[float]
        KS distance of non-zero A values between B-high and B-low.
    p_A_given_B : Optional[float]
        Permutation p-value for D_A_given_B.

    D_B_given_A : Optional[float]
        Total two-part distance for "B when A is high versus A is low".
    D_B_given_A_zero : Optional[float]
        Difference in zero-fraction of B between A-high and A-low.
    D_B_given_A_cont : Optional[float]
        KS distance of non-zero B values between A-high and A-low.
    p_B_given_A : Optional[float]
        Permutation p-value for D_B_given_A.

    asymmetry : Optional[float]
        Difference D_A_given_B - D_B_given_A.
        Positive values mean 'A|B' contrast is stronger than 'B|A';
        negative values mean the reverse.
        This is a *pattern* summary, not a causal effect size.

    p_asymmetry : Optional[float]
        Permutation p-value for the asymmetry (two-sided).
    notes : Optional[str]
        Optional human-readable notes (e.g. reasons for skipping).
    """

    cluster: int
    n_cells: int
    n_high_a: int
    n_high_b: int
    D_A_given_B: Optional[float]
    D_A_given_B_zero: Optional[float]
    D_A_given_B_cont: Optional[float]
    p_A_given_B: Optional[float]
    D_B_given_A: Optional[float]
    D_B_given_A_zero: Optional[float]
    D_B_given_A_cont: Optional[float]
    p_B_given_A: Optional[float]
    asymmetry: Optional[float]
    p_asymmetry: Optional[float]
    notes: Optional[str] = None


def _ks_distance_nonzero(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute the KS distance between two positive-valued samples."""
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
    min_nonzero: int = 10,
) -> TwoPartDistance:
    """
    Quantify how differently a gene is expressed in 'high' versus 'low' cells.

    The distance combines:
      (i) a difference in the fraction of zero-expression cells, and
      (ii) a Kolmogorov–Smirnov distance on the non-zero values.

    D = w0 * |Δ zero_fraction| + (1 - w0) * KS(nonzero)

    Parameters
    ----------
    expression
        1D expression values for a single gene (e.g. log-normalized counts).
    high_labels
        1D 0/1 array: 1 marks the 'high' group (e.g. B-high), 0 the 'low' group.
    weight_zero
        Weight in [0, 1] given to the difference in zero fractions.
        A value of 0.5 gives equal importance to on/off and magnitude.
    min_nonzero
        Minimum number of non-zero observations required in each group
        to compute the KS component. If this is not met, the KS term is
        omitted and the distance is based on zero fractions only.

    Returns
    -------
    TwoPartDistance
        Object containing d_zero, d_continuous, and total distance.
    """
    values = np.asarray(expression, dtype=float)
    labels = np.asarray(high_labels, dtype=int)

    if labels.shape != values.shape:
        raise ValueError("expression and high_labels must have the same shape")

    fg_mask = labels == 1
    bg_mask = labels == 0
    if fg_mask.sum() == 0 or bg_mask.sum() == 0:
        return TwoPartDistance(0.0, 0.0, 0.0)

    fg_values = values[fg_mask]
    bg_values = values[bg_mask]

    fg_zero = np.mean(fg_values == 0)
    bg_zero = np.mean(bg_values == 0)
    d_zero = abs(fg_zero - bg_zero)

    fg_positive = fg_values[fg_values > 0]
    bg_positive = bg_values[bg_values > 0]

    if fg_positive.size >= min_nonzero and bg_positive.size >= min_nonzero:
        d_cont = _ks_distance_nonzero(fg_positive, bg_positive)
    else:
        d_cont = 0.0

    w0 = float(np.clip(weight_zero, 0.0, 1.0))
    total = w0 * d_zero + (1.0 - w0) * d_cont
    return TwoPartDistance(d_zero, d_cont, total)


def compute_Ds_for_thresholds(
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    thres_a: ThresholdResult,
    thres_b: ThresholdResult,
    weight_zero: float,
    min_nonzero: int,
) -> Tuple[TwoPartDistance, TwoPartDistance, float]:
    """Compute D_A|B, D_B|A, and S for given thresholds."""
    d_ab = compute_two_part_distance(
        cluster_a, thres_b.labels, weight_zero, min_nonzero
    )
    d_ba = compute_two_part_distance(
        cluster_b, thres_a.labels, weight_zero, min_nonzero
    )
    s = d_ab.total - d_ba.total
    return d_ab, d_ba, s


def _permute_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a permuted copy of binary labels."""
    perm = rng.permutation(labels.shape[0])
    return labels[perm]


def _directional_single_cluster(
    cluster_id: int,
    idx: np.ndarray,
    expr_a: np.ndarray,
    expr_b: np.ndarray,
    q: float,
    n_min: int,
    min_high: int,
    weight_zero: float,
    min_nonzero: int,
    n_permutations: int,
    rng: np.random.Generator,
    embedding: Optional[np.ndarray] = None,
) -> PairwiseAssociationResult:
    cluster_a = expr_a[idx]
    cluster_b = expr_b[idx]

    local_groups = None
    if embedding is not None:
        cluster_emb = embedding[idx]
        local_groups = build_local_groups(cluster_emb, n_neighbors=20, rng=rng)

    thres_b = threshold_high_low(cluster_b, q=q, n_min=n_min, min_high=min_high)
    thres_a = threshold_high_low(cluster_a, q=q, n_min=n_min, min_high=min_high)

    if thres_b.skipped or thres_a.skipped:
        return PairwiseAssociationResult(
            cluster=cluster_id,
            n_cells=idx.size,
            n_high_a=thres_a.n_high,
            n_high_b=thres_b.n_high,
            D_A_given_B=None,
            D_A_given_B_zero=None,
            D_A_given_B_cont=None,
            p_A_given_B=None,
            D_B_given_A=None,
            D_B_given_A_zero=None,
            D_B_given_A_cont=None,
            p_B_given_A=None,
            asymmetry=None,
            p_asymmetry=None,
            notes=f"Skipped: {thres_a.reason or thres_b.reason}",
        )

    obs_d_ab, obs_d_ba, obs_s = compute_Ds_for_thresholds(
        cluster_a, cluster_b, thres_a, thres_b, weight_zero, min_nonzero
    )

    exceed_ab = 0
    exceed_ba = 0
    exceed_s_abs = 0

    indices = np.arange(cluster_a.size)

    for _ in range(n_permutations):
        if local_groups is not None:
            perm_idx = permute_indices_locally(indices, local_groups, rng)
            perm_labels_b = thres_b.labels[perm_idx]
            perm_labels_a = thres_a.labels[perm_idx]
        else:
            perm_labels_b = _permute_labels(thres_b.labels, rng)
            perm_labels_a = _permute_labels(thres_a.labels, rng)

        perm_d_ab = compute_two_part_distance(
            cluster_a, perm_labels_b, weight_zero, min_nonzero
        )

        perm_d_ba = compute_two_part_distance(
            cluster_b, perm_labels_a, weight_zero, min_nonzero
        )

        perm_s = perm_d_ab.total - perm_d_ba.total

        if perm_d_ab.total >= obs_d_ab.total:
            exceed_ab += 1
        if perm_d_ba.total >= obs_d_ba.total:
            exceed_ba += 1
        if abs(perm_s) >= abs(obs_s):
            exceed_s_abs += 1

    p_ab = (1 + exceed_ab) / (1 + n_permutations)
    p_ba = (1 + exceed_ba) / (1 + n_permutations)
    p_s = (1 + exceed_s_abs) / (1 + n_permutations)

    return PairwiseAssociationResult(
        cluster=cluster_id,
        n_cells=idx.size,
        n_high_a=thres_a.n_high,
        n_high_b=thres_b.n_high,
        D_A_given_B=obs_d_ab.total,
        D_A_given_B_zero=obs_d_ab.d_zero,
        D_A_given_B_cont=obs_d_ab.d_continuous,
        p_A_given_B=p_ab,
        D_B_given_A=obs_d_ba.total,
        D_B_given_A_zero=obs_d_ba.d_zero,
        D_B_given_A_cont=obs_d_ba.d_continuous,
        p_B_given_A=p_ba,
        asymmetry=obs_s,
        p_asymmetry=p_s,
        notes=None,
    )


def bidirectional_association(
    expression_a: Sequence[float],
    expression_b: Sequence[float],
    clusters: Sequence[int],
    q: float = 0.3,
    n_min: int = 80,
    min_high: int = 30,
    weight_zero: float = 0.5,
    min_nonzero: int = 10,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    embedding: Optional[np.ndarray] = None,
) -> List[PairwiseAssociationResult]:
    """
    Compute bidirectional conditional association metrics for each cluster.

    Parameters
    ----------
    expression_a, expression_b
        Expression values.
    clusters
        Cluster assignments.
    q
        Quantile to define high expression (default 0.3).
    n_min
        Minimum cells required in a cluster.
    min_high
        Minimum high cells required.
    weight_zero
        Weight for the zero-mass component.
    min_nonzero
        Minimum non-zero values for KS distance.
    n_permutations
        Number of permutations.
    random_state
        Seed.
    embedding
        (N, D) array for local permutations.
    """
    expr_a = np.asarray(expression_a, dtype=float)
    expr_b = np.asarray(expression_b, dtype=float)
    cluster_labels = np.asarray(clusters, dtype=int)

    validate_inputs(expr_a, expr_b, cluster_labels)

    if embedding is not None:
        embedding = np.asarray(embedding)
        if embedding.shape[0] != expr_a.shape[0]:
            raise ValueError("Embedding must have same number of rows as expression")

    rng = np.random.default_rng(random_state)

    unique_clusters = np.unique(cluster_labels)
    results = []

    for cluster_id in unique_clusters:
        idx = np.where(cluster_labels == cluster_id)[0]
        res = _directional_single_cluster(
            int(cluster_id),
            idx,
            expr_a,
            expr_b,
            q,
            n_min,
            min_high,
            weight_zero,
            min_nonzero,
            n_permutations,
            rng,
            embedding=embedding,
        )
        results.append(res)

    return results


def _qualitative_strength(D: float, p: Optional[float]) -> str:
    if D is None or p is None:
        return "not tested"
    if p >= 0.05:
        return "no statistically clear difference"
    if D < 0.1:
        return "very small but statistically detectable difference"
    if D < 0.25:
        return "weak but statistically clear difference"
    if D < 0.5:
        return "moderate difference"
    return "strong difference"


def summarize_gene_pair(
    expr_a: Sequence[float],
    expr_b: Sequence[float],
    clusters: Sequence[int],
    gene_a_name: str = "GeneA",
    gene_b_name: str = "GeneB",
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    fdr_correction: bool = True,
) -> pd.DataFrame:
    """
    Analyze how two genes co-vary in each cluster and summarize in a table.

    This is the main entry point for biologists.

    For each cluster, the table reports:
      - how many cells were analyzed,
      - how often each gene was 'high',
      - how strongly A differs when B is high vs low (A|B),
      - how strongly B differs when A is high vs low (B|A),
      - statistical significance (permutation p-values),
      - FDR-corrected p-values (if fdr_correction=True),
      - an easy-to-read qualitative interpretation.

    Parameters
    ----------
    expr_a, expr_b : array-like
        Expression values for genes A and B.
    clusters : array-like
        Cluster assignments.
    gene_a_name, gene_b_name : str
        Names for the genes (for interpretation).
    n_permutations : int
        Number of permutations for p-value calculation.
    random_state : int, optional
        Random seed.
    fdr_correction : bool, default True
        Whether to apply Benjamini-Hochberg FDR correction to p-values.

    Returns
    -------
    DataFrame
        One row per cluster with columns:

        ['cluster', 'n_cells', 'n_high_A', 'n_high_B',
         'D_A_given_B', 'p_A_given_B',
         'D_B_given_A', 'p_B_given_A',
         'asymmetry', 'p_asymmetry',
         'interpretation']
        If fdr_correction=True, also includes 'p_A_given_B_fdr', 'p_B_given_A_fdr', 'p_asymmetry_fdr'.
    """
    results = bidirectional_association(
        expr_a,
        expr_b,
        clusters=clusters,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    rows = []
    for res in results:
        if res.D_A_given_B is None:
            interp = f"Cluster {res.cluster}: not enough cells or high-expressing cells to analyze."
        else:
            strength_ab = _qualitative_strength(res.D_A_given_B, res.p_A_given_B)
            strength_ba = _qualitative_strength(res.D_B_given_A, res.p_B_given_A)

            interp = (
                f"In cluster {res.cluster}, {gene_a_name} shows {strength_ab} "
                f"between {gene_b_name}-high and -low cells, and {gene_b_name} shows "
                f"{strength_ba} between {gene_a_name}-high and -low cells. "
                "These reflect association patterns only; they do NOT prove one gene regulates the other."
            )

        rows.append(
            {
                "cluster": res.cluster,
                "n_cells": res.n_cells,
                "n_high_A": res.n_high_a,
                "n_high_B": res.n_high_b,
                "D_A_given_B": res.D_A_given_B,
                "D_A_given_B_zero": res.D_A_given_B_zero,
                "D_A_given_B_cont": res.D_A_given_B_cont,
                "p_A_given_B": res.p_A_given_B,
                "D_B_given_A": res.D_B_given_A,
                "D_B_given_A_zero": res.D_B_given_A_zero,
                "D_B_given_A_cont": res.D_B_given_A_cont,
                "p_B_given_A": res.p_B_given_A,
                "asymmetry": res.asymmetry,
                "p_asymmetry": res.p_asymmetry,
                "interpretation": interp,
            }
        )

    df = pd.DataFrame(rows)

    if fdr_correction:
        p_cols = ["p_A_given_B", "p_B_given_A", "p_asymmetry"]
        for col in p_cols:
            if col in df.columns:
                valid_mask = df[col].notna()
                if valid_mask.any():
                    _, p_adj, _, _ = multipletests(
                        df.loc[valid_mask, col], alpha=0.05, method="fdr_bh"
                    )
                    df.loc[valid_mask, col + "_fdr"] = p_adj
                else:
                    df[col + "_fdr"] = np.nan

    return df
