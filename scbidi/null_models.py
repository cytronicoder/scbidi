"""Null models and permutation schemes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.cluster.vq import kmeans2


def build_local_groups(
    embedding: np.ndarray,
    n_neighbors: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Cluster cells in embedding space into local groups for constrained permutations.

    Parameters
    ----------
    embedding
        (n_cells, dim) embedding coordinates (e.g. UMAP1, UMAP2).
    n_neighbors
        Approximate group size; smaller values create more, smaller groups.
    rng
        Optional Generator for k-means initialization.

    Returns
    -------
    group_ids : np.ndarray
        Integer group label for each cell (size n_cells). Cells are permuted
        within these groups.
    """
    rng = rng or np.random.default_rng()
    n_cells = embedding.shape[0]
    if n_cells <= n_neighbors:
        return np.zeros(n_cells, dtype=int)

    k = max(1, n_cells // n_neighbors)

    seed = int(rng.integers(0, 2**31 - 1))
    prev_state = np.random.get_state()
    np.random.seed(seed)
    try:
        _, group_ids = kmeans2(embedding, k, minit="points")
    finally:
        np.random.set_state(prev_state)

    return group_ids


def permute_indices_locally(
    indices: np.ndarray,
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute indices within local groups defined by group_ids.

    This preserves the local density structure in embedding space
    while breaking the specific mapping between cells and their values.
    """
    permuted = np.empty_like(indices)
    unique_groups = np.unique(group_ids)
    for gid in unique_groups:
        mask = group_ids == gid
        subset = indices[mask]
        permuted[mask] = rng.permutation(subset)
    return permuted


def global_permutation(
    indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simple global permutation of indices."""
    return rng.permutation(indices)
