"""Plotting utilities."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .metrics import PairwiseAssociationResult


def plot_association_summary(
    results: List[PairwiseAssociationResult],
    title: str = "Directional Association Summary",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot a summary of directional associations.

    Displays D_A|B and D_B|A for each cluster.
    """
    data = []
    for res in results:
        if res.D_A_given_B is not None:
            data.append(
                {
                    "Cluster": res.cluster,
                    "Direction": "A|B",
                    "Distance": res.D_A_given_B,
                    "P-value": res.p_A_given_B,
                }
            )
        if res.D_B_given_A is not None:
            data.append(
                {
                    "Cluster": res.cluster,
                    "Direction": "B|A",
                    "Distance": res.D_B_given_A,
                    "P-value": res.p_B_given_A,
                }
            )

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    if not df.empty:
        sns.barplot(data=df, x="Cluster", y="Distance", hue="Direction", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Two-Part Distance")
    else:
        ax.text(0.5, 0.5, "No valid associations found", ha="center")

    return fig
