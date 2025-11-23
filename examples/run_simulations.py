"""
Unified simulation and analysis runner for scbidi.

This script performs three main tasks:
1. Runs example simulations for all scenarios and prints summary statistics.
2. Runs a comprehensive benchmark to validate statistical properties (FDR, Power, etc.).
3. Generates publication-quality figures based on the simulation and benchmark results.

Usage:
    python examples/run_simulations.py
"""

from __future__ import annotations

from typing import List, Dict

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from scbidi.metrics import (
    bidirectional_association,
    threshold_high_low,
)
from scbidi.simulation import (
    SimulationConfig,
    generate_association_scenarios,
    simulate_scenarios,
    generate_spatial_data,
    SCENARIOS,
)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def summarise_result(record: Dict[str, object]) -> str:
    """Format a single simulation result record as a string."""
    association = record["associations"][0]
    corr = record.get("correlation", float("nan"))
    return (
        f"rep {record['replicate']:02d} | Corr={corr:.3f} | "
        f"D_A|B={association.D_A_given_B:.3f} p={association.p_A_given_B:.3f} | "
        f"D_B|A={association.D_B_given_A:.3f} p={association.p_B_given_A:.3f} | "
        f"S={association.asymmetry:.3f}"
    )


def compute_baselines(
    expr_a: np.ndarray, expr_b: np.ndarray, q: float = 0.3
) -> Dict[str, float]:
    """Compute baseline metrics for a gene pair (Pearson, Spearman, Simple DE)."""
    pearson_r, _ = pearsonr(expr_a, expr_b)
    spearman_r, _ = spearmanr(expr_a, expr_b)

    thres_a = threshold_high_low(expr_a, q=q)
    thres_b = threshold_high_low(expr_b, q=q)

    if thres_a.n_high < 5 or (thres_a.n_cells - thres_a.n_high) < 5:
        de_b_given_a = np.nan
    else:
        b_given_a_high = expr_b[thres_a.labels == 1]
        b_given_a_low = expr_b[thres_a.labels == 0]
        de_b_given_a = np.mean(b_given_a_high) - np.mean(b_given_a_low)

    if thres_b.n_high < 5 or (thres_b.n_cells - thres_b.n_high) < 5:
        de_a_given_b = np.nan
    else:
        a_given_b_high = expr_a[thres_b.labels == 1]
        a_given_b_low = expr_a[thres_b.labels == 0]
        de_a_given_b = np.mean(a_given_b_high) - np.mean(a_given_b_low)

    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "de_a_given_b": de_a_given_b,
        "de_b_given_a": de_b_given_a,
        "de_asymmetry": (
            abs(de_a_given_b) - abs(de_b_given_a)
            if not np.isnan(de_a_given_b) and not np.isnan(de_b_given_a)
            else np.nan
        ),
    }


def run_benchmark(
    n_replicates: int = 200,
    scenarios: List[str] = None,
    n_permutations: int = 200,
    q: float = 0.3,
    weight_zero: float = 0.5,
    min_nonzero: int = 10,
) -> pd.DataFrame:
    """Run simulations for multiple scenarios to gather benchmark statistics."""
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results = []
    rng = np.random.default_rng(42)
    config = SimulationConfig(n_cells=1000)

    print(f"Starting benchmark with {n_replicates} replicates per scenario...")

    for scenario in scenarios:
        print(f"  Running scenario: {scenario}")
        for i in range(n_replicates):
            seed = rng.integers(0, 1e9)
            expr_a, expr_b = generate_association_scenarios(
                scenario, config=config, random_state=seed
            )

            clusters = np.zeros(len(expr_a), dtype=int)
            res_list = bidirectional_association(
                expr_a,
                expr_b,
                clusters,
                q=q,
                n_permutations=n_permutations,
                weight_zero=weight_zero,
                min_nonzero=min_nonzero,
                random_state=seed,
            )
            res = res_list[0]

            baselines = compute_baselines(expr_a, expr_b, q=q)

            row = {
                "scenario": scenario,
                "replicate": i,
                "D_A_given_B": res.D_A_given_B,
                "p_A_given_B": res.p_A_given_B,
                "D_B_given_A": res.D_B_given_A,
                "p_B_given_A": res.p_B_given_A,
                "asymmetry": res.asymmetry,
                "p_asymmetry": res.p_asymmetry,
                **baselines,
            }
            results.append(row)

    return pd.DataFrame(results)


def plot_fig1_cdf_curves() -> None:
    """Figure 1: Simulated data examples showing A|B and B|A conditional distributions."""
    print("Generating Figure 1: CDF Curves...")

    config = SimulationConfig(n_cells=2000, gamma=2.0)
    expr_a, expr_b = generate_association_scenarios("a_to_b", config, random_state=42)

    res_a = threshold_high_low(expr_a, q=0.3)
    res_b = threshold_high_low(expr_b, q=0.3)

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    b_given_a_high = expr_b[res_a.labels == 1]
    b_given_a_low = expr_b[res_a.labels == 0]

    sns.ecdfplot(
        b_given_a_low, ax=axes[0], label="A-low cells", color="gray", linestyle="--"
    )
    sns.ecdfplot(
        b_given_a_high, ax=axes[0], label="A-high cells", color="blue", linewidth=2
    )
    axes[0].set_title("Does B increase when A is high?")
    axes[0].set_xlabel("B expression")
    axes[0].set_ylabel("Fraction of cells")
    axes[0].legend(title="Cell group")

    a_given_b_high = expr_a[res_b.labels == 1]
    a_given_b_low = expr_a[res_b.labels == 0]

    sns.ecdfplot(
        a_given_b_low, ax=axes[1], label="B-low cells", color="gray", linestyle="--"
    )
    sns.ecdfplot(
        a_given_b_high, ax=axes[1], label="B-high cells", color="red", linewidth=2
    )
    axes[1].set_title("Does A increase when B is high?")
    axes[1].set_xlabel("A expression")
    axes[1].set_ylabel("Fraction of cells")
    axes[1].legend(title="Cell group")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_cdf_curves.png"))
    plt.close()


def plot_fig2_scatter_scenarios() -> None:
    """Figure 2: Scatter plots of simulated expression values for all scenarios."""
    print("Generating Figure 2: Scatter Plots...")

    titles = {
        "null": "Null (Independent)",
        "shared_driver": "Shared Driver",
        "a_to_b": "Causal A -> B",
        "b_to_a": "Causal B -> A",
        "confounded_a_to_b": "Confounded A -> B",
        "confounded_b_to_a": "Confounded B -> A",
    }

    _fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, scenario in enumerate(SCENARIOS):
        config = SimulationConfig(n_cells=1000, gamma=1.5)
        expr_a, expr_b = generate_association_scenarios(
            scenario, config, random_state=42
        )

        clusters = np.zeros_like(expr_a, dtype=int)
        res = bidirectional_association(expr_a, expr_b, clusters, n_permutations=0)[0]

        sns.scatterplot(x=expr_a, y=expr_b, ax=axes[i], alpha=0.3, s=10, color="black")
        sns.kdeplot(x=expr_a, y=expr_b, ax=axes[i], levels=5, color="blue", alpha=0.5)

        axes[i].set_title(
            "{}\nD(A|B)={:.2f}, D(B|A)={:.2f}\nS={:.2f}".format(
                titles[scenario], res.D_A_given_B, res.D_B_given_A, res.asymmetry
            )
        )
        axes[i].set_xlabel("Gene A Expression")
        axes[i].set_ylabel("Gene B Expression")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_scatter_scenarios.png"))
    plt.close()


def plot_fig3_benchmark(df: pd.DataFrame) -> None:
    """Figure 3: Benchmark Performance (Distributions, FDR, Power)."""
    print("Generating Figure 3: Benchmark Performance...")

    g = sns.FacetGrid(df, col="scenario", col_wrap=3, sharex=False, sharey=False)
    g.map_dataframe(
        sns.histplot, x="D_A_given_B", color="blue", alpha=0.5, label="D(A|B)", kde=True
    )
    g.map_dataframe(
        sns.histplot, x="D_B_given_A", color="red", alpha=0.5, label="D(B|A)", kde=True
    )
    g.add_legend()
    g.set_titles("{col_name}")
    g.fig.suptitle("Distributions of Conditional Distances (D)", y=1.02)
    g.savefig(os.path.join(OUTPUT_DIR, "fig3a_dist_D.png"))
    plt.close()

    pvals = np.concatenate([df["p_A_given_B"].values, df["p_B_given_A"].values])
    df["reject_A_given_B"] = multipletests(
        df["p_A_given_B"], alpha=0.05, method="fdr_bh"
    )[0]
    df["reject_B_given_A"] = multipletests(
        df["p_B_given_A"], alpha=0.05, method="fdr_bh"
    )[0]

    power = df.groupby("scenario")[["reject_A_given_B", "reject_B_given_A"]].mean()

    _ax = power.plot(kind="bar", figsize=(10, 6), rot=45)
    plt.title("Power (Rejection Rate at FDR 0.05)")
    plt.ylabel("Fraction of Replicates Rejected")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3b_power.png"))
    plt.close()


def plot_fig4_spatial() -> None:
    """Figure 4: Spatial Analysis."""
    print("Generating Figure 4: Spatial Scenarios...")

    grey_red = LinearSegmentedColormap.from_list("grey_red", ["lightgrey", "red"])
    grey_blue = LinearSegmentedColormap.from_list("grey_blue", ["lightgrey", "blue"])

    n_scenarios = len(SCENARIOS)
    _fig, axes = plt.subplots(n_scenarios, 4, figsize=(20, 4 * n_scenarios))

    titles = {
        "null": "Null",
        "shared_driver": "Shared Driver",
        "a_to_b": "Causal A -> B",
        "b_to_a": "Causal B -> A",
        "confounded_a_to_b": "Confounded A -> B",
        "confounded_b_to_a": "Confounded B -> A",
    }

    rng = np.random.default_rng(42)

    for i, scenario in enumerate(SCENARIOS):
        embedding, expr_a, expr_b = generate_spatial_data(
            scenario, n_points=2000, rng=rng
        )

        clusters = np.zeros(len(expr_a), dtype=int)
        res = bidirectional_association(
            expr_a, expr_b, clusters, q=0.2, n_permutations=200, embedding=embedding
        )[0]

        sc1 = axes[i, 0].scatter(
            embedding[:, 0], embedding[:, 1], c=expr_a, cmap=grey_red, s=5
        )
        axes[i, 0].set_title("{}\nGene A".format(titles[scenario]))
        axes[i, 0].set_aspect("equal")
        axes[i, 0].axis("off")
        plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)

        sc2 = axes[i, 1].scatter(
            embedding[:, 0], embedding[:, 1], c=expr_b, cmap=grey_blue, s=5
        )
        axes[i, 1].set_title("{}\nGene B".format(titles[scenario]))
        axes[i, 1].set_aspect("equal")
        axes[i, 1].axis("off")
        plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        r_val = np.corrcoef(expr_a, expr_b)[0, 1]
        axes[i, 2].scatter(expr_a, expr_b, alpha=0.1, s=5, c="k")
        axes[i, 2].set_title(f"Correlation\nR = {r_val:.3f}")
        axes[i, 2].set_xlabel("Gene A")
        axes[i, 2].set_ylabel("Gene B")

        s_score = res.asymmetry
        axes[i, 3].axis("off")
        color = "purple" if s_score > 0 else "orange"
        if abs(s_score) < 0.005:
            color = "black"

        axes[i, 3].text(
            0.5,
            0.5,
            f"S = {s_score:.3f}\n\np(A|B)={res.p_A_given_B:.3f}\np(B|A)={res.p_B_given_A:.3f}",
            ha="center",
            va="center",
            fontsize=14,
            color=color,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.5"),
        )
        axes[i, 3].set_title("Metrics")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_spatial.png"))
    plt.close()


def plot_fig5_sensitivity() -> None:
    """Figure 5: Parameter Sensitivity."""
    print("Generating Figure 5: Parameter Sensitivity...")

    scenario = "a_to_b"
    n_reps = 50

    q_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    results_q = []
    for q in q_values:
        df = run_benchmark(
            n_replicates=n_reps, scenarios=[scenario], q=q, n_permutations=0
        )
        df["param_value"] = q
        results_q.append(df)

    df_q = pd.concat(results_q)

    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(
        data=df_q,
        x="param_value",
        y="D_A_given_B",
        label="D(A|B)",
        marker="o",
        ax=axes[0],
    )
    sns.lineplot(
        data=df_q,
        x="param_value",
        y="D_B_given_A",
        label="D(B|A)",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Sensitivity to Quantile Threshold (q)")
    axes[0].set_xlabel("q (quantile)")
    axes[0].set_ylabel("Distance Metric")

    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results_alpha = []
    rng = np.random.default_rng(42)

    for alpha in alpha_values:
        config = SimulationConfig(n_cells=1000, alpha_a=alpha, alpha_b=alpha)
        for _ in range(n_reps):
            seed = rng.integers(0, 1e9)
            expr_a, expr_b = generate_association_scenarios(
                scenario, config=config, random_state=seed
            )
            clusters = np.zeros(len(expr_a), dtype=int)
            res = bidirectional_association(
                expr_a, expr_b, clusters, n_permutations=0, random_state=seed
            )[0]

            results_alpha.append(
                {
                    "param_value": alpha,
                    "D_A_given_B": res.D_A_given_B,
                    "D_B_given_A": res.D_B_given_A,
                }
            )

    df_alpha = pd.DataFrame(results_alpha)

    sns.lineplot(
        data=df_alpha,
        x="param_value",
        y="D_A_given_B",
        label="D(A|B)",
        marker="o",
        ax=axes[1],
    )
    sns.lineplot(
        data=df_alpha,
        x="param_value",
        y="D_B_given_A",
        label="D(B|A)",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title(
        "Sensitivity to Mean Expression (Alpha)\n(Lower Alpha = More Sparsity)"
    )
    axes[1].set_xlabel("Alpha (Log Mean Expression)")
    axes[1].set_ylabel("Distance Metric")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_sensitivity.png"))
    plt.close()


def main() -> None:
    print("=== 1. Running Example Simulations ===")
    rng = np.random.default_rng(123)

    for scenario in SCENARIOS:
        print(f"\n--- Scenario: {scenario} ---")
        results = simulate_scenarios(
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

    print("\n=== 2. Running Benchmark (this may take a moment) ===")
    df_benchmark = run_benchmark(n_replicates=100, n_permutations=100)
    output_file = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    df_benchmark.to_csv(output_file, index=False)
    print(f"Benchmark results saved to {output_file}")

    print("\n=== 3. Generating Publication Figures ===")
    plot_fig1_cdf_curves()
    plot_fig2_scatter_scenarios()
    plot_fig3_benchmark(df_benchmark)
    plot_fig4_spatial()
    plot_fig5_sensitivity()

    print(f"All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
