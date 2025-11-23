### Zero-inflation in scRNA-seq

Single-cell RNA-seq data suffers from an excess of zero counts beyond what would be expected from technical noise alone. Zeros arise from 3 main sources:

1. Biological silence: Gene is truly not expressed in that cell
2. Technical dropout: mRNA was present but not captured/sequenced
3. Sampling zeros: Count is non-zero but below detection threshold

Traditional methods (Pearson correlation, t-tests) treat zeros as "low expression" and may:

- Underestimate associations when genes co-activate (0→positive transitions)
- Confuse magnitude changes with on/off switches
- Violate normality assumptions

Therefore, we use a **two-part distance** that explicitly separates:

$$
D = w_0 \cdot |\Delta P(\text{zero})| + (1 - w_0) \cdot \text{KS}(\text{nonzero})
$$

where:

- $|\Delta P(\text{zero})|$ = absolute difference in zero fractions (on/off component)
- $\text{KS}(\text{nonzero})$ = Kolmogorov-Smirnov distance on positive values (magnitude component)
- $w_0$ = weight parameter (default 0.5 for equal importance)

**Example:**

```text
Gene A in B-high cells: [0, 0, 0, 2.3, 4.1, 5.2]  → zero fraction = 50%
Gene A in B-low cells:  [0, 1.1, 2.2, 3.1, 4.0]   → zero fraction = 20%

Δ zero fraction = |0.5 - 0.2| = 0.3  (large on/off difference)
KS on nonzero = KS([2.3, 4.1, 5.2], [1.1, 2.2, 3.1, 4.0]) ≈ 0.4
Total distance = 0.5 × 0.3 + 0.5 × 0.4 = 0.35
```

### Conditional distributional comparison

We compare the full distribution of one gene conditional on another gene's expression level by constructing **foreground** and **background** groups.

**Setup:**

1. Split cells by gene B expression: **B-high** (top q%, default q=30%) vs **B-low** (bottom 70%)
2. Compare gene A's distribution in these two groups:
   - Foreground: A in B-high cells
   - Background: A in B-low cells

$H_0$: A's distribution is identical in B-high and B-low cells
$H_1$: A's distribution shifts when B is expressed at high levels

Unlike mean-difference tests (e.g., t-test), distributional tests detect:

- **Variance changes**: A becomes more variable when B is high
- **Shape changes**: A becomes bimodal when B is high
- **Zero-fraction changes**: A is more likely to be "on" when B is high

For genes A and B, the test statistic for "A given B" is:

$$
D_{A|B} = w_0 \cdot \left| P(A = 0 \mid B \in \text{high}) - P(A = 0 \mid B \in \text{low}) \right|
$$

$$
\quad\quad\quad + (1 - w_0) \cdot \sup_{x} \left| F_{A|B_{\text{high}}}(x) - F_{A|B_{\text{low}}}(x) \right|
$$

where $F_{A|B_{\text{high}}}(x)$ is the empirical CDF of non-zero A values in B-high cells.

### Permutation-based inference

We then use permutation tests instead of parametric tests (e.g., t-test, Wilcoxon) because:

1. No distributional assumptions: Works for any data shape (bimodal, heavy-tailed, zero-inflated)
2. Exact finite-sample validity: p-values are accurate even with small sample sizes
3. Custom test statistics: Two-part distance has no known null distribution

**Observed data:**

```text
B-high cells: indices [10, 23, 45, ...]  → observe D_A|B = 0.42
B-low cells:  indices [1, 5, 12, ...]
```

**Permutation procedure** (repeated `n_permutations` times, default 1000):

1. **Shuffle the "high" labels** among all cells (or within local neighborhoods if `embedding` provided)
2. Recompute $D_{A|B}^{\text{perm}}$ using shuffled labels
3. Check if $D_{A|B}^{\text{perm}} \geq D_{A|B}^{\text{obs}}$

**p-value calculation:**

$$
p = \frac{1 + \text{number of permutations where } D^{\text{perm}} \geq D^{\text{obs}}}{1 + n_{\text{permutations}}}
$$

Adding 1 to numerator and denominator ensures p > 0 (conservative adjustment).

- **p < 0.05**: The observed distributional difference is unlikely under the null (labels are exchangeable)
- **p ≥ 0.05**: Data are consistent with no association

Note that permutation p-values test association, not causation. **A significant A|B does not prove B causes A.**

### Bidirectional testing and asymmetry

We run two independent tests for each gene pair:

1. **A|B**: How does A's distribution change with B's expression level?
2. **B|A**: How does B's distribution change with A's expression level?

These are NOT symmetric:

- A|B strong, B|A weak → "A is responsive to B's state"
- B|A strong, A|B weak → "B is responsive to A's state"
- Both strong → "Mutual association (could be A→B, B→A, or A←L→B)"
- Both weak → "No clear association"

Thus, we define an **asymmetry score**:

$$
S = D_{A|B} - D_{B|A}
$$

**Properties:**

- $S > 0$: A changes more with B than B changes with A
- $S < 0$: B changes more with A than A changes with B
- $S \approx 0$: Symmetric association

**Statistical test:**

- Permutation test for $|S|$ (two-sided)
- Null: asymmetry is due to random noise
- Alternative: true directional pattern exists

❌ **Asymmetry ≠ causation**

Consider these scenarios that all produce $S > 0$:

1. Direct regulation: A → B (B responds to A)
2. Reverse regulation with confounding: B → A, but A and B share a common driver L
3. Threshold nonlinearity: A has a sigmoidal response to B, but B responds linearly to A
4. Measurement noise: B has higher technical variance than A

Thus, asymmetry detects patterns that are consistent with directed effects, but cannot distinguish causation from confounding.

In causal inference terminology:

- Observational association: $P(A|B)$ vs $P(A)$ (what we test)
- Causal effect: $P(A|\text{do}(B))$ vs $P(A)$ (requires intervention/randomization)

We provide observational evidence that may guide:

- Hypothesis generation for validation experiments
- Prioritization of gene pairs for perturbation studies
- Network inference (as one of multiple information sources)

### Cluster-aware analysis

Cell-type heterogeneity is a major confounder in scRNA-seq. If:

- Gene A is upregulated in cell type X
- Gene B is upregulated in cell type X
- You analyze pooled data across cell types

You will observe spurious association due to Simpson's paradox.

To address this, scbidi performs **within-cluster testing** by default:

1. Assign each cell to a cluster (e.g., from Louvain/Leiden clustering)
2. Perform A|B and B|A tests **independently in each cluster**
3. Report separate statistics per cluster
4. Optionally apply FDR correction across clusters

**Example:**

```python
results = bidirectional_association(expr_a, expr_b, clusters)
# Returns one PairwiseAssociationResult per unique cluster ID
```

#### When to pool vs. stratify

**Stratify (recommended):**

- Cell types have different expression programs
- Testing regulatory relationships that may vary by cell type
- Controlling for batch/donor effects (use cluster = batch × cell type)

**Pool (discouraged):**

- Homogeneous cell population (e.g., sorted monoculture)
- Already verified that association is consistent across clusters

### Robustness via local permutations (optional)

Global permutations assume cells are exchangeable after conditioning on cluster. But if:

- Cells have spatial structure (e.g., tissue slides)
- Pseudotime/trajectory effects exist within cluster
- Batch effects are present

Then global permutations may be too conservative or anti-conservative.

If you provide an `embedding` (e.g., UMAP coordinates), we:

1. Groups cells into local neighborhoods (via k-means in embedding space)
2. Permutes labels only within each neighborhood
3. Preserves local density structure while breaking cell-specific associations

**Usage:**

```python
results = bidirectional_association(
    expr_a, expr_b, clusters,
    embedding=adata.obsm['X_umap']  # (N, 2) array
)
```

**When to use:**

- Strong trajectory effects within cluster
- Spatial transcriptomics data
- You suspect local correlation structure

**When NOT to use:**

- Embedding is based on A and B themselves (circular reasoning)
- Cluster definitions are already fine-grained

### Key assumptions

We make the following assumptions:

1. **Cells are independent** (or permutations account for local structure)
2. **Expression values are on a comparable scale** (e.g., log-normalized counts)
3. **Clusters are biologically meaningful** (not arbitrary)
4. **Thresholding is reasonable** (q=30% captures "high" expressers)

We also assume the following (but not strictly required):

- Normality of expression distributions
- Homoscedasticity (equal variance)
- Linear relationships
- Absence of batch effects (if controlled via clustering or embedding)

#### Violations and consequences

| Violation                         | Consequence                   | Mitigation                                 |
| --------------------------------- | ----------------------------- | ------------------------------------------ |
| Cell correlation (e.g., doublets) | Inflated Type I error         | Use embedding-based permutations           |
| Arbitrary clusters                | Uninterpretable results       | Use data-driven clustering (Louvain, etc.) |
| Poor normalization                | Spurious differences          | Use log(CP10K+1) or similar                |
| Too few cells per cluster         | Low power, unstable estimates | Filter clusters with n < 80 (default)      |

### Method comparisons

| Method                      | Type                       | Handles Zeros?         | Bidirectional? | Causal Claims? |
| --------------------------- | -------------------------- | ---------------------- | -------------- | -------------- |
| **Pearson correlation**     | Linear association         | No (treats 0 as value) | Symmetric      | No             |
| **Spearman correlation**    | Monotone association       | Partially (rank-based) | Symmetric      | No             |
| **Differential expression** | Mean difference            | Yes (via models)       | Unidirectional | No             |
| **GRNBoost/SCENIC**         | Network inference          | Partially              | Directed graph | Weak           |
| **scBiDi**                  | Distributional association | **Yes (explicit)**     | **Yes**        | **No**         |
