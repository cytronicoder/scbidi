### When to use scBiDi

✅ **Use scBiDi when you want to:**

- Detect distributional associations beyond linear correlation
- Account for zero-inflation explicitly
- Test bidirectional patterns (e.g., is A|B stronger than B|A?)
- Control for cell-type heterogeneity via cluster-wise analysis
- Perform permutation-based inference without distributional assumptions

❌ **Do NOT use scBiDi when you want to:**

- Test if a gene is upregulated in a condition (→ use DE testing)
- Infer causal direction (→ scbidi cannot distinguish A→B from confounding)
- Build temporal models (→ use pseudotime/trajectory methods)
- Test for any association (→ correlation is faster for screening)

### Common use cases

1. Screening gene-gene associations

```python
# Test many gene pairs, correct for multiple testing
results = []
for gene_a, gene_b in gene_pairs:
    df = summarize_gene_pair(adata[:, gene_a].X.toarray().flatten(),
                             adata[:, gene_b].X.toarray().flatten(),
                             adata.obs['cluster'],
                             gene_a_name=gene_a, gene_b_name=gene_b)
    results.append(df)
```

2. Validating a suspected regulatory relationship

```python
# Hypothesis: TF_X regulates Target_Y
# Check if Target_Y|TF_X is stronger than TF_X|Target_Y
df = summarize_gene_pair(tf_expr, target_expr, clusters,
                         gene_a_name="TF_X", gene_b_name="Target_Y")
print(df[['asymmetry', 'p_asymmetry']])  # Positive = Target|TF stronger
```

3. Comparing associations across cell types

```python
# Run separately for each cluster, compare D values
results = bidirectional_association(expr_a, expr_b, clusters)
for res in results:
    print(f"Cluster {res.cluster}: D_A|B={res.D_A_given_B:.3f}")
```

### Installation requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- statsmodels ≥ 0.14

#### Install from source

```bash
git clone https://github.com/cytronicoder/scbidi.git
cd scbidi
pip install -e .
```

#### Verify installation

```python
import scbidi
print(scbidi.__all__)
```

### Quick start in 5 lines

```python
import numpy as np
from scbidi import summarize_gene_pair

# Generate synthetic data (1000 cells, 2 genes)
expr_a = np.random.rand(1000)
expr_b = np.random.rand(1000)
clusters = np.zeros(1000, dtype=int)  # single cluster

# Analyze bidirectional association
df = summarize_gene_pair(expr_a, expr_b, clusters,
                         gene_a_name="GeneA",
                         gene_b_name="GeneB",
                         n_permutations=200)
print(df[["cluster", "D_A_given_B", "p_A_given_B",
          "D_B_given_A", "p_B_given_A", "asymmetry"]])
```

**Output:**

```text
   cluster  D_A_given_B  p_A_given_B  D_B_given_A  p_B_given_A  asymmetry
0        0        0.034        0.634        0.029        0.711      0.005
```

In this null case (independent random data), both D values are near zero and p-values are non-significant.

### Parameter reference

| Parameter        | Default | Meaning                                 |
| ---------------- | ------- | --------------------------------------- |
| `q`              | 0.3     | Top 30% defined as "high"               |
| `n_min`          | 80      | Minimum cells per cluster               |
| `min_high`       | 30      | Minimum "high" cells required           |
| `weight_zero`    | 0.5     | Weight for zero-fraction vs KS distance |
| `min_nonzero`    | 10      | Minimum non-zero values for KS distance |
| `n_permutations` | 1000    | Number of permutations for p-value      |

> [!IMPORTANT]
>
> Parameters are fixed across all genes/clusters to avoid post-hoc tuning. Modify only if you have a principled reason.
