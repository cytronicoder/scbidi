### scbidi

(**s**ingle-**c**ell **bi**directional **di**stributional association testing)

This is a lightweight Python toolkit for bidirectional distributional association testing between gene pairs in single-cell RNA-seq data.

> [!NOTE]
> Yes, that name was on purpose. No, I will not change it.

#### Quick start

Create a virtual environment with NumPy and Pandas available, then run the example script:

```bash
python examples/run_simulations.py
```

To use the high-level summary function (returns a Pandas DataFrame):

```python
import numpy as np
from scbidi.metrics import summarize_gene_pair

# Generate synthetic data
expr_a = np.random.rand(1000)
expr_b = np.random.rand(1000)
clusters = np.zeros(1000, dtype=int)

# Get a summary table
df = summarize_gene_pair(expr_a, expr_b, clusters, n_permutations=200)
print(df[["cluster", "D_A_given_B", "p_A_given_B", "interpretation"]])
```

To use the core functions directly:

```python
from scbidi.metrics import bidirectional_association

results = bidirectional_association(expr_a, expr_b, clusters, n_permutations=200)
for res in results:
    print(f"Cluster {res.cluster}: D(A|B)={res.D_A_given_B:.3f}, D(B|A)={res.D_B_given_A:.3f}")
```

Key defaults (modifiable via arguments):

- `q=0.3` (top 30% for "high")
- `n_min=80` cells per cluster, `min_high=30` high cells
- `weight_zero=0.5` for mixing zero mass vs. KS on positives
- `n_permutations=1000` for permutation calibration

#### Notes

- The asymmetry score highlights patterns but is not a causal direction test.
- Thresholds and weights are fixed across genes/clusters to avoid post-hoc tuning.
