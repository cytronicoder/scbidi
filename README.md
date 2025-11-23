### scbidi

A lightweight Python toolkit for bidirectional distributional association testing between gene pairs in single-cell RNA-seq.

> [!NOTE]
> Yes, that name was on purpose. No, I will not change it.

- Work within each annotated cluster; skip clusters with too few cells.
- Define "high" vs "low" using a fixed upper-quantile rule (default top 30%) without ad hoc tuning.
- Use a two-part distance that blends zero-mass differences with a KS distance on positive expression.
- Calibrate with permutation tests that respect cluster membership.
- Mirror the test in both directions (A|B and B|A) and report an asymmetry score.

#### Quick start

Create a virtual environment with NumPy available, then run the example script:

```bash
python examples/run_simulations.py
```

To use the core functions directly:

```python
import numpy as np
from scbidi.metrics import directional_association

expr_a = np.random.rand(1000)
expr_b = np.random.rand(1000)
clusters = np.zeros_like(expr_a, dtype=int)
results = directional_association(expr_a, expr_b, clusters, n_permutations=200)
for res in results:
    print(res)
```

Key defaults (modifiable via arguments):

- `q=0.3` (top 30% for "high")
- `n_min=80` cells per cluster, `min_high=30` high cells
- `weight_zero=0.5` for mixing zero mass vs. KS on positives
- `n_permutations=1000` for permutation calibration

#### Notes

- The asymmetry score highlights patterns but is not a causal direction test.
- Thresholds and weights are fixed across genes/clusters to avoid post-hoc tuning.
