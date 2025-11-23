### Core functions

#### `bidirectional_association`

Compute bidirectional conditional association metrics for each cluster.

**Signature:**

```python
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
```

**Purpose:**

Tests bidirectional distributional associations between two genes across cell clusters. For each cluster, computes:

- How gene A's distribution differs between B-high and B-low cells (A|B)
- How gene B's distribution differs between A-high and A-low cells (B|A)
- Statistical significance via permutation testing
- Asymmetry score (D_A|B - D_B|A)

**Parameters:**

| Parameter        | Type                   | Default    | Description                                                              |
| ---------------- | ---------------------- | ---------- | ------------------------------------------------------------------------ |
| `expression_a`   | `Sequence[float]`      | _required_ | Expression values for gene A (N cells). Should be log-normalized counts. |
| `expression_b`   | `Sequence[float]`      | _required_ | Expression values for gene B (N cells). Same normalization as A.         |
| `clusters`       | `Sequence[int]`        | _required_ | Cluster assignments (N cells). Integer labels (0, 1, 2, ...).            |
| `q`              | `float`                | `0.3`      | Quantile threshold for "high" expressers. `q=0.3` means top 30%.         |
| `n_min`          | `int`                  | `80`       | Minimum cells required in a cluster to attempt analysis.                 |
| `min_high`       | `int`                  | `30`       | Minimum number of "high" cells required for valid threshold.             |
| `weight_zero`    | `float`                | `0.5`      | Weight for zero-fraction component in two-part distance. Range: [0, 1].  |
| `min_nonzero`    | `int`                  | `10`       | Minimum non-zero observations in each group to compute KS distance.      |
| `n_permutations` | `int`                  | `1000`     | Number of permutations for p-value calculation.                          |
| `random_state`   | `Optional[int]`        | `None`     | Random seed for reproducibility.                                         |
| `embedding`      | `Optional[np.ndarray]` | `None`     | (N, D) array of cell coordinates (e.g., UMAP) for local permutations.    |

**Returns:**

`List[PairwiseAssociationResult]`: One result object per cluster. See [`PairwiseAssociationResult`](#pairwiseassociationresult) for fields.

**Raises:**

- `ValueError`: If input arrays have mismatched shapes or contain NaNs.
- `ValueError`: If embedding shape doesn't match expression arrays.

**Example:**

```python
import numpy as np
from scbidi import bidirectional_association

# Simulate data: 500 cells, 2 genes, 2 clusters
np.random.seed(42)
expr_a = np.concatenate([np.random.exponential(2, 250),
                         np.random.exponential(5, 250)])
expr_b = np.concatenate([np.random.exponential(3, 250),
                         np.random.exponential(3, 250)])
clusters = np.array([0]*250 + [1]*250)

# Run analysis
results = bidirectional_association(expr_a, expr_b, clusters,
                                    n_permutations=500, random_state=42)

# Inspect results
for res in results:
    print(f"Cluster {res.cluster}:")
    print(f"  D_A|B = {res.D_A_given_B:.3f}, p = {res.p_A_given_B:.3f}")
    print(f"  D_B|A = {res.D_B_given_A:.3f}, p = {res.p_B_given_A:.3f}")
    print(f"  Asymmetry = {res.asymmetry:.3f}, p = {res.p_asymmetry:.3f}")
```

**Edge Cases:**

- **Too few cells**: If a cluster has < `n_min` cells, analysis is skipped and `notes` field explains why.
- **Too few high expressers**: If < `min_high` cells exceed threshold, analysis is skipped.
- **All zeros**: If gene is all-zero in a cluster, KS component is 0, distance is based on zero-fraction only.
- **Single cluster**: If `clusters` is all the same value, returns a list with one result.

**Notes:**

- Parameters (`q`, `weight_zero`, etc.) should be fixed across all gene pairs to avoid p-hacking.
- Use `embedding` for spatial transcriptomics or when trajectory effects are present within clusters.
- Permutation p-values are conservative: p = (1 + exceeds) / (1 + n_permutations).

#### `summarize_gene_pair`

High-level function that returns a pandas DataFrame with analysis results and human-readable interpretation.

**Signature:**

```python
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
```

**Purpose:**

Wrapper around `bidirectional_association` that adds:

- Gene names for interpretation
- Qualitative strength categories (e.g., "moderate difference")
- FDR-corrected p-values (Benjamini-Hochberg)
- Human-readable interpretation text

**Parameters:**

| Parameter        | Type              | Default    | Description                               |
| ---------------- | ----------------- | ---------- | ----------------------------------------- |
| `expr_a`         | `Sequence[float]` | _required_ | Expression for gene A.                    |
| `expr_b`         | `Sequence[float]` | _required_ | Expression for gene B.                    |
| `clusters`       | `Sequence[int]`   | _required_ | Cluster assignments.                      |
| `gene_a_name`    | `str`             | `"GeneA"`  | Name for gene A (used in interpretation). |
| `gene_b_name`    | `str`             | `"GeneB"`  | Name for gene B (used in interpretation). |
| `n_permutations` | `int`             | `1000`     | Number of permutations.                   |
| `random_state`   | `Optional[int]`   | `None`     | Random seed.                              |
| `fdr_correction` | `bool`            | `True`     | Apply FDR correction across clusters.     |

**Returns:**

`pd.DataFrame` with columns:

| Column             | Type    | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `cluster`          | `int`   | Cluster ID                                       |
| `n_cells`          | `int`   | Number of cells in cluster                       |
| `n_high_A`         | `int`   | Number of A-high cells                           |
| `n_high_B`         | `int`   | Number of B-high cells                           |
| `D_A_given_B`      | `float` | Total distance for A\|B                          |
| `D_A_given_B_zero` | `float` | Zero-fraction component of A\|B                  |
| `D_A_given_B_cont` | `float` | KS component of A\|B                             |
| `p_A_given_B`      | `float` | Permutation p-value for A\|B                     |
| `p_A_given_B_fdr`  | `float` | FDR-corrected p-value (if `fdr_correction=True`) |
| `D_B_given_A`      | `float` | Total distance for B\|A                          |
| `D_B_given_A_zero` | `float` | Zero-fraction component of B\|A                  |
| `D_B_given_A_cont` | `float` | KS component of B\|A                             |
| `p_B_given_A`      | `float` | Permutation p-value for B\|A                     |
| `p_B_given_A_fdr`  | `float` | FDR-corrected p-value (if `fdr_correction=True`) |
| `asymmetry`        | `float` | D_A_given_B - D_B_given_A                        |
| `p_asymmetry`      | `float` | Two-sided p-value for asymmetry                  |
| `p_asymmetry_fdr`  | `float` | FDR-corrected p-value (if `fdr_correction=True`) |
| `interpretation`   | `str`   | Human-readable summary                           |

**Example:**

```python
from scbidi import summarize_gene_pair
import numpy as np

# Example data
expr_a = np.random.rand(1000)
expr_b = np.random.rand(1000)
clusters = np.repeat([0, 1, 2], [300, 400, 300])

# Analyze
df = summarize_gene_pair(expr_a, expr_b, clusters,
                         gene_a_name="CD4", gene_b_name="IL2",
                         n_permutations=500)

# View compact summary
print(df[['cluster', 'D_A_given_B', 'p_A_given_B_fdr',
          'D_B_given_A', 'p_B_given_A_fdr', 'asymmetry']])

# Read interpretation
print(df.loc[0, 'interpretation'])
```

**Qualitative Strength Categories:**

| D value  | p-value | Interpretation                                       |
| -------- | ------- | ---------------------------------------------------- |
| Any      | ≥ 0.05  | "no statistically clear difference"                  |
| < 0.1    | < 0.05  | "very small but statistically detectable difference" |
| 0.1–0.25 | < 0.05  | "weak but statistically clear difference"            |
| 0.25–0.5 | < 0.05  | "moderate difference"                                |
| ≥ 0.5    | < 0.05  | "strong difference"                                  |

**Notes:**

- FDR correction is applied **across all clusters** for each test (A|B, B|A, asymmetry separately).
- Interpretation text emphasizes that associations do not prove causation.
- Use this function for exploratory analysis; use `bidirectional_association` for programmatic access.

#### `compute_two_part_distance`

Low-level function to compute the two-part distance between foreground and background groups.

**Signature:**

```python
def compute_two_part_distance(
    expression: Sequence[float],
    high_labels: Sequence[int],
    weight_zero: float = 0.5,
    min_nonzero: int = 10,
) -> TwoPartDistance:
```

**Purpose:**

Quantifies distributional difference between "high" and "low" groups for a single gene, separating on/off (zero-fraction) from magnitude (KS distance) components.

**Parameters:**

| Parameter     | Type              | Default    | Description                                                       |
| ------------- | ----------------- | ---------- | ----------------------------------------------------------------- |
| `expression`  | `Sequence[float]` | _required_ | Expression values for one gene (N cells).                         |
| `high_labels` | `Sequence[int]`   | _required_ | Binary labels: 1 = "high" group, 0 = "low" group.                 |
| `weight_zero` | `float`           | `0.5`      | Weight for zero-fraction component.                               |
| `min_nonzero` | `int`             | `10`       | Minimum non-zero values required in **each** group to compute KS. |

**Returns:**

`TwoPartDistance` object with fields:

- `d_zero` (float): Absolute difference in zero fractions
- `d_continuous` (float): KS distance on non-zero values
- `total` (float): Weighted sum = `weight_zero * d_zero + (1 - weight_zero) * d_continuous`

**Mathematical Definition:**

$$
D = w_0 \cdot \left| \frac{\#\{x_i^{\text{high}} = 0\}}{n_{\text{high}}} - \frac{\#\{x_i^{\text{low}} = 0\}}{n_{\text{low}}} \right|
$$

$$
\quad + (1 - w_0) \cdot \sup_{t} \left| \text{ECDF}_{\text{high}}(t) - \text{ECDF}_{\text{low}}(t) \right|
$$

where ECDF is computed only on non-zero values.

**Example:**

```python
import numpy as np
from scbidi import compute_two_part_distance

# Gene expression (10 cells)
expression = np.array([0, 0, 0, 1.2, 2.3, 3.1, 4.5, 5.2, 6.1, 7.3])

# Label top 40% as "high"
high_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# Compute distance
dist = compute_two_part_distance(expression, high_labels, weight_zero=0.5)

print(f"Zero-fraction component: {dist.d_zero:.3f}")
print(f"KS component: {dist.d_continuous:.3f}")
print(f"Total distance: {dist.total:.3f}")
```

**Edge Cases:**

- **All same label**: If all labels are 0 or all are 1, returns `TwoPartDistance(0, 0, 0)`.
- **Insufficient non-zeros**: If either group has < `min_nonzero` non-zero values, `d_continuous = 0`.
- **All zeros**: If gene is all-zero in both groups, `d_zero = 0`, `d_continuous = 0`.

**Notes:**

- This function is called internally by `bidirectional_association`.
- Use directly only if you need custom thresholding or group definitions.

#### `threshold_high_low`

Split cells into "high" and "low" expressers for a single gene.

**Signature:**

```python
def threshold_high_low(
    expression: Sequence[float],
    q: float = 0.3,
    n_min: int = 80,
    min_high: int = 30,
) -> ThresholdResult:
```

**Purpose:**

Defines foreground (high) and background (low) groups based on quantile thresholding. Performs validity checks to ensure sufficient sample size.

**Parameters:**

| Parameter    | Type              | Default    | Description                           |
| ------------ | ----------------- | ---------- | ------------------------------------- |
| `expression` | `Sequence[float]` | _required_ | Expression values for one gene.       |
| `q`          | `float`           | `0.3`      | Fraction of cells to label as "high". |
| `n_min`      | `int`             | `80`       | Minimum total cells required.         |
| `min_high`   | `int`             | `30`       | Minimum "high" cells required.        |

**Returns:**

`ThresholdResult` object with fields:

- `labels` (np.ndarray): Binary labels (1 = high, 0 = low)
- `threshold` (float): Expression cutoff value (q-th quantile)
- `n_cells` (int): Total number of cells
- `n_high` (int): Number of cells labeled "high"
- `skipped` (bool): `True` if analysis should be skipped
- `reason` (Optional[str]): Explanation if `skipped=True`

**Example:**

```python
import numpy as np
from scbidi import threshold_high_low

expr = np.random.exponential(2, 100)

result = threshold_high_low(expr, q=0.3, n_min=50, min_high=20)

print(f"Threshold: {result.threshold:.2f}")
print(f"High cells: {result.n_high}")
print(f"Skipped: {result.skipped}")
if result.skipped:
    print(f"Reason: {result.reason}")
```

**Edge Cases:**

- **Too few cells**: If `n_cells < n_min`, sets `skipped=True`, `reason="insufficient cells (n=X, min=Y)"`.
- **Too few high cells**: If `n_high < min_high`, sets `skipped=True`, `reason="insufficient high cells (n_high=X, min=Y)"`.
- **All same value**: If all expression values are identical, threshold is that value and all cells may be labeled "low".

**Notes:**

- Thresholding is based on **rank**, not absolute expression level.
- For genes with many zeros, "high" may include cells with low but non-zero expression.
- Use this function directly only for custom workflows; `bidirectional_association` calls it internally.

### Data classes

#### `PairwiseAssociationResult`

Container for bidirectional association results from one cluster.

**Fields:**

| Field              | Type              | Description                                |
| ------------------ | ----------------- | ------------------------------------------ |
| `cluster`          | `int`             | Cluster identifier                         |
| `n_cells`          | `int`             | Number of cells in cluster                 |
| `n_high_a`         | `int`             | Number of A-high cells                     |
| `n_high_b`         | `int`             | Number of B-high cells                     |
| `D_A_given_B`      | `Optional[float]` | Total distance for A\|B (None if skipped)  |
| `D_A_given_B_zero` | `Optional[float]` | Zero-fraction component of A\|B            |
| `D_A_given_B_cont` | `Optional[float]` | KS component of A\|B                       |
| `p_A_given_B`      | `Optional[float]` | Permutation p-value for A\|B               |
| `D_B_given_A`      | `Optional[float]` | Total distance for B\|A (None if skipped)  |
| `D_B_given_A_zero` | `Optional[float]` | Zero-fraction component of B\|A            |
| `D_B_given_A_cont` | `Optional[float]` | KS component of B\|A                       |
| `p_B_given_A`      | `Optional[float]` | Permutation p-value for B\|A               |
| `asymmetry`        | `Optional[float]` | D_A_given_B - D_B_given_A                  |
| `p_asymmetry`      | `Optional[float]` | Two-sided p-value for asymmetry            |
| `notes`            | `Optional[str]`   | Human-readable message (e.g., skip reason) |

**Example:**

```python
from scbidi import bidirectional_association
import numpy as np

expr_a = np.random.rand(200)
expr_b = np.random.rand(200)
clusters = np.zeros(200, dtype=int)

results = bidirectional_association(expr_a, expr_b, clusters, n_permutations=200)
res = results[0]  # Single cluster

# Access fields
print(f"Cluster: {res.cluster}")
print(f"Cells: {res.n_cells}")
print(f"A|B: D={res.D_A_given_B:.3f}, p={res.p_A_given_B:.3f}")
print(f"B|A: D={res.D_B_given_A:.3f}, p={res.p_B_given_A:.3f}")
print(f"Asymmetry: {res.asymmetry:.3f} (p={res.p_asymmetry:.3f})")
```

#### `TwoPartDistance`

Container for two-part distance components.

**Fields:**

| Field          | Type    | Description                    |
| -------------- | ------- | ------------------------------ |
| `d_zero`       | `float` | Difference in zero fractions   |
| `d_continuous` | `float` | KS distance on non-zero values |
| `total`        | `float` | Weighted combination           |

**Example:**

```python
from scbidi import compute_two_part_distance
import numpy as np

expr = np.array([0, 0, 1, 2, 3, 4, 5, 6])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

dist = compute_two_part_distance(expr, labels, weight_zero=0.6)

print(f"Zero component: {dist.d_zero:.3f}")
print(f"Continuous component: {dist.d_continuous:.3f}")
print(f"Total (0.6 × zero + 0.4 × cont): {dist.total:.3f}")
```

#### `ThresholdResult`

Container for thresholding results.

**Fields:**

| Field       | Type            | Description                        |
| ----------- | --------------- | ---------------------------------- |
| `labels`    | `np.ndarray`    | Binary labels (1=high, 0=low)      |
| `threshold` | `float`         | Expression cutoff value            |
| `n_cells`   | `int`           | Total cells                        |
| `n_high`    | `int`           | Number of high cells               |
| `skipped`   | `bool`          | Whether analysis should be skipped |
| `reason`    | `Optional[str]` | Skip reason (if applicable)        |

**Example:**

```python
from scbidi import threshold_high_low
import numpy as np

expr = np.random.exponential(2, 50)
result = threshold_high_low(expr, q=0.3, n_min=80)

if result.skipped:
    print(f"Cannot analyze: {result.reason}")
else:
    print(f"Threshold: {result.threshold:.2f}")
    print(f"High cells: {result.n_high}")
```

### Simulation functions

#### `generate_association_scenarios`

Generate synthetic gene pairs under different association scenarios.

**Signature:**

```python
def generate_association_scenarios(
    scenario: str,
    config: Optional[SimulationConfig] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

**Purpose:**

Create simulated scRNA-seq data for testing and benchmarking. Supports multiple scenarios:

| Scenario              | Description         | Expected Pattern             |
| --------------------- | ------------------- | ---------------------------- |
| `"null"`              | A ⟂ B (independent) | Both D values ≈ 0            |
| `"shared_driver"`     | L → A, L → B        | Both D values > 0, symmetric |
| `"a_to_b"`            | A → B (direct)      | D_B\|A > D_A\|B              |
| `"b_to_a"`            | B → A (direct)      | D_A\|B > D_B\|A              |
| `"confounded_a_to_b"` | L → A, L → B, A → B | Both > 0, B\|A stronger      |
| `"confounded_b_to_a"` | L → A, L → B, B → A | Both > 0, A\|B stronger      |

**Parameters:**

| Parameter      | Type                         | Default    | Description                       |
| -------------- | ---------------------------- | ---------- | --------------------------------- |
| `scenario`     | `str`                        | _required_ | One of the scenarios above        |
| `config`       | `Optional[SimulationConfig]` | `None`     | Simulation parameters (see below) |
| `random_state` | `Optional[int]`              | `None`     | Random seed                       |

**Returns:**

`Tuple[np.ndarray, np.ndarray]`: (expr_a, expr_b) arrays of log-normalized counts.

**Example:**

```python
from scbidi.simulation import generate_association_scenarios
from scbidi import bidirectional_association
import numpy as np

# Generate A → B scenario
expr_a, expr_b = generate_association_scenarios("a_to_b", random_state=42)

# Test it
clusters = np.zeros(len(expr_a), dtype=int)
results = bidirectional_association(expr_a, expr_b, clusters, n_permutations=200)

print(f"D_A|B: {results[0].D_A_given_B:.3f}")  # Should be small
print(f"D_B|A: {results[0].D_B_given_A:.3f}")  # Should be large
```

**SimulationConfig Fields:**

| Field                | Default | Description                       |
| -------------------- | ------- | --------------------------------- |
| `n_cells`            | 2000    | Number of cells                   |
| `theta_a`, `theta_b` | 1.0     | Negative binomial dispersion      |
| `alpha_a`, `alpha_b` | 2.5     | Baseline log-mean expression      |
| `sigma`              | 0.6     | Noise standard deviation          |
| `gamma`              | 1.0     | Effect size for direct regulation |
| `beta_a`, `beta_b`   | 1.0     | Latent driver effect size         |

**Notes:**

- Simulations use **negative binomial counts** + log-normalization to mimic real scRNA-seq.
- Adjust `gamma` to control strength of direct effects.
- Use for method validation, power analysis, and illustration.

### Utility functions

#### `validate_inputs`

Check input arrays for consistency.

**Signature:**

```python
def validate_inputs(
    expr_a: np.ndarray,
    expr_b: np.ndarray,
    clusters: np.ndarray
) -> None:
```

**Raises:**

- `ValueError` if shapes don't match
- `ValueError` if NaNs are present

**Example:**

```python
from scbidi.utils import validate_inputs
import numpy as np

a = np.random.rand(100)
b = np.random.rand(100)
c = np.zeros(100, dtype=int)

validate_inputs(a, b, c)  # OK

b_bad = np.random.rand(99)
# validate_inputs(a, b_bad, c)  # Raises ValueError
```
