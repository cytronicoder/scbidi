"""Simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from .metrics import bidirectional_association


@dataclass
class SimulationConfig:
    n_cells: int = 2000
    theta_a: float = 1.0
    theta_b: float = 1.0
    alpha_a: float = 2.5
    alpha_b: float = 2.5
    sigma: float = 0.6
    gamma: float = 1.0
    beta_a: float = 1.0
    beta_b: float = 1.0
    delta: float = 0.0


def _negative_binomial(
    mean: np.ndarray, theta: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw counts from NB with mean ``mean`` and dispersion ``theta``."""
    mean = np.asarray(mean, dtype=float)
    p = theta / (theta + mean)
    r = theta
    counts = rng.negative_binomial(r, p)
    return counts


def _log_normalize(counts: np.ndarray) -> np.ndarray:
    return np.log2(counts + 1.0)


def _shared_driver(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared driver: L → A, L → B."""
    latent = rng.normal(0.0, 1.0, size=config.n_cells)
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)

    mu_a = np.exp(config.alpha_a + config.beta_a * latent + eps_a)
    mu_b = np.exp(config.alpha_b + config.beta_b * latent + eps_b)

    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    return _log_normalize(a_counts), _log_normalize(b_counts)


def _independent(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Null scenario: A ⟂ B, no shared drivers."""
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)

    grad_a = config.delta * np.linspace(0, 1, config.n_cells)
    grad_b = config.delta * rng.permutation(grad_a)

    mu_a = np.exp(config.alpha_a + grad_a + eps_a)
    mu_b = np.exp(config.alpha_b + grad_b + eps_b)

    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    return _log_normalize(a_counts), _log_normalize(b_counts)


def _a_to_b(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure A → B: A baseline, B = f(A) + noise."""
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + eps_a)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    a_expr = _log_normalize(a_counts)

    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_b = np.exp(config.alpha_b + config.gamma * np.tanh(a_expr) + eps_b)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    b_expr = _log_normalize(b_counts)

    return a_expr, b_expr


def _b_to_a(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure B → A: B baseline, A = f(B) + noise."""
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_b = np.exp(config.alpha_b + eps_b)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    b_expr = _log_normalize(b_counts)

    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + config.gamma * np.tanh(b_expr) + eps_a)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    a_expr = _log_normalize(a_counts)

    return a_expr, b_expr


def _confounded_a_to_b(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Confounded A → B: L → A, L → B, and A → B."""
    latent = rng.normal(0.0, 1.0, size=config.n_cells)

    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + config.beta_a * latent + eps_a)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    a_expr = _log_normalize(a_counts)

    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_b = np.exp(
        config.alpha_b + config.gamma * np.tanh(a_expr) + config.beta_b * latent + eps_b
    )
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    b_expr = _log_normalize(b_counts)

    return a_expr, b_expr


def _confounded_b_to_a(
    config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Confounded B → A: L → A, L → B, and B → A."""
    latent = rng.normal(0.0, 1.0, size=config.n_cells)

    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_b = np.exp(config.alpha_b + config.beta_b * latent + eps_b)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    b_expr = _log_normalize(b_counts)

    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(
        config.alpha_a + config.gamma * np.tanh(b_expr) + config.beta_a * latent + eps_a
    )
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    a_expr = _log_normalize(a_counts)

    return a_expr, b_expr


SCENARIOS: Dict[
    str,
    Callable[[SimulationConfig, np.random.Generator], Tuple[np.ndarray, np.ndarray]],
] = {
    "null": _independent,
    "shared_driver": _shared_driver,
    "a_to_b": _a_to_b,
    "b_to_a": _b_to_a,
    "confounded_a_to_b": _confounded_a_to_b,
    "confounded_b_to_a": _confounded_b_to_a,
}


def generate_association_scenarios(
    scenario: str,
    config: Optional[SimulationConfig] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a pair of expression vectors for the requested scenario."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(random_state)
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}"
        )
    return SCENARIOS[scenario](config, rng)


def simulate_scenarios(
    scenario: str,
    n_replicates: int = 50,
    config: Optional[SimulationConfig] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Run repeated simulations and analysis for a scenario."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(random_state)
    results: List[Dict[str, Any]] = []

    for rep in range(n_replicates):
        expr_a, expr_b = generate_association_scenarios(
            scenario, config=config, random_state=rng.integers(0, 1e9)
        )

        corr = np.corrcoef(expr_a, expr_b)[0, 1]

        clusters = np.zeros_like(expr_a, dtype=int)
        associations = bidirectional_association(
            expr_a,
            expr_b,
            clusters=clusters,
            **kwargs,
            random_state=rng.integers(0, 1e9),
        )

        results.append(
            {
                "replicate": rep,
                "scenario": scenario,
                "expression_a": expr_a,
                "expression_b": expr_b,
                "correlation": corr,
                "associations": associations,
            }
        )
    return results


def run_parameter_sweep(
    scenario: str,
    param_name: str,
    param_values: List[float],
    n_replicates: int = 20,
    base_config: Optional[SimulationConfig] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Run simulations sweeping a parameter (e.g. gamma)."""
    base_config = base_config or SimulationConfig()
    all_results = []

    for val in param_values:
        current_config = SimulationConfig(**base_config.__dict__)
        setattr(current_config, param_name, val)

        res = simulate_scenarios(
            scenario, n_replicates=n_replicates, config=current_config, **kwargs
        )
        for r in res:
            r["param_name"] = param_name
            r["param_value"] = val
        all_results.extend(res)

    return all_results


def sigmoid(x):
    """Logistic sigmoid function."""
    return 1 / (1 + np.exp(-x))


def hill_function(x, k=0.5, n=4, vmax=1.0):
    """Hill function for gene activation."""
    return vmax * (x**n) / (k**n + x**n)


def generate_spatial_data(scenario: str, n_points: int = 2000, rng=None):
    """Generate spatial expression data using non-linear structural equation models.

    Returns:
        embedding: (N, 2) array of coordinates in unit circle
        expr_a: (N,) array of gene A expression [0, 1]
        expr_b: (N,) array of gene B expression [0, 10]
    """
    if rng is None:
        rng = np.random.default_rng()

    r = np.sqrt(rng.uniform(0, 1, n_points))
    theta = rng.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    embedding = np.column_stack([x, y])

    z_center = np.exp(-(x**2 + y**2) / 0.2)
    z_gradient = (x + 1) / 2

    noise_a = rng.normal(0, 0.5, n_points)
    noise_b = rng.normal(0, 0.5, n_points)

    if scenario == "null":
        logit_a = 6 * (z_center - 0.5) + noise_a
        expr_a = sigmoid(logit_a)

        logit_b = 6 * (z_gradient - 0.5) + noise_b
        expr_b = sigmoid(logit_b) * 10

    elif scenario == "shared_driver":
        logit_a = 6 * (z_center - 0.5) + noise_a
        expr_a = sigmoid(logit_a)

        logit_b = 6 * (z_center - 0.5) + noise_b
        expr_b = sigmoid(logit_b) * 10

    elif scenario == "a_to_b":
        logit_a = 6 * (z_center - 0.5) + noise_a
        expr_a = sigmoid(logit_a)

        expr_b = hill_function(expr_a, k=0.5, n=4, vmax=10) + noise_b
        expr_b = np.clip(expr_b, 0, 10)

    elif scenario == "b_to_a":
        logit_b = 6 * (z_center - 0.5) + noise_b
        expr_b = sigmoid(logit_b) * 10

        b_scaled = expr_b / 10.0
        expr_a = hill_function(b_scaled, k=0.5, n=4, vmax=1.0) + noise_a
        expr_a = np.clip(expr_a, 0, 1)

    elif scenario == "confounded_a_to_b":
        logit_a = 6 * (z_center - 0.5) + noise_a
        expr_a = sigmoid(logit_a)

        term_z = hill_function(z_center, k=0.5, n=4, vmax=10)
        term_a = hill_function(expr_a, k=0.5, n=4, vmax=10)
        expr_b = 0.5 * term_z + 0.5 * term_a + noise_b
        expr_b = np.clip(expr_b, 0, 10)

    elif scenario == "confounded_b_to_a":
        logit_b = 6 * (z_center - 0.5) + noise_b
        expr_b = sigmoid(logit_b) * 10

        b_scaled = expr_b / 10.0
        term_z = hill_function(z_center, k=0.5, n=4, vmax=1.0)
        term_b = hill_function(b_scaled, k=0.5, n=4, vmax=1.0)

        expr_a = 0.5 * term_z + 0.5 * term_b + noise_a
        expr_a = np.clip(expr_a, 0, 1)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return embedding, expr_a, expr_b
