"""Simulation utilities for BiDEA scenarios.

Implements the four scenarios from the specification using negative
binomial sampling followed by log-normalisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics import directional_association


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


def _shared_driver(config: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    latent = rng.normal(0.0, 1.0, size=config.n_cells)
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + config.beta_a * latent + config.delta * np.linspace(0, 1, config.n_cells) + eps_a)
    mu_b = np.exp(config.alpha_b + config.beta_b * latent + config.delta * np.linspace(0, 1, config.n_cells) + eps_b)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    return _log_normalize(a_counts), _log_normalize(b_counts)


def _independent(config: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + config.delta * np.linspace(0, 1, config.n_cells) + eps_a)
    mu_b = np.exp(config.alpha_b + config.delta * np.linspace(0, 1, config.n_cells) + eps_b)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    return _log_normalize(a_counts), _log_normalize(b_counts)


def _a_to_b(config: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    a_expr, _ = _independent(config, rng)
    eps_b = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_b = np.exp(config.alpha_b + config.gamma * np.tanh(a_expr) + config.delta * np.linspace(0, 1, config.n_cells) + eps_b)
    b_counts = _negative_binomial(mu_b, config.theta_b, rng)
    return a_expr, _log_normalize(b_counts)


def _b_to_a(config: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    b_expr, _ = _independent(config, rng)
    eps_a = rng.normal(0.0, config.sigma, size=config.n_cells)
    mu_a = np.exp(config.alpha_a + config.gamma * np.tanh(b_expr) + config.delta * np.linspace(0, 1, config.n_cells) + eps_a)
    a_counts = _negative_binomial(mu_a, config.theta_a, rng)
    return _log_normalize(a_counts), b_expr


SCENARIOS: Dict[str, Callable[[SimulationConfig, np.random.Generator], Tuple[np.ndarray, np.ndarray]]] = {
    "null": _independent,
    "shared_driver": _shared_driver,
    "a_to_b": _a_to_b,
    "b_to_a": _b_to_a,
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
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}")
    return SCENARIOS[scenario](config, rng)


def simulate_bidea_scenarios(
    scenario: str,
    n_replicates: int = 50,
    config: Optional[SimulationConfig] = None,
    random_state: Optional[int] = None,
    **bidea_kwargs,
) -> List[Dict[str, object]]:
    """Run repeated simulations and BiDEA analysis for a scenario.

    Returns a list of dictionaries containing both simulated data and results.
    """

    config = config or SimulationConfig()
    rng = np.random.default_rng(random_state)
    results: List[Dict[str, object]] = []

    for rep in range(n_replicates):
        expr_a, expr_b = generate_association_scenarios(scenario, config=config, random_state=rng.integers(0, 1e9))
        clusters = np.zeros_like(expr_a, dtype=int)
        associations = directional_association(expr_a, expr_b, clusters=clusters, **bidea_kwargs, random_state=rng.integers(0, 1e9))
        results.append(
            {
                "replicate": rep,
                "scenario": scenario,
                "expression_a": expr_a,
                "expression_b": expr_b,
                "associations": associations,
            }
        )
    return results
