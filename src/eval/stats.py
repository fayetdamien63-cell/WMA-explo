"""Statistiques du banc d'évaluation : moyennes, intervalles de confiance par bootstrap, comparaisons
appariées, décomposition de variance et nombre de seeds nécessaire."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float          # écart-type entre épisodes (mondes)
    sem: float          # erreur standard de la moyenne
    ci_low: float       # intervalle de confiance à 95 % (bootstrap percentile)
    ci_high: float
    n: int


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """IC bootstrap de la moyenne (déterministe : le rééchantillonnage utilise une seed fixe)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarize(values) -> Summary:
    v = np.asarray(values, dtype=float)
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    lo, hi = bootstrap_ci(v)
    return Summary(float(v.mean()), std, std / np.sqrt(len(v)), lo, hi, len(v))


def paired_difference(a, b) -> Summary:
    """Résumé de a − b épisode par épisode (mêmes mondes). Un IC strictement > 0 : a bat b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("comparaison appariée : mêmes mondes, même ordre")
    return summarize(a - b)


def seeds_needed(std: float, half_width: float, z: float = 1.96) -> int:
    """Nombre d'épisodes pour que l'IC à 95 % de la moyenne ait une demi-largeur ≤ `half_width`
    (approximation normale : n = (z·σ / h)²)."""
    if half_width <= 0:
        raise ValueError("half_width doit être > 0")
    return int(np.ceil((z * std / half_width) ** 2))


def variance_decomposition(returns: np.ndarray) -> dict[str, float]:
    """`returns` : (n_mondes, n_seeds_agent). Sépare la variance due au MONDE de celle due à l'aléa de l'agent.

    Renvoie la variance entre mondes (de la moyenne par monde, corrigée du bruit résiduel), la variance
    intra-monde (aléa de l'agent) et la part « monde » (ICC).
    """
    r = np.asarray(returns, dtype=float)
    n_worlds, n_agent = r.shape
    within = float(r.var(axis=1, ddof=1).mean()) if n_agent > 1 else 0.0
    between_obs = float(r.mean(axis=1).var(ddof=1))
    between = max(between_obs - within / n_agent, 0.0)
    return {"var_between_worlds": between, "var_within_world": within,
            "world_share": between / (between + within) if between + within > 0 else float("nan")}
