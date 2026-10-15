"""Analyse des courbes d'apprentissage : efficacité en échantillons (pas d'environnement) et en temps réel."""

from __future__ import annotations

import numpy as np


def steps_to_reach(steps, values, target: float) -> float | None:
    """Premier nombre de pas où la courbe atteint `target` (interpolation linéaire entre deux points), sinon None."""
    steps, values = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    for i, v in enumerate(values):
        if v >= target:
            if i == 0:
                return float(steps[0])
            v0, v1 = values[i - 1], v
            frac = (target - v0) / (v1 - v0) if v1 != v0 else 1.0
            return float(steps[i - 1] + frac * (steps[i] - steps[i - 1]))
    return None


def per_seed_curves(records: list[dict], agent: str = "ppo") -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{seed: (pas d'entraînement, retour moyen, temps d'entraînement)} triés par pas."""
    out = {}
    for seed in sorted({r["seed"] for r in records if r["agent"] == agent}):
        rows = sorted((r for r in records if r["agent"] == agent and r["seed"] == seed), key=lambda r: r["train_env_steps"])
        out[seed] = (np.array([r["train_env_steps"] for r in rows]), np.array([r["mean_return"] for r in rows]),
                     np.array([r.get("wall_s", np.nan) for r in rows]))
    return out


def sample_efficiency(records: list[dict], targets: dict[str, float], agent: str = "ppo") -> dict[str, dict]:
    """Pour chaque cible (p. ex. « 90 % de l'heuristique »), pas et temps nécessaires, par seed.

    Renvoie {nom: {"steps": [par seed, None si jamais atteint], "wall_s": [...], "reached": k/n, "median_steps": ...}}.
    """
    curves = per_seed_curves(records, agent)
    out = {}
    for name, target in targets.items():
        steps, walls = [], []
        for s, (x, y, w) in curves.items():
            st = steps_to_reach(x, y, target)
            steps.append(st)
            walls.append(None if st is None else float(np.interp(st, x, w)))
        reached = [s for s in steps if s is not None]
        out[name] = {"target": target, "steps": steps, "wall_s": walls, "reached": f"{len(reached)}/{len(steps)}",
                     "median_steps": float(np.median(reached)) if reached else None}
    return out
