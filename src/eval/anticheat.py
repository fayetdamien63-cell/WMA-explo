"""Test anti-triche de la récompense : des stratégies absurdes codées à la main.

Une bonne récompense donne un score *inférieur* aux stratégies absurdes qu'à une stratégie sensée, et
ne permet aucun score hors de la borne théorique. Les stratégies lisent l'état du vrai monde
(`env.unwrapped.world`) : ce sont des politiques de test, pas des agents.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from antworld.config import AntWorldConfig
from antworld.env import AntWorldEnv
from antworld.reward import potential
from antworld.world import AntWorld

Policy = Callable[[AntWorld, int, np.random.Generator], np.ndarray]


def _act(*a: int) -> np.ndarray:
    return np.array(a, dtype=np.int64)


def sensible(w: AntWorld, t: int, rng) -> np.ndarray:
    """Référence sensée : 10 % d'éclaireuses ; du minerai tant que la nurserie n'est pas finie ;
    des bâtisseuses seulement quand on construit, la nurserie d'abord puis le grenier."""
    c = w.cfg.colony
    ore_share = 1 if w.nursery_level < c.max_building_level else 0
    building = w.col[5] > 0   # C_BUILD_TYPE
    choice = 0
    if not building:
        for kind, level, cost in ((1, w.nursery_level, c.nursery_ore_cost), (2, w.granary_level, c.granary_ore_cost)):
            if level < c.max_building_level and w.ore_stock >= cost * (level + 1):
                choice = kind
                break
    return _act(1, 2 if (building or choice) else 0, ore_share, choice, 0)


STRATEGIES: dict[str, tuple[str, Policy]] = {
    # --- références --------------------------------------------------------------------------
    "référence sensée": ("référence", sensible),
    "10 % d'éclaireuses, rien d'autre": ("référence", lambda w, t, r: _act(1, 0, 0, 0, 0)),
    # --- stratégies absurdes -----------------------------------------------------------------
    "ne rien faire (tous cueillent la nourriture)": ("absurde", lambda w, t, r: _act(0, 0, 0, 0, 0)),
    "thésauriser : 25 % minerai, ne jamais construire": ("absurde", lambda w, t, r: _act(1, 0, 1, 0, 0)),
    "tous au minerai (accumuler du minerai)": ("absurde", lambda w, t, r: _act(0, 0, 3, 0, 0)),
    "minerai + 10 % éclaireuses": ("absurde", lambda w, t, r: _act(1, 0, 3, 0, 0)),
    "personne ne cueille (éclaireuses + bâtisseuses)": ("absurde", lambda w, t, r: _act(3, 3, 0, 0, 0)),
    "50 % de bâtisseuses oisives": ("absurde", lambda w, t, r: _act(0, 3, 0, 0, 0)),
    "spam construction": ("absurde", lambda w, t, r: _act(1, 3, 1, 1 + t % 2, 0)),
    "oscillation extrême des rôles": ("absurde", lambda w, t, r: _act(3, 3, 3, 1, 0) if t % 2 else _act(0, 0, 0, 0, 0)),
    "aléatoire": ("absurde", lambda w, t, r: r.integers(0, w.action_nvec)),
}


def episode_return_bound(env: AntWorldEnv) -> float:
    """Borne supérieure théorique du retour (sans aucune hypothèse sur l'agent).

    retour = w_pop_level · Σ fourmis + Φ(fin) − Φ(début) + w_survive · pas_vivants avec
    fourmis ≤ population_max, Φ(fin) ≤ w_food · stockage_max + w_ore · minerai_total_du_monde,
    donc retour ≤ T · (w_pop_level · population_max + w_survive) + Φ_max − Φ_0.
    """
    w, cfg = env.world, env.cfg
    c, rc = cfg.colony, cfg.reward
    pop_max = c.base_pop_cap + c.max_building_level * c.nursery_gain
    storage_max = c.base_food_storage + c.max_building_level * c.granary_gain
    ore_total = float(w.layout.ore_amount.sum()) + c.initial_ore
    phi_max = rc.w_food * storage_max + rc.w_ore * ore_total
    return cfg.sim.max_steps * (rc.w_pop_level * pop_max + rc.w_survive) + phi_max - potential(w, rc)


def run_episode(cfg: AntWorldConfig, policy: Policy, seed: int) -> dict[str, float]:
    env = AntWorldEnv(cfg)
    env.reset(seed=seed)
    bound = episode_return_bound(env)
    phi0 = potential(env.world, cfg.reward)
    rng = np.random.default_rng(seed)
    total, t, terminated, pop_sum = 0.0, 0, False, 0
    while True:
        _, r, terminated, truncated, info = env.step(policy(env.world, t, rng))
        total += r
        pop_sum += info["n_ants"]
        t += 1
        if terminated or truncated:
            break
    w = env.world
    return {"return": total, "bound": bound, "steps": t, "dead": float(terminated), "pop": w.n_ants,
            "ore": w.ore_stock, "levels": w.nursery_level + w.granary_level,
            "telescoped": cfg.reward.w_pop_level * pop_sum + potential(w, cfg.reward) - phi0
                          + cfg.reward.w_survive * (t - (1 if terminated else 0))}


def evaluate(cfg: AntWorldConfig, seeds=range(20)) -> dict[str, dict[str, float]]:
    """Moyennes par stratégie sur les seeds (+ extrêmes utiles aux assertions)."""
    table = {}
    for name, (kind, policy) in STRATEGIES.items():
        runs = [run_episode(cfg, policy, s) for s in seeds]
        table[name] = {"kind": kind,
                       **{k: float(np.mean([r[k] for r in runs])) for k in ("return", "steps", "dead", "pop", "ore", "levels")},
                       "max_return": max(r["return"] for r in runs),
                       "min_margin_to_bound": min(r["bound"] - r["return"] for r in runs),
                       "max_telescope_error": max(abs(r["return"] - r["telescoped"]) for r in runs)}
    return table
