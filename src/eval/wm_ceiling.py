"""Plafond de prédictibilité du world model : l'erreur MINIMALE qu'aucun modèle ne peut éviter à cause de l'aléa du simulateur.

Méthode (exacte, sans modèle) : pour chaque fenêtre d'évaluation (épisode e, départ s) :
  1. on REJOUE l'épisode enregistré (même seed de monde, mêmes actions) jusqu'à s : le simulateur est déterministe, on retrouve l'état
     complet, y compris ce que le modèle ne voit pas (caps des éclaireuses, charges, contenu du brouillard). Vérifié contre la carte enregistrée ;
  2. à partir de cet état on lance N futurs avec les MÊMES actions enregistrées a_s, a_{s+1}, ... mais des aléas de simulation différents ;
  3. la VARIANCE entre ces N futurs est l'erreur quadratique du meilleur prédicteur possible (la moyenne des futurs) qui connaîtrait tout l'état
     caché. Aucun modèle qui ne voit que les observations ne peut faire mieux.
Compétence plafond = 1 − (cette variance) / (erreur de « rien ne change »), comparable terme à terme à la compétence du modèle.
Les actions sont celles de la trajectoire réelle (boucle ouverte : le modèle ne réagit pas aux observations futures, ni le plafond).
"""

from __future__ import annotations

import numpy as np

from antworld import AntWorldConfig
from antworld.env import AntWorldEnv
from common.seeding import SeedManager
from eval.wm_metrics import CONTEXT, HORIZONS, STARTS, WindowStats, _r2, _skill_idx, valid_windows
from models.data import TrajectoryDataset


def replay_to(env: AntWorldEnv, ds: TrajectoryDataset, episode: int, start: int) -> None:
    """Rejoue l'épisode jusqu'à l'instant `start` et vérifie que l'observation retrouvée est celle enregistrée."""
    env.reset(seed=int(ds.a["world_seeds"][episode]))
    for t in range(start):
        env.step(ds.a["actions"][episode, t].astype(np.int64))
    got = env._observe()["map"]
    want = ds.a["maps"][episode, start].astype(np.float32) / 255.0
    if np.abs(got - want).max() > 0.5 / 255 + 1e-6:
        raise RuntimeError(f"le rejeu de l'épisode {episode} diverge à l'instant {start} : le simulateur n'est plus déterministe ?")


def branch_futures(env: AntWorldEnv, ds: TrajectoryDataset, episode: int, start: int, horizons, n_branches: int, seed: int):
    """Cartes, états et récompenses cumulées aux horizons demandés, pour `n_branches` futurs aléatoires. Env déjà rejoué jusqu'à `start`."""
    snap = env.world.snapshot()
    H = max(horizons)
    acts = ds.a["actions"][episode, start: start + H].astype(np.int64)
    keep = {h: i for i, h in enumerate(horizons)}
    maps = np.zeros((n_branches, len(horizons), 8, 32, 32), np.float32)
    states = np.zeros((n_branches, len(horizons), 24), np.float32)
    cum = np.zeros((n_branches, len(horizons)), np.float64)
    for b in range(n_branches):
        env.world.restore(snap)
        env.world._rng = SeedManager(seed).rng(f"ceiling/{episode}/{start}/{b}")
        total = 0.0
        for k, a in enumerate(acts, 1):
            obs, r, *_ = env.step(a)
            total += r
            if k in keep:
                maps[b, keep[k]], states[b, keep[k]], cum[b, keep[k]] = obs["map"], obs["state"], total
    env.world.restore(snap)
    return maps, states, cum


def bayes_stats(world_cfg: AntWorldConfig, ds: TrajectoryDataset, stats: WindowStats, n_branches: int = 32, seed: int = 0,
                starts=STARTS, context=CONTEXT) -> dict[str, np.ndarray]:
    """Erreurs du meilleur prédicteur possible, alignées fenêtre par fenêtre sur `stats` (mêmes fenêtres, même ordre)."""
    wins = valid_windows(ds, starts, context, max(stats.horizons))
    assert len(wins) == len(stats.episode), "les fenêtres du plafond doivent être celles de l'évaluation"
    env = AntWorldEnv(world_cfg)
    W, nh = len(wins), len(stats.horizons)
    sse_m, sse_s, rew_pred = np.zeros((W, nh, 8)), np.zeros((W, nh, 24)), np.zeros((W, nh))
    cells = 32 * 32
    for w, (e, s) in enumerate(wins):
        replay_to(env, ds, e, s)
        maps, states, cum = branch_futures(env, ds, e, s, stats.horizons, n_branches, seed)
        f = n_branches / (n_branches - 1)                                   # variance sans biais
        sse_m[w] = (maps.var(axis=0) * f).sum(axis=(2, 3))                  # (nh, 8) : somme sur les cases
        sse_s[w] = states.var(axis=0) * f
        rew_pred[w] = cum.mean(axis=0)
    return {"sse_map_bayes": sse_m, "sse_state_bayes": sse_s, "rew_bayes": rew_pred}


def ceiling_summary(stats: WindowStats, bayes: dict[str, np.ndarray]) -> dict:
    """Compétences plafond par horizon (mêmes définitions que `wm_metrics.summarize`)."""
    out = {}
    for j, h in enumerate(stats.horizons):
        map_macro, map_total = _skill_idx(bayes["sse_map_bayes"][:, j], stats.sse_map_persist[:, j], 1e-6 * stats.n_cells)
        st_macro, st_total = _skill_idx(bayes["sse_state_bayes"][:, j], stats.sse_state_persist[:, j], 1e-9)
        per_ch = 1.0 - bayes["sse_map_bayes"][:, j].sum(0) / np.maximum(stats.sse_map_persist[:, j].sum(0), 1e-12)
        out[str(h)] = {"map_skill_macro": map_macro, "map_skill_total": map_total, "state_skill_macro": st_macro,
                       "reward_r2": _r2(stats.rew_true[:, j], bayes["rew_bayes"][:, j]),
                       "map_skill_by_channel": [float(x) for x in per_ch]}
    return out
