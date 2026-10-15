"""Planificateur CEM (Cross-Entropy Method) générique pour actions discrètes.

Il ne sait RIEN du simulateur : il reçoit une fonction `rollout_fn(séquences) -> retours`. À l'étape 3 cette
fonction utilise le vrai simulateur (oracle) ; à l'étape 6 elle utilisera le world model appris.

Principe de CEM : garder, pour chaque (bloc de pas, dimension de l'action), une distribution de
probabilité sur les valeurs possibles. On tire N séquences, on les évalue, on garde les meilleures
(« élites »), et on rapproche la distribution de la fréquence observée chez les élites. On répète quelques
itérations, puis on exécute le début de la meilleure séquence (MPC : on replanifie ensuite).

« Bloc » : une même action est maintenue `block` pas macro. Les rôles des fourmis changent lentement, donc
cela réduit beaucoup la dimension de recherche sans perdre grand-chose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CEMConfig:
    horizon: int = 30            # pas macro planifiés
    block: int = 5               # un même vecteur d'action tenu `block` pas
    n_samples: int = 64          # séquences évaluées par itération
    n_iters: int = 3
    n_elites: int = 8
    smoothing: float = 0.5       # poids de l'ancienne distribution dans la mise à jour
    replan_every: int = 5        # actions exécutées avant de replanifier (multiple de `block` conseillé)
    min_prob: float = 0.02       # plancher de probabilité : on continue d'explorer


RolloutFn = Callable[[np.ndarray], np.ndarray]    # (N, H, 5) actions -> (N,) retours


def _expand(blocks: np.ndarray, block: int, horizon: int) -> np.ndarray:
    """(N, n_blocks, 5) -> (N, horizon, 5) en répétant chaque bloc `block` fois."""
    return np.repeat(blocks, block, axis=1)[:, :horizon]


def cem_plan(rollout_fn: RolloutFn, nvec: np.ndarray, cfg: CEMConfig, rng: np.random.Generator,
             horizon: int, init_probs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Renvoie (meilleure séquence (horizon, 5), distribution finale, meilleur retour évalué)."""
    nvec = np.asarray(nvec)
    n_dims, kmax = len(nvec), int(nvec.max())
    n_blocks = -(-horizon // cfg.block)
    valid = np.arange(kmax)[None, :] < nvec[:, None]                      # (5, kmax) : valeurs autorisées
    uniform = valid / valid.sum(1, keepdims=True)
    probs = np.tile(uniform, (n_blocks, 1, 1)) if init_probs is None else init_probs.copy()

    best_blocks, best_ret = None, -np.inf
    for _ in range(cfg.n_iters):
        # Tirage par inversion de la fonction de répartition, pour tous (échantillon, bloc, dimension).
        cdf = np.cumsum(probs, axis=-1)                                   # (n_blocks, 5, kmax)
        u = rng.random((cfg.n_samples, n_blocks, n_dims, 1))
        blocks = np.minimum((u > cdf[None]).sum(-1), nvec - 1)            # (N, n_blocks, 5)
        if best_blocks is not None:                                       # élitisme : on réévalue le meilleur connu
            blocks[0] = best_blocks
        returns = rollout_fn(_expand(blocks, cfg.block, horizon))
        top = np.argsort(-returns, kind="stable")[: cfg.n_elites]
        if returns[top[0]] > best_ret:
            best_ret, best_blocks = float(returns[top[0]]), blocks[top[0]].copy()
        # Fréquences chez les élites -> nouvelle distribution (lissée, avec plancher).
        counts = np.zeros_like(probs)
        for b in range(n_blocks):
            for d in range(n_dims):
                counts[b, d] = np.bincount(blocks[top, b, d], minlength=kmax)
        freq = counts / cfg.n_elites
        probs = cfg.smoothing * probs + (1 - cfg.smoothing) * freq
        probs = np.where(valid[None], np.maximum(probs, cfg.min_prob), 0.0)
        probs /= probs.sum(-1, keepdims=True)
    return _expand(best_blocks[None], cfg.block, horizon)[0], probs, best_ret


def shift_probs(probs: np.ndarray, n_blocks_executed: int, uniform: np.ndarray) -> np.ndarray:
    """Réutilise la distribution de la planification précédente : on décale de `n` blocs et on complète en uniforme."""
    out = np.roll(probs, -n_blocks_executed, axis=0)
    out[-n_blocks_executed:] = uniform
    return out
