"""Jeux de trajectoires pour le world model : collecte, stockage compact, échantillonnage de fenêtres.

Une trajectoire (épisode de T ≤ 400 pas macro) contient : T+1 observations (carte 8×32×32 + état 24), T actions,
T récompenses, et un drapeau « la colonie est morte à ce pas ». Les cartes sont stockées en uint8 (valeurs × 255) : 3,3 Mo
par épisode au lieu de 13 Mo ; l'erreur de quantification (≤ 0,002) est la MÊME pour le modèle et pour la baseline
« rien ne change » puisque les deux sont comparées aux cartes déquantifiées.

Quatre jeux, tirés dans des plages de seeds de mondes disjointes :
  train        mondes d'entraînement (seeds [0, 1000)), mélange de politiques
  val          mondes de validation (seeds 500 000+), pour suivre l'entraînement
  eval_seen    MÊMES mondes que train mais NOUVELLES trajectoires : généralisation à d'autres comportements
  eval_unseen  mondes de test (seeds 1 000 000+), jamais vus : généralisation à d'autres mondes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.data_policies import POLICY_NAMES, make_data_policy
from antworld import AntWorldConfig
from antworld.env import AntWorldEnv
from antworld.observation import N_MAP_CHANNELS, N_STATE
from common.seeding import SeedManager
from eval.seeds import test_seeds, train_seeds, val_seeds

MAX_STEPS = 400


@dataclass
class DataConfig:
    n_train: int = 400                  # épisodes d'entraînement
    n_val: int = 40
    n_eval_worlds: int = 20             # mondes par jeu d'évaluation (× une trajectoire par politique)
    train_world_pool: int = 1000        # les mondes d'entraînement sont tirés parmi les seeds 0..pool-1
    policies: tuple[str, ...] = POLICY_NAMES
    weights: tuple[float, ...] = (0.25, 0.15, 0.15, 0.15, 0.20, 0.10)
    ppo_dir: str = "runs/step4/ppo_full_2M"
    ppo_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seed: int = 0                       # seed de collecte (choix des mondes/politiques/actions)


def _quantize(m: np.ndarray) -> np.ndarray:
    return np.rint(m * 255.0).astype(np.uint8)


def play_episode(agent, env: AntWorldEnv, world_seed: int, agent_seed: int) -> dict[str, np.ndarray]:
    """Joue un épisode complet et renvoie ses tableaux (non rembourrés)."""
    obs, _ = env.reset(seed=world_seed)
    agent.reset(agent_seed)
    maps, states, actions, rewards, dones = [_quantize(obs["map"])], [obs["state"]], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        a = np.asarray(agent.act(obs, env if agent.privileged else None), dtype=np.int64)
        obs, r, terminated, truncated, _ = env.step(a)
        maps.append(_quantize(obs["map"])); states.append(obs["state"])
        actions.append(a); rewards.append(r); dones.append(terminated)
    return {"maps": np.stack(maps), "states": np.stack(states).astype(np.float32), "actions": np.stack(actions).astype(np.int8),
            "rewards": np.array(rewards, dtype=np.float32), "dones": np.array(dones, dtype=bool)}


def collect(world_cfg: AntWorldConfig, world_seeds: list[int], policy_names: list[str], data_cfg: DataConfig,
            episode_seed0: int, log=print) -> dict[str, np.ndarray]:
    """Un épisode par (monde, politique) ; résultat rembourré à MAX_STEPS pas."""
    models = [Path(data_cfg.ppo_dir) / f"ppo_seed{s}.zip" for s in data_cfg.ppo_seeds]
    models = [m for m in models if m.exists()] if "ppo" in policy_names else []
    policies = {n: make_data_policy(n, world_cfg, models) for n in set(policy_names)}
    env = AntWorldEnv(world_cfg)
    n = len(world_seeds)
    out = {"maps": np.zeros((n, MAX_STEPS + 1, N_MAP_CHANNELS, world_cfg.world.size, world_cfg.world.size), np.uint8),
           "states": np.zeros((n, MAX_STEPS + 1, N_STATE), np.float32),
           "actions": np.zeros((n, MAX_STEPS, 5), np.int8), "rewards": np.zeros((n, MAX_STEPS), np.float32),
           "dones": np.zeros((n, MAX_STEPS), bool), "lengths": np.zeros(n, np.int32),
           "world_seeds": np.array(world_seeds, dtype=np.int64),
           "policies": np.array([POLICY_NAMES.index(p) for p in policy_names], dtype=np.int8)}
    for i, (ws, pol) in enumerate(zip(world_seeds, policy_names)):
        ep = play_episode(policies[pol], env, ws, episode_seed0 + i)
        t = len(ep["actions"])
        out["maps"][i, : t + 1], out["states"][i, : t + 1] = ep["maps"], ep["states"]
        out["actions"][i, :t], out["rewards"][i, :t], out["dones"][i, :t] = ep["actions"], ep["rewards"], ep["dones"]
        out["lengths"][i] = t
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{n} épisodes")
    return out


def build_splits(world_cfg: AntWorldConfig, data_cfg: DataConfig, log=print) -> dict[str, dict[str, np.ndarray]]:
    """Construit les quatre jeux (train, val, eval_seen, eval_unseen)."""
    seeds = SeedManager(data_cfg.seed)
    rng = seeds.rng("data/choice")
    p = np.asarray(data_cfg.weights, dtype=float) / np.sum(data_cfg.weights)
    pool = train_seeds(data_cfg.train_world_pool)
    splits = {}

    def mixture(n, worlds_pool, tag):
        w = [int(x) for x in rng.choice(worlds_pool, size=n)]
        pol = [str(x) for x in rng.choice(list(data_cfg.policies), size=n, p=p)]
        log(f"[{tag}] {n} épisodes ({len(set(w))} mondes distincts)")
        return w, pol

    w, pol = mixture(data_cfg.n_train, pool, "train")
    splits["train"] = collect(world_cfg, w, pol, data_cfg, seeds.int_seed("episodes/train"), log)
    train_worlds = sorted(set(w))
    w, pol = mixture(data_cfg.n_val, val_seeds(1000), "val")
    splits["val"] = collect(world_cfg, w, pol, data_cfg, seeds.int_seed("episodes/val"), log)

    # Jeux d'évaluation : une trajectoire par (monde, politique) pour toutes les politiques SAUF « degenerate » (rarement vivante).
    eval_pols = [x for x in data_cfg.policies if x != "degenerate"]
    seen_worlds = [int(x) for x in rng.choice(train_worlds, size=min(data_cfg.n_eval_worlds, len(train_worlds)), replace=False)]
    for tag, worlds in (("eval_seen", seen_worlds), ("eval_unseen", test_seeds(data_cfg.n_eval_worlds))):
        w = [ws for ws in worlds for _ in eval_pols]
        pol = [pn for _ in worlds for pn in eval_pols]
        log(f"[{tag}] {len(w)} épisodes ({len(worlds)} mondes × {len(eval_pols)} politiques)")
        splits[tag] = collect(world_cfg, w, pol, data_cfg, seeds.int_seed(f"episodes/{tag}"), log)
    return splits


def save_split(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class TrajectoryDataset:
    """Accès aux épisodes d'un jeu et échantillonnage de fenêtres de `length` pas consécutifs."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.a = arrays
        self.lengths = arrays["lengths"]
        self.n = len(self.lengths)

    def __len__(self) -> int:
        return self.n

    def window(self, episode: int, start: int, length: int) -> dict[str, np.ndarray]:
        """Pas start..start+length : observations (length+1), actions/récompenses/morts (length)."""
        a, s, e = self.a, start, start + length
        if e > self.lengths[episode]:
            raise ValueError("fenêtre hors de l'épisode")
        return {"maps": a["maps"][episode, s: e + 1].astype(np.float32) / 255.0, "states": a["states"][episode, s: e + 1],
                "actions": a["actions"][episode, s:e].astype(np.int64), "rewards": a["rewards"][episode, s:e],
                "dones": a["dones"][episode, s:e]}

    def sample_batch(self, batch: int, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Fenêtres tirées uniformément parmi les épisodes assez longs (pondérées par le nombre de positions)."""
        ok = np.flatnonzero(self.lengths >= length)
        if len(ok) == 0:
            raise ValueError(f"aucun épisode d'au moins {length} pas")
        weights = (self.lengths[ok] - length + 1).astype(float)
        eps = rng.choice(ok, size=batch, p=weights / weights.sum())
        starts = [int(rng.integers(0, self.lengths[e] - length + 1)) for e in eps]
        wins = [self.window(int(e), s, length) for e, s in zip(eps, starts)]
        return {k: np.stack([w[k] for w in wins]) for k in wins[0]}

    def persistence_stats(self) -> dict[str, np.ndarray]:
        """Variance de changement à 1 pas (= erreur de la baseline « rien ne change » à 1 pas), par canal / dimension d'état.

        Sert à NORMALISER les pertes : chaque canal compte selon son importance dynamique, pas selon sa rareté.
        """
        m, st, L = self.a["maps"], self.a["states"], self.lengths
        sse_m, sse_s, cnt = np.zeros(m.shape[2]), np.zeros(st.shape[2]), 0
        for e in range(self.n):
            T = int(L[e])
            if T < 1:
                continue
            d = (m[e, 1: T + 1].astype(np.float32) - m[e, :T].astype(np.float32)) / 255.0
            sse_m += (d ** 2).sum(axis=(0, 2, 3))
            sse_s += ((st[e, 1: T + 1] - st[e, :T]) ** 2).sum(axis=0)
            cnt += T
        cells = m.shape[3] * m.shape[4]
        return {"map_var": sse_m / (cnt * cells), "state_var": sse_s / cnt}
