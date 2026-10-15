"""PPO (stable-baselines3) : la baseline SANS modèle du monde, et son agent d'évaluation.

Protocole d'entraînement :
  - les mondes d'entraînement sont tirés dans un pool de seeds d'ENTRAÎNEMENT (jamais de validation/test) ;
  - toutes les `eval_every` pas d'environnement, la politique (déterministe) est évaluée sur des mondes de
    VALIDATION fixes : c'est la courbe d'apprentissage (retour vs pas d'environnement consommés, et vs temps réel) ;
  - à la fin, l'évaluation finale sur les seeds de TEST est faite par le script (harnais commun à tous les agents).
Aucune seed n'est confiée à SB3 pour l'environnement : tout dérive de `SeedManager(seed)` (voir DECISIONS.md).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from antworld.config import AntWorldConfig
from antworld.env import AntWorldEnv
from common.device import get_device
from common.seeding import SeedManager, seed_everything
from eval.harness import run_episode
from eval.seeds import train_seeds, val_seeds
from models.features import AntFeaturesExtractor


@dataclass
class PPOConfig:
    total_steps: int = 300_000          # pas d'environnement d'entraînement (tous environnements confondus)
    n_envs: int = 8                     # environnements en parallèle (DummyVecEnv : même processus)
    n_steps: int = 128                  # pas par environnement et par mise à jour (lot = n_envs × n_steps)
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.995                # horizon effectif ~200 pas : l'épisode en compte 400 et les bâtiments paient tard
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_norm: bool = True            # VecNormalize sur la récompense seulement (l'évaluation utilise la vraie)
    cnn_channels: tuple[int, ...] = (16, 32, 32)
    map_dim: int = 128
    state_dim: int = 64
    net_arch: tuple[int, ...] = (128, 128)
    train_pool_size: int = 1000         # nombre de mondes d'entraînement distincts (seeds 0..n-1)
    eval_every: int = 20_000            # pas d'entraînement entre deux évaluations de la courbe
    eval_worlds: int = 10               # mondes de validation de la courbe
    device: str = "auto"                # auto | cuda | mps | cpu (via get_device)
    torch_threads: int = 1              # 1 par processus quand on entraîne plusieurs seeds en parallèle


class PPOAgent:
    """Enveloppe d'un modèle SB3 à l'interface `Agent` du banc d'évaluation (politique déterministe)."""

    name = "ppo"
    privileged = False
    sim_steps = 0

    def __init__(self, model: PPO, train_env_steps: int = 0) -> None:
        self.model = model
        self.train_env_steps = train_env_steps

    def reset(self, agent_seed: int) -> None:
        pass

    def act(self, obs, env=None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.int64)

    @classmethod
    def from_file(cls, path: str | Path, device: str = "cpu", train_env_steps: int = 0) -> "PPOAgent":
        return cls(PPO.load(str(path), device=get_device(device)), train_env_steps)


def make_vec_env(world_cfg: AntWorldConfig, cfg: PPOConfig, seed: int):
    seeds = SeedManager(seed)
    pool = train_seeds(cfg.train_pool_size)

    def factory(i: int):
        return lambda: Monitor(AntWorldEnv(world_cfg, world_seeds=pool, rng_seed=seeds.int_seed(f"env/{i}")))

    venv = DummyVecEnv([factory(i) for i in range(cfg.n_envs)])
    if cfg.reward_norm:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
    return venv


def build_model(world_cfg: AntWorldConfig, cfg: PPOConfig, seed: int) -> PPO:
    torch.set_num_threads(cfg.torch_threads)
    seed_everything(seed)                       # torch / numpy globaux : initialisation des poids et échantillonnage d'actions
    venv = make_vec_env(world_cfg, cfg, seed)
    policy_kwargs = dict(
        features_extractor_class=AntFeaturesExtractor,
        features_extractor_kwargs=dict(channels=tuple(cfg.cnn_channels), map_dim=cfg.map_dim, state_dim=cfg.state_dim),
        net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)))
    return PPO("MultiInputPolicy", venv, n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
               learning_rate=cfg.learning_rate, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
               ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
               policy_kwargs=policy_kwargs, device=get_device(cfg.device), seed=None, verbose=0)


def evaluate_on(model: PPO, world_cfg: AntWorldConfig, world_seeds, agent_seed: int = 0, train_env_steps: int = 0):
    agent = PPOAgent(model, train_env_steps)
    return [run_episode(agent, world_cfg, ws, agent_seed) for ws in world_seeds]


def train_ppo(world_cfg: AntWorldConfig, cfg: PPOConfig, seed: int, out_dir: str | Path, total_steps: int | None = None,
              log=print) -> dict:
    """Entraîne un PPO (une seed) ; écrit la courbe d'apprentissage et le modèle. Renvoie un petit résumé."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = total_steps or cfg.total_steps
    model = build_model(world_cfg, cfg, seed)
    curve_path = out_dir / f"ppo_seed{seed}_curve.jsonl"
    val_worlds = val_seeds(cfg.eval_worlds)
    state = {"train_wall": 0.0, "next_eval": 0}
    records = []

    def evaluate_now() -> None:
        t0 = time.perf_counter()
        res = evaluate_on(model, world_cfg, val_worlds, seed, model.num_timesteps)
        rets = np.array([r.ret for r in res])
        rec = {"agent": "ppo", "seed": seed, "train_env_steps": int(model.num_timesteps),
               "mean_return": float(rets.mean()), "std_return": float(rets.std(ddof=1)),
               "mean_pop": float(np.mean([r.mean_pop for r in res])), "wall_s": state["train_wall"],
               "eval_wall_s": time.perf_counter() - t0}
        records.append(rec)
        with curve_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"[ppo seed {seed}] {rec['train_env_steps']:>9d} pas  retour val {rec['mean_return']:7.1f}  "
            f"pop moy {rec['mean_pop']:5.1f}  ({rec['wall_s']:.0f} s d'entraînement)")
        state["next_eval"] += cfg.eval_every

    curve_path.write_text("", encoding="utf-8")
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:          # une fois par mise à jour, avant l'optimisation
            if model.num_timesteps >= state["next_eval"]:
                state["train_wall"] += time.perf_counter() - state["t_chunk"]
                evaluate_now()
                state["t_chunk"] = time.perf_counter()

    evaluate_now()                                  # point de départ de la courbe (politique non entraînée)
    state["t_chunk"] = time.perf_counter()
    model.learn(total_timesteps=total, callback=CurveCallback(), progress_bar=False)
    state["train_wall"] += time.perf_counter() - state["t_chunk"]
    evaluate_now()                                  # point final de la courbe
    model.save(str(out_dir / f"ppo_seed{seed}.zip"))
    return {"seed": seed, "train_env_steps": int(model.num_timesteps), "train_wall_s": state["train_wall"],
            "final_val_return": records[-1]["mean_return"], "curve": str(curve_path)}
