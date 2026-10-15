"""Diagnostic d'un PPO entraîné : politique déterministe vs stochastique, entropie, actions choisies.

    python scripts/diagnose_ppo.py runs/step4/ppo_full --world full --seeds 0 1 2 3 4

Sert à comprendre un plateau : politique qui s'est effondrée sur une action constante ? évaluation argmax trompeuse ?
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from agents.ppo_agent import PPOAgent
from antworld import AntWorldConfig
from antworld.env import AntWorldEnv
from common.config import load_config
from common.device import get_device
from eval.harness import run_episode
from eval.seeds import val_seeds

ROOT = Path(__file__).resolve().parent.parent
WORLDS = {"simple": "antworld_simple.yaml", "full": "antworld.yaml"}


class StochasticPPOAgent(PPOAgent):
    name = "ppo_stochastic"

    def act(self, obs, env=None):
        action, _ = self.model.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--world", choices=WORLDS, default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--worlds", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(AntWorldConfig, ROOT / "configs" / WORLDS[args.world])
    worlds = val_seeds(args.worlds)

    for seed in args.seeds:
        model = PPO.load(str(args.run_dir / f"ppo_seed{seed}.zip"), device=get_device("cpu"))
        det = [run_episode(PPOAgent(model), cfg, w, seed) for w in worlds]
        sto = [run_episode(StochasticPPOAgent(model), cfg, w, seed) for w in worlds]
        # Entropie moyenne de la politique et actions les plus fréquentes (politique déterministe) sur un épisode.
        env = AntWorldEnv(cfg)
        obs, _ = env.reset(seed=worlds[0])
        ent, acts, done = [], [], False
        while not done:
            t = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
            with torch.no_grad():
                dist = model.policy.get_distribution(t)
                ent.append(float(dist.entropy().mean()))
            a = PPOAgent(model).act(obs)
            acts.append(tuple(a.tolist()))
            obs, _, te, tr, _ = env.step(a)
            done = te or tr
        top = Counter(acts).most_common(3)
        print(f"seed {seed}: déterministe {np.mean([r.ret for r in det]):6.1f} (pop moy {np.mean([r.mean_pop for r in det]):4.1f}) | "
              f"stochastique {np.mean([r.ret for r in sto]):6.1f} (pop moy {np.mean([r.mean_pop for r in sto]):4.1f}) | "
              f"entropie moy {np.mean(ent):.2f} (max {np.log(4) * 3 + np.log(3) + np.log(5):.2f}) | actions déterministes les + fréquentes : {top}")


if __name__ == "__main__":
    main()
