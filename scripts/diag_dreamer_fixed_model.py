"""Diagnostic (étape 7) : acteur-critique entraîné dans l'imagination d'un modèle FIXE (par défaut : celui de l'étape 5).

Sépare deux questions : (1) l'acteur-critique sait-il apprendre dans l'imagination d'un BON modèle ? (2) est-ce l'apprentissage EN LIGNE du
modèle qui limite Dreamer-lite ? Ce n'est PAS un résultat d'efficacité en échantillons : le modèle de l'étape 5 a été entraîné sur des données
(dont des trajectoires de PPO) qu'un agent honnête ne possède pas. Les croyances de départ viennent des trajectoires d'entraînement de l'étape 5.

    python scripts/diag_dreamer_fixed_model.py --updates 3000 --out runs/step7/diag_fixed_model
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agents.dreamer_agent import DreamerAgent, DreamerConfig, evaluate_dreamer
from antworld import AntWorldConfig
from common.config import load_config
from common.seeding import SeedManager, seed_everything
from eval.harness import run_episode
from eval.seeds import val_seeds
from models.data import load_split
from models.dreamer import ActorCritic, ModelWorld, ReplayBuffer, sample_start_beliefs
from models.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "runs/step5/wm_residual/model.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "runs/step5/data/train.npz")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/dreamer.yaml")
    ap.add_argument("--world-config", type=Path, default=ROOT / "configs/antworld.yaml")
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/step7/diag_fixed_model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    cfg, wcfg = load_config(DreamerConfig, args.config), load_config(AntWorldConfig, args.world_config)
    model = load_checkpoint(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    arrays = load_split(args.data)
    buf = ReplayBuffer(len(arrays["lengths"]), wcfg.world.size, wcfg.sim.max_steps)
    for k in ("maps", "states", "actions", "rewards", "dones", "lengths"):
        buf.arrays[k][...] = arrays[k]
    world = ModelWorld(model, wcfg.sim.max_steps)
    ac = ActorCritic(world.feat_dim, cfg.ac, "cpu")
    agent = DreamerAgent(model, ac, max_steps=wcfg.sim.max_steps)
    seeds = SeedManager(args.seed)
    rng, gen = seeds.rng("starts"), torch.Generator().manual_seed(1)
    val, curve, t0 = val_seeds(cfg.eval_worlds), [], time.perf_counter()
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            ev = evaluate_dreamer(agent, wcfg, cfg, val, args.seed)
            rec = {"updates": u, "wall_s": time.perf_counter() - t0, **ev}
            curve.append(rec)
            print(f"{u:6d} mises à jour  retour val {ev['mean_return_stochastic']:6.1f}  pop {ev['mean_pop']:5.1f}  écart rêve−réel politique "
                  f"{ev['gap_policy']:+.2f} / aléatoire {ev['gap_random']:+.2f}  biais critique {ev['value_bias']:+.2f}  ({rec['wall_s']:.0f} s)", flush=True)
            (args.out / "curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")
        if u == args.updates:
            break
        belief = world.start(sample_start_beliefs(model, buf, cfg.ac_batch, cfg.ac_context, rng, "cpu"))
        info = ac.update(ac.imagine(world, belief, cfg.ac.horizon, gen))
        if u % 100 == 0:
            print(f"   [{u}] entropie {info['entropy']:.2f} rendement rêvé {info['dream_return']:.2f} récompense rêvée {info['dream_reward']:.3f}", flush=True)


if __name__ == "__main__":
    main()
