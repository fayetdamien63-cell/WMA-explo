"""Collecte les jeux de trajectoires du world model (train, val, eval_seen, eval_unseen).

    python scripts/collect_data.py --out runs/step5/data
    python scripts/collect_data.py --out runs/step5/data --n-train 2000     # plus de données (machine avec plus de RAM)
    python scripts/collect_data.py --out runs/step5/data --no-ppo           # sans les modèles PPO (étape 4)

Les modèles PPO viennent de scripts/train_ppo.py (par défaut runs/step4/ppo_full_2M/ppo_seed*.zip).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from antworld import AntWorldConfig
from common.config import load_config
from models.data import DataConfig, build_splits, save_split
from agents.data_policies import POLICY_NAMES

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "step5" / "data")
    ap.add_argument("--world-config", type=Path, default=ROOT / "configs" / "antworld.yaml")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-eval-worlds", type=int, default=None)
    ap.add_argument("--ppo-dir", default=None)
    ap.add_argument("--no-ppo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world_cfg = load_config(AntWorldConfig, args.world_config)
    cfg = DataConfig(seed=args.seed)
    for k, v in (("n_train", args.n_train), ("n_val", args.n_val), ("n_eval_worlds", args.n_eval_worlds), ("ppo_dir", args.ppo_dir)):
        if v is not None:
            setattr(cfg, k, v)
    if args.no_ppo:                       # retire PPO du mélange (les autres poids sont renormalisés)
        keep = [i for i, p in enumerate(cfg.policies) if p != "ppo"]
        cfg.policies, cfg.weights = tuple(cfg.policies[i] for i in keep), tuple(cfg.weights[i] for i in keep)
    t0 = time.perf_counter()
    splits = build_splits(world_cfg, cfg)
    for name, arrays in splits.items():
        save_split(args.out / f"{name}.npz", arrays)
        L = arrays["lengths"]
        pols = [str(x) for x in np.array(POLICY_NAMES)[arrays["policies"]]]
        mix = {p: pols.count(p) for p in sorted(set(pols))}
        print(f"{name:12s} {len(L)} épisodes, longueur moyenne {L.mean():.0f} pas, colonies mortes {int((L < 400).sum())}, politiques {mix}")
    print(f"terminé en {time.perf_counter() - t0:.0f} s ; fichiers dans {args.out}")


if __name__ == "__main__":
    main()
