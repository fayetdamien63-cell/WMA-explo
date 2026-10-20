"""Calcule le plafond de prédictibilité (voir src/eval/wm_ceiling.py) et le compare à un modèle entraîné.

    python scripts/bayes_ceiling.py --model runs/step5/wm_residual --data runs/step5/data --split eval_unseen --branches 32
"""

import argparse
import json
from pathlib import Path

import torch

from antworld import AntWorldConfig
from common.config import load_config
from eval.wm_ceiling import bayes_stats, ceiling_summary
from eval.wm_metrics import CHANNELS, HORIZONS, summarize, window_stats
from models.data import TrajectoryDataset, load_split
from models.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "runs" / "step5" / "data")
    ap.add_argument("--split", default="eval_unseen")
    ap.add_argument("--branches", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    world_cfg = load_config(AntWorldConfig, ROOT / "configs" / "antworld.yaml")
    ds = TrajectoryDataset(load_split(args.data / f"{args.split}.npz"))
    model = load_checkpoint(args.model / "model.pt")
    stats = window_stats(model, ds)
    res = summarize(stats, n_boot=100)
    ceil = ceiling_summary(stats, bayes_stats(world_cfg, ds, stats, args.branches))
    out = {"split": args.split, "branches": args.branches, "ceiling": ceil,
           "model": {str(h): {k: res["horizons"][str(h)][k]["value"] for k in ("map_skill_macro", "state_skill_macro", "reward_r2")} for h in HORIZONS}}
    (args.model / f"ceiling_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Le plafond ne dépend PAS du modèle (seulement des fenêtres et du simulateur) : copie mise en cache à côté des données, lue par
    # scripts/eval_world_model.py pour le critère « part du plafond ».
    (args.data / f"ceiling_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{args.split}] plafond de prédictibilité (futurs aléatoires du simulateur, état complet connu) contre le modèle {args.model.name}")
    print("horizon | carte : plafond / modèle / part atteinte | état : plafond / modèle | récompense R² : plafond / modèle")
    for h in HORIZONS:
        c, m = ceil[str(h)], out["model"][str(h)]
        print(f"{h:7d} | {c['map_skill_macro']:+.2f} / {m['map_skill_macro']:+.2f} / {100 * m['map_skill_macro'] / c['map_skill_macro']:3.0f} % | "
              f"{c['state_skill_macro']:+.2f} / {m['state_skill_macro']:+.2f} | {c['reward_r2']:+.2f} / {m['reward_r2']:+.2f}")
    for h in (1, 10):
        c = ceil[str(h)]["map_skill_by_channel"]
        mc = res["horizons"][str(h)]["map_skill_by_channel"]
        print(f"par canal à {h:2d} pas (plafond / modèle) : " + ", ".join(f"{n} {c[i]:+.2f}/{mc[n]:+.2f}" for i, n in enumerate(CHANNELS) if n != "nest"))


if __name__ == "__main__":
    main()
