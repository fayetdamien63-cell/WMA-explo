"""Compare Dreamer-lite et PPO : retour de validation contre pas d'environnement (gauche) et contre temps d'entraînement réel (droite).

    python scripts/plot_dreamer.py --dreamer runs/step7/dev_b/dreamer_seed0_curve.jsonl runs/step7/dreamer_h8/dreamer_seed*_curve.jsonl \
        --ppo runs/step4/ppo_full_2M/ppo_seed*_curve.jsonl --refs runs/step4/full_baselines/val/summary.json --out runs/step7/dreamer_vs_ppo.png

Une ligne fine par seed, une ligne épaisse pour la moyenne des seeds (aux pas où toutes les seeds ont un point). Politique STOCHASTIQUE des deux côtés.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"PPO": "#2ca02c", "Dreamer-lite": "#ff7f0e"}


def load(files):
    return {Path(f).stem: [json.loads(l) for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()] for f in files}


def draw(ax_steps, ax_wall, name, curves, max_steps):
    xs_all = []
    for rows in curves.values():
        rows = [r for r in rows if r["train_env_steps"] <= max_steps]
        x, y, w = (np.array([r[k] for r in rows]) for k in ("train_env_steps", "mean_return_stochastic", "wall_s"))
        ax_steps.plot(x, y, color=COLORS[name], alpha=0.25, lw=1)
        ax_wall.plot(w, y, color=COLORS[name], alpha=0.25, lw=1)
        xs_all.append((x, y, w))
    # moyenne sur les seeds : interpolation sur la grille commune (pas) / (temps)
    grid = np.linspace(max(x[0] for x, _, _ in xs_all), min(x[-1] for x, _, _ in xs_all), 60)
    mean = np.mean([np.interp(grid, x, y) for x, y, _ in xs_all], axis=0)
    ax_steps.plot(grid, mean, color=COLORS[name], lw=2.5, label=f"{name} ({len(curves)} seeds, moyenne)")
    wall_mean = np.mean([np.interp(grid, x, w) for x, _, w in xs_all], axis=0)
    ax_wall.plot(wall_mean, mean, color=COLORS[name], lw=2.5, label=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer", nargs="+", type=Path, required=True)
    ap.add_argument("--ppo", nargs="+", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=300_000)
    ap.add_argument("--out", type=Path, default=Path("runs/step7/dreamer_vs_ppo.png"))
    args = ap.parse_args()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    draw(a1, a2, "PPO", load(args.ppo), args.max_steps)
    draw(a1, a2, "Dreamer-lite", load(args.dreamer), args.max_steps)
    if args.refs:
        for n, c in (("random", "gray"), ("heuristic", "#1f77b4")):
            v = json.loads(args.refs.read_text(encoding="utf-8"))["agents"][n]["mean"]
            for a in (a1, a2):
                a.axhline(v, ls="--", color=c, alpha=0.8, label=f"{n} (référence)")
    a1.set_xlabel("pas d'environnement d'entraînement consommés"); a1.set_ylabel("retour moyen (10 mondes de validation, politique stochastique)")
    a1.set_title("Efficacité en échantillons"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.set_xlabel("temps d'entraînement réel (s, hors évaluation, CPU)"); a2.set_title("... et en temps réel"); a2.set_xscale("log"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("écrit", args.out)


if __name__ == "__main__":
    main()
