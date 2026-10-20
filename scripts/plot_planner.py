"""Trace le score du planificateur en fonction du budget (nombre de séquences simulées par plan), contre les agents de référence.

    python scripts/plot_planner.py runs/step6 --out runs/step6/budget_curve.png

Lit runs/step6/*/summary.json (produits par scripts/run_planner.py). Courbes : planificateur dans le MODÈLE et dans le vrai SIMULATEUR au même
horizon ; lignes horizontales : aléatoire, heuristique, oracle (horizon complet) sur les mêmes mondes. Deuxième panneau : même score contre le
temps de calcul par épisode (axe temps réel), troisième : écart prédit − réel de la récompense cumulée à 25 pas (plan choisi contre séquences aléatoires).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sums = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.runs.glob("*/summary.json"))]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"model": "#d62728", "sim": "#1f77b4"}
    for mode, label in (("sim", "planificateur, vrai simulateur (H25)"), ("model", "planificateur, modèle appris (H25)")):
        rows = sorted((s for s in sums if s["mode"] == mode), key=lambda s: s["sequences_per_plan"])
        if not rows:
            continue
        x = [s["sequences_per_plan"] for s in rows]
        y = [s["return_mean"] for s in rows]
        lo = [s["return_ci"][0] for s in rows]; hi = [s["return_ci"][1] for s in rows]
        axes[0].plot(x, y, "o-", color=colors[mode], label=label); axes[0].fill_between(x, lo, hi, color=colors[mode], alpha=0.15)
        axes[1].plot([s["wall_s_per_episode"] for s in rows], y, "o-", color=colors[mode], label=label)
    refs = {}
    for s in sums:
        for n, v in s.get("refs", {}).items():
            refs[n] = v
    for ax in axes[:2]:
        for n, c in (("random", "gray"), ("heuristic", "#2ca02c"), ("oracle", "black")):
            if n in refs:
                ax.axhline(refs[n], ls="--", color=c, alpha=0.8, label=f"{n} (mêmes mondes)")
        ax.grid(alpha=0.3); ax.set_ylabel("retour d'épisode")
    axes[0].set_xscale("log"); axes[0].set_xlabel("séquences simulées par plan (échantillons × itérations)"); axes[0].set_title("Score selon le budget"); axes[0].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("temps de calcul par épisode (s, CPU)"); axes[1].set_title("Score selon le temps réel")
    rows = sorted((s for s in sums if s["mode"] == "model" and "prediction_vs_reality" in s and "25" in s["prediction_vs_reality"]), key=lambda s: s["sequences_per_plan"])
    if rows:
        x = np.arange(len(rows)); w = 0.38
        axes[2].bar(x - w / 2, [s["prediction_vs_reality"]["25"]["gap_chosen"] for s in rows], w, color="#d62728", label="plan choisi")
        axes[2].bar(x + w / 2, [s["prediction_vs_reality"]["25"]["gap_random"] for s in rows], w, color="gray", label="séquences aléatoires")
        axes[2].set_xticks(x, [str(s["sequences_per_plan"]) for s in rows]); axes[2].axhline(0, color="black", lw=1)
        axes[2].set_xlabel("séquences par plan"); axes[2].set_ylabel("récompense cumulée sur 25 pas : prédit − réel")
        axes[2].set_title("Le planificateur exploite-t-il les erreurs du modèle ?"); axes[2].legend()
    fig.tight_layout()
    out = args.out or args.runs / "budget_curve.png"
    fig.savefig(out, dpi=110)
    print("écrit", out)


if __name__ == "__main__":
    main()
