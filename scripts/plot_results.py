"""Trace les courbes à partir des journaux du banc d'évaluation.

    python scripts/plot_results.py runs/step3/test                  # résultats finaux (barres, courbes temporelles, appariement)
    python scripts/plot_results.py runs/step3/test --curves runs/ppo/*_curve.jsonl   # courbes d'apprentissage (étape 4+)

Un fichier de courbe d'apprentissage contient une ligne JSON par point :
    {"agent": "ppo", "seed": 0, "train_env_steps": 100000, "mean_return": 71.2}
Les courbes sont tracées en fonction des pas d'environnement consommés (axe d'efficacité en échantillons).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.harness import load_results  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402

COLORS = {"random": "#888888", "heuristic": "#1f77b4", "oracle": "#d62728", "ppo": "#2ca02c",
          "planner": "#9467bd", "dreamer": "#ff7f0e"}


def plot_returns(rows_by_agent: dict[str, list[dict]], ax) -> None:
    for i, (agent, rows) in enumerate(rows_by_agent.items()):
        by_world: dict[int, list[float]] = {}
        for r in rows:
            by_world.setdefault(r["world_seed"], []).append(r["ret"])
        vals = np.array([np.mean(v) for v in by_world.values()])
        lo, hi = bootstrap_ci(vals)
        ax.bar(i, vals.mean(), color=COLORS.get(agent, "#444444"), alpha=0.85)
        ax.errorbar(i, vals.mean(), yerr=[[vals.mean() - lo], [hi - vals.mean()]], color="black", capsize=5)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.25, 0.25, len(vals)), vals, s=8, color="black", alpha=0.4, zorder=3)
    ax.set_xticks(range(len(rows_by_agent)), list(rows_by_agent))
    ax.set_ylabel("retour d'épisode"); ax.set_title("Retour par agent (IC 95 % bootstrap, un point = un monde)")


def plot_series(series_by_agent: dict[str, dict[str, np.ndarray]], axes) -> None:
    for ax, key, title in zip(axes, ("pop", "food", "ore"), ("Population", "Nourriture en stock", "Minerai en stock")):
        for agent, s in series_by_agent.items():
            a = s[key]
            m, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
            ax.plot(m, label=agent, color=COLORS.get(agent, "#444444"))
            ax.fill_between(range(len(m)), m - sd, m + sd, color=COLORS.get(agent, "#444444"), alpha=0.12)
        ax.set_title(title); ax.set_xlabel("pas macro"); ax.grid(alpha=0.3)
    axes[0].legend()


def plot_learning_curves(files: list[Path], ax, ax_time=None, refs: dict[str, float] | None = None,
                         key: str = "mean_return") -> None:
    """Courbes d'apprentissage : moyenne et IC 95 % (bootstrap) entre seeds, vs pas d'environnement (et vs temps réel)."""
    rows = [json.loads(l) for f in files for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
    for agent in sorted({r["agent"] for r in rows}):
        pts: dict[int, list[tuple[float, float]]] = {}
        for r in rows:
            if r["agent"] == agent:
                pts.setdefault(r["train_env_steps"], []).append((r[key], r.get("wall_s", np.nan)))
        xs = sorted(pts)
        m = np.array([np.mean([v for v, _ in pts[x]]) for x in xs])
        lo, hi = np.array([bootstrap_ci(np.array([v for v, _ in pts[x]])) for x in xs]).T
        ax.plot(xs, m, label=agent, color=COLORS.get(agent)); ax.fill_between(xs, lo, hi, alpha=0.2, color=COLORS.get(agent))
        wall = np.array([np.nanmean([w for _, w in pts[x]]) for x in xs])
        if ax_time is not None and not np.isnan(wall).all():
            ax_time.plot(wall, m, label=agent, color=COLORS.get(agent)); ax_time.fill_between(wall, lo, hi, alpha=0.2, color=COLORS.get(agent))
    for name, value in (refs or {}).items():
        for a in (ax, ax_time):
            if a is not None:
                a.axhline(value, linestyle="--", color=COLORS.get(name, "gray"), alpha=0.8, label=f"{name} (référence)")
    ax.set_xlabel("pas d'environnement d'entraînement consommés"); ax.set_ylabel("retour moyen (mondes de validation)")
    ax.set_title("Courbes d'apprentissage : efficacité en échantillons"); ax.legend(); ax.grid(alpha=0.3)
    if ax_time is not None:
        ax_time.set_xlabel("temps d'entraînement réel (s)"); ax_time.set_title("... et en temps réel"); ax_time.grid(alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--curves", nargs="*", type=Path, default=[])
    ap.add_argument("--key", default="mean_return", help="champ des courbes : mean_return (déterministe) ou mean_return_stochastic")
    ap.add_argument("--refs", type=Path, default=None, help="summary.json des agents de référence (lignes horizontales)")
    args = ap.parse_args()
    rows = {}
    for p in sorted(args.results_dir.glob("*.jsonl")):
        records = load_results(p)
        if records and "world_seed" in records[0]:          # journaux d'épisodes ; on ignore les fichiers de courbes
            rows[p.stem] = records
    series = {p.name.replace("_series.npz", ""): dict(np.load(p)) for p in sorted(args.results_dir.glob("*_series.npz"))}

    if args.curves:                                              # mode courbes d'apprentissage
        refs = None
        if args.refs:
            refs = {n: a["mean"] for n, a in json.loads(args.refs.read_text(encoding="utf-8"))["agents"].items()}
        fig, (ax, ax_time) = plt.subplots(1, 2, figsize=(15, 5))
        plot_learning_curves(args.curves, ax, ax_time, refs, args.key)
        fig.tight_layout()
        out = args.results_dir / ("learning_curves.png" if args.key == "mean_return" else f"learning_curves_{args.key.replace('mean_return_', '')}.png")
        fig.savefig(out, dpi=110)
        print("écrit", out)
        return

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    plot_returns(rows, fig.add_subplot(gs[0, 0]))
    plot_series(series, [fig.add_subplot(gs[1, i]) for i in range(3)])
    ax = fig.add_subplot(gs[0, 1:])
    if {"heuristic", "oracle"} <= set(rows):                       # appariement par monde : heuristique vs oracle
        h = {r["world_seed"]: r["ret"] for r in rows["heuristic"]}
        o = {}
        for r in rows["oracle"]:
            o.setdefault(r["world_seed"], []).append(r["ret"])
        w = sorted(h)
        ax.scatter([h[k] for k in w], [np.mean(o[k]) for k in w], color="black")
        lim = [min(min(h.values()), 0), max(max(np.mean(v) for v in o.values()), max(h.values())) * 1.05]
        ax.plot(lim, lim, "--", color="gray"); ax.set_xlabel("retour de l'heuristique"); ax.set_ylabel("retour de l'oracle")
        ax.set_title("Appariement par monde (au-dessus de la diagonale : l'oracle gagne)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.results_dir / "overview.png"
    fig.savefig(out, dpi=110)
    print("écrit", out)


if __name__ == "__main__":
    main()
