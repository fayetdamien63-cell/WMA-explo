"""Évalue les agents de référence (aléatoire, heuristique, oracle) et affiche le verdict de la porte.

    python scripts/run_baselines.py --split test --n-worlds 20 --workers 4
    python scripts/run_baselines.py --split val --n-worlds 8 --agents random,heuristic     # réglages : seeds de validation

Sorties dans --out (par défaut runs/step3/<split>/) : <agent>.jsonl (une ligne par épisode), <agent>_series.npz
(courbes par pas), summary.json et summary.md. Utiliser --split test uniquement pour les résultats FINAUX.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agents.factory import AGENT_NAMES, BaselinesConfig
from antworld import AntWorldConfig
from common.config import load_config
from eval.harness import evaluate, exploitation_warnings, log_tensorboard, returns_by_world, save_results
from eval.seeds import test_seeds, train_seeds, val_seeds
from eval.stats import paired_difference, seeds_needed, summarize, variance_decomposition

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": train_seeds, "val": val_seeds, "test": test_seeds}
# Critère de la porte, fixé AVANT de regarder les résultats :
GATE_MAX_NORMALIZED_HEURISTIC = 0.85   # (heuristique − aléatoire) / (oracle − aléatoire) doit rester <= 0.85


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--n-worlds", type=int, default=20)
    ap.add_argument("--agents", default=",".join(AGENT_NAMES))
    ap.add_argument("--random-agent-seeds", type=int, default=5, help="seeds d'agent pour l'agent aléatoire (mesure de variance)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--world-config", type=Path, default=ROOT / "configs" / "antworld.yaml",
                    help="monde évalué (configs/antworld_simple.yaml : version simplifiée de l'étape 4)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    world_cfg = load_config(AntWorldConfig, args.world_config)
    baselines = load_config(BaselinesConfig, ROOT / "configs" / "baselines.yaml")
    worlds = SPLITS[args.split](args.n_worlds)
    out = args.out or ROOT / "runs" / "step3" / args.split
    agents = args.agents.split(",")

    results = {}
    for name in agents:
        agent_seeds = tuple(range(args.random_agent_seeds)) if name == "random" else (0,)
        t0 = time.perf_counter()
        results[name] = evaluate(name, world_cfg, baselines, worlds, agent_seeds, workers=args.workers)
        save_results(results[name], out, name)
        log_tensorboard(results[name], out / "tensorboard" / name)
        print(f"[{name}] {len(results[name])} épisodes en {time.perf_counter() - t0:.0f} s", flush=True)

    # ---- tableau principal --------------------------------------------------------------------
    per_world = {n: returns_by_world(results[n], n, worlds) for n in agents}
    lines = [f"Split : {args.split} ({args.n_worlds} mondes, seeds {worlds[0]}..{worlds[-1]})", "",
             "| agent | retour moyen [IC 95 %] | écart-type entre mondes | survie (pas) | colonies mortes | pop. moy. / creux (après pas 100) / finale | pas d'env (éval) | pas simulés/épisode | temps/épisode | ms/décision |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {"split": args.split, "n_worlds": args.n_worlds, "agents": {}}
    for n in agents:
        s, rs = summarize(per_world[n]), results[n]
        row = {"mean": s.mean, "ci_low": s.ci_low, "ci_high": s.ci_high, "std_between_worlds": s.std,
               "steps": float(np.mean([r.steps for r in rs])), "dead": float(np.mean([r.terminated for r in rs])),
               "final_pop": float(np.mean([r.final_pop for r in rs])), "mean_pop": float(np.mean([r.mean_pop for r in rs])),
               "min_pop": float(np.mean([r.min_pop for r in rs])), "steady_mean_pop": float(np.mean([r.steady_mean_pop for r in rs])), "late_return_share": float(np.mean([r.late_return_share for r in rs])), "env_steps": float(np.mean([r.env_steps for r in rs])),
               "sim_steps": float(np.mean([r.sim_steps for r in rs])), "wall_s": float(np.mean([r.wall_s for r in rs])),
               "decision_ms": float(np.mean([r.decision_ms for r in rs])),
               "seeds_for_5pct_ci": seeds_needed(s.std, 0.05 * abs(s.mean))}
        summary["agents"][n] = row
        lines.append(f"| {n} | {s.mean:.1f} [{s.ci_low:.1f}, {s.ci_high:.1f}] | {s.std:.1f} | {row['steps']:.0f} | "
                     f"{100 * row['dead']:.0f} % | {row['mean_pop']:.1f} / {row['min_pop']:.1f} / {row['final_pop']:.1f} | {row['env_steps']:.0f} | {row['sim_steps']:.0f} | "
                     f"{row['wall_s']:.1f} s | {row['decision_ms']:.1f} |")

    # ---- comparaisons appariées et porte ------------------------------------------------------
    lines += ["", "Comparaisons appariées (mêmes mondes) :"]
    comparisons = [("heuristic", "random"), ("oracle", "heuristic"), ("oracle", "random")]
    for a, b in comparisons:
        if a in per_world and b in per_world:
            d = paired_difference(per_world[a], per_world[b])
            summary[f"{a}_minus_{b}"] = {"mean": d.mean, "ci_low": d.ci_low, "ci_high": d.ci_high, "std": d.std,
                                         "seeds_to_resolve": seeds_needed(d.std, max(abs(d.mean) / 2, 1e-9))}
            lines.append(f"- {a} − {b} : {d.mean:+.1f} [IC 95 % : {d.ci_low:+.1f}, {d.ci_high:+.1f}], "
                         f"{'net' if d.ci_low > 0 else 'PAS net'} ; mondes nécessaires pour le résoudre à ±50 % : "
                         f"{summary[f'{a}_minus_{b}']['seeds_to_resolve']}")
    if {"random", "heuristic", "oracle"} <= set(agents):
        m = {n: summary["agents"][n]["mean"] for n in agents}
        norm = (m["heuristic"] - m["random"]) / (m["oracle"] - m["random"])
        ordered = all(summary[k]["ci_low"] > 0 for k in ("heuristic_minus_random", "oracle_minus_heuristic"))
        passed = ordered and norm <= GATE_MAX_NORMALIZED_HEURISTIC
        summary["gate"] = {"normalized_heuristic": norm, "order_net": ordered, "threshold": GATE_MAX_NORMALIZED_HEURISTIC,
                           "passed": bool(passed)}
        lines += ["", f"Score normalisé de l'heuristique (aléatoire = 0, oracle = 1) : {norm:.2f} (seuil {GATE_MAX_NORMALIZED_HEURISTIC})",
                  f"Ordre aléatoire < heuristique < oracle net (IC 95 % des différences appariées > 0) : {'oui' if ordered else 'NON'}",
                  f"PORTE : {'ATTEINTE' if passed else 'NON ATTEINTE'}"]

    # ---- alertes d'exploitation de la récompense ----------------------------------------------------
    warnings = [w for n in agents for w in exploitation_warnings(results[n])]
    summary["exploitation_warnings"] = warnings
    lines += ["", "Diagnostic anti-exploitation (part du retour dans les 50 derniers pas, creux de population) : " +
              ("RAS" if not warnings else "ALERTE")] + [f"- {w}" for w in warnings]
    if warnings and "gate" in summary:
        summary["gate"]["passed"] = False
        lines.append("PORTE INVALIDÉE : un agent exploite la récompense, l'écart heuristique/oracle n'est pas fiable.")

    # ---- variance entre seeds (agent aléatoire : plusieurs seeds d'agent par monde) --------------
    if "random" in results and args.random_agent_seeds > 1:
        grid = np.array([[next(r.ret for r in results["random"] if r.world_seed == w and r.agent_seed == a)
                          for a in range(args.random_agent_seeds)] for w in worlds])
        vd = variance_decomposition(grid)
        summary["variance_random_agent"] = vd
        lines += ["", "Variance du retour de l'agent aléatoire : "
                      f"entre mondes {vd['var_between_worlds']:.1f}, intra-monde (aléa de l'agent) {vd['var_within_world']:.1f}, "
                      f"part « monde » {100 * vd['world_share']:.0f} %"]
    lines += ["", "Nombre de mondes pour un IC à 95 % de demi-largeur ≤ 5 % du retour moyen : " +
              ", ".join(f"{n} : {summary['agents'][n]['seeds_for_5pct_ci']}" for n in agents)]

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
