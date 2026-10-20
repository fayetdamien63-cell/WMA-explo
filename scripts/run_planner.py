"""Évalue le planificateur CEM/MPC, dans le world model appris (mode `model`) ou dans le vrai simulateur au même horizon (mode `sim`).

    # validation du planificateur seul : vrai simulateur, horizon 25 (comparer à l'oracle, horizon 400)
    python scripts/run_planner.py --mode sim --horizon 25 --samples 64 --iters 3 --n-worlds 8 --workers 4 --tag sim_h25_64x3
    # planification dans le modèle appris, budget de 32 × 2 séquences par plan
    python scripts/run_planner.py --mode model --model runs/step5/wm_residual --samples 32 --iters 2 --n-worlds 8 --workers 4 --tag model_32x2

Sorties dans runs/step6/<tag>/ : <tag>.jsonl (épisodes), summary.json (retours, comparaison appariée aux agents de référence de l'étape 3 sur les
MÊMES mondes, coût, diagnostics prédit-contre-réel). Les agents de référence doivent avoir été évalués (scripts/run_baselines.py --split test).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from agents.model_planner import ModelPlannerAgent, ModelPlannerConfig
from agents.oracle import OracleAgent, OracleConfig
from agents.planning import CEMConfig
from antworld import AntWorldConfig
from common.config import load_config
from eval.harness import load_results, returns_by_world, run_episode, save_results
from eval.plan_diagnostics import PlanDiagnostics
from eval.seeds import test_seeds, train_seeds, val_seeds
from eval.stats import paired_difference, summarize

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": train_seeds, "val": val_seeds, "test": test_seeds}


def build_agent(a: dict):
    cem = CEMConfig(horizon=a["horizon"], block=5, n_samples=a["samples"], n_iters=a["iters"], n_elites=a["elites"], replan_every=a["replan"])
    if a["mode"] == "sim":
        agent = OracleAgent(OracleConfig(cem=cem))
        agent.name = a["tag"]
        return agent
    cfg = ModelPlannerConfig(model_path=str(Path(a["model"]) / "model.pt"), cem=cem, torch_threads=a["threads"], record_plans=a["diag_every"] > 0,
                             device=a["device"])
    agent = ModelPlannerAgent(cfg)
    agent.name = a["tag"]
    return PlanDiagnostics(agent, every=a["diag_every"]) if a["diag_every"] > 0 else agent


def job(args):
    a, world_seed = args
    world_cfg = load_config(AntWorldConfig, ROOT / "configs" / "antworld.yaml")
    agent = build_agent(a)
    res = run_episode(agent, world_cfg, world_seed, 0)
    diag = None
    if isinstance(agent, PlanDiagnostics):
        diag = {"summary": agent.summary(), "records": [{k: v for k, v in r.items()} for r in agent.records]}
    return res, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("model", "sim"), required=True)
    ap.add_argument("--model", default=str(ROOT / "runs" / "step5" / "wm_residual"))
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--n-worlds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="threads PyTorch par processus (mode model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--diag-every", type=int, default=8, help="mesure prédit-contre-réel sur un plan sur n (0 = aucune ; mode model)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "step6")
    args = ap.parse_args()
    a = {k: getattr(args, k) for k in ("mode", "model", "horizon", "samples", "iters", "elites", "replan", "threads", "device", "diag_every", "tag")}
    a["diag_every"] = args.diag_every if args.mode == "model" else 0
    worlds = SPLITS[args.split](args.n_worlds)
    out = args.out / args.tag
    t0 = time.perf_counter()
    jobs = [(a, w) for w in worlds]
    if args.workers <= 1:
        done = [job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(job, jobs, chunksize=1))
    results = [d[0] for d in done]
    save_results(results, out, args.tag)

    rets = np.array([r.ret for r in results])
    s = summarize(rets)
    summary = {"tag": args.tag, "mode": args.mode, "split": args.split, "n_worlds": len(worlds), "config": a,
               "return_mean": s.mean, "return_ci": [s.ci_low, s.ci_high], "returns": rets.tolist(),
               "sequences_per_plan": args.samples * args.iters, "imagined_or_simulated_steps_per_episode": float(np.mean([r.sim_steps for r in results])),
               "wall_s_per_episode": float(np.mean([r.wall_s for r in results])), "decision_ms": float(np.mean([r.decision_ms for r in results])),
               "mean_pop": float(np.mean([r.mean_pop for r in results])), "total_wall_s": time.perf_counter() - t0}
    lines = [f"{args.tag} : retour {s.mean:.1f} [{s.ci_low:.1f}, {s.ci_high:.1f}] sur {len(worlds)} mondes ({args.split}) ; "
             f"{summary['sequences_per_plan']} séquences/plan, {summary['imagined_or_simulated_steps_per_episode']:.0f} pas simulés/imaginés par épisode, "
             f"{summary['wall_s_per_episode']:.0f} s/épisode, {summary['decision_ms']:.0f} ms/décision"]
    # Comparaison appariée aux agents de référence (mêmes mondes)
    refs = {}
    for name in ("random", "heuristic", "oracle"):
        path = ROOT / "runs" / "step3" / args.split / f"{name}.jsonl"
        if path.exists():
            try:
                refs[name] = returns_by_world(load_results(path), name, worlds)
            except KeyError:
                pass
    summary["refs"] = {n: float(v.mean()) for n, v in refs.items()}
    summary["paired"] = {}
    for name, v in refs.items():
        d = paired_difference(rets, v)
        summary["paired"][name] = {"mean": d.mean, "ci_low": d.ci_low, "ci_high": d.ci_high}
        lines.append(f"  − {name:9s} (mêmes mondes, retour {v.mean():.1f}) : {d.mean:+.1f} [IC 95 % {d.ci_low:+.1f}, {d.ci_high:+.1f}]")
    if {"random", "oracle"} <= set(refs):
        summary["normalized"] = float((rets.mean() - refs["random"].mean()) / (refs["oracle"].mean() - refs["random"].mean()))
        lines.append(f"  score normalisé (aléatoire = 0, oracle = 1) : {summary['normalized']:.2f}")
    # Diagnostics prédit-contre-réel (agrégés sur les épisodes)
    diags = [d[1] for d in done if d[1] is not None]
    if diags:
        agg = {}
        for k in ("5", "10", "25"):
            rows = [x["summary"][k] for x in diags if x["summary"][k]["n_plans"] > 0]
            if rows:
                w = np.array([r["n_plans"] for r in rows], dtype=float)
                agg[k] = {"n_plans": int(w.sum()), "gap_chosen": float(np.average([r["gap_chosen"] for r in rows], weights=w)),
                          "gap_random": float(np.average([r["gap_random"] for r in rows], weights=w))}
        pc = [(r["pred_chosen"].sum(), r["true_chosen"].sum()) for x in diags for r in x["records"] if len(r["pred_chosen"]) >= 25]
        if len(pc) > 3:
            pred, true = np.array(pc).T
            agg["r2_chosen_25"] = float(1 - ((true - pred) ** 2).sum() / max(((true - true.mean()) ** 2).sum(), 1e-12))
            agg["corr_chosen_25"] = float(np.corrcoef(pred, true)[0, 1])
        summary["prediction_vs_reality"] = agg
        for k in ("5", "10", "25"):
            if k in agg:
                lines.append(f"  récompense cumulée sur {k:>2s} pas, prédit − réel : plan CHOISI {agg[k]['gap_chosen']:+.2f} | séquences ALÉATOIRES {agg[k]['gap_random']:+.2f}  ({agg[k]['n_plans']} plans)")
        if "corr_chosen_25" in agg:
            lines.append(f"  plans choisis, 25 pas : corrélation prédit/réel {agg['corr_chosen_25']:+.2f}, R² {agg['r2_chosen_25']:+.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
