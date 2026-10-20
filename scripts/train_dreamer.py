"""Entraîne Dreamer-lite sur plusieurs seeds (en parallèle), évalue sur les mondes de TEST et compare à PPO / aux références.

    python scripts/train_dreamer.py --seeds 0 1 2 --workers 3 --steps 200000 --out runs/step7/dreamer \
        --refs-val runs/step4/full_baselines/val/summary.json --refs-test runs/step3/test/summary.json

Par seed : `dreamer_seed<k>_curve.jsonl` (courbe de validation : retour, temps d'entraînement, écart rêve-réalité, biais du critique),
`dreamer_seed<k>_train.jsonl` (pertes, un point par épisode d'entraînement), `dreamer_seed<k>.pt` (modèle + politique), puis évaluation
FINALE sur les seeds de test (`dreamer_seed<k>.jsonl`, politique stochastique : même convention que la référence PPO).
Dans la config : `torch_threads: 1` par processus ; `device: cuda` sur la machine avec GPU.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from agents.dreamer_agent import DreamerAgent, DreamerConfig, load_dreamer, train_dreamer
from antworld import AntWorldConfig
from common.config import load_config
from common.device import get_device
from eval.curves import sample_efficiency
from eval.harness import run_episode, save_results
from eval.seeds import test_seeds
from eval.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent


def train_one(args):
    world_path, cfg_path, seed, steps, out_dir, n_test, eval_every, threads, overrides = args
    world_cfg = load_config(AntWorldConfig, world_path)
    cfg = load_config(DreamerConfig, cfg_path, overrides)
    if eval_every:
        cfg.eval_every = eval_every
    if threads:
        cfg.torch_threads = threads
    summary = train_dreamer(world_cfg, cfg, seed, out_dir, total_steps=steps)
    if n_test == 0:                    # run de développement : on ne touche PAS aux mondes de test (réglages sur validation seulement)
        return {**summary, "test_returns": [], "test_mean": float("nan"), "test_min_pop": float("nan"), "test_late_share": float("nan")}
    model, ac = load_dreamer(Path(out_dir) / f"dreamer_seed{seed}.pt", get_device(cfg.device))
    agent = DreamerAgent(model, ac, deterministic=False, max_steps=world_cfg.sim.max_steps)
    agent.train_env_steps = summary["train_env_steps"]
    res = [run_episode(agent, world_cfg, ws, seed) for ws in test_seeds(n_test)]
    save_results(res, out_dir, f"dreamer_seed{seed}")
    summary["test_returns"] = [r.ret for r in res]
    summary["test_mean"] = float(np.mean(summary["test_returns"]))
    summary["test_min_pop"] = float(np.mean([r.min_pop for r in res]))
    summary["test_late_share"] = float(np.mean([r.late_return_share for r in res]))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-config", type=Path, default=ROOT / "configs" / "antworld.yaml")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dreamer.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="threads torch par processus (défaut : valeur de la config)")
    ap.add_argument("--set", nargs="*", default=[], help="surcharges de config, ex. ac.ent_coef=0.03 wm_updates_per_1k=40")
    ap.add_argument("--test-worlds", type=int, default=20, help="0 : pas d'évaluation de test (développement)")
    ap.add_argument("--refs-val", type=Path, default=None)
    ap.add_argument("--refs-test", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "step7" / "dreamer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(args.world_config), str(args.config), s, args.steps, str(args.out), args.test_worlds, args.eval_every, args.threads, list(args.set))
            for s in args.seeds]
    t0 = time.perf_counter()
    if args.workers <= 1:
        summaries = [train_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            summaries = list(pool.map(train_one, jobs, chunksize=1))
    total_wall = time.perf_counter() - t0

    records = [json.loads(l) for s in args.seeds for l in (args.out / f"dreamer_seed{s}_curve.jsonl").read_text(encoding="utf-8").splitlines()]
    finals = np.array([s["test_mean"] for s in summaries])
    lo, hi = bootstrap_ci(finals) if len(finals) > 1 and not np.isnan(finals).any() else (finals[0], finals[0])
    lines = [f"Dreamer-lite, seeds {args.seeds}, {summaries[0]['train_env_steps']} pas par seed, {total_wall:.0f} s au total", "",
             f"Retour final sur {args.test_worlds} mondes de TEST (politique stochastique) : {finals.mean():.1f} [IC 95 % entre seeds : {lo:.1f}, {hi:.1f}] ; "
             "par seed : " + ", ".join(f"{v:.1f}" for v in finals),
             f"  population minimale après l'échauffement (moy.) : {np.mean([s['test_min_pop'] for s in summaries]):.1f} ; "
             f"part du retour dans les 50 derniers pas : {np.mean([s['test_late_share'] for s in summaries]):.2f}"]
    summary = {"seeds": args.seeds, "steps": summaries[0]["train_env_steps"], "train_wall_s": [s["train_wall_s"] for s in summaries],
               "test": {"by_seed": finals.tolist(), "mean": float(finals.mean()), "ci": [float(lo), float(hi)]}}
    refs_test = json.loads(args.refs_test.read_text(encoding="utf-8"))["agents"] if args.refs_test else {}
    for name, r in refs_test.items():
        lines.append(f"  référence {name:10s} (test) : {r['mean']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]")
    if args.refs_val:
        ref = json.loads(args.refs_val.read_text(encoding="utf-8"))["agents"]
        targets = {"90 % de l'heuristique": 0.9 * ref["heuristic"]["mean"], "100 % de l'heuristique": ref["heuristic"]["mean"]}
        eff = sample_efficiency(records, targets, agent="dreamer", key="mean_return_stochastic")
        summary["efficiency"] = eff
        lines += ["", "Efficacité en échantillons (courbe de validation, politique stochastique ; pas d'environnement pour atteindre la cible) :"]
        for name, e in eff.items():
            med = f"{e['median_steps']:,.0f}".replace(",", " ") if e["median_steps"] is not None else "jamais atteint"
            walls = [w for w in e["wall_s"] if w is not None]
            lines.append(f"  - {name} (≥ {e['target']:.1f}) : atteint par {e['reached']} seeds, médiane {med} pas"
                         + (f", temps d'entraînement médian {np.median(walls):.0f} s" if walls else ""))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
