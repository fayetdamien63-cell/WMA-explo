"""Entraîne PPO sur plusieurs seeds (en parallèle), trace les courbes et compare aux agents de référence.

Monde simplifié (1 ressource, sans brouillard) — étape 4, première partie :
    python scripts/train_ppo.py --world simple --seeds 0 1 2 3 4 --workers 4 --out runs/ppo/simple

Monde complet (long : à lancer sur la machine avec GPU ; ~3 M pas par seed est un bon point de départ) :
    python scripts/train_ppo.py --world full --steps 3000000 --seeds 0 1 2 3 4 --workers 1 --out runs/ppo/full

Par seed d'entraînement : `ppo_seed<k>_curve.jsonl` (courbe : pas d'env, retour de validation, temps), `ppo_seed<k>.zip`
(modèle), puis évaluation FINALE sur les seeds de test (`ppo_seed<k>.jsonl`). `--refs-val` / `--refs-test` pointent vers les
`summary.json` produits par scripts/run_baselines.py sur le MÊME monde (splits val et test) : ils fournissent les niveaux de
référence (aléatoire, heuristique, oracle) pour les cibles d'efficacité et la comparaison finale.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from agents.ppo_agent import PPOConfig, evaluate_on, train_ppo
from antworld import AntWorldConfig
from common.config import load_config
from eval.curves import sample_efficiency
from eval.harness import save_results
from eval.seeds import test_seeds
from eval.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
WORLDS = {"simple": ROOT / "configs" / "antworld_simple.yaml", "full": ROOT / "configs" / "antworld.yaml"}


def train_one(args):
    world_path, ppo_path, seed, steps, out_dir, n_test, eval_every = args
    world_cfg = load_config(AntWorldConfig, world_path)
    cfg = load_config(PPOConfig, ppo_path)
    if eval_every:
        cfg.eval_every = eval_every
    summary = train_ppo(world_cfg, cfg, seed, out_dir, total_steps=steps)
    # Évaluation finale sur les mondes de TEST, avec le harnais commun à tous les agents.
    from stable_baselines3 import PPO
    from common.device import get_device
    model = PPO.load(str(Path(out_dir) / f"ppo_seed{seed}.zip"), device=get_device(cfg.device))
    res = evaluate_on(model, world_cfg, test_seeds(n_test), seed, summary["train_env_steps"])
    save_results(res, out_dir, f"ppo_seed{seed}")
    summary["test_returns"] = [r.ret for r in res]
    summary["test_mean"] = float(np.mean(summary["test_returns"]))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=WORLDS, default="simple")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=None, help="pas d'entraînement par seed (défaut : configs/ppo.yaml)")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--test-worlds", type=int, default=20)
    ap.add_argument("--refs-val", type=Path, default=None, help="summary.json des agents de référence sur les mondes de validation")
    ap.add_argument("--refs-test", type=Path, default=None, help="summary.json des agents de référence sur les mondes de test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "runs" / "ppo" / args.world
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(WORLDS[args.world]), str(args.config), s, args.steps, str(out), args.test_worlds, args.eval_every)
            for s in args.seeds]
    t0 = time.perf_counter()
    if args.workers <= 1:
        summaries = [train_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            summaries = list(pool.map(train_one, jobs, chunksize=1))
    total_wall = time.perf_counter() - t0

    records = [json.loads(l) for s in args.seeds for l in (out / f"ppo_seed{s}_curve.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = [f"PPO, monde « {args.world} », seeds {args.seeds}, {summaries[0]['train_env_steps']} pas par seed, {total_wall:.0f} s au total", ""]
    finals = np.array([s["test_mean"] for s in summaries])
    lo, hi = bootstrap_ci(finals)
    lines.append(f"Retour final sur {args.test_worlds} mondes de TEST (moyenne des seeds d'entraînement) : {finals.mean():.1f} "
                 f"[IC 95 % entre seeds : {lo:.1f}, {hi:.1f}] ; par seed : " + ", ".join(f"{v:.1f}" for v in finals))
    summary = {"world": args.world, "seeds": args.seeds, "steps": summaries[0]["train_env_steps"], "test_by_seed": finals.tolist(),
               "test_mean": float(finals.mean()), "test_ci": [lo, hi], "train_wall_s": [s["train_wall_s"] for s in summaries]}

    if args.refs_test:
        ref = json.loads(args.refs_test.read_text(encoding="utf-8"))["agents"]
        for name, r in ref.items():
            lines.append(f"  référence {name:10s} (test) : {r['mean']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]")
        summary["refs_test"] = {n: r["mean"] for n, r in ref.items()}
        if "random" in ref:
            lines.append(f"  PPO − aléatoire : {finals.mean() - ref['random']['mean']:+.1f}")
    if args.refs_val:
        ref = json.loads(args.refs_val.read_text(encoding="utf-8"))["agents"]
        targets = {}
        if "heuristic" in ref:
            targets = {"90 % de l'heuristique": 0.9 * ref["heuristic"]["mean"], "100 % de l'heuristique": ref["heuristic"]["mean"]}
        if "random" in ref:
            targets["aléatoire + 50 % (heur. − aléa.)"] = ref["random"]["mean"] + 0.5 * (ref["heuristic"]["mean"] - ref["random"]["mean"])
        eff = sample_efficiency(records, targets)
        summary["efficiency"] = eff
        lines += ["", "Efficacité en échantillons (courbe de validation ; pas d'environnement pour atteindre la cible) :"]
        for name, e in eff.items():
            med = f"{e['median_steps']:,.0f}".replace(",", " ") if e["median_steps"] is not None else "jamais atteint"
            lines.append(f"  - {name} (≥ {e['target']:.1f}) : atteint par {e['reached']} seeds, médiane {med} pas")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
