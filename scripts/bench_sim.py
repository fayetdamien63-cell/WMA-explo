"""Mesure la vitesse du simulateur (à lancer sur chaque machine : MacBook, PC RTX...).

    python scripts/bench_sim.py
"""

import time
from pathlib import Path

import numpy as np

from antworld import AntWorld, AntWorldConfig
from antworld.layout import generate_world
from common.config import load_config
from common.seeding import SeedManager

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "antworld.yaml"


def measure(cfg: AntWorldConfig, seeds=range(8)) -> dict[str, float]:
    # Échauffement : déclenche (ou charge depuis le cache) la compilation Numba.
    w = AntWorld.from_seed(cfg, 10_000)
    w.step(np.zeros(5, dtype=int))

    steps, t_step, t_gen = 0, 0.0, 0.0
    pops = []
    for seed in seeds:
        t0 = time.perf_counter()
        layout = generate_world(cfg.world, seed)
        w = AntWorld(layout, cfg, SeedManager(seed).rng("sim"))
        t_gen += time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        actions = rng.integers(0, w.action_nvec, size=(cfg.sim.max_steps, 5))
        t0 = time.perf_counter()
        for a in actions:
            if w.done:
                break
            w.step(a)
            steps += 1
        t_step += time.perf_counter() - t0
        pops.append(w.n_ants)
    n = len(list(seeds))
    return {"steps_per_s": steps / t_step, "ticks_per_s": steps * cfg.sim.ticks_per_step / t_step,
            "world_gen_ms": 1000 * t_gen / n, "mean_final_pop": float(np.mean(pops))}


if __name__ == "__main__":
    cfg = load_config(AntWorldConfig, CONFIG)
    r = measure(cfg)
    print(f"{r['steps_per_s']:8.0f} pas macro/s  ({r['ticks_per_s']:.0f} ticks/s, {cfg.sim.ticks_per_step} ticks/pas)")
    print(f"{r['world_gen_ms']:8.2f} ms par génération de monde (coût d'un reset)")
    print(f"population finale moyenne pendant la mesure : {r['mean_final_pop']:.1f}")
