"""Banc d'évaluation : fait jouer des agents sur des mondes (seeds) et journalise des métriques.

    results = evaluate("heuristic", world_cfg, baselines, world_seeds=test_seeds(20), agent_seeds=(0,))

Une exécution = (agent, seed du monde, seed de l'agent). L'aléa de l'agent d'un épisode dérive de
(seed de l'agent, seed du monde) : le résultat ne dépend ni de l'ordre d'évaluation ni du parallélisme.

Deux axes de coût sont mesurés, car un world model gagne en pas d'environnement mais coûte du calcul :
  - pas d'environnement : `env_steps` (évaluation), `train_env_steps` (entraînement de l'agent),
    `sim_steps` (pas simulés pour planifier : le « coût caché » de l'oracle/des planificateurs)
  - temps réel : `wall_s` (épisode) et `decision_ms` (temps moyen d'une décision).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from agents.factory import BaselinesConfig, make_agent
from antworld.config import AntWorldConfig
from antworld.env import AntWorldEnv
from common.seeding import SeedManager

SERIES_KEYS = ("pop", "food", "ore", "reward")
WARMUP = 100                      # phase de croissance initiale ignorée pour le creux de population
LATE_WINDOW = 50                  # « fin d'épisode » = les 50 derniers pas macro

# Seuils d'alerte d'exploitation (voir `exploitation_warnings`) :
MAX_LATE_SHARE = 0.25             # plus d'un quart du retour gagné dans les 50 derniers pas (sur 400) : suspect
MIN_POP_RATIO = 0.35              # population minimale < 35 % de la population moyenne : colonie « sacrifiée »


@dataclass
class EpisodeResult:
    agent: str
    world_seed: int
    agent_seed: int
    ret: float                  # somme des récompenses
    steps: int                  # pas macro joués = durée de survie (400 si la colonie survit)
    terminated: bool            # la colonie est morte
    final_pop: int
    mean_pop: float
    max_pop: int
    min_pop: int                # population minimale APRÈS la phase de croissance (pas WARMUP..fin) : détecte l'« hibernation »
    steady_mean_pop: float      # population moyenne sur la même fenêtre (pas WARMUP..fin)
    late_return_share: float    # part du retour gagnée dans les 50 derniers pas (détecte l'exploitation de la fin d'épisode)
    final_food: float
    final_ore: float
    nursery: int
    granary: int
    env_steps: int              # pas d'environnement consommés par cette évaluation
    train_env_steps: int        # pas d'environnement consommés pour entraîner l'agent (0 hors apprentissage)
    sim_steps: int              # pas de simulateur consommés pour planifier
    wall_s: float
    decision_ms: float
    series: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def scalars(self) -> dict:
        d = asdict(self)
        d.pop("series")
        return d


def run_episode(agent, cfg: AntWorldConfig, world_seed: int, agent_seed: int = 0) -> EpisodeResult:
    env = AntWorldEnv(cfg)
    obs, _ = env.reset(seed=world_seed)
    agent.reset(SeedManager(agent_seed).int_seed(f"world/{world_seed}"))
    sim0 = agent.sim_steps
    series = {k: [] for k in SERIES_KEYS}
    ret, decision_time, t0 = 0.0, 0.0, time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        d0 = time.perf_counter()
        action = agent.act(obs, env if agent.privileged else None)
        decision_time += time.perf_counter() - d0
        obs, r, terminated, truncated, info = env.step(action)
        ret += r
        for k, v in zip(SERIES_KEYS, (info["n_ants"], info["food"], info["ore"], r)):
            series[k].append(v)
    steps = info["step"]
    pop = np.array(series["pop"], dtype=float)
    late = float(np.sum(series["reward"][-LATE_WINDOW:]))
    steady = pop[WARMUP:] if len(pop) > WARMUP else pop      # toutes les colonies démarrent à 12 fourmis : on ignore la croissance initiale
    return EpisodeResult(
        agent=agent.name, world_seed=world_seed, agent_seed=agent_seed, ret=float(ret), steps=steps,
        terminated=bool(terminated), final_pop=int(pop[-1]), mean_pop=float(pop.mean()), max_pop=int(pop.max()), min_pop=int(steady.min()), steady_mean_pop=float(steady.mean()),
        late_return_share=late / ret if abs(ret) > 1e-9 else 0.0,
        final_food=float(info["food"]), final_ore=float(info["ore"]), nursery=info["nursery"],
        granary=info["granary"], env_steps=steps, train_env_steps=int(agent.train_env_steps),
        sim_steps=int(agent.sim_steps - sim0), wall_s=time.perf_counter() - t0,
        decision_ms=1000 * decision_time / max(steps, 1),
        series={k: np.array(v, dtype=np.float32) for k, v in series.items()})


def _job(args) -> EpisodeResult:
    name, world_cfg, baselines, world_seed, agent_seed = args
    return run_episode(make_agent(name, world_cfg, baselines), world_cfg, world_seed, agent_seed)


def evaluate(name: str, world_cfg: AntWorldConfig, baselines: BaselinesConfig | None, world_seeds,
             agent_seeds=(0,), workers: int = 1) -> list[EpisodeResult]:
    """Évalue l'agent `name` sur chaque (monde, seed d'agent). `workers > 1` : processus parallèles
    (méthode « spawn », portable Mac/Windows/Linux). Ordre des résultats = ordre des seeds."""
    jobs = [(name, world_cfg, baselines, ws, a) for ws in world_seeds for a in agent_seeds]
    if workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(_job, jobs, chunksize=1))


# ----------------------------------------------------------------------------------------------
# Journalisation et relecture
# ----------------------------------------------------------------------------------------------

def save_results(results: list[EpisodeResult], out_dir: str | Path, tag: str) -> Path:
    """Écrit `<tag>.jsonl` (un épisode par ligne : métriques scalaires) et `<tag>_series.npz`
    (courbes par pas : population, nourriture, minerai, récompense ; NaN après la fin d'un épisode)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.scalars()) + "\n")
    horizon = max(len(r.series["pop"]) for r in results)
    arrays = {k: np.full((len(results), horizon), np.nan, dtype=np.float32) for k in SERIES_KEYS}
    for i, r in enumerate(results):
        for k in SERIES_KEYS:
            arrays[k][i, : len(r.series[k])] = r.series[k]
    np.savez_compressed(out_dir / f"{tag}_series.npz", world_seeds=[r.world_seed for r in results],
                        agent_seeds=[r.agent_seed for r in results], agents=[r.agent for r in results], **arrays)
    return path


def load_results(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def log_tensorboard(results: list[EpisodeResult], logdir: str | Path) -> bool:
    """Écrit les métriques dans TensorBoard (courbes par agent). Renvoie False si TensorBoard est absent."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False
    with SummaryWriter(str(logdir)) as w:
        for i, r in enumerate(results):
            for key in ("ret", "steps", "final_pop", "mean_pop", "sim_steps", "wall_s", "decision_ms"):
                w.add_scalar(f"eval/{r.agent}/{key}", getattr(r, key), i)
    return True


def returns_by_world(results, agent: str, world_seeds) -> np.ndarray:
    """Retour moyen (sur les seeds d'agent) de `agent` pour chaque monde, dans l'ordre de `world_seeds`."""
    out = []
    for ws in world_seeds:
        vals = [(r["ret"] if isinstance(r, dict) else r.ret) for r in results
                if (r["agent"] if isinstance(r, dict) else r.agent) == agent
                and (r["world_seed"] if isinstance(r, dict) else r.world_seed) == ws]
        if not vals:
            raise KeyError(f"pas de résultat pour {agent} sur le monde {ws}")
        out.append(float(np.mean(vals)))
    return np.array(out)


def exploitation_warnings(results: list[EpisodeResult]) -> list[str]:
    """Signaux qu'un agent exploite la définition de la récompense plutôt que de bien gérer la colonie.

    Leçon de l'étape 3 : un oracle optimisant l'ancienne récompense (état final seul) faisait tomber la colonie à
    2-8 fourmis puis la faisait exploser dans les derniers pas. On surveille donc, par agent :
      - la part du retour gagnée dans les 50 derniers pas,
      - le creux de population par rapport à la population moyenne.
    """
    warnings = []
    for agent in sorted({r.agent for r in results}):
        rs = [r for r in results if r.agent == agent]
        late = float(np.mean([r.late_return_share for r in rs]))
        ratio = float(np.mean([r.min_pop / max(r.steady_mean_pop, 1e-9) for r in rs]))
        if late > MAX_LATE_SHARE:
            warnings.append(f"{agent} : {100 * late:.0f} % du retour gagné dans les {LATE_WINDOW} derniers pas (seuil {100 * MAX_LATE_SHARE:.0f} %)")
        if ratio < MIN_POP_RATIO:
            warnings.append(f"{agent} : population minimale (après le pas {WARMUP}) = {100 * ratio:.0f} % de la population moyenne sur la même période (seuil {100 * MIN_POP_RATIO:.0f} %)")
    return warnings
