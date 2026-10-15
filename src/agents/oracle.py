"""Oracle : planifie avec le VRAI simulateur et connaît tout le monde (pas de brouillard).

C'est la borne haute (approchée) du banc d'évaluation. Il utilise le même planificateur CEM que celui de
l'étape 6 (`cem_plan`), mais `rollout_fn` simule dans le vrai monde : copie de l'état, application de la
séquence d'actions, mesure de la récompense cumulée.

Deux avantages « interdits » aux autres agents : il lit l'état complet (ressources cachées dans le
brouillard) et il simule la dynamique exacte. Par défaut il NE connaît PAS les futurs tirages aléatoires
(`perfect_foresight=False` : chaque simulation de planification utilise un autre aléa que celui de la
partie réelle) ; sinon il pourrait exploiter le hasard futur, ce qui n'a aucun sens pour un agent réel.
Limite : horizon fini (30 pas macro) et budget limité → borne haute approchée, pas optimum prouvé.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agents.planning import CEMConfig, cem_plan, shift_probs
from antworld.reward import potential, sequence_return
from common.seeding import SeedManager


@dataclass
class OracleConfig:
    cem: CEMConfig = field(default_factory=CEMConfig)
    perfect_foresight: bool = False


class OracleAgent:
    name = "oracle"
    privileged = True

    def __init__(self, cfg: OracleConfig | None = None) -> None:
        self.cfg = cfg or OracleConfig()
        self.sim_steps = 0
        self.train_env_steps = 0
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        self._seeds = SeedManager(agent_seed)
        self._rng = self._seeds.rng("agent/oracle/cem")
        self._plan_queue: list[np.ndarray] = []
        self._probs = None
        self._n_plans = 0

    # ------------------------------------------------------------------------------------------
    def act(self, obs, env=None) -> np.ndarray:
        if not self._plan_queue:
            self._plan(env)
        return self._plan_queue.pop(0)

    def _plan(self, env) -> None:
        world, cfg = env.world, self.cfg.cem
        horizon = min(cfg.horizon, world.cfg.sim.max_steps - world.macro_step)
        work = world.clone()                       # monde de travail : l'état réel n'est jamais touché
        start = world.snapshot()
        rc = env.cfg.reward
        phi0 = potential(world, rc)
        sim_rng_seeds = self._seeds.child(f"plan/{self._n_plans}")
        counter = [0]

        def rollout(seqs: np.ndarray) -> np.ndarray:
            returns = np.empty(len(seqs))
            for i, seq in enumerate(seqs):
                work.restore(start)
                if not self.cfg.perfect_foresight:
                    work._rng = sim_rng_seeds.rng(f"rollout/{counter[0]}")
                counter[0] += 1
                alive_steps, pop_sum = 0, 0
                for a in seq:
                    work.step(a)
                    alive_steps += work.alive
                    pop_sum += work.n_ants
                returns[i] = sequence_return(phi0, work, pop_sum, alive_steps, rc)
            self.sim_steps += len(seqs) * len(seqs[0])
            return returns

        nvec = np.array(world.action_nvec)
        init = None
        if self._probs is not None:                 # réutilise la planification précédente (décalée)
            n_blocks = -(-horizon // cfg.block)
            if self._probs.shape[0] >= n_blocks:
                kmax = int(nvec.max())
                valid = np.arange(kmax)[None, :] < nvec[:, None]
                uniform = valid / valid.sum(1, keepdims=True)
                shifted = shift_probs(self._probs, max(cfg.replan_every // cfg.block, 1), uniform)
                init = shifted[:n_blocks]
        seq, probs, _ = cem_plan(rollout, nvec, cfg, self._rng, horizon, init)
        self._probs = probs
        self._n_plans += 1
        k = max(min(cfg.replan_every, horizon), 1)
        self._plan_queue = [a.copy() for a in seq[:k]]
