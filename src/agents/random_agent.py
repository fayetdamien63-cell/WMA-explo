"""Agent aléatoire : tire chaque dimension de l'action uniformément. Borne basse du banc d'évaluation."""

from __future__ import annotations

import numpy as np

from common.seeding import SeedManager


class RandomAgent:
    name = "random"
    privileged = False

    def __init__(self, nvec=(4, 4, 4, 3, 5)) -> None:
        self.nvec = np.asarray(nvec)
        self.sim_steps = 0
        self.train_env_steps = 0
        self._rng = SeedManager(0).rng("agent/random")

    def reset(self, agent_seed: int) -> None:
        self._rng = SeedManager(agent_seed).rng("agent/random")

    def act(self, obs, env=None) -> np.ndarray:
        return self._rng.integers(0, self.nvec)
