"""Politiques servant à COLLECTER des trajectoires pour le world model (étape 5).

Un world model ne sait prédire que ce qu'il a vu : si les données ne contiennent que de bons joueurs, il ne saura rien
des situations atypiques que le planificateur (étape 6) ira explorer. D'où un mélange volontairement varié :
aléatoire, aléatoire « par blocs » (une action tenue 5 à 20 pas, comme la structure de plan du CEM), heuristique,
heuristique bruitée, PPO stochastique, et des politiques dégénérées (qui font mourir ou stagner la colonie).
Toutes respectent l'interface `Agent` (`reset`, `act`, `privileged`, compteurs de pas).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.heuristic import HeuristicAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from antworld.config import AntWorldConfig
from common.seeding import SeedManager

NVEC = np.array([4, 4, 4, 3, 5])

# Actions constantes « dégénérées » : certaines condamnent la colonie, d'autres la laissent stagner.
DEGENERATE_ACTIONS = (
    (0, 0, 0, 0, 0),   # tous cueillent la nourriture, aucune exploration
    (3, 3, 0, 0, 0),   # personne ne cueille : la colonie meurt de faim
    (3, 0, 3, 0, 0),   # éclaireuses + minerai seulement
    (0, 3, 0, 0, 0),   # bâtisseuses oisives
    (0, 0, 3, 0, 0),   # tous au minerai
    (1, 0, 3, 1, 2),   # minerai + constructions
)


class _Base:
    privileged = False
    sim_steps = 0
    train_env_steps = 0
    name = "policy"

    def reset(self, agent_seed: int) -> None:
        pass


class BlockRandomAgent(_Base):
    """Tire une action aléatoire et la TIENT pendant 5 à 20 pas macro, puis en tire une autre."""

    name = "random_block"

    def __init__(self, min_hold: int = 5, max_hold: int = 20) -> None:
        self.min_hold, self.max_hold = min_hold, max_hold
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        self._rng = SeedManager(agent_seed).rng("agent/random_block")
        self._left, self._action = 0, None

    def act(self, obs, env=None) -> np.ndarray:
        if self._left <= 0:
            self._action = self._rng.integers(0, NVEC)
            self._left = int(self._rng.integers(self.min_hold, self.max_hold + 1))
        self._left -= 1
        return self._action.copy()


class NoisyAgent(_Base):
    """Remplace chaque dimension de l'action de l'agent `base` par une valeur aléatoire avec probabilité `epsilon`."""

    def __init__(self, base, epsilon: float = 0.25, name: str | None = None) -> None:
        self.base, self.epsilon = base, epsilon
        self.name = name or f"{base.name}_noisy"
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        self.base.reset(agent_seed)
        self._rng = SeedManager(agent_seed).rng(f"agent/{self.name}")

    def act(self, obs, env=None) -> np.ndarray:
        a = np.asarray(self.base.act(obs, env), dtype=np.int64).copy()
        flip = self._rng.random(len(a)) < self.epsilon
        a[flip] = self._rng.integers(0, NVEC)[flip]
        return a


class DegenerateAgent(_Base):
    """Une action constante choisie (selon la seed) parmi `DEGENERATE_ACTIONS`, tenue tout l'épisode."""

    name = "degenerate"

    def reset(self, agent_seed: int) -> None:
        rng = SeedManager(agent_seed).rng("agent/degenerate")
        self._action = np.array(DEGENERATE_ACTIONS[int(rng.integers(0, len(DEGENERATE_ACTIONS)))])

    def act(self, obs, env=None) -> np.ndarray:
        return self._action.copy()


class RandomPPOAgent(_Base):
    """PPO STOCHASTIQUE ; à chaque épisode, un des modèles entraînés (une des seeds) est tiré au hasard."""

    name = "ppo"

    def __init__(self, model_paths: list[Path]) -> None:
        if not model_paths:
            raise FileNotFoundError("Aucun modèle PPO trouvé (voir scripts/train_ppo.py) ; utiliser --no-ppo pour s'en passer")
        self.agents = [PPOAgent.from_file(p, deterministic=False) for p in model_paths]
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        rng = SeedManager(agent_seed).rng("agent/ppo_choice")
        self._agent = self.agents[int(rng.integers(0, len(self.agents)))]
        self._agent.reset(agent_seed)

    def act(self, obs, env=None) -> np.ndarray:
        return self._agent.act(obs, env)


POLICY_NAMES = ("random", "random_block", "heuristic", "heuristic_noisy", "ppo", "degenerate")


def make_data_policy(name: str, world_cfg: AntWorldConfig, ppo_models: list[Path] | None = None):
    if name == "random":
        return RandomAgent()
    if name == "random_block":
        return BlockRandomAgent()
    if name == "heuristic":
        return HeuristicAgent(world_cfg=world_cfg)
    if name == "heuristic_noisy":
        return NoisyAgent(HeuristicAgent(world_cfg=world_cfg), 0.25, "heuristic_noisy")
    if name == "degenerate":
        return DegenerateAgent()
    if name == "ppo":
        return RandomPPOAgent(ppo_models or [])
    raise ValueError(f"politique de données inconnue : {name!r} (attendu : {POLICY_NAMES})")
