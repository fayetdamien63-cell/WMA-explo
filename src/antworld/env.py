"""Environnement Gymnasium : l'interface standard entre le monde et les agents.

    env = AntWorldEnv(cfg)
    obs, info = env.reset(seed=3)          # seed = seed du monde (déterministe)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

- action : MultiDiscrete([4, 4, 4, 3, 5]) — voir docs/WORLD_SPEC.md §7
- observation : dict {"map": (8, 32, 32), "state": (24,)} en float32 dans [0, 1] — voir observation.py
- récompense : voir reward.py
- terminated : la colonie est morte ; truncated : `max_steps` pas macro écoulés
- get_state() / set_state() : sauvegarde/restauration complète (pour planifier avec le vrai simulateur)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from antworld.config import AntWorldConfig
from antworld.observation import N_MAP_CHANNELS, N_STATE, observe_map, state_vector
from antworld.render import render_rgb
from antworld.reward import potential_terms, step_reward
from antworld.world import AntWorld, WorldState


@dataclass(frozen=True)
class EnvState:
    """État complet d'un épisode : seed du monde (→ layout) + photographie de l'état dynamique."""

    world_seed: int
    world: WorldState


class AntWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: AntWorldConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg) if cfg is not None else AntWorldConfig()
        if not self.cfg.obs.fog:
            # Sans brouillard : le nid « voit » toute la carte à chaque tick (ablation de l'étape 8).
            self.cfg.colony.nest_vision = self.cfg.world.size
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode inconnu : {render_mode!r}")
        self.render_mode = render_mode

        s = self.cfg.world.size
        c = self.cfg.colony
        self.action_space = spaces.MultiDiscrete(
            [len(c.scout_shares), len(c.build_shares), len(c.ore_shares), 3, 5])
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0.0, 1.0, (N_MAP_CHANNELS, s, s), dtype=np.float32),
            "state": spaces.Box(0.0, 1.0, (N_STATE,), dtype=np.float32),
        })
        self.world: AntWorld | None = None
        self.world_seed: int | None = None

    # ------------------------------------------------------------------ cycle de vie
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Nouveau monde. `seed` fixe exactement le monde généré (et l'aléa de la partie) ;
        sans seed, le monde est tiré du générateur de l'env (reproductible via `reset(seed=...)` initial)."""
        super().reset(seed=seed)
        self.world_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.world = AntWorld.from_seed(self.cfg, self.world_seed)
        return self._observe(), self._info()

    def step(self, action):
        if self.world is None:
            raise RuntimeError("Appeler reset() avant step().")
        before = potential_terms(self.world, self.cfg.reward)
        self.world.step(action)
        reward, terms = step_reward(before, self.world, self.cfg.reward)
        terminated = not self.world.alive
        truncated = self.world.truncated and not terminated
        info = self._info()
        info["reward_terms"] = terms
        return self._observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ sauvegarde / restauration
    def get_state(self) -> EnvState:
        return EnvState(self.world_seed, self.world.snapshot())

    def set_state(self, state: EnvState) -> None:
        """Restaure un état (même d'un autre env) : le monde est régénéré si la seed diffère."""
        if self.world is None or self.world_seed != state.world_seed:
            self.world = AntWorld.from_seed(self.cfg, state.world_seed)
            self.world_seed = state.world_seed
        self.world.restore(state.world)

    # ------------------------------------------------------------------ observation / info / rendu
    def _observe(self) -> dict[str, np.ndarray]:
        return {"map": observe_map(self.world), "state": state_vector(self.world)}

    def _info(self) -> dict:
        w = self.world
        return {"world_seed": self.world_seed, "step": w.macro_step, "n_ants": w.n_ants, "food": w.food_stock,
                "ore": w.ore_stock, "pop_cap": w.pop_cap, "nursery": w.nursery_level,
                "granary": w.granary_level, "known_fraction": w.known_fraction}

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb(self.world)
        return None
