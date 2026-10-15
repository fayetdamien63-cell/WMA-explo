"""`AntWorld` : l'état d'une partie et son interface de pas macro.

Cette classe est un mince habillage Python autour du noyau Numba (kernels.py) : elle possède les
tableaux d'état, tire les nombres aléatoires, et expose des propriétés lisibles.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from antworld import indices as ix
from antworld import kernels
from antworld.config import AntWorldConfig
from antworld.layout import WorldLayout, generate_world
from common.seeding import SeedManager

# Taille de chaque dimension de l'action macro (ordre : ACT_SCOUT, ACT_BUILD_SHARE, ACT_ORE_SHARE,
# ACT_BUILD_CHOICE, ACT_TARGET).
N_BUILD_CHOICES = 3
N_TARGETS = 5


@dataclass(frozen=True)
class WorldState:
    """Photographie complète de l'état dynamique d'une partie (le layout et la config sont immuables).

    Contient aussi l'état du générateur aléatoire : restaurer puis rejouer les mêmes actions redonne
    exactement la même suite. Nécessaire à l'oracle (planifier avec le vrai simulateur).
    """

    grid: np.ndarray
    flow: np.ndarray
    ants: np.ndarray
    col: np.ndarray
    rng_state: dict
    macro_step: int


def build_params(cfg: AntWorldConfig) -> np.ndarray:
    """Convertit la config en vecteur de paramètres float64 pour le noyau."""
    c, w = cfg.colony, cfg.world
    P = np.zeros(ix.N_PARAMS)
    P[ix.P_SIZE] = w.size
    P[ix.P_CARRY] = c.carry_capacity
    P[ix.P_EAT] = c.eat_rate
    P[ix.P_STARVE] = c.starve_rate
    P[ix.P_REGROW] = w.food_regrow_rate
    P[ix.P_REGROW_SEED] = w.food_regrow_seed
    P[ix.P_FORAGER_VIS] = c.forager_vision
    P[ix.P_SCOUT_VIS] = c.scout_vision
    P[ix.P_NEST_VIS] = c.nest_vision
    P[ix.P_TURN_PROB] = c.scout_turn_prob
    P[ix.P_BIAS] = c.scout_bias
    P[ix.P_LAY_RATE] = c.lay_rate
    P[ix.P_BIRTH_COST] = c.birth_cost
    P[ix.P_RESERVE] = c.food_reserve
    P[ix.P_BASE_CAP] = c.base_pop_cap
    P[ix.P_NURSERY_GAIN] = c.nursery_gain
    P[ix.P_BASE_STORAGE] = c.base_food_storage
    P[ix.P_GRANARY_GAIN] = c.granary_gain
    P[ix.P_MAX_LEVEL] = c.max_building_level
    P[ix.P_NURSERY_COST] = c.nursery_ore_cost
    P[ix.P_GRANARY_COST] = c.granary_ore_cost
    P[ix.P_BUILD_WORK] = c.build_work
    P[ix.P_MAX_ANTS] = c.max_ants
    P[ix.P_SCOUT_SHARES:ix.P_SCOUT_SHARES + 4] = c.scout_shares
    P[ix.P_BUILD_SHARES:ix.P_BUILD_SHARES + 4] = c.build_shares
    P[ix.P_ORE_SHARES:ix.P_ORE_SHARES + 4] = c.ore_shares
    return P


class AntWorld:
    """Une partie : un layout figé + l'état dynamique (grille, fourmis, colonie)."""

    def __init__(self, layout: WorldLayout, cfg: AntWorldConfig, sim_rng: np.random.Generator) -> None:
        if layout.size != cfg.world.size:
            raise ValueError("layout.size et cfg.world.size diffèrent")
        self.cfg = cfg
        self.layout = layout
        self.size = layout.size
        self._rng = sim_rng
        self._P = build_params(cfg)
        self._food_cells = np.flatnonzero(layout.food_cap > 0).astype(np.int32)

        n = layout.size**2
        c = cfg.colony
        self.grid = np.zeros((ix.N_GRID, n))
        self.grid[ix.G_AMOUNT + ix.R_FOOD] = layout.food_cap          # cases de nourriture pleines
        self.grid[ix.G_AMOUNT + ix.R_ORE] = layout.ore_amount
        self.flow = np.full((2, n), -1, dtype=np.int32)
        self.ants = np.zeros((c.max_ants, ix.N_ANT_FIELDS), dtype=np.int32)
        self.col = np.zeros(ix.N_COL)

        self.ants[: c.initial_ants, ix.A_CELL] = layout.nest
        self.ants[: c.initial_ants, ix.A_ROLE] = ix.ROLE_FOOD
        self.col[ix.C_N_ANTS] = c.initial_ants
        self.col[ix.C_FOOD_STOCK] = c.initial_food
        self.col[ix.C_ORE_STOCK] = c.initial_ore

        # La colonie voit d'emblée les alentours du nid ; on calcule les champs de flux initiaux.
        kernels.reveal(self.grid, self.col, layout.nest, c.nest_vision, self.size)
        for r in range(2):
            kernels.recompute_flow(self.grid, self.flow, layout.terrain, self.size, r)
        self.col[ix.C_DIRTY:ix.C_DIRTY + 2] = 0.0

        self.macro_step = 0
        # Photographie initiale pour les statistiques de conservation.
        self.initial_world_food = float(self.grid[ix.G_AMOUNT + ix.R_FOOD].sum())
        self.initial_world_ore = float(self.grid[ix.G_AMOUNT + ix.R_ORE].sum())
        self.initial_food_stock = float(c.initial_food)
        self.initial_ore_stock = float(c.initial_ore)
        self.initial_ants = int(c.initial_ants)

    @classmethod
    def from_seed(cls, cfg: AntWorldConfig, seed: int) -> "AntWorld":
        """Monde + aléa de simulation entièrement déterminés par `seed`."""
        layout = generate_world(cfg.world, seed)
        return cls(layout, cfg, SeedManager(seed).rng("sim"))

    # ------------------------------------------------------------------ sauvegarde / restauration
    def snapshot(self) -> WorldState:
        """Copie indépendante de tout l'état dynamique."""
        return WorldState(self.grid.copy(), self.flow.copy(), self.ants.copy(), self.col.copy(),
                          copy.deepcopy(self._rng.bit_generator.state), self.macro_step)

    def restore(self, state: WorldState) -> None:
        """Remet la partie exactement dans l'état photographié (le snapshot n'est pas modifié)."""
        np.copyto(self.grid, state.grid)
        np.copyto(self.flow, state.flow)
        np.copyto(self.ants, state.ants)
        np.copyto(self.col, state.col)
        self._rng.bit_generator.state = copy.deepcopy(state.rng_state)
        self.macro_step = state.macro_step

    def clone(self) -> "AntWorld":
        """Partie indépendante, dans le même état (le layout immuable est partagé : pas de régénération)."""
        new = copy.copy(self)
        new._P = self._P.copy()
        new._rng = np.random.Generator(type(self._rng.bit_generator)())
        new.grid, new.flow, new.ants, new.col = (self.grid.copy(), self.flow.copy(),
                                                 self.ants.copy(), self.col.copy())
        new._rng.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        return new

    # ------------------------------------------------------------------ action et avancement
    @property
    def action_nvec(self) -> tuple[int, ...]:
        """Taille de chaque dimension de l'action (pour `MultiDiscrete`)."""
        c = self.cfg.colony
        return (len(c.scout_shares), len(c.build_shares), len(c.ore_shares), N_BUILD_CHOICES, N_TARGETS)

    def step(self, action) -> None:
        """Un pas macro : applique l'action puis joue `ticks_per_step` ticks."""
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (5,) or np.any(action < 0) or np.any(action >= np.array(self.action_nvec)):
            raise ValueError(f"Action invalide {action.tolist()} (attendu 5 entiers dans {self.action_nvec})")
        if not self.alive:
            return
        T = self.cfg.sim.ticks_per_step
        kernels.apply_action(self.ants, self.col, self._P, action)
        # Tous les tirages du pas macro d'un coup (une naissance par tick au plus => +T lignes).
        rand = self._rng.random((T, self.n_ants + T, 3))
        kernels.run_ticks(self.grid, self.flow, self.ants, self.col, self.layout.terrain,
                          self.layout.food_cap, self._food_cells, self.layout.nest_next,
                          self.layout.nest, self._P, rand)
        self.macro_step += 1

    # ------------------------------------------------------------------ lecture de l'état
    @property
    def alive(self) -> bool:
        return self.col[ix.C_N_ANTS] >= 1.0

    @property
    def truncated(self) -> bool:
        return self.macro_step >= self.cfg.sim.max_steps

    @property
    def done(self) -> bool:
        return (not self.alive) or self.truncated

    @property
    def n_ants(self) -> int:
        return int(self.col[ix.C_N_ANTS])

    @property
    def food_stock(self) -> float:
        return float(self.col[ix.C_FOOD_STOCK])

    @property
    def ore_stock(self) -> float:
        return float(self.col[ix.C_ORE_STOCK])

    @property
    def nursery_level(self) -> int:
        return int(self.col[ix.C_NURSERY_LVL])

    @property
    def granary_level(self) -> int:
        return int(self.col[ix.C_GRANARY_LVL])

    @property
    def pop_cap(self) -> int:
        c = self.cfg.colony
        return c.base_pop_cap + self.nursery_level * c.nursery_gain

    @property
    def storage_cap(self) -> float:
        c = self.cfg.colony
        return c.base_food_storage + self.granary_level * c.granary_gain

    @property
    def known_fraction(self) -> float:
        """Part des cases libres déjà vues."""
        free = self.layout.terrain == 0
        return float(self.grid[ix.G_KNOWN][free].sum() / free.sum())

    @property
    def tick(self) -> int:
        return int(self.col[ix.C_TICK])

    def role_counts(self) -> np.ndarray:
        return np.bincount(self.ants[: self.n_ants, ix.A_ROLE], minlength=ix.N_ROLES)

    def world_amount(self, r: int) -> float:
        return float(self.grid[ix.G_AMOUNT + r].sum())

    def carried(self, r: int) -> int:
        a = self.ants[: self.n_ants]
        return int(a[(a[:, ix.A_LOAD_TYPE] == r), ix.A_LOAD].sum())

    def stats(self) -> dict[str, float]:
        """Instantané lisible des grandeurs principales."""
        return {
            "step": self.macro_step, "tick": self.tick, "ants": self.n_ants, "pop_cap": self.pop_cap,
            "food": self.food_stock, "ore": self.ore_stock,
            "nursery": self.nursery_level, "granary": self.granary_level,
            "known": self.known_fraction,
            "births": float(self.col[ix.C_BIRTHS]), "deaths": float(self.col[ix.C_DEATHS]),
        }

    def state_arrays(self) -> tuple[np.ndarray, ...]:
        """Tous les tableaux qui constituent l'état dynamique (pour hachage / copie)."""
        return self.grid, self.flow, self.ants[: self.n_ants], self.col
