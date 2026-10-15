"""Outils partagés par les tests du simulateur."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from antworld import AntWorld, AntWorldConfig
from antworld import indices as ix
from antworld.config import ColonyConfig, SimConfig, WorldGenConfig
from antworld.layout import make_layout
from common.config import load_config
from common.seeding import SeedManager

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "antworld.yaml"


def default_cfg() -> AntWorldConfig:
    return load_config(AntWorldConfig, CONFIG_PATH)


def make_arena(*, size=12, nest=(2, 5), food=None, ore=None, rocks=(), ticks=5, seed=0, **colony_kw) -> AntWorld:
    """Petit monde fait main pour tester une règle précise.

    food / ore : dict {(x, y): quantité}. Par défaut : colonie sans épuisement de stock ni
    naissance, pour isoler la règle testée (on remet ce qu'on veut via colony_kw).
    """
    colony = dict(initial_ants=1, initial_food=30.0, eat_rate=0.0, lay_rate=0.0, nest_vision=5)
    colony.update(colony_kw)
    cfg = AntWorldConfig(
        world=WorldGenConfig(size=size, nest_margin=3),
        colony=ColonyConfig(**colony),
        sim=SimConfig(ticks_per_step=ticks, max_steps=1000),
    )
    n = size * size
    terrain = np.zeros(n, np.int8)
    for (x, y) in rocks:
        terrain[y * size + x] = 1
    food_cap, ore_amount = np.zeros(n), np.zeros(n)
    for (x, y), v in (food or {}).items():
        food_cap[y * size + x] = v
    for (x, y), v in (ore or {}).items():
        ore_amount[y * size + x] = v
    layout = make_layout(size, terrain, nest[1] * size + nest[0], food_cap, ore_amount)
    return AntWorld(layout, cfg, SeedManager(seed).rng("sim"))


def state_hash(w: AntWorld) -> str:
    h = hashlib.sha256()
    for a in w.state_arrays():
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def random_play(w: AntWorld, rng: np.random.Generator, steps: int | None = None):
    """Joue des actions aléatoires ; génère le monde après chaque pas."""
    for _ in range(steps or w.cfg.sim.max_steps):
        if w.done:
            return
        w.step(rng.integers(0, w.action_nvec))
        yield w


def check_invariants(w: AntWorld, tol: float = 1e-6) -> None:
    """Invariants qui doivent TOUJOURS être vrais (voir WORLD_SPEC §9)."""
    g, col, lay, c = w.grid, w.col, w.layout, w.cfg.colony
    n = w.n_ants
    ants = w.ants[:n]

    # Positivité et bornes.
    assert (g[ix.G_AMOUNT:ix.G_AMOUNT + 4] >= 0).all(), "quantité négative dans le monde"
    assert (g[ix.G_AMOUNT + ix.R_FOOD] <= lay.food_cap + tol).all(), "nourriture au-dessus de la capacité"
    assert (col >= 0).all(), f"valeur négative dans l'état colonie : {np.flatnonzero(col < 0)}"
    assert w.food_stock <= w.storage_cap + tol, "stock de nourriture au-dessus de la capacité de stockage"
    assert (ants[:, ix.A_LOAD] >= 0).all() and (ants[:, ix.A_LOAD] <= c.carry_capacity).all()
    assert 0 <= n <= c.max_ants and n <= max(w.pop_cap, c.initial_ants)

    # Les fourmis sont sur des cases libres de la grille.
    assert ((ants[:, ix.A_CELL] >= 0) & (ants[:, ix.A_CELL] < w.size**2)).all()
    assert (lay.terrain[ants[:, ix.A_CELL]] == 0).all(), "fourmi dans la roche"
    assert set(np.unique(ants[:, ix.A_ROLE])) <= {0, 1, 2, 3}

    # Conservation (rien ne se crée sans règle).
    f, o = ix.R_FOOD, ix.R_ORE
    assert abs(w.world_amount(f) - (w.initial_world_food + col[ix.C_FOOD_REGROWN] - col[ix.C_HARVESTED + f])) < 1e-6 * max(1, w.initial_world_food)
    assert abs(col[ix.C_HARVESTED + f] - (col[ix.C_DELIVERED + f] + col[ix.C_FOOD_OVERFLOW] + col[ix.C_LOST + f] + w.carried(f))) < tol
    assert abs(w.food_stock - (w.initial_food_stock + col[ix.C_DELIVERED + f] - col[ix.C_FOOD_EATEN] - col[ix.C_FOOD_BROOD])) < 1e-6 * max(1, w.initial_food_stock + col[ix.C_DELIVERED + f])
    assert abs(w.world_amount(o) - (w.initial_world_ore - col[ix.C_HARVESTED + o])) < 1e-6 * max(1, w.initial_world_ore)
    assert abs(col[ix.C_HARVESTED + o] - (col[ix.C_DELIVERED + o] + col[ix.C_LOST + o] + w.carried(o))) < tol
    assert abs(w.ore_stock - (w.initial_ore_stock + col[ix.C_DELIVERED + o] - col[ix.C_ORE_SPENT])) < 1e-6 * max(1, w.initial_ore_stock + col[ix.C_DELIVERED + o])
    assert n == w.initial_ants + col[ix.C_BIRTHS] - col[ix.C_DEATHS]
