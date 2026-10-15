"""Disposition statique d'un monde (terrain, nid, zones de ressources) et son générateur.

Un `WorldLayout` ne change jamais pendant une partie : c'est le « plan » du monde. L'état qui évolue
(quantités restantes, fourmis, stocks) est dans `AntWorld`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antworld.config import WorldGenConfig
from antworld.kernels import bfs_from
from antworld.noise import fractal_noise
from common.seeding import SeedManager


@dataclass(frozen=True)
class WorldLayout:
    size: int
    terrain: np.ndarray       # (N,) int8 : 0 libre, 1 roche
    nest: int                 # indice de la case du nid
    food_cap: np.ndarray      # (N,) float64 : capacité de nourriture (0 = pas de nourriture ici)
    ore_amount: np.ndarray    # (N,) float64 : minerai initial
    nest_next: np.ndarray     # (N,) int32 : case suivante sur le plus court chemin vers le nid (−1 au nid)
    nest_dist: np.ndarray     # (N,) int32 : distance de chemin au nid (−1 = inaccessible)
    seed: int = -1
    attempts: int = 1         # nombre de tentatives de génération nécessaires


def make_layout(size: int, terrain: np.ndarray, nest: int, food_cap: np.ndarray,
                ore_amount: np.ndarray, seed: int = -1, attempts: int = 1) -> WorldLayout:
    """Assemble un layout et calcule le champ « chemin vers le nid » (utile aussi aux tests)."""
    terrain = np.ascontiguousarray(terrain, dtype=np.int8)
    src = np.zeros(size * size, dtype=np.uint8)
    src[nest] = 1
    nest_next, nest_dist = bfs_from(terrain, size, src)
    return WorldLayout(size, terrain, nest, np.asarray(food_cap, dtype=np.float64),
                       np.asarray(ore_amount, dtype=np.float64), nest_next, nest_dist, seed, attempts)


def _select_zone(noise: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """Indices des `count` cases candidates où le bruit est le plus élevé (→ zones d'un seul tenant)."""
    order = np.argsort(-noise[candidates], kind="stable")
    return candidates[order[:count]]


def _try_generate(cfg: WorldGenConfig, rng: np.random.Generator, seed: int, attempt: int) -> WorldLayout | None:
    s = cfg.size
    n = s * s

    # 1. Terrain : roche où le bruit est le plus haut.
    noise = fractal_noise(s, cfg.terrain_scale, cfg.noise_octaves, rng).ravel()
    terrain = (noise > np.quantile(noise, 1.0 - cfg.rock_fraction)).astype(np.int8)

    # 2. Nid dans la zone centrale + carré libre autour.
    nx = int(rng.integers(cfg.nest_margin, s - cfg.nest_margin))
    ny = int(rng.integers(cfg.nest_margin, s - cfg.nest_margin))
    nest = ny * s + nx
    r = cfg.nest_clear_radius
    terrain2d = terrain.reshape(s, s)
    terrain2d[max(0, ny - r): ny + r + 1, max(0, nx - r): nx + r + 1] = 0

    # 3. Les poches inaccessibles depuis le nid deviennent roche.
    src = np.zeros(n, dtype=np.uint8)
    src[nest] = 1
    _, dist = bfs_from(terrain, s, src)
    terrain[dist < 0] = 1
    reachable = np.flatnonzero(dist >= 0)
    if len(reachable) < cfg.min_open_fraction * n:
        return None

    # 4. Nourriture puis minerai, à distance du nid, sur des cases distinctes.
    candidates = reachable[dist[reachable] >= cfg.min_resource_dist]
    n_food = int(round(cfg.food_fraction * len(reachable)))
    n_ore = int(round(cfg.ore_fraction * len(reachable)))
    if len(candidates) < n_food + n_ore:
        return None

    food_noise = fractal_noise(s, cfg.food_scale, cfg.noise_octaves, rng).ravel()
    food_cells = _select_zone(food_noise, candidates, n_food)
    food_cap = np.zeros(n)
    f = food_noise[food_cells]
    food_cap[food_cells] = cfg.food_cap_min + (cfg.food_cap_max - cfg.food_cap_min) * (f - f.min()) / max(f.max() - f.min(), 1e-9)

    ore_noise = fractal_noise(s, cfg.ore_scale, cfg.noise_octaves, rng).ravel()
    rest = np.setdiff1d(candidates, food_cells)
    ore_cells = _select_zone(ore_noise, rest, n_ore)
    ore_amount = np.zeros(n)
    o = ore_noise[ore_cells]
    ore_amount[ore_cells] = cfg.ore_min + (cfg.ore_max - cfg.ore_min) * (o - o.min()) / max(o.max() - o.min(), 1e-9)

    layout = make_layout(s, terrain, nest, food_cap, ore_amount, seed=seed, attempts=attempt + 1)
    return layout if playability_problem(layout, cfg) is None else None


def playability_problem(layout: WorldLayout, cfg: WorldGenConfig) -> str | None:
    """None si le monde est jouable, sinon la raison de rejet."""
    reach = layout.nest_dist >= 0
    food = (layout.food_cap > 0) & reach
    ore = (layout.ore_amount > 0) & reach
    if food.sum() < cfg.min_food_cells:
        return f"pas assez de cases de nourriture accessibles ({food.sum()})"
    if ore.sum() < cfg.min_ore_cells:
        return f"pas assez de cases de minerai accessibles ({ore.sum()})"
    if layout.nest_dist[food].min() > cfg.max_food_dist:
        return "nourriture trop éloignée du nid"
    if layout.nest_dist[ore].min() > cfg.max_ore_dist:
        return "minerai trop éloigné du nid"
    if reach.sum() < cfg.min_open_fraction * layout.size**2:
        return "trop peu de cases libres"
    return None


def generate_world(cfg: WorldGenConfig, seed: int) -> WorldLayout:
    """Génère un monde jouable à partir d'une seed (déterministe)."""
    seeds = SeedManager(seed)
    for attempt in range(cfg.max_attempts):
        layout = _try_generate(cfg, seeds.rng(f"worldgen/{attempt}"), seed, attempt)
        if layout is not None:
            return layout
    raise RuntimeError(f"Aucun monde jouable trouvé pour la seed {seed} en {cfg.max_attempts} tentatives")
