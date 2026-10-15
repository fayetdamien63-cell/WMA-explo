"""Observation partielle de la colonie : carte (8 canaux) + état interne (24 scalaires), tout dans [0, 1].

La colonie n'observe PAS la vérité terrain, mais sa propre connaissance :
  - la carte qu'elle a construite (cases déjà vues, dernières quantités vues — éventuellement périmées),
  - quelles cases ses fourmis voient EN CE MOMENT (zone visible),
  - la position de ses fourmis et son état interne.
Ce qui est dans le brouillard est à 0 : l'agent doit se souvenir de ce qu'il a vu (mémoire récurrente).

Canaux de `map` (forme (8, size, size)) :
  0 visible  : 1 si une fourmi (ou le nid) voit la case à cet instant
  1 known    : 1 si la case a déjà été vue
  2 rock     : 1 si la case est de la roche ET connue
  3 food     : dernière quantité de nourriture vue / capacité maximale
  4 ore      : dernière quantité de minerai vue / quantité maximale
  5 nest     : 1 sur le nid
  6 scouts   : nombre d'éclaireuses par case / 4 (borné à 1)
  7 others   : nombre des autres fourmis par case / 8 (borné à 1)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from antworld import indices as ix
from antworld.world import AntWorld

N_MAP_CHANNELS = 8
N_STATE = 24

# Indices du vecteur `state` (valeurs normalisées dans [0, 1]). Les grandeurs de même nature partagent la
# même référence de normalisation : on peut donc les comparer directement (p. ex. S_ORE >= S_NEXT_NURSERY).
S_POP, S_POP_CAP, S_FOOD, S_STORAGE_CAP, S_ORE = 0, 1, 2, 3, 4
S_NURSERY, S_GRANARY = 5, 6
S_BUILD_ACTIVE, S_BUILD_NURSERY, S_BUILD_GRANARY, S_BUILD_PROGRESS = 7, 8, 9, 10
S_ROLES = slice(11, 15)                  # parts : cueilleuses nourriture, minerai, éclaireuses, bâtisseuses
S_CARRYING, S_TIME, S_NEXT_NURSERY, S_NEXT_GRANARY = 15, 16, 17, 18
S_TARGET = slice(19, 24)                 # cible d'exploration courante (one-hot)
CH_VISIBLE, CH_KNOWN, CH_ROCK, CH_FOOD, CH_ORE, CH_NEST, CH_SCOUT, CH_OTHER = range(8)


@njit(cache=True)
def _mark_visible(out, cell, radius, size):
    x = cell % size
    y = cell // size
    for yy in range(max(0, y - radius), min(size, y + radius + 1)):
        for xx in range(max(0, x - radius), min(size, x + radius + 1)):
            out[CH_VISIBLE, yy * size + xx] = 1.0


@njit(cache=True)
def build_map(grid, ants, n, terrain, nest, P, food_norm, ore_norm, out):
    """Remplit `out` (8, N) float32 à partir de l'état. Tout est recalculé : out est d'abord remis à 0."""
    size = int(P[ix.P_SIZE])
    out[:, :] = 0.0
    _mark_visible(out, nest, int(P[ix.P_NEST_VIS]), size)
    for i in range(n):
        cell = ants[i, ix.A_CELL]
        if ants[i, ix.A_ROLE] == ix.ROLE_SCOUT:
            _mark_visible(out, cell, int(P[ix.P_SCOUT_VIS]), size)
            out[CH_SCOUT, cell] += 0.25
        else:
            _mark_visible(out, cell, int(P[ix.P_FORAGER_VIS]), size)
            out[CH_OTHER, cell] += 0.125
    for c in range(size * size):
        if grid[ix.G_KNOWN, c] > 0.5:
            out[CH_KNOWN, c] = 1.0
            out[CH_ROCK, c] = float(terrain[c])
        out[CH_FOOD, c] = min(grid[ix.G_SEEN + ix.R_FOOD, c] / food_norm, 1.0)
        out[CH_ORE, c] = min(grid[ix.G_SEEN + ix.R_ORE, c] / ore_norm, 1.0)
        out[CH_SCOUT, c] = min(out[CH_SCOUT, c], 1.0)
        out[CH_OTHER, c] = min(out[CH_OTHER, c], 1.0)
    out[CH_NEST, nest] = 1.0


def observe_map(world: AntWorld) -> np.ndarray:
    out = np.empty((N_MAP_CHANNELS, world.size**2), dtype=np.float32)
    w = world.cfg.world
    build_map(world.grid, world.ants, world.n_ants, world.layout.terrain, world.layout.nest,
              world._P, w.food_cap_max, w.ore_max, out)
    return out.reshape(N_MAP_CHANNELS, world.size, world.size)


def state_vector(world: AntWorld) -> np.ndarray:
    """État interne de la colonie, normalisé dans [0, 1] (24 valeurs, ordre ci-dessous)."""
    c, col = world.cfg.colony, world.col
    max_lvl = c.max_building_level
    pop_ref = c.base_pop_cap + max_lvl * c.nursery_gain
    storage_ref = c.base_food_storage + max_lvl * c.granary_gain
    # minerai total pour tout construire = somme des coûts des niveaux 0..max-1
    tri = max_lvl * (max_lvl + 1) / 2
    ore_ref = (c.nursery_ore_cost + c.granary_ore_cost) * tri
    n = world.n_ants
    roles = world.role_counts() / max(n, 1)
    carrying = float((world.ants[:n, ix.A_LOAD] > 0).mean()) if n else 0.0
    btype = int(col[ix.C_BUILD_TYPE])
    required = col[ix.C_BUILD_REQUIRED]
    next_nursery = c.nursery_ore_cost * (world.nursery_level + 1) if world.nursery_level < max_lvl else 0.0
    next_granary = c.granary_ore_cost * (world.granary_level + 1) if world.granary_level < max_lvl else 0.0
    target = np.zeros(5)
    target[int(col[ix.C_EXPLORE_TARGET])] = 1.0
    v = np.concatenate([
        [n / pop_ref, world.pop_cap / pop_ref,                                   # 0-1  population
         world.food_stock / storage_ref, world.storage_cap / storage_ref,        # 2-3  nourriture
         world.ore_stock / ore_ref,                                              # 4    minerai
         world.nursery_level / max_lvl, world.granary_level / max_lvl,           # 5-6  bâtiments
         float(btype > 0), float(btype == 1), float(btype == 2),                 # 7-9  construction en cours
         col[ix.C_BUILD_PROGRESS] / required if required > 0 else 0.0],          # 10   avancement
        roles,                                                                   # 11-14 rôles (nourr., minerai, écl., bât.)
        [carrying,                                                               # 15   part de fourmis chargées
         world.macro_step / world.cfg.sim.max_steps,                             # 16   temps écoulé
         next_nursery / ore_ref, next_granary / ore_ref],                        # 17-18 coût du prochain niveau
        target,                                                                  # 19-23 cible d'exploration (one-hot)
    ])
    return np.clip(v, 0.0, 1.0).astype(np.float32)
