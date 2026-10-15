"""Fonction de récompense : différence de potentiel + bonus de survie.

    r_t = Φ(s_{t+1}) − Φ(s_t) + w_survive · [colonie vivante]

    Φ(s) = w_pop · fourmis  +  w_food · nourriture_en_stock  +  w_ore · valeur_du_minerai

Pourquoi une différence de potentiel ? La somme des récompenses d'un épisode vaut alors
Φ(fin) − Φ(début) + bonus de survie : elle ne dépend que de l'état atteint, pas du chemin. Aucune
boucle (stocker puis dépenser, construire puis ...) ne peut produire de récompense « gratuite ».
C'est la garantie anti-exploitation de la conception ; le test anti-triche la vérifie.

Valeur du minerai : le minerai n'a d'usage que pour construire. On compte
  - le stock « utile » (de quoi payer les prochains niveaux de nurserie et de grenier) à pleine valeur,
  - le stock en SURPLUS (au-delà) à une valeur négative −ore_surplus_penalty · w_ore (pénalité légère),
  - le minerai déjà investi dans des bâtiments à pleine valeur : construire est neutre pour la récompense,
    son bénéfice vient de la capacité gagnée (plus de fourmis possibles, plus de stockage).
"""

from __future__ import annotations

import numpy as np

from antworld import indices as ix
from antworld.config import RewardConfig
from antworld.world import AntWorld


def useful_ore(world: AntWorld) -> float:
    """Minerai encore utile : coût du prochain niveau de nurserie + du prochain niveau de grenier."""
    c = world.cfg.colony
    useful = 0.0
    if world.nursery_level < c.max_building_level:
        useful += c.nursery_ore_cost * (world.nursery_level + 1)
    if world.granary_level < c.max_building_level:
        useful += c.granary_ore_cost * (world.granary_level + 1)
    return useful


def ore_value(world: AntWorld, rc: RewardConfig) -> float:
    """Valeur du minerai en « unités de minerai » (avant pondération w_ore)."""
    stock = world.ore_stock
    useful = useful_ore(world)
    surplus = max(stock - useful, 0.0)
    return min(stock, useful) - rc.ore_surplus_penalty * surplus + float(world.col[ix.C_ORE_SPENT])


def potential_terms(world: AntWorld, rc: RewardConfig) -> np.ndarray:
    """Les trois termes pondérés de Φ : [population, nourriture, minerai]."""
    return np.array([rc.w_pop * world.n_ants,
                     rc.w_food * world.food_stock,
                     rc.w_ore * ore_value(world, rc)])


def potential(world: AntWorld, rc: RewardConfig) -> float:
    return float(potential_terms(world, rc).sum())


def step_reward(before: np.ndarray, world_after: AntWorld, rc: RewardConfig) -> tuple[float, dict[str, float]]:
    """Récompense du pas qui mène à `world_after`, `before` étant `potential_terms` avant le pas."""
    delta = potential_terms(world_after, rc) - before
    survive = rc.w_survive if world_after.alive else 0.0
    terms = {"population": float(delta[0]), "food": float(delta[1]), "ore": float(delta[2]), "survive": survive}
    return float(delta.sum() + survive), terms
