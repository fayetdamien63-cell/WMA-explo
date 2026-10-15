"""Fonction de récompense « prospérité » (v2).

    r_t = w_pop_level · fourmis_{t+1}  +  [Φ(s_{t+1}) − Φ(s_t)]  +  w_survive · [colonie vivante]

    Φ(s) = w_food · nourriture_en_stock  +  w_ore · valeur_du_minerai

Pourquoi cette forme ? (historique : docs/DECISIONS.md, étape 3)
  - Version 1 : r = Δ(w_pop·fourmis + ...) + survie. Le retour d'un épisode ne dépendait que de l'état FINAL ;
    l'oracle en a tiré une stratégie « hiberner puis exploser » (colonie réduite à 2-8 fourmis pendant l'essentiel de la
    partie, explosion dans les 50 derniers pas). Un objectif qui ne regarde que la fin n'est pas une prospérité.
  - Version 2 (ici) : la population est récompensée à chaque pas (aire sous la courbe). Une colonie qui s'éteint
    perd tout le futur ; une explosion finale ne rapporte presque rien. Les STOCKS restent des différences de
    potentiel : stocker puis dépenser ne crée aucune récompense (somme télescopique), seul le niveau final compte.

Valeur du minerai : le minerai n'a d'usage que pour construire. On compte
  - le stock « utile » (de quoi payer les prochains niveaux de nurserie et de grenier) à pleine valeur,
  - le stock en SURPLUS (au-delà) à une valeur négative −ore_surplus_penalty · w_ore par unité (pénalité légère),
  - le minerai déjà investi dans des bâtiments à pleine valeur : construire est neutre pour la récompense,
    son bénéfice vient de la capacité gagnée (plus de fourmis possibles, plus de stockage).

Retour d'un épisode = w_pop_level · Σ_t fourmis_t + Φ(fin) − Φ(début) + w_survive · pas_vivants.
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
    """Termes pondérés du potentiel Φ des ressources : [nourriture, minerai]."""
    return np.array([rc.w_food * world.food_stock, rc.w_ore * ore_value(world, rc)])


def potential(world: AntWorld, rc: RewardConfig) -> float:
    return float(potential_terms(world, rc).sum())


def step_reward(before: np.ndarray, world_after: AntWorld, rc: RewardConfig) -> tuple[float, dict[str, float]]:
    """Récompense du pas qui mène à `world_after`, `before` étant `potential_terms` avant le pas."""
    delta = potential_terms(world_after, rc) - before
    level = rc.w_pop_level * world_after.n_ants
    survive = rc.w_survive if world_after.alive else 0.0
    terms = {"population": float(level), "food": float(delta[0]), "ore": float(delta[1]), "survive": survive}
    return float(level + delta.sum() + survive), terms


def sequence_return(phi_start: float, world_end: AntWorld, pop_sum: float, alive_steps: int, rc: RewardConfig) -> float:
    """Retour cumulé d'une séquence de pas, sans refaire la somme pas à pas (identité télescopique).

    `pop_sum` = Σ fourmis après chaque pas ; `alive_steps` = nombre de pas terminés colonie vivante.
    Identique à la somme des `step_reward` (vérifié par test) ; sert aux planificateurs (oracle).
    """
    return (rc.w_pop_level * pop_sum + potential(world_end, rc) - phi_start + rc.w_survive * alive_steps)
