"""Agent heuristique codé à la main : n'utilise QUE l'observation partielle (comme un agent appris).

Idées derrière les règles (toutes les constantes sont dans `HeuristicConfig`, donc dans la config) :
  - Explorer d'abord : sans nourriture connue la colonie meurt. Beaucoup d'éclaireuses tant que peu de
    nourriture est connue, puis peu, puis aucune une fois la carte quasi explorée.
  - Orienter les éclaireuses vers la moitié de la carte (autour du nid) la moins explorée.
  - Le minerai ne sert qu'à construire : on en cueille s'il y en a de connu et qu'il en manque pour le
    prochain bâtiment utile, jamais au-delà.
  - Nurserie utile quand la population approche son plafond ; grenier utile quand le stock de nourriture
    approche sa capacité. On ne construit rien d'inutile.
  - Des bâtisseuses seulement pendant qu'une construction est en cours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antworld import observation as ob
from antworld.config import AntWorldConfig


@dataclass
class HeuristicConfig:
    # Éclaireuses : paliers d'action (0: 0 %, 1: 10 %, 2: 25 %, 3: 50 %)
    scout_level_early: int = 2         # peu de nourriture connue
    scout_level_mid: int = 1           # nourriture connue mais carte peu explorée
    scout_level_late: int = 1          # carte quasi explorée
    few_food_cells: int = 10           # « peu de nourriture connue » : moins de n cases
    explored_enough: float = 0.80      # part de carte connue au-delà de laquelle on arrête d'explorer
    hungry_food: float = 0.10          # stock de nourriture (part de la capacité) sous lequel on explore plus
    # Minerai
    ore_level: int = 1                 # palier de la part de minerai parmi les cueilleuses (1 : 25 %)
    ore_surplus_margin: float = 1.0    # on arrête le minerai quand le stock couvre margin × le prochain besoin
    # Constructions
    nursery_pop_ratio: float = 0.85    # nurserie quand fourmis >= ratio × capacité
    granary_food_ratio: float = 0.80   # grenier quand nourriture >= ratio × capacité de stockage
    builders_level: int = 2            # palier de bâtisseuses pendant une construction (2 : 25 %)
    # Exploration dirigée
    directed_exploration: bool = True


class HeuristicAgent:
    name = "heuristic"
    privileged = False

    def __init__(self, cfg: HeuristicConfig | None = None, world_cfg: AntWorldConfig | None = None) -> None:
        self.cfg = cfg or HeuristicConfig()
        w = (world_cfg or AntWorldConfig()).world
        self._one_food, self._one_ore = 1.0 / w.food_cap_max, 1.0 / w.ore_max   # « 1 unité » dans l'observation normalisée
        self.sim_steps = 0
        self.train_env_steps = 0

    def reset(self, agent_seed: int) -> None:   # déterministe : la seed ne sert pas
        pass

    def act(self, obs, env=None) -> np.ndarray:
        c = self.cfg
        m, s = obs["map"], obs["state"]
        known = m[ob.CH_KNOWN]
        food_seen = int((m[ob.CH_FOOD] >= self._one_food).sum())   # cases où l'on croit qu'il reste ≥ 1 unité
        ore_seen = int((m[ob.CH_ORE] >= self._one_ore).sum())
        known_frac = float(known.mean())

        # --- éclaireuses -----------------------------------------------------------------------
        if known_frac >= c.explored_enough:
            scout = c.scout_level_late
        elif food_seen < c.few_food_cells or s[ob.S_FOOD] < c.hungry_food * s[ob.S_STORAGE_CAP]:
            scout = c.scout_level_early
        else:
            scout = c.scout_level_mid

        # --- constructions ---------------------------------------------------------------------
        building = s[ob.S_BUILD_ACTIVE] > 0.5
        nursery_open, granary_open = s[ob.S_NEXT_NURSERY] > 0, s[ob.S_NEXT_GRANARY] > 0   # 0 = niveau max atteint
        want_nursery = nursery_open and s[ob.S_POP] >= c.nursery_pop_ratio * s[ob.S_POP_CAP]
        want_granary = granary_open and s[ob.S_FOOD] >= c.granary_food_ratio * s[ob.S_STORAGE_CAP]
        choice = 0
        if not building:
            if want_nursery and s[ob.S_ORE] >= s[ob.S_NEXT_NURSERY]:
                choice = 1
            elif want_granary and s[ob.S_ORE] >= s[ob.S_NEXT_GRANARY]:
                choice = 2
        builders = c.builders_level if (building or choice) else 0

        # --- minerai : seulement ce qui manque pour le bâtiment voulu ------------------------------
        need = 0.0
        if want_nursery:
            need = s[ob.S_NEXT_NURSERY]
        elif want_granary:
            need = s[ob.S_NEXT_GRANARY]
        ore_share = c.ore_level if (ore_seen > 0 and need > 0 and s[ob.S_ORE] < c.ore_surplus_margin * need) else 0

        # --- cible d'exploration : la moitié de la carte (autour du nid) la moins explorée ----------
        target = 0
        if c.directed_exploration and scout > 0 and known_frac < 0.97:
            ny, nx = divmod(int(np.argmax(m[ob.CH_NEST])), m.shape[2])
            halves = [known[:ny].mean() if ny > 0 else 1.0,           # nord
                      known[:, nx + 1:].mean() if nx < m.shape[2] - 1 else 1.0,   # est
                      known[ny + 1:].mean() if ny < m.shape[1] - 1 else 1.0,      # sud
                      known[:, :nx].mean() if nx > 0 else 1.0]        # ouest
            target = 1 + int(np.argmin(halves))
        return np.array([scout, builders, ore_share, choice, target])
