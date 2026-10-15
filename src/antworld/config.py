"""Paramètres du monde et de la colonie (valeurs par défaut = celles de docs/WORLD_SPEC.md).

Les valeurs réelles vivent dans `configs/antworld.yaml` ; ces dataclasses fournissent les défauts
et la structure. Chargement : `load_config(AntWorldConfig, "configs/antworld.yaml")`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldGenConfig:
    """Génération procédurale de la carte."""

    size: int = 32                      # grille size × size
    rock_fraction: float = 0.16         # part de roche avant retrait des poches isolées
    terrain_scale: float = 8.0          # taille caractéristique des zones de roche (en cases)
    noise_octaves: int = 2
    nest_margin: int = 8                # le nid est tiré dans [margin, size - margin)
    nest_clear_radius: int = 1          # carré libre autour du nid
    min_open_fraction: float = 0.60     # part minimale de cases libres accessibles
    food_fraction: float = 0.14         # part des cases accessibles en zone de nourriture
    food_scale: float = 6.0
    food_cap_min: float = 8.0
    food_cap_max: float = 16.0
    food_regrow_rate: float = 0.01      # g : vitesse de repousse logistique (par tick)
    food_regrow_seed: float = 0.05      # s : « graine » permettant de repartir de 0 (fraction de cap)
    ore_fraction: float = 0.04
    ore_scale: float = 5.0
    ore_min: float = 10.0
    ore_max: float = 25.0
    min_resource_dist: int = 3          # pas de ressource à moins de n pas (chemin) du nid
    max_food_dist: int = 14             # critères de jouabilité
    max_ore_dist: int = 22
    min_food_cells: int = 12
    min_ore_cells: int = 4
    max_attempts: int = 50


@dataclass
class ColonyConfig:
    """Règles de la colonie et des fourmis."""

    max_ants: int = 256                 # capacité des tableaux (doit dépasser la population maximale)
    initial_ants: int = 12
    initial_food: float = 30.0
    initial_ore: float = 0.0
    carry_capacity: int = 2             # unités portées par voyage
    eat_rate: float = 0.02              # nourriture mangée par fourmi et par tick
    starve_rate: float = 0.01           # fraction de la population qui meurt par tick en famine totale
    forager_vision: int = 1
    scout_vision: int = 2
    nest_vision: int = 5
    scout_turn_prob: float = 0.15
    scout_bias: float = 0.6
    lay_rate: float = 0.25              # avancée du compteur de ponte par tick
    birth_cost: float = 4.0
    food_reserve: float = 10.0          # stock minimal conservé avant de pondre
    base_pop_cap: int = 40
    nursery_gain: int = 20              # capacité de population gagnée par niveau de nurserie
    base_food_storage: float = 60.0
    granary_gain: float = 60.0          # capacité de stockage gagnée par niveau de grenier
    max_building_level: int = 4
    nursery_ore_cost: float = 15.0      # coût du niveau L : cost * (L + 1)
    granary_ore_cost: float = 10.0
    build_work: float = 200.0           # travail du niveau L : work * (L + 1)
    # Paliers de l'action macro
    scout_shares: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
    build_shares: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
    ore_shares: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)


@dataclass
class SimConfig:
    ticks_per_step: int = 5             # ticks de simulation par pas macro
    max_steps: int = 400                # longueur max d'un épisode (pas macro)


@dataclass
class AntWorldConfig:
    world: WorldGenConfig = field(default_factory=WorldGenConfig)
    colony: ColonyConfig = field(default_factory=ColonyConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self) -> None:
        c = self.colony
        max_pop = c.base_pop_cap + c.max_building_level * c.nursery_gain
        if c.max_ants < max_pop + self.sim.ticks_per_step:
            raise ValueError(f"colony.max_ants={c.max_ants} trop petit pour une population max de {max_pop}")
        if c.initial_ants > c.base_pop_cap:
            raise ValueError("colony.initial_ants dépasse colony.base_pop_cap")
        if c.initial_food > c.base_food_storage:
            raise ValueError("colony.initial_food dépasse colony.base_food_storage")
        if not (len(c.scout_shares) == len(c.build_shares) == len(c.ore_shares) == 4):
            raise ValueError("Les paliers scout_shares, build_shares, ore_shares doivent avoir 4 valeurs")
        if self.world.size < 2 * self.world.nest_margin + 2:
            raise ValueError("world.size trop petit pour nest_margin")
