"""Étape 1 : chaque règle du monde (docs/WORLD_SPEC.md) testée sur un mini-monde fait main."""

import numpy as np
import pytest

from antworld import indices as ix
from antworld.world import AntWorld
from helpers import check_invariants, make_arena

NOOP = np.array([0, 0, 0, 0, 0])


def test_forager_makes_a_round_trip_and_delivers():
    # Nid en (2,5), nourriture en (6,5) : 4 pas pour y aller, 1 tick de récolte, 4 pas retour, 1 dépôt.
    w = make_arena(food={(6, 5): 10.0}, ticks=10)
    w.step(NOOP)
    assert w.food_stock == pytest.approx(30.0 + 2)                  # carry_capacity = 2
    assert w.world_amount(ix.R_FOOD) == pytest.approx(10.0 - 2 + w.col[ix.C_FOOD_REGROWN])
    check_invariants(w)


def test_forager_ignores_food_it_has_never_seen():
    # Nid avec une vision de 2 seulement : la nourriture à 6 cases est dans le brouillard.
    w = make_arena(food={(8, 5): 10.0}, nest_vision=2, ticks=20)
    for _ in range(5):
        w.step(NOOP)
    assert w.food_stock == pytest.approx(30.0)
    assert (w.ants[0, ix.A_CELL] == w.layout.nest)                  # elle attend au nid


def test_overflow_is_wasted_when_storage_is_full():
    w = make_arena(food={(6, 5): 10.0}, ticks=10, base_food_storage=31.0)
    w.step(NOOP)
    assert w.food_stock == pytest.approx(31.0)
    assert w.col[ix.C_FOOD_OVERFLOW] == pytest.approx(1.0)          # 2 livrées, 1 accepté
    check_invariants(w)


def test_starvation_kills_the_colony():
    w = make_arena(initial_ants=10, initial_food=0.0, eat_rate=0.05, starve_rate=0.05)
    alive = []
    for _ in range(400):
        w.step(NOOP)
        alive.append(w.n_ants)
        check_invariants(w)
        if w.done:
            break
    assert w.n_ants == 0 and not w.alive and w.done and not w.truncated
    assert all(a >= b for a, b in zip(alive, alive[1:])), "la population ne doit pas remonter sans nourriture"
    before = w.tick
    w.step(NOOP)                                                    # un pas sur une colonie morte ne fait rien
    assert w.tick == before


def test_no_starvation_while_food_lasts():
    w = make_arena(initial_ants=10, initial_food=100.0, base_food_storage=200.0, eat_rate=0.01)
    for _ in range(20):
        w.step(NOOP)
    assert w.n_ants == 10 and w.col[ix.C_DEATHS] == 0
    assert w.food_stock == pytest.approx(100.0 - 10 * 0.01 * 100)   # 20 pas * 5 ticks = 100 ticks


def test_births_need_food_and_room():
    w = make_arena(initial_food=50.0, lay_rate=1.0, base_pop_cap=40, ticks=10)
    w.step(NOOP)                                                    # 1 naissance par tick
    assert w.n_ants == 11 and w.col[ix.C_FOOD_BROOD] == pytest.approx(40.0)
    check_invariants(w)

    small = make_arena(initial_food=50.0, lay_rate=1.0, base_pop_cap=5, ticks=10)
    small.step(NOOP)
    assert small.n_ants == 5                                        # plafonné par la capacité

    poor = make_arena(initial_food=13.0, lay_rate=1.0, ticks=10)    # < birth_cost + reserve = 14
    poor.step(NOOP)
    assert poor.n_ants == 1


def test_role_assignment_matches_shares():
    w = make_arena(initial_ants=20, ticks=1)
    # éclaireuses palier 2 = 25 % -> 5 ; bâtisseuses palier 1 = 10 % -> 2 ; reste 13 ;
    # minerai palier 2 = 50 % -> round(6.5) = 7 ; nourriture 6
    w.step(np.array([2, 1, 2, 0, 0]))
    assert w.role_counts().tolist() == [6, 7, 5, 2]
    # Réaffectation minimale : les premières fourmis gardent leur rôle de départ (nourriture).
    assert (w.ants[:6, ix.A_ROLE] == ix.ROLE_FOOD).all()
    w.step(NOOP)
    assert w.role_counts().tolist() == [20, 0, 0, 0]


def test_building_costs_ore_then_raises_population_cap():
    w = make_arena(initial_ants=10, ticks=10, ore={}, initial_ore=100.0)
    build = np.array([0, 3, 0, 1, 0])        # 50 % de bâtisseuses -> 5, construire une nurserie
    w.step(build)
    assert w.ore_stock == pytest.approx(100.0 - 15.0)               # niveau 0 : 15 * (0 + 1)
    assert w.col[ix.C_BUILD_TYPE] == 1 and w.nursery_level == 0
    w.step(build)                                                   # redemander n'est pas payé deux fois
    assert w.ore_stock == pytest.approx(85.0)
    for _ in range(2):
        w.step(build)
    # travail requis 200, 5 bâtisseuses au nid -> 40 ticks = 4 pas de 10 ticks, terminé à la fin du 4e
    assert w.nursery_level == 1 and w.pop_cap == 60
    assert w.col[ix.C_BUILD_TYPE] == 0
    # Une nouvelle demande lance le niveau suivant : coût 15 * 2 = 30, travail 200 * 2 = 400.
    ore_before = w.ore_stock
    w.step(build)
    assert ore_before - w.ore_stock == pytest.approx(30.0) and w.col[ix.C_BUILD_REQUIRED] == pytest.approx(400.0)
    check_invariants(w)


def test_cannot_build_without_ore_or_beyond_max_level():
    w = make_arena(initial_ants=10, ticks=5, initial_ore=5.0)
    w.step(np.array([0, 3, 0, 1, 0]))
    assert w.col[ix.C_BUILD_TYPE] == 0 and w.ore_stock == 5.0
    cap = make_arena(initial_ants=10, ticks=5, initial_ore=1000.0, max_building_level=1, build_work=1.0)
    for _ in range(6):
        cap.step(np.array([0, 3, 0, 2, 0]))                         # granaire à répétition
    assert cap.granary_level == 1 and cap.storage_cap == pytest.approx(60 + 60)


def test_builders_without_a_site_stay_idle():
    w = make_arena(initial_ants=10, ticks=10)
    w.step(np.array([0, 3, 0, 0, 0]))                               # 50 % de bâtisseuses mais rien à construire
    assert w.col[ix.C_BUILD_PROGRESS] == 0


def test_scouts_reveal_the_map():
    explore = make_arena(size=20, nest=(10, 10), initial_ants=10, nest_vision=2, ticks=10)
    idle = make_arena(size=20, nest=(10, 10), initial_ants=10, nest_vision=2, ticks=10)
    for _ in range(10):
        explore.step(np.array([3, 0, 0, 0, 0]))                     # 50 % d'éclaireuses
        idle.step(NOOP)
    assert explore.known_fraction > idle.known_fraction + 0.2


def test_scout_target_direction_biases_the_walk():
    xs = {}
    for target, name in ((2, "est"), (4, "ouest")):
        w = make_arena(size=30, nest=(15, 15), initial_ants=20, ticks=10, seed=3)
        for _ in range(6):
            w.step(np.array([3, 0, 0, 0, target]))
        scouts = w.ants[: w.n_ants][w.ants[: w.n_ants, ix.A_ROLE] == ix.ROLE_SCOUT]
        xs[name] = (scouts[:, ix.A_CELL] % 30).mean()
    assert xs["est"] > 15 + 3 and xs["ouest"] < 15 - 3


def test_ore_is_finite_and_food_regrows_slowly():
    w = make_arena(ore={(6, 5): 3.0}, food={(8, 8): 10.0}, ticks=1, initial_ants=1)
    w.grid[ix.G_AMOUNT + ix.R_FOOD, 8 * 12 + 8] = 0.0               # case de nourriture vide
    for _ in range(300):
        w.step(NOOP)
    regrown = w.grid[ix.G_AMOUNT + ix.R_FOOD, 8 * 12 + 8]
    assert 4.0 < regrown < 6.0                                      # ≈ mi-capacité après ~300 ticks
    w2 = make_arena(ore={(6, 5): 3.0}, ticks=10, initial_ants=1)
    ore_role = np.array([0, 0, 3, 0, 0])                            # 100 % des cueilleuses sur le minerai
    for _ in range(10):
        w2.step(ore_role)
    # 2 unités au premier voyage, ⌊1.0⌋ = 1 au second : tout le minerai est livré, puis plus rien.
    assert w2.ore_stock == pytest.approx(3.0)
    assert w2.world_amount(ix.R_ORE) == pytest.approx(0.0)


def test_invalid_actions_rejected():
    w = make_arena()
    for bad in ([0, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 5], [-1, 0, 0, 0, 0]):
        with pytest.raises(ValueError):
            w.step(bad)
