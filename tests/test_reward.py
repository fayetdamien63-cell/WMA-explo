"""Étape 2/3 : propriétés de la fonction de récompense « prospérité » (v2)."""

import numpy as np
import pytest

from antworld import indices as ix
from antworld.env import AntWorldEnv
from antworld.reward import ore_value, potential, potential_terms, sequence_return, step_reward, useful_ore
from helpers import default_cfg, make_arena

CFG = default_cfg()
RC = CFG.reward


def test_return_equals_population_area_plus_potential_change_plus_survival():
    """retour = w_pop_level · Σ fourmis + Φ(fin) − Φ(début) + survie, quelles que soient les actions."""
    for seed in range(6):
        env = AntWorldEnv(CFG)
        env.reset(seed=seed)
        phi0, rng = potential(env.world, RC), np.random.default_rng(seed)
        total, alive_steps, pop_sum, done = 0.0, 0, 0, False
        while not done:
            _, r, te, tr, info = env.step(rng.integers(0, env.action_space.nvec))
            total += r
            pop_sum += info["n_ants"]
            alive_steps += 0 if te else 1
            done = te or tr
        expected = RC.w_pop_level * pop_sum + potential(env.world, RC) - phi0 + RC.w_survive * alive_steps
        assert total == pytest.approx(expected, abs=1e-9)


def test_stocks_cannot_be_farmed_by_cycling():
    """Les stocks sont des potentiels : monter puis redescendre ne rapporte rien (somme nulle)."""
    w = make_arena(initial_ants=10, initial_food=20.0, base_food_storage=200.0)
    before = potential_terms(w, RC)
    gains = 0.0
    for stock in (50.0, 20.0, 120.0, 20.0):
        w.col[ix.C_FOOD_STOCK] = stock
        after = potential_terms(w, RC)
        gains += (after - before).sum()
        before = after
    assert gains == pytest.approx(0.0, abs=1e-12)


def test_population_is_rewarded_at_every_step_not_only_at_the_end():
    """Le terme de population est proportionnel au nombre de fourmis, à chaque pas (pas une différence)."""
    for n in (1, 10, 40):
        w = make_arena(initial_ants=n)
        before = potential_terms(w, RC)
        r, terms = step_reward(before, w, RC)
        assert terms["population"] == pytest.approx(RC.w_pop_level * n)
        assert r == pytest.approx(RC.w_pop_level * n + RC.w_survive)     # rien d'autre n'a changé
    # Deux historiques de même population finale, l'un à 40 fourmis tout du long, l'autre à 4 puis 40 :
    flat = 40 * 100 * RC.w_pop_level
    boom = (4 * 90 + 40 * 10) * RC.w_pop_level
    assert flat > 5 * boom, "hiberner puis exploser doit coûter cher"


def test_sequence_return_matches_the_sum_of_step_rewards():
    """L'identité utilisée par l'oracle pour évaluer ses séquences = somme des récompenses de l'env."""
    env = AntWorldEnv(CFG)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(25):
        env.step(rng.integers(0, env.action_space.nvec))
    work = env.world.clone()
    phi0 = potential(work, RC)
    actions = rng.integers(0, env.action_space.nvec, size=(60, 5))
    pop_sum, alive_steps, env_total = 0, 0, 0.0
    for a in actions:
        work.step(a)
        pop_sum += work.n_ants
        alive_steps += work.alive
    for a in actions:
        env_total += env.step(a)[1]
    assert sequence_return(phi0, work, pop_sum, alive_steps, RC) == pytest.approx(env_total, abs=1e-9)


def test_ore_surplus_is_mildly_penalised():
    w = make_arena(initial_ore=0.0)
    useful = useful_ore(w)
    assert useful == 15.0 + 10.0                       # prochain niveau de nurserie + de grenier
    values = {}
    for stock in (0, 10, useful, useful + 20, useful + 100):
        w.col[ix.C_ORE_STOCK] = stock
        values[stock] = ore_value(w, RC)
    assert values[10] == 10 and values[useful] == useful                       # utile : valeur pleine
    assert values[useful + 20] == pytest.approx(useful - 0.5 * 20)            # surplus : valeur négative
    assert values[useful + 20] < values[useful] and values[useful + 100] < values[useful + 20]
    assert RC.ore_surplus_penalty * RC.w_ore < RC.w_ore        # « un peu » : moitié moins que la valeur utile


def test_stored_ore_counts_when_useful():
    w = make_arena(initial_ore=0.0)
    p0 = potential(w, RC)
    w.col[ix.C_ORE_STOCK] = 10.0
    assert potential(w, RC) - p0 == pytest.approx(RC.w_ore * 10.0)


def test_investing_ore_in_a_building_is_reward_neutral():
    """Dépenser du minerai pour construire ne coûte rien à la récompense (le gain vient de la capacité)."""
    w = make_arena(initial_ants=10, initial_ore=25.0, ticks=1)
    before = potential_terms(w, RC)[1]
    w.step([0, 3, 0, 1, 0])                              # construction d'une nurserie (coût 15)
    assert w.ore_stock == 10.0
    assert potential_terms(w, RC)[1] == pytest.approx(before)


def test_survival_bonus_and_death():
    env = AntWorldEnv(CFG)
    env.reset(seed=0)
    _, _, te, _, info = env.step(np.zeros(5, int))
    terms = info["reward_terms"]
    assert not te and terms["survive"] == RC.w_survive and terms["population"] > 0   # vivante : survie + population
    while True:
        _, r, te, tr, info = env.step([3, 3, 0, 0, 0])   # personne ne cueille : la colonie meurt
        if te or tr:
            break
    # Le pas de la mort ne rapporte ni survie ni population : tout le futur de prospérité est perdu.
    assert te and info["reward_terms"]["survive"] == 0.0 and info["reward_terms"]["population"] == 0.0
    assert r <= 1e-12


def test_weights_come_from_the_config():
    cfg = default_cfg()
    cfg.reward.w_pop_level = 0.01
    w = make_arena(initial_ants=10)
    r, terms = step_reward(potential_terms(w, cfg.reward), w, cfg.reward)
    assert terms["population"] == pytest.approx(0.1)
