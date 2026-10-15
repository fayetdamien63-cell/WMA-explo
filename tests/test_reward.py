"""Étape 2 : propriétés de la fonction de récompense."""

import numpy as np
import pytest

from antworld import indices as ix
from antworld.env import AntWorldEnv
from antworld.reward import ore_value, potential, potential_terms, useful_ore
from helpers import default_cfg, make_arena

CFG = default_cfg()
RC = CFG.reward


def test_return_telescopes_to_final_potential_minus_initial_plus_survival():
    """Garantie anti-boucle : retour = Φ(fin) − Φ(début) + bonus de survie, quelles que soient les actions."""
    for seed in range(6):
        env = AntWorldEnv(CFG)
        env.reset(seed=seed)
        phi0, rng, total, alive_steps = potential(env.world, RC), np.random.default_rng(seed), 0.0, 0
        done = False
        while not done:
            _, r, te, tr, _ = env.step(rng.integers(0, env.action_space.nvec))
            total += r
            alive_steps += 0 if te else 1
            done = te or tr
        assert total == pytest.approx(potential(env.world, RC) - phi0 + RC.w_survive * alive_steps, abs=1e-9)


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
    # « un peu » : le surplus coûte moitié moins qu'il ne rapporte quand il est utile.
    assert RC.ore_surplus_penalty * RC.w_ore < RC.w_ore


def test_stored_ore_counts_when_useful():
    w = make_arena(initial_ore=0.0)
    p0 = potential(w, RC)
    w.col[ix.C_ORE_STOCK] = 10.0
    assert potential(w, RC) - p0 == pytest.approx(RC.w_ore * 10.0)


def test_investing_ore_in_a_building_is_reward_neutral():
    """Dépenser du minerai pour construire ne coûte rien à la récompense (le gain vient de la capacité)."""
    w = make_arena(initial_ants=10, initial_ore=25.0, ticks=1)
    before = potential_terms(w, RC)[2]
    w.step([0, 3, 0, 1, 0])                              # construction d'une nurserie (coût 15)
    assert w.ore_stock == 10.0
    assert potential_terms(w, RC)[2] == pytest.approx(before)


def test_survival_bonus_and_death_penalty():
    env = AntWorldEnv(CFG)
    env.reset(seed=0)
    _, r, te, _, info = env.step(np.zeros(5, int))
    assert not te and info["reward_terms"]["survive"] == RC.w_survive
    rewards = []
    while True:
        _, r, te, tr, info = env.step([3, 3, 0, 0, 0])   # personne ne cueille : la colonie meurt
        rewards.append(r)
        if te or tr:
            break
    assert te and info["reward_terms"]["survive"] == 0.0
    assert rewards[-1] < 0, "mourir donne une récompense négative (la population disparaît du potentiel)"


def test_weights_come_from_the_config():
    cfg = default_cfg()
    cfg.reward.w_pop = 2.0
    w1 = make_arena(initial_ants=10)
    assert potential_terms(w1, cfg.reward)[0] == 2.0 * 10
