"""Étape 3 : agents de référence (aléatoire, heuristique)."""

import numpy as np
import pytest

from agents.factory import BaselinesConfig, make_agent
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from antworld import indices as ix
from antworld.env import AntWorldEnv
from eval.harness import run_episode
from eval.seeds import train_seeds
from helpers import default_cfg

CFG = default_cfg()


def valid(env, a):
    return env.action_space.contains(np.asarray(a, dtype=np.int64))


def test_random_agent_valid_reproducible_and_covers_the_action_space():
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=0)
    a1, a2, a3 = RandomAgent(), RandomAgent(), RandomAgent()
    a1.reset(5); a2.reset(5); a3.reset(6)
    s1 = np.array([a1.act(obs) for _ in range(2000)])
    s2 = np.array([a2.act(obs) for _ in range(2000)])
    s3 = np.array([a3.act(obs) for _ in range(2000)])
    assert np.array_equal(s1, s2) and not np.array_equal(s1, s3)
    assert all(valid(env, a) for a in s1[:50])
    for d, n in enumerate(env.action_space.nvec):          # chaque valeur de chaque dimension apparaît
        assert set(s1[:, d]) == set(range(n))


def test_heuristic_is_deterministic_and_only_needs_the_observation():
    ag = HeuristicAgent(world_cfg=CFG)
    assert not ag.privileged
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=2)
    for _ in range(60):
        a = ag.act(obs, None)
        assert valid(env, a) and np.array_equal(a, HeuristicAgent(world_cfg=CFG).act(obs, None))
        obs, *_ = env.step(a)


def _obs_with(env, **changes):
    """Observation de l'état courant après modification de l'état interne (scénarios de règles)."""
    w = env.world
    for key, val in changes.items():
        w.col[getattr(ix, key)] = val
    return env._observe()


def test_heuristic_builds_a_nursery_only_when_useful():
    ag = HeuristicAgent(world_cfg=CFG)
    env = AntWorldEnv(CFG)
    env.reset(seed=1)
    w = env.world
    # Population loin du plafond : pas de nurserie même avec du minerai.
    assert ag.act(_obs_with(env, C_ORE_STOCK=100.0), None)[3] == 0
    # Population au plafond + minerai suffisant : nurserie, et des bâtisseuses.
    w.col[ix.C_N_ANTS] = 38
    w.ants[:38, ix.A_ROLE] = ix.ROLE_FOOD
    a = ag.act(_obs_with(env, C_ORE_STOCK=100.0), None)
    assert a[3] == 1 and a[1] > 0
    # ... mais pas sans le minerai.
    assert ag.act(_obs_with(env, C_ORE_STOCK=0.0), None)[3] == 0
    # Une construction déjà en cours : pas de nouvelle demande, bâtisseuses maintenues.
    a = ag.act(_obs_with(env, C_ORE_STOCK=100.0, C_BUILD_TYPE=1, C_BUILD_REQUIRED=200.0), None)
    assert a[3] == 0 and a[1] > 0


def test_heuristic_collects_ore_only_when_known_and_needed():
    ag = HeuristicAgent(world_cfg=CFG)
    env = AntWorldEnv(CFG)
    env.reset(seed=1)
    w = env.world
    w.col[ix.C_N_ANTS] = 38
    w.ants[:38, ix.A_ROLE] = ix.ROLE_FOOD
    w.col[ix.C_ORE_STOCK] = 0.0
    w.grid[ix.G_SEEN + ix.R_ORE] = 0.0                       # aucun minerai connu
    assert ag.act(env._observe(), None)[2] == 0
    w.grid[ix.G_SEEN + ix.R_ORE, w.layout.nest] = 10.0       # du minerai connu et nécessaire
    assert ag.act(env._observe(), None)[2] > 0
    w.col[ix.C_ORE_STOCK] = 500.0                            # déjà plus qu'assez : on arrête
    assert ag.act(env._observe(), None)[2] == 0


def test_heuristic_never_builds_beyond_max_level_and_stays_valid_through_whole_episodes():
    ag = HeuristicAgent(world_cfg=CFG)
    for seed in train_seeds(3):
        r = run_episode(ag, CFG, seed)
        assert r.nursery <= CFG.colony.max_building_level and r.granary <= CFG.colony.max_building_level


def test_heuristic_beats_random_on_training_worlds():
    seeds = train_seeds(12)
    h = np.mean([run_episode(HeuristicAgent(world_cfg=CFG), CFG, s).ret for s in seeds])
    r = np.mean([run_episode(RandomAgent(), CFG, s, 0).ret for s in seeds])
    assert h > r + 15


def test_factory():
    assert make_agent("random", CFG).name == "random"
    assert make_agent("heuristic", CFG, BaselinesConfig()).name == "heuristic"
    assert make_agent("oracle", CFG).privileged
    with pytest.raises(ValueError):
        make_agent("inconnu", CFG)
