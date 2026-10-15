"""Étape 3 : planificateur CEM et oracle."""

import copy

import numpy as np
import pytest

from agents.oracle import OracleAgent, OracleConfig
from agents.planning import CEMConfig, cem_plan, shift_probs
from agents.random_agent import RandomAgent
from antworld.env import AntWorldEnv
from eval.harness import run_episode
from helpers import default_cfg, state_hash

CFG = default_cfg()
NVEC = np.array([4, 4, 4, 3, 5])


def test_cem_finds_the_optimum_of_a_known_toy_problem():
    """Valide le planificateur seul : retour = −(distance L1 à une séquence cible), optimum = 0.

    Mesuré pendant le développement : 64 échantillons, élites 8 -> 55 % du chemin « aléatoire -> optimum » après
    3 itérations, 90 % après 10, optimum exact après 20. Le budget d'itérations compte beaucoup !
    """
    target = np.repeat(np.random.default_rng(0).integers(0, NVEC, size=(6, 5)), 5, axis=0)

    def rollout(seqs):
        return -np.abs(seqs - target[None]).sum(axis=(1, 2)).astype(float)

    random_returns = rollout(np.random.default_rng(2).integers(0, NVEC, size=(2000, 30, 5)))
    results = {}
    for n_iters in (3, 20):
        cfg = CEMConfig(horizon=30, block=5, n_samples=64, n_iters=n_iters, n_elites=8)
        seq, probs, best = cem_plan(rollout, NVEC, cfg, np.random.default_rng(1), 30)
        results[n_iters] = best
        assert seq.shape == (30, 5) and (seq >= 0).all() and (seq < NVEC).all()
        assert np.allclose(probs.sum(-1), 1.0)
    assert results[3] > random_returns.max(), "même 3 itérations battent 2000 séquences aléatoires"
    assert results[20] > results[3]
    assert results[20] == 0.0, "avec assez d'itérations, CEM retrouve la séquence optimale exacte"


def test_cem_evaluation_count_matches_budget():
    cfg = CEMConfig(horizon=10, block=5, n_samples=12, n_iters=3, n_elites=3)
    calls = []
    cem_plan(lambda s: (calls.append(len(s)), np.zeros(len(s)))[1], NVEC, cfg, np.random.default_rng(0), 10)
    assert calls == [12, 12, 12]


def test_cem_never_samples_invalid_action_values():
    cfg = CEMConfig(horizon=10, block=5, n_samples=200, n_iters=2)
    seen = []

    def rollout(seqs):
        seen.append(seqs)
        return np.zeros(len(seqs))

    cem_plan(rollout, NVEC, cfg, np.random.default_rng(0), 10)
    allseq = np.concatenate(seen)
    assert (allseq >= 0).all() and (allseq < NVEC).all()


def test_shift_probs_moves_the_plan_forward():
    p = np.random.default_rng(0).random((6, 5, 5)); p /= p.sum(-1, keepdims=True)
    uniform = np.full((5, 5), 0.2)
    out = shift_probs(p, 1, uniform)
    assert np.array_equal(out[:5], p[1:]) and np.array_equal(out[5], uniform)


SMALL = OracleConfig(cem=CEMConfig(horizon=10, block=5, n_samples=16, n_iters=2, n_elites=4, replan_every=5))


def test_planning_never_touches_the_real_game():
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=3)
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
    before = state_hash(env.world), env.world.snapshot().rng_state
    oracle = OracleAgent(copy.deepcopy(SMALL))
    action = oracle.act(obs, env)
    assert state_hash(env.world) == before[0] and env.world.snapshot().rng_state == before[1]
    assert env.action_space.contains(np.asarray(action, dtype=np.int64))


def test_oracle_budget_accounting_and_replanning_cadence():
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=3)
    oracle = OracleAgent(copy.deepcopy(SMALL))
    c = SMALL.cem
    for step in range(10):
        obs, *_ = env.step(oracle.act(obs, env))
    # 10 actions, replanification tous les 5 pas -> 2 planifications.
    assert oracle.sim_steps == 2 * c.n_samples * c.n_iters * c.horizon


def test_oracle_is_deterministic_given_its_seed():
    def play(agent_seed):
        env = AntWorldEnv(CFG)
        obs, _ = env.reset(seed=4)
        oracle = OracleAgent(copy.deepcopy(SMALL))
        oracle.reset(agent_seed)
        acts = []
        for _ in range(20):
            a = oracle.act(obs, env); acts.append(a); obs, *_ = env.step(a)
        return np.array(acts)
    assert np.array_equal(play(1), play(1))
    assert not np.array_equal(play(1), play(2))


def test_even_a_tiny_oracle_beats_random_play():
    seed = 5
    oracle = run_episode(OracleAgent(copy.deepcopy(SMALL)), CFG, seed)
    rand = np.mean([run_episode(RandomAgent(), CFG, seed, a).ret for a in range(3)])
    assert oracle.ret > rand + 15
    assert oracle.sim_steps > 0 and oracle.decision_ms > 0


def test_oracle_plans_near_the_episode_end_without_overrunning():
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=3)
    env.world.macro_step = CFG.sim.max_steps - 3          # il ne reste que 3 pas
    oracle = OracleAgent(copy.deepcopy(SMALL))
    a = oracle.act(obs, env)
    assert env.action_space.contains(np.asarray(a, dtype=np.int64))
