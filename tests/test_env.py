"""Étape 2 : interface Gymnasium (espaces, observations partielles, fin des épisodes, déterminisme)."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from antworld import AntWorldConfig
from antworld import observation as obs_mod
from antworld.env import AntWorldEnv
from helpers import default_cfg

CFG = default_cfg()


def run_episode(env, policy, seed, record=False):
    obs, info = env.reset(seed=seed)
    seq = [obs]
    rewards, t = [], 0
    while True:
        obs, r, terminated, truncated, info = env.step(policy(t))
        seq.append(obs); rewards.append(r); t += 1
        if terminated or truncated:
            return seq, rewards, terminated, truncated, info


def test_gymnasium_check_env_passes():
    check_env(AntWorldEnv(CFG), skip_render_check=True)
    check_env(AntWorldEnv(CFG, render_mode="rgb_array"))          # y compris le rendu


def test_spaces_shapes_and_bounds_at_every_step():
    env = AntWorldEnv(CFG)
    assert env.action_space.nvec.tolist() == [4, 4, 4, 3, 5]
    assert env.observation_space["map"].shape == (8, 32, 32) and env.observation_space["state"].shape == (24,)
    rng = np.random.default_rng(0)
    for seed in range(4):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        done = False
        while not done:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), "observation hors bornes ou mauvais dtype"
            assert np.isfinite(r) and np.isfinite(obs["map"]).all() and np.isfinite(obs["state"]).all()
            done = te or tr


def test_observation_never_leaks_the_fog():
    """Rien n'est visible dans le brouillard : toute information (nourriture, minerai, roche) est sur une case connue."""
    env = AntWorldEnv(CFG)
    for seed in range(1, 9):
        obs, _ = env.reset(seed=seed)
        done, steps = False, 0
        while not done and steps < 200:
            m = obs["map"]
            known = m[obs_mod.CH_KNOWN] > 0
            assert ((m[obs_mod.CH_FOOD] > 0) <= known).all() and ((m[obs_mod.CH_ORE] > 0) <= known).all()
            assert ((m[obs_mod.CH_ROCK] > 0) <= known).all()
            assert ((m[obs_mod.CH_VISIBLE] > 0) <= known).all(), "une case visible doit être connue"
            obs, _, te, tr, _ = env.step(env.action_space.sample())
            done, steps = te or tr, steps + 1
    obs, _ = env.reset(seed=1)
    assert 0.0 < obs["map"][obs_mod.CH_KNOWN].mean() < 0.3, "au départ on ne voit que les alentours du nid"


def test_without_fog_the_whole_map_is_known():
    cfg = default_cfg()
    cfg.obs.fog = False
    env = AntWorldEnv(cfg)
    obs, _ = env.reset(seed=1)
    assert (obs["map"][obs_mod.CH_KNOWN] == 1).all()
    true_food = env.world.grid[0].reshape(32, 32) / cfg.world.food_cap_max
    assert np.allclose(obs["map"][obs_mod.CH_FOOD], np.minimum(true_food, 1), atol=1e-6)


def test_episodes_always_end():
    env = AntWorldEnv(CFG)
    rng = np.random.default_rng(0)
    policies = {"aléatoire": lambda t: rng.integers(0, env.action_space.nvec),
                "tout nourriture": lambda t: np.zeros(5, int),
                "personne ne cueille": lambda t: np.array([3, 3, 0, 0, 0])}
    for name, policy in policies.items():
        for seed in range(5):
            _, _, te, tr, info = run_episode(env, policy, seed)
            assert te != tr, "exactement une des deux fins (terminée XOR tronquée)"
            assert info["step"] <= CFG.sim.max_steps
            if tr:
                assert info["step"] == CFG.sim.max_steps
    # Une colonie condamnée termine bien en « terminated », pas en « truncated ».
    _, _, te, tr, info = run_episode(env, policies["personne ne cueille"], 0)
    assert te and not tr and info["n_ants"] == 0 and info["step"] < CFG.sim.max_steps


def test_env_is_deterministic():
    def play(seed):
        env = AntWorldEnv(CFG)
        rng = np.random.default_rng(5)
        seq, rewards, *_ = run_episode(env, lambda t: rng.integers(0, env.action_space.nvec), seed)
        return seq, rewards
    (s1, r1), (s2, r2) = play(3), play(3)
    assert r1 == r2
    assert all(np.array_equal(a["map"], b["map"]) and np.array_equal(a["state"], b["state"]) for a, b in zip(s1, s2))
    assert not np.array_equal(s1[0]["map"], play(4)[0][0]["map"]), "deux seeds, deux mondes différents"


def test_reset_without_seed_is_reproducible_through_the_first_seed():
    seeds = []
    for _ in range(2):
        env = AntWorldEnv(CFG)
        env.reset(seed=11)
        seeds.append([env.reset()[1]["world_seed"] for _ in range(3)])
    assert seeds[0] == seeds[1] and len(set(seeds[0])) == 3


def test_misuse_is_reported():
    env = AntWorldEnv(CFG)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(5, int))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([9, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        AntWorldEnv(CFG, render_mode="human")


def test_render_rgb_array():
    env = AntWorldEnv(CFG, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3


def test_env_step_speed_is_reported():
    import time
    env = AntWorldEnv(CFG)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = rng.integers(0, env.action_space.nvec, size=(300, 5))
    t0, n = time.perf_counter(), 0
    for a in actions:
        _, _, te, tr, _ = env.step(a); n += 1
        if te or tr:
            break
    print(f"\n[env perf] {n / (time.perf_counter() - t0):.0f} pas/s (observation et récompense comprises)")
    assert n / (time.perf_counter() - t0) > 2000
