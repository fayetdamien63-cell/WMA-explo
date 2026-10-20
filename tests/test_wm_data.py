"""Étape 5 : politiques de collecte, jeux de trajectoires, échantillonnage."""

from pathlib import Path

import numpy as np
import pytest

from agents.data_policies import (DEGENERATE_ACTIONS, NVEC, BlockRandomAgent, DegenerateAgent, NoisyAgent,
                                  RandomPPOAgent, make_data_policy)
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from antworld.env import AntWorldEnv
from eval.seeds import TRAIN_START, VAL_START, test_seeds
from helpers import default_cfg
from models.data import DataConfig, TrajectoryDataset, build_splits, collect, load_split, play_episode, save_split

CFG = default_cfg()
PPO_MODEL = Path(__file__).resolve().parent.parent / "runs" / "step4" / "ppo_full_2M" / "ppo_seed0.zip"
DUMMY_OBS = {"map": np.zeros((8, 32, 32), np.float32), "state": np.zeros(24, np.float32)}


# ------------------------------------------------------------------ politiques de collecte
def test_block_random_holds_each_action_between_5_and_20_steps():
    agent = BlockRandomAgent()
    agent.reset(3)
    acts = [tuple(agent.act(DUMMY_OBS)) for _ in range(400)]
    runs, prev, length = [], acts[0], 0
    for a in acts:
        if a == prev:
            length += 1
        else:
            runs.append(length); prev, length = a, 1
    assert 3 <= len(runs) and all(5 <= r for r in runs[:-1]), "chaque action est tenue au moins 5 pas"
    assert max(runs) <= 40, "deux tirages identiques d'affilée sont rares, mais pas plus de 2 blocs de 20 pas"
    assert all(0 <= v < n for a in acts for v, n in zip(a, NVEC))
    agent.reset(3)
    assert acts == [tuple(agent.act(DUMMY_OBS)) for _ in range(400)], "reproductible"


def test_noisy_agent_interpolates_between_base_and_random():
    heur = HeuristicAgent(world_cfg=CFG)
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=1)
    base = heur.act(obs)
    clean = NoisyAgent(HeuristicAgent(world_cfg=CFG), 0.0)
    clean.reset(0)
    assert np.array_equal(clean.act(obs), base)
    for eps in (0.25, 1.0):
        noisy = NoisyAgent(HeuristicAgent(world_cfg=CFG), eps)
        noisy.reset(0)
        flips = np.mean([(noisy.act(obs) != base).mean() for _ in range(400)])
        assert flips == pytest.approx(eps * np.mean(1 - 1 / NVEC), abs=0.05)
    assert noisy.name == "heuristic_noisy"


def test_degenerate_agent_is_constant_per_episode_and_covers_the_list():
    agent, seen = DegenerateAgent(), set()
    for seed in range(60):
        agent.reset(seed)
        a = agent.act(DUMMY_OBS)
        assert all(np.array_equal(agent.act(DUMMY_OBS), a) for _ in range(5))
        seen.add(tuple(a.tolist()))
    assert seen == set(DEGENERATE_ACTIONS)


def test_ppo_policy_needs_models_and_uses_them_stochastically():
    with pytest.raises(FileNotFoundError):
        RandomPPOAgent([])
    if not PPO_MODEL.exists():
        pytest.skip("modèle PPO de l'étape 4 absent")
    agent = RandomPPOAgent([PPO_MODEL])
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=2)
    agent.reset(0)
    acts = {tuple(agent.act(obs)) for _ in range(30)}
    assert len(acts) > 3, "politique stochastique : actions variées sur une même observation"


def test_factory_rejects_unknown_policy():
    assert make_data_policy("random", CFG).name == "random"
    with pytest.raises(ValueError):
        make_data_policy("inconnue", CFG)


# ------------------------------------------------------------------ épisodes et jeux de données
@pytest.fixture(scope="module")
def tiny_arrays():
    cfg = DataConfig()
    return collect(CFG, [1, 2, 3, 4, 5], ["random", "heuristic", "random_block", "degenerate", "heuristic_noisy"], cfg, 0, log=lambda *_: None)


def test_play_episode_shapes_and_quantization():
    env = AntWorldEnv(CFG)
    ep = play_episode(HeuristicAgent(world_cfg=CFG), env, 3, 0)
    T = len(ep["actions"])
    assert T == 400 and ep["maps"].shape == (T + 1, 8, 32, 32) and ep["maps"].dtype == np.uint8
    assert ep["states"].shape == (T + 1, 24) and ep["rewards"].shape == (T,) and not ep["dones"].any()
    env2 = AntWorldEnv(CFG)
    obs, _ = env2.reset(seed=3)
    assert np.abs(ep["maps"][0].astype(np.float32) / 255 - obs["map"]).max() <= 0.5 / 255 + 1e-6      # erreur de quantification bornée
    assert np.array_equal(ep["states"][0], obs["state"])


def test_dead_colony_episode_is_marked_terminated_at_its_last_step():
    env = AntWorldEnv(CFG)
    agent = DegenerateAgent()
    agent._action = np.array([3, 3, 0, 0, 0])                         # personne ne cueille
    agent.reset = lambda seed: None
    ep = play_episode(agent, env, 0, 0)
    assert len(ep["actions"]) < 400 and ep["dones"][-1] and not ep["dones"][:-1].any()


def test_collected_arrays_are_padded_consistent_and_reproducible(tiny_arrays):
    a = tiny_arrays
    assert a["maps"].shape[:2] == (5, 401) and a["actions"].shape == (5, 400, 5) and (a["lengths"] <= 400).all()
    for i, T in enumerate(a["lengths"]):
        assert not a["maps"][i, T + 1:].any() and not a["actions"][i, T:].any() and (a["rewards"][i, T:] == 0).all()
        assert a["dones"][i].sum() == (1 if T < 400 else 0)
    again = collect(CFG, [1, 2, 3, 4, 5], ["random", "heuristic", "random_block", "degenerate", "heuristic_noisy"], DataConfig(), 0, log=lambda *_: None)
    assert all(np.array_equal(a[k], again[k]) for k in a)


def test_splits_use_disjoint_world_ranges_and_new_trajectories_on_seen_worlds():
    cfg = DataConfig(n_train=8, n_val=3, n_eval_worlds=3, policies=("random", "heuristic", "random_block"), weights=(1, 1, 1))
    s = build_splits(CFG, cfg, log=lambda *_: None)
    assert set(s) == {"train", "val", "eval_seen", "eval_unseen"}
    tw, vw, sw, uw = (set(s[k]["world_seeds"].tolist()) for k in ("train", "val", "eval_seen", "eval_unseen"))
    assert max(tw) < cfg.train_world_pool and tw.isdisjoint(vw | uw) and min(vw) >= VAL_START and min(uw) >= test_seeds(1)[0]
    assert sw <= tw, "eval_seen : mêmes mondes que l'entraînement"
    assert uw == set(test_seeds(3)) and len(s["eval_unseen"]["lengths"]) == 9   # 3 mondes × 3 politiques
    assert set(s["eval_unseen"]["policies"].tolist()) == {0, 2, 3} or len(set(s["eval_unseen"]["policies"].tolist())) == 3


def test_save_and_load_roundtrip(tmp_path, tiny_arrays):
    save_split(tmp_path / "x.npz", tiny_arrays)
    back = load_split(tmp_path / "x.npz")
    assert all(np.array_equal(back[k], tiny_arrays[k]) for k in tiny_arrays)


# ------------------------------------------------------------------ échantillonnage
def test_windows_dequantize_and_stay_within_episodes(tiny_arrays):
    ds = TrajectoryDataset(tiny_arrays)
    w = ds.window(0, 10, 7)
    assert w["maps"].shape == (8, 8, 32, 32) and w["maps"].dtype == np.float32 and 0 <= w["maps"].min() and w["maps"].max() <= 1
    assert w["actions"].shape == (7, 5) and w["rewards"].shape == (7,)
    assert np.array_equal(w["actions"][0], tiny_arrays["actions"][0, 10])
    short = int(np.argmin(ds.lengths))
    with pytest.raises(ValueError):
        ds.window(short, int(ds.lengths[short]) - 2, 5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = ds.sample_batch(6, 30, rng)
        assert b["maps"].shape == (6, 31, 8, 32, 32) and b["actions"].shape == (6, 30, 5)
    with pytest.raises(ValueError):
        ds.sample_batch(2, 10_000, rng)
    b1, b2 = ds.sample_batch(4, 12, np.random.default_rng(5)), ds.sample_batch(4, 12, np.random.default_rng(5))
    assert all(np.array_equal(b1[k], b2[k]) for k in b1)


def test_windows_never_cross_the_end_of_a_short_episode(tiny_arrays):
    ds = TrajectoryDataset(tiny_arrays)
    rng = np.random.default_rng(1)
    short = int(np.argmin(ds.lengths))
    L = int(ds.lengths[short])
    if L >= 400:
        pytest.skip("aucun épisode court dans ce jeu")
    for _ in range(300):
        b = ds.sample_batch(1, L, rng)           # ne peut venir que d'épisodes d'au moins L pas
        assert b["maps"].shape[1] == L + 1


def test_persistence_stats_match_brute_force(tiny_arrays):
    ds = TrajectoryDataset(tiny_arrays)
    st = ds.persistence_stats()
    sse, n = np.zeros(8), 0
    for e in range(len(ds)):
        T = int(ds.lengths[e])
        w = ds.window(e, 0, T)
        sse += ((w["maps"][1:] - w["maps"][:-1]) ** 2).sum(axis=(0, 2, 3)); n += T
    assert np.allclose(st["map_var"], sse / (n * 1024), rtol=1e-4, atol=1e-9)
    assert (st["state_var"] >= 0).all() and st["map_var"][5] == pytest.approx(0.0, abs=1e-9), "le nid ne bouge jamais"
    assert st["map_var"][0] > 0 and st["map_var"][6] > 0, "la vision et les éclaireuses changent"
