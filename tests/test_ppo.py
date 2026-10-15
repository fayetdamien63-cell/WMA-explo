"""Étape 4 : monde simplifié, pool de mondes d'entraînement, extracteur, PPO (plomberie), reproductibilité, courbes."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.heuristic import HeuristicAgent
from agents.ppo_agent import PPOAgent, PPOConfig, make_vec_env, train_ppo
from antworld import AntWorldConfig
from antworld import observation as ob
from antworld.env import AntWorldEnv
from antworld.layout import generate_world, playability_problem
from common.config import load_config
from eval.curves import per_seed_curves, sample_efficiency, steps_to_reach
from eval.harness import run_episode
from eval.seeds import TRAIN_START, VAL_START, train_seeds
from helpers import CONFIG_PATH, default_cfg

ROOT = CONFIG_PATH.parent.parent
SIMPLE = load_config(AntWorldConfig, ROOT / "configs" / "antworld_simple.yaml")


def tiny_cfg(**kw) -> PPOConfig:
    cfg = load_config(PPOConfig, ROOT / "configs" / "ppo.yaml")
    for k, v in dict(n_envs=2, n_steps=32, batch_size=32, n_epochs=2, eval_every=64, eval_worlds=2, total_steps=128).items():
        setattr(cfg, k, v)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------------ monde simplifié
def test_simple_world_has_food_only_and_no_fog():
    for seed in range(100):
        lay = generate_world(SIMPLE.world, seed)
        assert lay.ore_amount.sum() == 0 and (lay.food_cap > 0).sum() >= SIMPLE.world.min_food_cells
        assert playability_problem(lay, SIMPLE.world) is None
    env = AntWorldEnv(SIMPLE)
    obs, _ = env.reset(seed=1)
    assert (obs["map"][ob.CH_KNOWN] == 1).all() and (obs["map"][ob.CH_VISIBLE] == 1).all()
    assert (obs["map"][ob.CH_ORE] == 0).all()


def test_heuristic_does_not_scout_when_everything_is_visible():
    env = AntWorldEnv(SIMPLE)
    obs, _ = env.reset(seed=1)
    a = HeuristicAgent(world_cfg=SIMPLE).act(obs, None)
    assert a[0] == 0 and a[3] == 0 and a[2] == 0           # ni éclaireuses, ni construction, ni minerai
    full = AntWorldEnv(default_cfg())
    obs, _ = full.reset(seed=1)
    assert HeuristicAgent(world_cfg=default_cfg()).act(obs, None)[0] > 0   # avec brouillard : on explore


# ------------------------------------------------------------------ pool de mondes
def test_world_pool_only_draws_from_the_pool_and_is_reproducible():
    pool = [11, 12, 13]
    draws = []
    for rng_seed in (1, 1, 2):
        env = AntWorldEnv(SIMPLE, world_seeds=pool, rng_seed=rng_seed)
        draws.append([env.reset()[1]["world_seed"] for _ in range(30)])
    assert set(draws[0]) <= set(pool) and set(draws[0]) == set(pool)
    assert draws[0] == draws[1] and draws[0] != draws[2]
    env = AntWorldEnv(SIMPLE, world_seeds=pool)
    assert env.reset(seed=999)[1]["world_seed"] == 999      # une seed explicite l'emporte
    with pytest.raises(ValueError):
        AntWorldEnv(SIMPLE, world_seeds=[])


def test_training_environments_never_see_validation_or_test_worlds():
    cfg = tiny_cfg(train_pool_size=50)
    venv = make_vec_env(SIMPLE, cfg, seed=3)
    seen = set()
    for _ in range(40):
        venv.reset()
        seen |= set(venv.get_attr("world_seed"))
    assert seen and max(seen) < 50 and min(seen) >= TRAIN_START and max(seen) < VAL_START
    assert len(seen) > 10, "les mondes d'entraînement doivent varier"


def test_each_training_seed_has_its_own_environment_streams():
    a = [e for e in make_vec_env(SIMPLE, tiny_cfg(), seed=0).get_attr("world_seed")]
    b = make_vec_env(SIMPLE, tiny_cfg(), seed=1).get_attr("world_seed")
    assert a == [None, None] and b == [None, None]            # avant reset : rien de tiré
    seeds = {}
    for s in (0, 1):
        v = make_vec_env(SIMPLE, tiny_cfg(train_pool_size=1000), seed=s)
        seq = []
        for _ in range(6):
            v.reset(); seq.append(tuple(v.get_attr("world_seed")))
        seeds[s] = seq
    assert seeds[0] != seeds[1]


# ------------------------------------------------------------------ extracteur
def test_features_extractor_shapes_gradients_and_size():
    from models.features import AntFeaturesExtractor
    env = AntWorldEnv(SIMPLE)
    ext = AntFeaturesExtractor(env.observation_space)
    obs = {"map": torch.rand(5, 8, 32, 32), "state": torch.rand(5, 24)}
    out = ext(obs)
    assert out.shape == (5, ext.features_dim) == (5, 192)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ext.parameters())
    assert sum(p.numel() for p in ext.parameters()) < 100_000


# ------------------------------------------------------------------ PPO (plomberie)
def test_ppo_training_writes_curve_model_and_a_loadable_agent(tmp_path):
    summary = train_ppo(SIMPLE, tiny_cfg(), seed=0, out_dir=tmp_path, log=lambda *_: None)
    rows = [json.loads(l) for l in (tmp_path / "ppo_seed0_curve.jsonl").read_text().splitlines()]
    steps = [r["train_env_steps"] for r in rows]
    assert steps[0] == 0 and steps == sorted(steps) and steps[-1] == summary["train_env_steps"] >= 128
    assert all(np.isfinite(r["mean_return"]) and np.isfinite(r["mean_return_stochastic"]) and r["wall_s"] >= 0
               and r["agent"] == "ppo" and r["seed"] == 0 for r in rows)
    agent = PPOAgent.from_file(tmp_path / "ppo_seed0.zip", train_env_steps=steps[-1])
    res = run_episode(agent, SIMPLE, 5, agent_seed=0)
    assert np.isfinite(res.ret) and res.train_env_steps == steps[-1] and res.sim_steps == 0
    again = run_episode(agent, SIMPLE, 5, agent_seed=0)
    assert again.ret == res.ret, "politique déterministe"


def test_stochastic_agent_is_reproducible_given_its_seed_and_differs_from_argmax(tmp_path):
    train_ppo(SIMPLE, tiny_cfg(), seed=0, out_dir=tmp_path, log=lambda *_: None)
    det = PPOAgent.from_file(tmp_path / "ppo_seed0.zip")
    sto = PPOAgent.from_file(tmp_path / "ppo_seed0.zip", deterministic=False)
    assert det.name == "ppo" and sto.name == "ppo_stochastic"
    a, b, c = (run_episode(sto, SIMPLE, 5, agent_seed=s) for s in (0, 0, 1))
    assert a.ret == b.ret, "même seed d'agent => même partie stochastique"
    d = run_episode(det, SIMPLE, 5, 0)
    assert (a.ret, c.ret) != (d.ret, d.ret) and a.series["pop"].tolist() != d.series["pop"].tolist()


def test_ppo_is_reproducible_given_the_seed_and_differs_across_seeds(tmp_path):
    def curve(seed, sub):
        train_ppo(SIMPLE, tiny_cfg(), seed=seed, out_dir=tmp_path / sub, log=lambda *_: None)
        return [json.loads(l)["mean_return"] for l in (tmp_path / sub / f"ppo_seed{seed}_curve.jsonl").read_text().splitlines()]
    a, b, c = curve(0, "a"), curve(0, "b"), curve(1, "c")
    assert a == b, "même seed => même courbe"
    assert a != c, "seeds différentes => courbes différentes"


# ------------------------------------------------------------------ courbes
def test_steps_to_reach_interpolates_and_handles_unreached_targets():
    x, y = [0, 100, 200, 300], [0.0, 10.0, 30.0, 30.0]
    assert steps_to_reach(x, y, 20.0) == pytest.approx(150.0)
    assert steps_to_reach(x, y, 10.0) == 100.0 and steps_to_reach(x, y, 0.0) == 0.0
    assert steps_to_reach(x, y, 31.0) is None


def test_sample_efficiency_per_seed_and_median():
    recs = []
    for seed, speed in ((0, 1.0), (1, 2.0), (2, 0.1)):
        for step in (0, 1000, 2000, 3000):
            recs.append({"agent": "ppo", "seed": seed, "train_env_steps": step, "mean_return": 10 * speed * step / 1000,
                         "wall_s": step / 100})
    for r in recs:
        r["mean_return_stochastic"] = 2 * r["mean_return"]            # courbe stochastique deux fois plus rapide
    recs.append({"agent": "autre", "seed": 0, "train_env_steps": 0, "mean_return": 1e9, "mean_return_stochastic": 1e9})
    curves = per_seed_curves(recs)
    assert sorted(curves) == [0, 1, 2] and list(curves[0][0]) == [0, 1000, 2000, 3000]
    eff = sample_efficiency(recs, {"cible 10": 10.0})["cible 10"]
    assert eff["reached"] == "2/3"                          # la seed lente (×0,1) n'atteint jamais 10
    assert eff["steps"][0] == pytest.approx(1000) and eff["steps"][1] == pytest.approx(500) and eff["steps"][2] is None
    assert eff["median_steps"] == pytest.approx(750) and eff["wall_s"][0] == pytest.approx(10.0)
    sto = sample_efficiency(recs, {"cible 10": 10.0}, key="mean_return_stochastic")["cible 10"]
    assert sto["steps"][0] == pytest.approx(500) and sto["reached"] == "2/3"   # clé choisie : la courbe stochastique va 2x plus vite ; la seed lente n'atteint toujours pas


# ------------------------------------------------------------------ test long : PPO apprend vraiment (monde simplifié)
@pytest.mark.slow
def test_ppo_learns_on_the_simple_world(tmp_path):
    from agents.random_agent import RandomAgent
    cfg = load_config(PPOConfig, ROOT / "configs" / "ppo.yaml")
    cfg.eval_every, cfg.eval_worlds = 15_000, 5
    train_ppo(SIMPLE, cfg, seed=0, out_dir=tmp_path, total_steps=30_000, log=lambda *_: None)
    rows = [json.loads(l) for l in (tmp_path / "ppo_seed0_curve.jsonl").read_text().splitlines()]
    from eval.seeds import val_seeds
    rand = np.mean([run_episode(RandomAgent(), SIMPLE, w, 0).ret for w in val_seeds(5)])
    assert rows[-1]["mean_return"] > rand + 10 and rows[-1]["mean_return"] > rows[0]["mean_return"] + 10
