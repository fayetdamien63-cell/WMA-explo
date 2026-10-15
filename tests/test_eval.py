"""Étape 3 : banc d'évaluation (seeds, statistiques, harnais, journalisation)."""

import numpy as np
import pytest

from agents.factory import BaselinesConfig
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
from eval import seeds as sd
from eval.harness import (evaluate, load_results, log_tensorboard, returns_by_world, run_episode,
                          save_results)
from eval.stats import (bootstrap_ci, paired_difference, seeds_needed, summarize,
                        variance_decomposition)
from helpers import default_cfg

CFG = default_cfg()
BASE = BaselinesConfig()


# ------------------------------------------------------------------ seeds
def test_train_val_test_seed_ranges_are_disjoint():
    train, val, test = sd.train_seeds(1000), sd.val_seeds(1000), sd.test_seeds(1000)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert max(train) < min(val) < min(test)
    assert sd.test_seeds(5, start=10) == list(range(sd.TEST_START + 10, sd.TEST_START + 15))
    with pytest.raises(ValueError):
        sd.test_seeds(sd.RANGE_SIZE + 1)


# ------------------------------------------------------------------ statistiques
def test_summary_matches_numpy_and_bootstrap_is_deterministic():
    v = np.random.default_rng(0).normal(50, 10, 40)
    s = summarize(v)
    assert s.mean == pytest.approx(v.mean()) and s.std == pytest.approx(v.std(ddof=1))
    assert s.sem == pytest.approx(v.std(ddof=1) / np.sqrt(40)) and s.n == 40
    assert s.ci_low < s.mean < s.ci_high and (s.ci_low, s.ci_high) == bootstrap_ci(v)
    # L'IC bootstrap a la bonne largeur (≈ 1,96 × erreur standard).
    assert (s.ci_high - s.ci_low) / 2 == pytest.approx(1.96 * s.sem, rel=0.15)


def test_ci_coverage_is_close_to_95_percent():
    """Sur des échantillons de vraie moyenne connue, l'IC contient la vérité ~95 % du temps."""
    rng, hits, trials = np.random.default_rng(1), 0, 300
    for _ in range(trials):
        lo, hi = bootstrap_ci(rng.normal(10, 3, 30), n_boot=500, seed=int(rng.integers(1 << 30)))
        hits += lo <= 10 <= hi
    assert 0.90 <= hits / trials <= 0.99


def test_paired_difference_detects_a_consistent_gap_hidden_by_world_variance():
    rng = np.random.default_rng(2)
    world = rng.normal(100, 30, 20)                        # forte variance entre mondes
    a, b = world + 5 + rng.normal(0, 1, 20), world + rng.normal(0, 1, 20)
    d = paired_difference(a, b)
    assert d.ci_low > 0 and d.mean == pytest.approx(5, abs=1)
    unpaired = summarize(a).ci_low < summarize(b).ci_high   # sans appariement, les IC se chevauchent
    assert unpaired
    with pytest.raises(ValueError):
        paired_difference([1, 2, 3], [1, 2])


def test_seeds_needed_formula_and_monotonicity():
    assert seeds_needed(10.0, 2.0) == int(np.ceil((1.96 * 10 / 2) ** 2))
    assert seeds_needed(10.0, 1.0) > seeds_needed(10.0, 2.0) > seeds_needed(10.0, 4.0)
    assert seeds_needed(20.0, 2.0) > seeds_needed(10.0, 2.0)
    with pytest.raises(ValueError):
        seeds_needed(1.0, 0.0)


def test_variance_decomposition_recovers_known_variances():
    rng = np.random.default_rng(3)
    world = rng.normal(0, 10, (400, 1))                    # σ_monde = 10 -> variance 100
    returns = world + rng.normal(0, 3, (400, 6))           # σ_agent = 3 -> variance 9
    vd = variance_decomposition(returns)
    assert vd["var_between_worlds"] == pytest.approx(100, rel=0.2)
    assert vd["var_within_world"] == pytest.approx(9, rel=0.2)
    assert vd["world_share"] == pytest.approx(100 / 109, abs=0.03)


# ------------------------------------------------------------------ harnais
def test_run_episode_record_is_consistent():
    r = run_episode(HeuristicAgent(world_cfg=CFG), CFG, 7)
    assert r.steps == r.env_steps == len(r.series["pop"]) == len(r.series["reward"]) == CFG.sim.max_steps
    assert r.ret == pytest.approx(float(np.sum(r.series["reward"])), rel=1e-4)
    assert r.final_pop == int(r.series["pop"][-1]) and r.max_pop >= r.final_pop
    assert r.train_env_steps == 0 and r.sim_steps == 0 and r.wall_s > 0 and r.decision_ms >= 0
    assert not r.terminated


def test_results_are_reproducible_and_independent_of_evaluation_order():
    a = evaluate("random", CFG, BASE, [3, 4], agent_seeds=(0, 1))
    b = evaluate("random", CFG, BASE, [4, 3], agent_seeds=(1, 0))
    key = lambda r: (r.world_seed, r.agent_seed)
    ra, rb = {key(r): r.ret for r in a}, {key(r): r.ret for r in b}
    assert ra == rb
    assert ra[(3, 0)] != ra[(3, 1)], "deux seeds d'agent donnent deux parties différentes"
    again = evaluate("random", CFG, BASE, [3, 4], agent_seeds=(0, 1))
    assert [r.ret for r in a] == [r.ret for r in again]


def test_parallel_evaluation_matches_sequential():
    seq = evaluate("heuristic", CFG, BASE, [10, 11, 12], workers=1)
    par = evaluate("heuristic", CFG, BASE, [10, 11, 12], workers=2)
    assert [r.ret for r in seq] == [r.ret for r in par] and [r.world_seed for r in par] == [10, 11, 12]


def test_only_privileged_agents_receive_the_environment():
    class Spy:
        sim_steps = train_env_steps = 0
        def __init__(self, privileged): self.privileged, self.name, self.seen = privileged, "spy", []
        def reset(self, seed): pass
        def act(self, obs, env=None):
            self.seen.append(env is not None)
            return np.zeros(5, dtype=int)
    for privileged in (False, True):
        spy = Spy(privileged)
        run_episode(spy, CFG, 1)
        assert set(spy.seen) == {privileged}


def test_save_and_load_roundtrip_with_nan_padding(tmp_path):
    res = [run_episode(RandomAgent(), CFG, 1), run_episode(_Doomed(), CFG, 2)]
    path = save_results(res, tmp_path, "demo")
    rows = load_results(path)
    assert [r["world_seed"] for r in rows] == [1, 2] and rows[1]["terminated"] and not rows[0]["terminated"]
    assert set(rows[0]) >= {"ret", "steps", "final_pop", "env_steps", "sim_steps", "train_env_steps", "wall_s", "decision_ms"}
    z = np.load(tmp_path / "demo_series.npz")
    assert z["pop"].shape == (2, CFG.sim.max_steps)
    assert np.isnan(z["pop"][1, rows[1]["steps"]:]).all() and not np.isnan(z["pop"][0]).any()
    assert z["reward"][0].sum() == pytest.approx(rows[0]["ret"], rel=1e-4)
    assert returns_by_world(rows, "random", [1]) [0] == pytest.approx(rows[0]["ret"])
    with pytest.raises(KeyError):
        returns_by_world(rows, "random", [99])


class _Doomed:
    """Agent qui condamne la colonie (personne ne cueille) : sert à tester l'épisode terminé tôt."""
    name, privileged, sim_steps, train_env_steps = "doomed", False, 0, 0
    def reset(self, seed): pass
    def act(self, obs, env=None): return np.array([3, 3, 0, 0, 0])


def test_tensorboard_logging_writes_event_files(tmp_path):
    res = [run_episode(RandomAgent(), CFG, 1)]
    assert log_tensorboard(res, tmp_path / "tb")
    assert list((tmp_path / "tb").glob("events.*"))


def test_plot_scripts_run_on_saved_results(tmp_path, monkeypatch):
    import json
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"))
    import plot_results
    for name, agent in (("random", RandomAgent()), ("heuristic", HeuristicAgent(world_cfg=CFG)), ("oracle", HeuristicAgent(world_cfg=CFG))):
        res = [run_episode(agent, CFG, s) for s in (1, 2)]
        for r in res:
            r.agent = name
        save_results(res, tmp_path, name)
    curve = tmp_path / "ppo_curve.jsonl"
    curve.write_text("\n".join(json.dumps({"agent": "ppo", "seed": s, "train_env_steps": x, "mean_return": 50 + x / 1000 + s})
                               for s in (0, 1) for x in (1000, 2000, 3000)), encoding="utf-8")
    for extra in ([], ["--curves", str(curve)]):
        monkeypatch.setattr(sys, "argv", ["plot_results", str(tmp_path), *extra])
        plot_results.main()
        assert (tmp_path / "overview.png").stat().st_size > 5000
