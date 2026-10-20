"""Étape 6 : planification dans le world model (interface modèle, agent, diagnostics)."""

import copy

import numpy as np
import pytest
import torch

from agents.model_planner import ModelPlannerAgent, ModelPlannerConfig, ModelRollout
from agents.planning import CEMConfig
from antworld import observation as ob
from antworld.env import AntWorldEnv
from eval.harness import run_episode
from eval.plan_diagnostics import PlanDiagnostics, true_step_rewards
from helpers import default_cfg
from models.data import DataConfig, TrajectoryDataset, collect
from models.worldmodel import WMConfig, WorldModel

CFG = default_cfg()
TINY = dict(hidden=64, embed=96, head_channels=8)
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def ds():
    return TrajectoryDataset(collect(CFG, [1, 2, 3], ["random", "heuristic", "random_block"], DataConfig(), 0, log=lambda *_: None))


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = WorldModel(WMConfig(residual=True, **TINY)).eval()
    for p in m.parameters():                            # sorties nulles à l'init : on perturbe pour avoir des prédictions non triviales
        p.data += 0.02 * torch.randn_like(p)
    return m


def window(ds, e=0, start=20, length=14):
    w = ds.window(e, start, length)
    return {k: torch.as_tensor(v)[None] for k, v in w.items()}


# ------------------------------------------------------------------ interface modèle : imagine / advance
def test_imagine_equals_the_open_loop_part_of_rollout(tiny_model, ds):
    b = window(ds)
    n = 4
    ref = tiny_model.rollout(b["maps"], b["states"], b["actions"], n_forced=n)
    belief = tiny_model.init_belief(b["maps"][:, 0], b["states"][:, 0])
    for t in range(n):                                    # mêmes pas observés que rollout(n_forced=n)
        belief = tiny_model.advance(belief, b["actions"][:, t], b["maps"][:, t + 1], b["states"][:, t + 1])
    out = tiny_model.imagine(belief, b["actions"][:, n:])
    for k in out:
        assert torch.allclose(out[k], ref[k][:, n:], atol=1e-5), f"imagine diverge de rollout sur « {k} »"


def test_advance_is_the_observed_step_of_rollout(tiny_model, ds):
    """La croyance construite pas à pas avec `advance` est celle de `rollout` : prédire depuis elle redonne la prédiction de rollout."""
    b = window(ds, e=1, start=40, length=10)
    ref = tiny_model.rollout(b["maps"], b["states"], b["actions"], n_forced=6)
    belief = tiny_model.init_belief(b["maps"][:, 0], b["states"][:, 0])
    for t in range(6):
        belief = tiny_model.advance(belief, b["actions"][:, t], b["maps"][:, t + 1], b["states"][:, t + 1])
    pred, _ = tiny_model.predict(belief, b["actions"][:, 6])
    assert torch.allclose(pred["map"], ref["map"][:, 6], atol=1e-5) and torch.allclose(pred["reward"], ref["reward"][:, 6], atol=1e-5)


def test_imagine_handles_a_batch_of_distinct_sequences(tiny_model, ds):
    b = window(ds)
    belief = tiny_model.init_belief(b["maps"][:, 0], b["states"][:, 0])
    rng = np.random.default_rng(0)
    seqs = torch.as_tensor(rng.integers(0, [4, 4, 4, 3, 5], size=(6, 9, 5)))
    out = tiny_model.imagine({k: v.expand(6, *v.shape[1:]) for k, v in belief.items()}, seqs)
    assert out["reward"].shape == (6, 9) and out["map"].shape == (6, 9, 8, 32, 32)
    one = tiny_model.imagine(belief, seqs[2:3])                  # le lot ne mélange pas les séquences
    assert torch.allclose(out["reward"][2], one["reward"][0], atol=1e-5)
    assert len({round(float(x), 5) for x in out["reward"].sum(1)}) > 1, "des séquences différentes donnent des récompenses différentes"


# ------------------------------------------------------------------ fonction d'évaluation de CEM
class _Stub:
    """Faux modèle dont on connaît la bonne réponse : récompense 1 par pas quand la 1re dimension de l'action vaut 2."""

    def __init__(self, death_logit=-20.0):
        self.imagined_shapes, self.death_logit = [], death_logit

    def to(self, device):
        return self

    def eval(self):
        return self

    def init_belief(self, m, s):
        return {"h": torch.zeros(1, 4), "map": m, "state": s}

    def advance(self, belief, action, m, s):
        return {"h": belief["h"], "map": m, "state": s}

    def imagine(self, belief, actions):
        self.imagined_shapes.append(tuple(actions.shape))
        B, H = actions.shape[:2]
        return {"reward": (actions[:, :, 0] == 2).float(), "done_logit": torch.full((B, H), self.death_logit)}


def fake_obs(time=0.0):
    state = np.zeros(24, np.float32)
    state[ob.S_TIME] = time
    return {"map": np.zeros((8, 32, 32), np.float32), "state": state}


def test_rollout_fn_returns_one_value_per_sequence_and_counts_imagined_steps():
    stub = _Stub()
    belief = stub.init_belief(torch.zeros(1, 8, 32, 32), torch.zeros(1, 24))
    fn = ModelRollout(stub, belief, use_done=False)
    seqs = np.zeros((7, 10, 5), dtype=int); seqs[:3, :, 0] = 2
    ret = fn(seqs)
    assert ret.shape == (7,) and ret[:3].tolist() == [10.0] * 3 and ret[3:].tolist() == [0.0] * 4 and fn.n_imagined == 70


def test_survival_weighting_discounts_rewards_after_a_predicted_death():
    belief = _Stub().init_belief(torch.zeros(1, 8, 32, 32), torch.zeros(1, 24))
    seqs = np.zeros((1, 10, 5), dtype=int); seqs[:, :, 0] = 2
    alive = ModelRollout(_Stub(death_logit=-20.0), belief, use_done=True)(seqs)[0]
    doomed = ModelRollout(_Stub(death_logit=20.0), belief, use_done=True)(seqs)[0]
    assert alive == pytest.approx(10.0, abs=1e-3)
    assert doomed == pytest.approx(1.0, abs=1e-3), "la colonie meurt au 1er pas : seule la 1re récompense compte"


# ------------------------------------------------------------------ agent
def stub_agent(cem=None):
    cfg = ModelPlannerConfig(cem=cem or CEMConfig(horizon=25, block=5, n_samples=32, n_iters=3, n_elites=6, replan_every=5), use_done=False)
    return ModelPlannerAgent(cfg, model=_Stub())


def test_planner_picks_the_action_the_model_rewards_and_accounts_for_its_budget():
    agent = stub_agent()
    agent.reset(1)
    actions = [agent.act(fake_obs(t / 400.0)) for t in range(40)]
    assert np.mean([a[0] == 2 for a in actions]) >= 0.95, "CEM + modèle doivent trouver l'action récompensée"
    c = agent.cfg.cem
    assert agent.plan_count == 8 and agent.sim_steps == 8 * c.n_samples * c.n_iters * c.horizon
    assert all(a.shape == (5,) and (a >= 0).all() and (a < [4, 4, 4, 3, 5]).all() for a in actions)


def test_planner_never_uses_the_environment_and_is_reproducible():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("l'agent a touché l'environnement")
    a, b = stub_agent(), stub_agent()
    assert a.privileged is False
    a.reset(3); b.reset(3)
    acts_a = [a.act(fake_obs(t / 400), Forbidden()) for t in range(15)]
    acts_b = [b.act(fake_obs(t / 400)) for t in range(15)]
    assert all(np.array_equal(x, y) for x, y in zip(acts_a, acts_b))
    b.reset(4)
    assert not all(np.array_equal(x, y) for x, y in zip(acts_a, [b.act(fake_obs(t / 400)) for t in range(15)]))


def test_planning_horizon_is_capped_by_the_remaining_steps():
    agent = stub_agent()
    agent.reset(0)
    agent.act(fake_obs(396 / 400))                    # il reste 4 pas
    assert all(shape[1] == 4 for shape in agent.model.imagined_shapes) and len(agent._queue) == 3   # 4 pas planifiés, 1 déjà exécuté


def test_reset_clears_the_belief_between_episodes():
    agent = stub_agent()
    agent.reset(0)
    agent.act(fake_obs())
    assert agent.belief is not None
    agent.reset(0)
    assert agent.belief is None and agent._queue == []


def test_full_episode_with_a_real_architecture_runs_and_stays_valid(tiny_model):
    cfg = ModelPlannerConfig(cem=CEMConfig(horizon=10, block=5, n_samples=8, n_iters=2, n_elites=2, replan_every=5))
    agent = ModelPlannerAgent(cfg, model=copy.deepcopy(tiny_model))
    env = AntWorldEnv(CFG)
    obs, _ = env.reset(seed=3)
    agent.reset(0)
    for _ in range(25):
        a = agent.act(obs, None)
        assert env.action_space.contains(np.asarray(a, dtype=np.int64))
        obs, *_ = env.step(a)
    assert agent.sim_steps == 5 * 8 * 2 * 10 and agent.plan_count == 5


# ------------------------------------------------------------------ diagnostics prédit contre réel
def test_true_rewards_match_the_environment_exactly_when_the_random_stream_is_shared():
    import copy as cp
    from numpy.random import Generator, PCG64
    env = AntWorldEnv(CFG)
    env.reset(seed=2)
    rng0 = np.random.default_rng(0)
    for _ in range(12):
        env.step(rng0.integers(0, [4, 4, 4, 3, 5]))
    seq = rng0.integers(0, [4, 4, 4, 3, 5], size=(9, 5))
    twin = Generator(PCG64())
    twin.bit_generator.state = cp.deepcopy(env.world._rng.bit_generator.state)     # même flux aléatoire que l'env
    predicted_true = true_step_rewards(env.world, seq, CFG.reward, twin)
    real = np.array([env.step(a)[1] for a in seq])
    assert np.allclose(predicted_true, real, atol=1e-12)
    assert env.world.macro_step == 21, "l'état réel n'est pas touché par la mesure"


def test_plan_diagnostics_records_chosen_and_random_sequences_without_leaking_the_env(tiny_model):
    cfg = ModelPlannerConfig(cem=CEMConfig(horizon=10, block=5, n_samples=8, n_iters=2, n_elites=2, replan_every=5), record_plans=True)
    inner = ModelPlannerAgent(cfg, model=copy.deepcopy(tiny_model))
    diag = PlanDiagnostics(inner, every=1, n_random=3, n_seeds=2)
    assert diag.privileged and not inner.privileged
    res = run_episode(_Truncated(diag), CFG, 5, 0)
    assert len(diag.records) == 80, "un plan tous les 5 pas sur 400 pas, mesuré à chaque fois (every=1)"
    r = diag.records[0]
    assert r["pred_chosen"].shape == (10,) and r["true_chosen"].shape == (10,) and r["pred_random"].shape == (3, 10) and r["true_random"].shape == (3, 10)
    s = diag.summary(horizons=(5, 10))
    full = [r for r in diag.records if len(r["pred_chosen"]) >= 10]
    assert len(full) == 79, "le dernier plan est plus court (il reste moins de 10 pas) et n'entre pas dans le résumé à 10 pas"
    assert set(s) == {"5", "10"} and s["10"]["n_plans"] == len(full) and np.isfinite(s["10"]["gap_chosen"])
    assert res.sim_steps > 0


class _Truncated:
    """Limite un épisode à quelques pas pour garder le test rapide (le harnais joue sinon 400 pas)."""
    def __init__(self, agent):
        self.agent, self.name, self.privileged, self.train_env_steps = agent, agent.name, agent.privileged, 0
        self.n = 0
    sim_steps = property(lambda self: self.agent.sim_steps)

    def reset(self, seed):
        self.agent.reset(seed)

    def act(self, obs, env=None):
        self.n += 1
        return self.agent.act(obs, env) if self.n <= 400 else np.zeros(5, dtype=int)


def test_diagnostic_summary_math():
    diag = PlanDiagnostics(_Stub(), every=1)
    diag.records = [{"pred_chosen": np.full(10, 2.0), "true_chosen": np.full(10, 1.0),
                     "pred_random": np.full((4, 10), 1.5), "true_random": np.full((4, 10), 1.0)}] * 3
    s = diag.summary(horizons=(5, 10))
    assert s["5"]["gap_chosen"] == pytest.approx(5.0) and s["5"]["gap_random"] == pytest.approx(2.5)
    assert s["10"]["gap_chosen"] == pytest.approx(10.0) and s["10"]["gap_random"] == pytest.approx(5.0)
