"""Étape 7 : Dreamer-lite (noyau d'apprentissage dans l'imagination, tampon, agent, boucle complète)."""

import numpy as np
import pytest
import torch

from agents.dreamer_agent import DreamerAgent, DreamerConfig, GapMeter, load_dreamer, train_dreamer
from antworld.config import AntWorldConfig, SimConfig
from antworld.env import AntWorldEnv
from antworld import observation as ob
from eval.harness import run_episode
from models.data import play_episode
from models.dreamer import (ACConfig, ActorCritic, ModelWorld, ReplayBuffer, lambda_returns, rescale_reward_head, sample_start_beliefs)
from models.worldmodel import WMConfig, WorldModel

torch.set_num_threads(2)
TINY_WM = WMConfig(hidden=32, embed=96, action_embed=16, enc_channels=(4, 8, 8), head_channels=4)


def tiny_world_cfg(steps=30) -> AntWorldConfig:
    return AntWorldConfig(sim=SimConfig(ticks_per_step=5, max_steps=steps))


# ---------------------------------------------------------------------------------------------- rendements λ

def test_lambda_returns_match_hand_computation():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    c = torch.ones(1, 3)
    g = 0.9
    # λ = 0 : TD(0), R_t = r_t + γ v_{t+1}
    np.testing.assert_allclose(lambda_returns(r, c, v, g, 0.0).numpy(), [[1 + 9 * 2.0, 2 + 0.9 * 30, 3 + 0.9 * 40]])
    # λ = 1 : somme actualisée puis valeur finale
    np.testing.assert_allclose(lambda_returns(r, c, v, g, 1.0).numpy(), [[1 + 0.9 * 2 + 0.81 * 3 + 0.729 * 40, 2 + 0.9 * 3 + 0.81 * 40, 3 + 0.9 * 40]])
    # mélange : R_2 = 3 + γ v_3 ; R_1 = 2 + γ((1−λ) v_2 + λ R_2) ; R_0 = 1 + γ((1−λ) v_1 + λ R_1)
    lam = 0.5
    R2 = 3 + 0.9 * 40
    R1 = 2 + 0.9 * ((1 - lam) * 30 + lam * R2)
    R0 = 1 + 0.9 * ((1 - lam) * 20 + lam * R1)
    np.testing.assert_allclose(lambda_returns(r, c, v, g, lam).numpy(), [[R0, R1, R2]], rtol=1e-6)


def test_lambda_returns_stop_at_death():
    """Si la colonie meurt (continuation 0), plus rien après la récompense de ce pas : ni valeur ni récompenses suivantes."""
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.full((1, 4), 100.0)
    c = torch.tensor([[1.0, 0.0, 1.0]])
    R = lambda_returns(r, c, v, 0.9, 0.95)
    assert R[0, 1].item() == pytest.approx(2.0)                                       # ni valeur finale ni récompense suivante
    assert R[0, 0].item() == pytest.approx(1 + 0.9 * ((1 - 0.95) * 100 + 0.95 * 2.0))   # R_0 = r_0 + γ((1−λ) v_1 + λ R_1)


# ---------------------------------------------------------------------------------------------- tampon

def fake_episode(T, fill, size=8):
    return {"maps": np.full((T + 1, 8, size, size), fill, np.uint8), "states": np.full((T + 1, 24), fill / 255, np.float32),
            "actions": np.zeros((T, 5), np.int8), "rewards": np.full(T, float(fill), np.float32), "dones": np.zeros(T, bool)}


def test_replay_buffer_is_circular_and_samples_valid_windows():
    buf = ReplayBuffer(3, size=8, max_steps=20)
    assert len(buf) == 0
    for i in range(5):
        buf.add(fake_episode(20 if i != 3 else 12, i + 1))
    assert len(buf) == 3 and buf.n_added == 5 and buf.env_steps == 20 * 4 + 12
    assert sorted(set(buf.arrays["rewards"][:, 0])) == [3.0, 4.0, 5.0]               # les épisodes 1 et 2 ont été écrasés
    rng = np.random.default_rng(0)
    b = buf.sample_batch(16, 10, rng)
    assert b["maps"].shape == (16, 11, 8, 8, 8) and b["actions"].shape == (16, 10, 5)
    assert set(np.unique(b["rewards"])) <= {3.0, 4.0, 5.0}                           # jamais un emplacement vide ou écrasé


# ---------------------------------------------------------------------------------------------- apprentissage : mondes jouets

class ToyWorld:
    """Monde jouet sans modèle appris. Contexte c ∈ {0..3} visible ; indicateur f visible.
    Récompense : 1·[a0 = c]  (pour apprendre à LIRE le contexte)  +  1·f  (f = [a1 = 1] du pas précédent : crédit RETARDÉ)."""

    feat_dim = 8

    def __init__(self, generator):
        self.gen = generator

    def start(self, B):
        return {"c": torch.randint(0, 4, (B,), generator=self.gen), "f": torch.zeros(B)}

    def features(self, b):
        return torch.cat([torch.nn.functional.one_hot(b["c"], 4).float(), b["f"][:, None], torch.zeros(len(b["c"]), 3)], dim=1)

    def step(self, b, a):
        r = (a[:, 0] == b["c"]).float() + b["f"]
        return r, torch.zeros(len(r)), {"c": torch.randint(0, 4, (len(r),), generator=self.gen), "f": (a[:, 1] == 1).float()}


def test_actor_critic_learns_toy_problem_with_delayed_credit():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    world = ToyWorld(gen)
    ac = ActorCritic(world.feat_dim, ACConfig(hidden=32, horizon=8, ent_coef=0.003, actor_lr=3e-3, critic_lr=3e-3, value_scale=5.0), "cpu")
    for _ in range(300):
        info = ac.update(ac.imagine(world, world.start(64), 8, gen))
    with torch.no_grad():
        eval_b = world.start(512)
        a = ac.actor.sample(world.features(eval_b), gen)
    assert (a[:, 0] == eval_b["c"]).float().mean() > 0.9          # lit le contexte
    assert (a[:, 1] == 1).float().mean() > 0.9                    # a compris qu'une récompense arrive au pas suivant
    assert info["entropy"] < 4.0                                  # la politique s'est resserrée (entropie max ≈ 6,9)
    # le critique a appris la valeur d'une politique quasi optimale : ~ (1 + 1) par pas sur 8 pas, amorcé par la valeur finale
    f = world.features({"c": torch.zeros(4, dtype=torch.long), "f": torch.ones(4)})
    assert ac.critic.value(f).mean().item() > 10.0


def test_actor_critic_does_not_learn_from_noise():
    """Contrôle négatif : si la récompense ne dépend pas des actions, la politique reste proche de l'uniforme (pas de dérive spontanée)."""
    class NoiseWorld(ToyWorld):
        def step(self, b, a):
            return torch.rand(len(a), generator=self.gen), torch.zeros(len(a)), self.start(len(a))
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    world = NoiseWorld(gen)
    ac = ActorCritic(world.feat_dim, ACConfig(hidden=32, horizon=8, ent_coef=0.01, actor_lr=1e-3, value_scale=5.0), "cpu")
    for _ in range(200):
        info = ac.update(ac.imagine(world, world.start(64), 8, gen))
    assert info["entropy"] > 6.0


# ---------------------------------------------------------------------------------------------- adaptateur du vrai modèle

@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = WorldModel(TINY_WM).eval()
    for p in m.parameters():
        p.data += 0.02 * torch.randn_like(p)
    return m


def test_model_world_ends_the_dream_at_episode_end_and_tracks_time(tiny_model):
    world = ModelWorld(tiny_model, max_steps=400)
    maps, states = torch.rand(2, 8, 32, 32), torch.rand(2, 24)
    states[:, ob.S_TIME] = torch.tensor([0.5, 398 / 400])
    b = world.start(tiny_model.init_belief(maps, states))
    assert b["t"].tolist() == [200.0, 398.0]
    a = torch.zeros(2, 5, dtype=torch.long)
    _, end, b1 = world.step(b, a)
    assert b1["t"].tolist() == [201.0, 399.0] and end[1] < 0.5                    # pas 399 : l'épisode continue encore
    _, end, b2 = world.step(b1, a)
    assert b2["t"].tolist() == [202.0, 400.0] and end[1].item() == pytest.approx(1.0)   # le pas 400 est le dernier : le rêve s'arrête


def test_imagination_gives_no_gradient_to_the_world_model(tiny_model):
    """L'acteur ne « pousse » jamais le modèle : seule la mise à jour supervisée sur données réelles le modifie."""
    world = ModelWorld(tiny_model)
    ac = ActorCritic(world.feat_dim, ACConfig(hidden=16, horizon=4), "cpu")
    b = world.start(tiny_model.init_belief(torch.rand(4, 8, 32, 32), torch.rand(4, 24)))
    tiny_model.zero_grad(set_to_none=True)
    ac.update(ac.imagine(world, b, 4))
    assert all(p.grad is None for p in tiny_model.parameters())


def test_rescale_reward_head_keeps_predictions(tiny_model):
    m = WorldModel(TINY_WM).eval()
    m.load_state_dict(tiny_model.state_dict())
    m.reward_mean.fill_(0.2); m.reward_std.fill_(0.3)
    maps, states, act = torch.rand(3, 8, 32, 32), torch.rand(3, 24), torch.zeros(3, 5, dtype=torch.long)
    with torch.no_grad():
        before = m.predict(m.init_belief(maps, states), act)[0]["reward"]
        rescale_reward_head(m, new_mean=1.5, new_std=0.07)
        after = m.predict(m.init_belief(maps, states), act)[0]["reward"]
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    assert float(m.reward_std) == pytest.approx(0.07)


def test_start_beliefs_come_from_real_windows(tiny_model):
    buf = ReplayBuffer(2, size=32, max_steps=30)
    cfg = tiny_world_cfg()
    env = AntWorldEnv(cfg)
    from agents.random_agent import RandomAgent
    for s in (1, 2):
        buf.add(play_episode(RandomAgent(), env, s, s))
    b = sample_start_beliefs(tiny_model, buf, 6, 5, np.random.default_rng(0), "cpu")
    assert b["h"].shape == (6, 32) and b["map"].shape == (6, 8, 32, 32)
    assert torch.isfinite(b["h"]).all()


# ---------------------------------------------------------------------------------------------- agent et boucle complète

def make_agent(model, deterministic=False):
    ac = ActorCritic(ModelWorld(model).feat_dim, ACConfig(hidden=16), "cpu")
    return DreamerAgent(model, ac, deterministic=deterministic, max_steps=30)


def test_agent_acts_validly_reproducibly_and_without_privileged_access(tiny_model):
    cfg = tiny_world_cfg()
    a1, a2 = make_agent(tiny_model), make_agent(tiny_model)
    assert a1.privileged is False
    a2.ac.actor.load_state_dict(a1.ac.actor.state_dict())
    env = AntWorldEnv(cfg)
    obs, _ = env.reset(seed=5)
    a1.reset(3); a2.reset(3)
    acts = []
    for _ in range(6):
        x, y = a1.act(obs, None), a2.act(obs, None)
        assert (x == y).all() and x.shape == (5,) and (x >= 0).all() and (x < np.array([4, 4, 4, 3, 5])).all()
        acts.append(x)
        obs, *_ = env.step(x)
    a1.reset(4)
    obs, _ = env.reset(seed=5)
    assert any((a1.act(obs, None) != acts[0]).any() for _ in range(30))          # une autre seed d'agent change les tirages


def test_gap_meter_measures_policy_and_random_sequences(tiny_model):
    cfg = tiny_world_cfg(30)
    agent = make_agent(tiny_model)
    meter = GapMeter(agent, every=10, n_seq=3, horizon=5, n_rollouts=2)
    res = run_episode(meter, cfg, 7, 0)
    s = meter.summary()
    assert len(meter.records) == 3 and res.steps == 30                           # mesures aux pas 0, 10, 20 (le pas 30 manque de place)
    assert meter.records[0]["pred_policy"].shape == (3, 5) and meter.records[0]["true_random"].shape == (3, 5)
    assert np.isfinite(list(s.values())).all()


def test_train_dreamer_end_to_end_is_deterministic_and_counts_steps(tmp_path):
    cfg = DreamerConfig(total_steps=120, warmup_steps=60, buffer_episodes=6, eval_every=60, eval_worlds=1, gap_every=10, gap_sequences=2,
                        wm=TINY_WM, wm_batch=4, wm_pretrain_updates=3, wm_phase_updates=(2, 2, 1000), wm_phase_context=(4, 4, 4),
                        wm_phase_horizon=(0, 3, 5), wm_updates_per_1k=50, ac=ACConfig(hidden=16, horizon=5), ac_updates_per_1k=100,
                        ac_batch=4, ac_context=3, refit_norm_every=60, torch_threads=2)
    wcfg = tiny_world_cfg(30)
    out = []
    for run in (0, 1):
        res = train_dreamer(wcfg, cfg, 3, tmp_path / f"r{run}", log=lambda *_: None)
        out.append([__import__("json").loads(line) for line in (tmp_path / f"r{run}" / "dreamer_seed3_curve.jsonl").read_text().splitlines()])
    assert res["train_env_steps"] == 120                                          # 4 épisodes de 30 pas, tous comptés (échauffement compris)
    assert [r["train_env_steps"] for r in out[0]] == [60, 120]                    # point de départ (fin d'échauffement) puis une évaluation
    assert [r["mean_return_stochastic"] for r in out[0]] == [r["mean_return_stochastic"] for r in out[1]]     # mêmes seeds -> mêmes résultats
    assert out[0][-1]["ac_updates"] > 0 and out[0][-1]["wm_updates"] >= 3
    model, ac = load_dreamer(tmp_path / "r0" / "dreamer_seed3.pt")
    assert isinstance(model, WorldModel)
