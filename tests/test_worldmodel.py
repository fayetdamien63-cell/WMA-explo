"""Étape 5 : architecture, absence de fuite, sur-apprentissage d'un petit lot, métriques, entraînement, rendu."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.data_policies import DegenerateAgent
from antworld.render import render_obs_map
from eval import wm_metrics as M
from helpers import default_cfg
from models.data import DataConfig, TrajectoryDataset, collect
from models.train import WMTrainConfig, fit_norm, load_checkpoint, save_checkpoint, to_tensors, train_world_model
from models.worldmodel import WMConfig, WorldModel, count_parameters

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

CFG = default_cfg()
TINY = dict(hidden=64, embed=96, head_channels=8)
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def ds():
    arrays = collect(CFG, [1, 2, 3, 4, 5, 6], ["random", "heuristic", "random_block", "heuristic", "random", "heuristic_noisy"],
                     DataConfig(), 0, log=lambda *_: None)
    return TrajectoryDataset(arrays)


def batch_of(ds, B=3, L=8, seed=0):
    return to_tensors(ds.sample_batch(B, L, np.random.default_rng(seed)), "cpu")


@pytest.fixture(params=[True, False], ids=["résiduel", "latent seul"])
def model(request):
    torch.manual_seed(0)
    return WorldModel(WMConfig(residual=request.param, **TINY)).eval()


# ------------------------------------------------------------------ forme et initialisation
def test_output_shapes_ranges_and_size(model, ds):
    b = batch_of(ds)
    out = model.rollout(b["maps"], b["states"], b["actions"], n_forced=3)
    assert out["map"].shape == (3, 8, 8, 32, 32) and out["state"].shape == (3, 8, 24)
    assert out["reward"].shape == (3, 8) and out["done_logit"].shape == (3, 8)
    assert (out["map"] >= 0).all() and (out["map"] <= 1).all()
    assert all(torch.isfinite(v).all() for v in out.values())
    assert count_parameters(WorldModel(WMConfig())) < 2_000_000


def test_residual_model_starts_exactly_at_persistence(ds):
    """Sortie nulle à l'initialisation : le modèle prédit « rien ne change » avant tout entraînement (cadre de comparaison honnête)."""
    m = WorldModel(WMConfig(residual=True, **TINY)).eval()
    b = batch_of(ds)
    out = m.rollout(b["maps"], b["states"], b["actions"], n_forced=8)
    assert torch.allclose(out["map"], b["maps"][:, :-1], atol=1e-6)
    assert torch.allclose(out["state"], b["states"][:, :-1], atol=1e-6)


# ------------------------------------------------------------------ pas de fuite d'information du futur
def test_open_loop_predictions_do_not_depend_on_future_observations(model, ds):
    b = batch_of(ds, L=10)
    n = 4
    ref = model.rollout(b["maps"], b["states"], b["actions"], n_forced=n)
    for kind in ("bruit", "zéro"):
        pert = {k: v.clone() for k, v in b.items()}
        if kind == "bruit":
            pert["maps"][:, n + 1:] = torch.rand_like(pert["maps"][:, n + 1:])
            pert["states"][:, n + 1:] = torch.rand_like(pert["states"][:, n + 1:])
        else:
            pert["maps"][:, n + 1:] = 0
            pert["states"][:, n + 1:] = 0
        out = model.rollout(pert["maps"], pert["states"], pert["actions"], n_forced=n)
        for k in ref:
            assert torch.equal(ref[k], out[k]), f"fuite : « {k} » dépend d'observations futures ({kind})"
    sens = {k: v.clone() for k, v in b.items()}
    sens["maps"][:, n] = 1 - sens["maps"][:, n]                       # la DERNIÈRE observation autorisée compte, elle
    out = model.rollout(sens["maps"], sens["states"], sens["actions"], n_forced=n)
    assert not torch.equal(ref["map"][:, n:], out["map"][:, n:])


def test_a_prediction_never_uses_its_own_target(model, ds):
    """La prédiction du pas t est faite AVANT d'observer o_{t+1} : changer o_1 ne change pas la prédiction du pas 0."""
    b = batch_of(ds)
    ref = model.rollout(b["maps"], b["states"], b["actions"], n_forced=5)
    pert = {k: v.clone() for k, v in b.items()}
    pert["maps"][:, 1] = 1 - pert["maps"][:, 1]
    out = model.rollout(pert["maps"], pert["states"], pert["actions"], n_forced=5)
    assert torch.equal(ref["map"][:, 0], out["map"][:, 0]) and not torch.equal(ref["map"][:, 1], out["map"][:, 1])


def test_first_open_loop_step_equals_the_teacher_forced_prediction(model, ds):
    """Observer ou non APRÈS une prédiction ne change pas cette prédiction : cohérence entre les deux modes."""
    b = batch_of(ds, L=9)
    a = model.rollout(b["maps"], b["states"], b["actions"], n_forced=3)
    c = model.rollout(b["maps"], b["states"], b["actions"], n_forced=4)
    for k in a:
        assert torch.equal(a[k][:, :4], c[k][:, :4])


def test_actions_matter(model, ds):
    b = batch_of(ds)
    ref = model.rollout(b["maps"], b["states"], b["actions"], n_forced=2)
    other = (b["actions"] + 1) % torch.tensor([4, 4, 4, 3, 5])
    out = model.rollout(b["maps"], b["states"], other, n_forced=2)
    assert not torch.allclose(ref["state"][:, 3:], out["state"][:, 3:]) or not torch.allclose(ref["reward"][:, 3:], out["reward"][:, 3:])


# ------------------------------------------------------------------ apprentissage
@pytest.mark.parametrize("residual, max_ratio", [(True, 0.15), (False, 0.6)], ids=["résiduel", "latent seul"])
def test_can_overfit_a_small_batch(ds, residual, max_ratio):
    """Vérification du code : sur un lot fixe, la perte (1 pas puis boucle ouverte) doit s'effondrer."""
    torch.manual_seed(0)
    m = WorldModel(WMConfig(residual=residual, **TINY))
    fit_norm(m, ds)
    b = batch_of(ds, B=4, L=8, seed=1)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for i in range(300):
        loss, info = m.loss(b, n_forced=4)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        first = first if first is not None else info["loss"]
    assert info["loss"] < max_ratio * first, f"perte {first:.4f} -> {info['loss']:.4f}"
    assert info["loss_open"] < 0.6 * first


def test_residual_head_can_flip_binary_channels(ds):
    """Régression : les canaux BINAIRES (valeurs exactement 0/1 : visible, connu, roche) doivent pouvoir changer. Avec une
    paramétrisation σ(logit(base)+δ) leur gradient était atténué d'un facteur ~1000 et le modèle restait à « rien ne change »."""
    torch.manual_seed(0)
    m = WorldModel(WMConfig(residual=True, **TINY))
    fit_norm(m, ds)
    b = batch_of(ds, B=4, L=8, seed=1)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(400):
        loss, _ = m.loss(b, n_forced=8)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        out = m.rollout(b["maps"], b["states"], b["actions"], n_forced=8)["map"]
    err = ((out - b["maps"][:, 1:]) ** 2).mean(dim=(0, 1, 3, 4))
    persist = ((b["maps"][:, :-1] - b["maps"][:, 1:]) ** 2).mean(dim=(0, 1, 3, 4))
    for c, name in ((0, "visible"), (1, "known")):
        assert persist[c] > 1e-6
        assert err[c] < 0.6 * persist[c], f"canal binaire « {name} » bloqué à la persistance : {float(err[c] / persist[c]):.2f}"


def test_gradients_flow_through_open_loop_to_every_module(ds):
    m = WorldModel(WMConfig(residual=True, **TINY))
    fit_norm(m, ds)
    for p in m.parameters():                       # sorties nulles à l'init : on les perturbe pour que les gradients circulent
        p.data += 0.01 * torch.randn_like(p)
    loss, _ = m.loss(batch_of(ds), n_forced=2)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, f"paramètres sans gradient : {dead}"
    assert m.encoder.map_fc[0].weight.grad.abs().sum() > 0 and m.dyn_cell.weight_hh.grad.abs().sum() > 0


def test_loss_normalisation_makes_persistence_cost_about_one_per_dynamic_channel(ds):
    """w = 1/(variance de changement + ε) : une prédiction « rien ne change » coûte ≈ 1 par canal dynamique (et 0 pour le nid immobile),
    donc carte, état et récompense (normalisée par son écart-type) ont des termes de perte du même ordre de grandeur."""
    m = WorldModel(WMConfig(**TINY))
    fit_norm(m, ds)
    st = ds.persistence_stats()
    var = torch.as_tensor(st["map_var"], dtype=torch.float32)
    assert (m.map_w > 0).all() and m.reward_std > 0
    importance = m.map_w * var                       # coût de la persistance par canal
    assert importance[5] < 1e-6, "le nid est immobile : coût nul"
    dynamic = importance[var > 1e-5]                  # TOUS les canaux qui changent, y compris les moins dynamiques (minerai, nourriture, roche)
    assert len(dynamic) >= 6, "attendu : tous les canaux sauf le nid"
    assert (dynamic > 0.6).all() and (dynamic <= 1.0 + 1e-6).all(), f"coût de la persistance par canal dynamique : {dynamic.tolist()}"
    # Même ordre de grandeur que le terme de récompense, mesuré sur un lot : la persistance coûte ~1 sur la carte comme sur l'état.
    b = batch_of(ds, B=8, L=6, seed=3)
    persist_map = (((b["maps"][:, 1:] - b["maps"][:, :-1]) ** 2).mean(dim=(3, 4)) * m.map_w).mean()
    persist_state = (((b["states"][:, 1:] - b["states"][:, :-1]) ** 2) * m.state_w).mean()
    assert 0.1 < float(persist_map) < 3.0 and 0.1 < float(persist_state) < 3.0, (float(persist_map), float(persist_state))


def test_checkpoint_roundtrip(tmp_path, model, ds):
    save_checkpoint(tmp_path / "m.pt", model, {"iter": 7})
    back = load_checkpoint(tmp_path / "m.pt")
    assert back.cfg == model.cfg
    b = batch_of(ds)
    a, c = (x.rollout(b["maps"], b["states"], b["actions"], 3) for x in (model, back))
    assert all(torch.equal(a[k], c[k]) for k in a)


def test_training_loop_runs_logs_and_is_reproducible(tmp_path, ds):
    cfg = WMTrainConfig(batch_size=4, phase_iters=(3, 3, 3), phase_context=(6, 3, 3), phase_horizon=(0, 3, 5), val_every=3,
                        val_batches=1, log_every=1, torch_threads=2, seed=4)
    losses = []
    for sub in ("a", "b"):
        train_world_model(ds, ds, WMConfig(**TINY), cfg, tmp_path / sub, log=lambda *_: None)
        rows = [__import__("json").loads(l) for l in (tmp_path / sub / "train_log.jsonl").read_text().splitlines()]
        losses.append([r["loss"] for r in rows])
    assert len(rows) == 9 and (tmp_path / "a" / "model.pt").exists() and "val_loss" in rows[2]
    assert losses[0] == losses[1], "même seed => même entraînement"


# ------------------------------------------------------------------ métriques
class _Replay:
    """Faux modèle : renvoie une prédiction construite à partir de la vérité (pour tester les métriques)."""

    def __init__(self, mode, reward_mean=0.0):
        self.mode, self.reward_mean = mode, torch.tensor(reward_mean)

    def eval(self):
        return self

    def rollout(self, maps, states, actions, n_forced, detach_base=False):
        L = actions.shape[1]
        tm, ts = maps[:, 1:], states[:, 1:]
        if self.mode == "perfect":
            return {"map": tm.clone(), "state": ts.clone(), "reward": torch.zeros(actions.shape[:2]) + 0.0,
                    "done_logit": torch.full(actions.shape[:2], -9.0)}
        # « rien ne change » : on répète la dernière observation fournie (o_{n_forced}) pour tous les pas d'imagination
        base_m = maps[:, n_forced: n_forced + 1].expand(-1, L, -1, -1, -1)
        base_s = states[:, n_forced: n_forced + 1].expand(-1, L, -1)
        return {"map": base_m.clone(), "state": base_s.clone(), "reward": torch.zeros(actions.shape[:2]),
                "done_logit": torch.full(actions.shape[:2], -9.0)}


def test_persistence_model_has_exactly_zero_skill_and_perfect_model_skill_one(ds):
    class Perfect(_Replay):
        def rollout(self, maps, states, actions, n_forced, detach_base=False):
            out = super().rollout(maps, states, actions, n_forced)
            return out

    # Le modèle « parfait » renvoie aussi les vraies récompenses : on les fournit via un sous-classement
    class PerfectWithRewards(_Replay):
        def rollout(self, maps, states, actions, n_forced, detach_base=False):
            out = super().rollout(maps, states, actions, n_forced)
            return out
    ds_small = ds
    res_p = M.evaluate_model(_Replay("persist"), ds_small, n_boot=20)
    for h in M.HORIZONS:
        x = res_p["horizons"][str(h)]
        assert x["map_skill_macro"]["value"] == pytest.approx(0.0, abs=1e-6)
        assert x["state_skill_total"]["value"] == pytest.approx(0.0, abs=1e-6)
        assert x["map_mse_model"]["value"] == pytest.approx(x["map_mse_persist"]["value"], rel=1e-6)
    res_o = M.evaluate_model(_Replay("perfect"), ds_small, n_boot=20)
    for h in M.HORIZONS:
        x = res_o["horizons"][str(h)]
        assert x["map_skill_macro"]["value"] == pytest.approx(1.0, abs=1e-6) and x["state_skill_macro"]["value"] == pytest.approx(1.0, abs=1e-6)
        assert x["map_mse_model"]["value"] == pytest.approx(0.0, abs=1e-9)


def test_reward_r2_and_auc_helpers():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert M._r2(t, t) == pytest.approx(1.0) and M._r2(t, np.full(4, t.mean())) == pytest.approx(0.0)
    assert M._r2(t, np.zeros(4)) < 0 and np.isnan(M._r2(np.ones(4), np.ones(4)))
    assert M._auc(np.array([1, 1, 0, 0.0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert M._auc(np.array([1, 0.0]), np.array([0.5, 0.5])) == 0.5 and np.isnan(M._auc(np.zeros(3), np.ones(3)))


def test_windows_respect_episode_bounds_and_stats_are_consistent(ds):
    wins = M.valid_windows(ds)
    assert wins and all(s - M.CONTEXT >= 0 and s + 50 <= ds.lengths[e] for e, s in wins)
    st = M.window_stats(_Replay("persist"), ds, batch=7)
    W = len(wins)
    assert st.sse_map_model.shape == (W, 5, 8) and st.sse_state_model.shape == (W, 5, 24) and st.rew_true.shape == (W, 5)
    e0, s0 = wins[0]
    w = ds.window(e0, s0 - M.CONTEXT, M.CONTEXT + 50)
    r = np.cumsum(w["rewards"][M.CONTEXT:])
    assert st.rew_true[0, 2] == pytest.approx(r[9], rel=1e-5)               # cumul sur 10 pas
    assert st.sse_map_persist[0, 1, 3] == pytest.approx(((w["maps"][M.CONTEXT] - w["maps"][M.CONTEXT + 5])[3] ** 2).sum(), rel=1e-4)
    assert summarize_twice_equal(st)


def summarize_twice_equal(st):
    import json
    return json.dumps(M.summarize(st, n_boot=30), sort_keys=True) == json.dumps(M.summarize(st, n_boot=30), sort_keys=True)   # NaN != NaN : on compare le texte


def test_untrained_residual_model_is_the_persistence_baseline_for_map_and_state(ds):
    m = WorldModel(WMConfig(residual=True, **TINY)).eval()
    res = M.evaluate_model(m, ds, n_boot=10)
    for h in (1, 10):
        x = res["horizons"][str(h)]
        assert x["map_skill_macro"]["value"] == pytest.approx(0.0, abs=1e-4) and x["state_skill_macro"]["value"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ rendu et script d'évaluation
def test_render_obs_map_colours():
    m = np.zeros((8, 32, 32), np.float32)
    img = render_obs_map(m, scale=4)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (38, 38, 44), "brouillard"
    m[1] = 1; m[3, 5, 5] = 1; m[5, 10, 10] = 1; m[6, 20, 20] = 1
    img = render_obs_map(m, scale=4)
    assert img[5 * 4, 5 * 4, 1] > img[5 * 4, 5 * 4, 0] * 0.6 and tuple(img[10 * 4, 10 * 4]) == (255, 215, 0)
    assert img[20 * 4, 20 * 4, 0] == 230 and img[20 * 4, 20 * 4, 1] == 30
    blur = render_obs_map(np.full((8, 32, 32), 0.5, np.float32), scale=2)
    assert blur.min() >= 0 and blur.max() <= 255


def test_gate_verdict_and_dream_figure(tmp_path, ds):
    import eval_world_model as E

    def fake(skill_map, skill_state, r2):
        return {"horizons": {str(h): {k: {"value": v} for k, v in (("map_skill_macro", skill_map), ("state_skill_macro", skill_state), ("reward_r2", r2))}
                             for h in M.HORIZONS}}
    good = E.gate_verdict(fake(0.6, 0.6, 0.8), fake(0.55, 0.6, 0.8))
    assert good["passed"] and all(c["ok"] for c in good["checks"].values())
    bad = E.gate_verdict(fake(0.6, 0.6, 0.8), fake(0.1, 0.6, 0.8))
    assert not bad["passed"] and sum(not c["ok"] for c in bad["checks"].values()) >= 2      # carte à 10 pas et à 1 pas (+ écart)
    gap = E.gate_verdict(fake(0.9, 0.6, 0.8), fake(0.6, 0.6, 0.8))
    assert not gap["passed"] and [k for k, c in gap["checks"].items() if not c["ok"]] == [k for k in gap["checks"] if "écart" in k]
    m = WorldModel(WMConfig(residual=True, **TINY)).eval()
    wins = M.valid_windows(ds, starts=(120,))
    E.dream_vs_reality(m, ds, wins[0][0], wins[0][1], tmp_path / "dream.png")
    assert (tmp_path / "dream.png").stat().st_size > 2000
    E.plot_curves({"résiduel": {"eval_seen": M.evaluate_model(m, ds, n_boot=5), "eval_unseen": M.evaluate_model(m, ds, n_boot=5)}}, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 2000
