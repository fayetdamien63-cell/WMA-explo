"""« Dreamer-lite » (étape 7) : agent + boucle d'entraînement complète.

Boucle (rien n'est préchargé : ni données de PPO, ni le modèle de l'étape 5 — chaque pas d'environnement consommé est compté) :

    échauffement   quelques épisodes joués au hasard  ->  premières données
    répéter        JOUER un épisode réel avec la politique courante (stochastique)      -> tampon de rejeu
                   METTRE À JOUR LE MODÈLE du monde sur le tampon (prédiction à 1 pas puis en boucle ouverte)
                   RÊVER depuis des croyances réelles et METTRE À JOUR acteur + critique sur ces rêves

À chaque évaluation (mondes de VALIDATION, politique stochastique : même protocole que la référence PPO) on journalise :
  - le retour, contre pas d'environnement et contre temps réel (hors évaluation) ;
  - l'écart RÊVE-RÉALITÉ : récompense prédite par le modèle pour une séquence d'actions de la politique, contre la récompense obtenue en
    rejouant exactement ces actions dans le vrai simulateur ; même mesure pour des séquences aléatoires (si l'écart est plus grand pour la
    politique, elle exploite les erreurs du modèle) ;
  - le biais du critique : valeur prédite d'une croyance contre le retour réellement obtenu ensuite.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from agents.data_policies import BlockRandomAgent
from agents.random_agent import RandomAgent
from antworld import observation as ob
from antworld.config import AntWorldConfig
from antworld.env import AntWorldEnv
from common.device import get_device
from common.seeding import SeedManager, seed_everything
from eval.harness import run_episode
from eval.plan_diagnostics import NVEC, true_step_rewards
from eval.seeds import train_seeds, val_seeds
from models.data import play_episode
from models.dreamer import (ACConfig, ActorCritic, ModelWorld, ReplayBuffer, refit_norm, sample_start_beliefs)
from models.train import fit_norm
from models.worldmodel import WMConfig, WorldModel, count_parameters


@dataclass
class DreamerConfig:
    total_steps: int = 200_000          # pas d'environnement d'entraînement (échauffement compris)
    warmup_steps: int = 4_000           # épisodes aléatoires avant de commencer à apprendre
    buffer_episodes: int = 300          # tampon circulaire (≈ 1 Go pour 300 épisodes)
    train_pool_size: int = 1000         # mondes d'entraînement : seeds 0..n-1 (comme PPO)
    eval_every: int = 10_000
    eval_worlds: int = 10               # mondes de validation de la courbe (comme PPO)
    gap_every: int = 40                 # pas entre deux mesures « rêve contre réalité » pendant l'évaluation
    gap_sequences: int = 8
    # --- world model (entraîné de zéro, en ligne) ---
    wm: WMConfig = field(default_factory=WMConfig)
    wm_batch: int = 16
    wm_updates_per_1k: float = 10.0     # mises à jour du modèle par 1000 pas d'environnement
    wm_pretrain_updates: int = 300      # après l'échauffement, avant la première mise à jour de la politique
    wm_phase_updates: tuple[int, ...] = (100, 100, 1_000_000)   # durée (en mises à jour) des phases : 1 pas observé -> boucle ouverte courte -> longue
    wm_phase_context: tuple[int, ...] = (16, 8, 8)
    wm_phase_horizon: tuple[int, ...] = (0, 8, 16)
    wm_phase_lr: tuple[float, ...] = (1e-3, 5e-4, 3e-4)
    wm_grad_clip: float = 1.0
    wm_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 0.1)
    wm_weight_floor: float = 5e-6
    refit_norm_every: int = 25_000      # pas entre deux ré-estimations de la normalisation du modèle
    # --- acteur-critique dans l'imagination ---
    ac: ACConfig = field(default_factory=ACConfig)
    ac_updates_per_1k: float = 25.0
    ac_batch: int = 64                  # croyances de départ par mise à jour
    ac_context: int = 8                 # pas réels rejoués pour construire une croyance de départ
    device: str = "auto"
    torch_threads: int = 1


# ----------------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------------

class DreamerAgent:
    """Politique de Dreamer-lite : croyance du modèle (mise à jour avec les vraies observations partielles) -> acteur.

    N'utilise que l'observation partielle (`privileged = False`). `deterministic=False` : action tirée de la politique (celle qu'on
    optimise ; même convention que la référence PPO). Pas de planification : `sim_steps` reste 0.
    """

    privileged = False
    sim_steps = 0

    def __init__(self, model: WorldModel, ac: ActorCritic, deterministic: bool = False, name: str | None = None,
                 max_steps: int = 400) -> None:
        self.model, self.ac, self.deterministic = model, ac, deterministic
        self.world = ModelWorld(model, max_steps)
        self.device = ac.device
        self.name = name or ("dreamer" if deterministic else "dreamer_stochastic")
        self.train_env_steps = 0
        self.record_values = False
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        self._gen = torch.Generator(device=self.device).manual_seed(int(agent_seed))
        self.belief, self._prev, self.values = None, None, []

    @torch.no_grad()
    def act(self, obs, env=None) -> np.ndarray:
        m = torch.as_tensor(obs["map"], device=self.device)[None]
        s = torch.as_tensor(obs["state"], device=self.device)[None]
        self.belief = self.model.init_belief(m, s) if self.belief is None else self.model.advance(self.belief, self._prev, m, s)
        f = self.world.features(self.belief)
        if self.record_values:
            self.values.append(float(self.ac.critic.value(f)))
        if self.deterministic:
            a = torch.stack([lp.argmax(1) for lp in self.ac.actor.log_probs(f)], dim=1)
        else:
            a = self.ac.actor.sample(f, self._gen)
        self._prev = a
        return a[0].cpu().numpy().astype(np.int64)


class GapMeter:
    """Enveloppe de mesure « rêve contre réalité » (lit l'état réel via `env`, ne le donne jamais à l'agent).

    Toutes les `every` décisions : depuis la croyance courante, on fait RÊVER `n_seq` séquences à la politique (boucle ouverte), puis on
    rejoue exactement ces actions dans le vrai simulateur (3 aléas) depuis l'état réel. Contrôle : séquences ALÉATOIRES, mêmes mesures.
    """

    privileged = True
    sim_steps = 0

    def __init__(self, inner: DreamerAgent, every: int = 40, n_seq: int = 8, horizon: int | None = None, n_rollouts: int = 3,
                 seed: int = 0) -> None:
        self.inner, self.every, self.n_seq, self.n_rollouts = inner, every, n_seq, n_rollouts
        self.horizon = horizon or inner.ac.cfg.horizon
        self.name = inner.name
        self._seeds = SeedManager(seed)
        self.records: list[dict] = []
        self._count = 0

    @property
    def train_env_steps(self) -> int:
        return self.inner.train_env_steps

    def reset(self, agent_seed: int) -> None:
        self.inner.reset(agent_seed)
        self.records, self._count = [], 0

    def act(self, obs, env=None):
        a = self.inner.act(obs, None)
        if self._count % self.every == 0 and env.world.macro_step + self.horizon <= env.cfg.sim.max_steps:
            self._measure(env)
        self._count += 1
        return a

    @torch.no_grad()
    def _measure(self, env) -> None:
        inner, world, rc, H, K = self.inner, env.world, env.cfg.reward, self.horizon, self.n_seq
        rng = self._seeds.rng(f"gap/{self._count}")
        base = inner.world.start(inner.belief)
        belief = {k: v.expand(K, *v.shape[1:]) for k, v in base.items()}
        gen = torch.Generator(device=inner.device).manual_seed(int(rng.integers(2**31)))
        dream = inner.ac.imagine(inner.world, belief, H, gen)
        pol_seqs, pol_pred = dream["actions"].cpu().numpy(), dream["rewards"].double().cpu().numpy()
        rand_seqs = rng.integers(0, NVEC, size=(K, H, 5))
        rand_pred = inner.model.imagine({k: v for k, v in belief.items() if k != "t"},
                                        torch.as_tensor(rand_seqs, dtype=torch.long, device=inner.device))["reward"].double().cpu().numpy()
        seed0 = self._seeds.int_seed(f"rollout/{self._count}")

        def real(seq):
            return np.mean([true_step_rewards(world, seq, rc, SeedManager(seed0 + s).rng("gap")) for s in range(self.n_rollouts)], axis=0)

        self.records.append({"step": world.macro_step, "pred_policy": pol_pred, "true_policy": np.stack([real(s) for s in pol_seqs]),
                             "pred_random": rand_pred, "true_random": np.stack([real(s) for s in rand_seqs])})

    def summary(self) -> dict[str, float]:
        """Écarts moyens (prédit − réel) de la récompense cumulée sur l'horizon, et corrélation prédit/réel entre séquences."""
        if not self.records:
            return {}
        out = {}
        for tag in ("policy", "random"):
            pred = np.concatenate([r[f"pred_{tag}"].sum(1) for r in self.records])
            true = np.concatenate([r[f"true_{tag}"].sum(1) for r in self.records])
            out[f"gap_{tag}"] = float((pred - true).mean())
            out[f"abs_gap_{tag}"] = float(np.abs(pred - true).mean())
            out[f"true_{tag}"] = float(true.mean())
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Sauvegarde
# ----------------------------------------------------------------------------------------------------------------------

def save_dreamer(path: Path, model: WorldModel, ac: ActorCritic, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"wm_cfg": asdict(model.cfg), "ac_cfg": asdict(ac.cfg), "wm": model.state_dict(), "actor": ac.actor.state_dict(),
                "critic": ac.critic.state_dict(), "extra": extra or {}}, path)


def load_dreamer(path: str | Path, device="cpu") -> tuple[WorldModel, ActorCritic]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = WorldModel(WMConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["wm_cfg"].items()}))
    model.load_state_dict(ck["wm"])
    model.to(device).eval()
    ac = ActorCritic(ModelWorld(model).feat_dim, ACConfig(**ck["ac_cfg"]), device)
    ac.actor.load_state_dict(ck["actor"]); ac.critic.load_state_dict(ck["critic"]); ac.critic_slow.load_state_dict(ck["critic"])
    return model, ac


# ----------------------------------------------------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------------------------------------------------

class WorldModelUpdater:
    """Mises à jour EN LIGNE du modèle sur le tampon, avec le même programme progressif qu'à l'étape 5 (compté en mises à jour)."""

    def __init__(self, model: WorldModel, cfg: DreamerConfig, device) -> None:
        self.model, self.cfg, self.device = model, cfg, device
        self.n = 0
        self._phase, self._opt = -1, None

    def _phase_now(self) -> int:
        edges = np.cumsum(self.cfg.wm_phase_updates)
        return int(min(np.searchsorted(edges, self.n, side="right"), len(edges) - 1))

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict[str, float]:
        c = self.cfg
        ph = self._phase_now()
        if ph != self._phase:                                          # nouvelle phase : nouvel optimiseur (pas, comme à l'étape 5)
            self._phase, self._opt = ph, torch.optim.Adam(self.model.parameters(), lr=c.wm_phase_lr[ph])
        ctx, hor = c.wm_phase_context[ph], c.wm_phase_horizon[ph]
        ctx = min(ctx, buffer.max_length())
        hor = min(hor, buffer.max_length() - ctx)
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in buffer.sample_batch(c.wm_batch, ctx + hor, rng).items()}
        loss, info = self.model.loss(batch, ctx, c.wm_loss_weights, detach_base=True)
        self._opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.wm_grad_clip)
        self._opt.step()
        self.n += 1
        return {**info, "phase": ph}


def _value_bias(agent: DreamerAgent, series_reward: np.ndarray, gamma: float) -> tuple[float, float]:
    """(biais moyen, erreur absolue moyenne) de la valeur prédite contre le retour actualisé RÉEL jusqu'à la fin de l'épisode."""
    r = np.asarray(series_reward, dtype=np.float64)
    G = np.zeros(len(r))
    acc = 0.0
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        G[t] = acc
    v = np.asarray(agent.values[: len(r)])
    return float((v - G[: len(v)]).mean()), float(np.abs(v - G[: len(v)]).mean())


def evaluate_dreamer(agent: DreamerAgent, world_cfg: AntWorldConfig, cfg: DreamerConfig, val_worlds, seed: int) -> dict:
    """Évalue la politique stochastique sur les mondes de validation + mesures rêve/réalité et biais du critique."""
    meter = GapMeter(agent, cfg.gap_every, cfg.gap_sequences, seed=seed)
    agent.record_values = True
    rets, pops, gaps, biases = [], [], [], []
    for ws in val_worlds:
        res = run_episode(meter, world_cfg, ws, seed)
        rets.append(res.ret); pops.append(res.mean_pop)
        gaps.append(meter.summary())
        biases.append(_value_bias(agent, res.series["reward"], cfg.ac.gamma))
    agent.record_values = False
    out = {"mean_return_stochastic": float(np.mean(rets)), "std_return": float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
           "mean_pop": float(np.mean(pops)), "value_bias": float(np.mean([b[0] for b in biases])),
           "value_abs_err": float(np.mean([b[1] for b in biases]))}
    for k in ("gap_policy", "gap_random", "abs_gap_policy", "abs_gap_random", "true_policy", "true_random"):
        vals = [g[k] for g in gaps if k in g]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def train_dreamer(world_cfg: AntWorldConfig, cfg: DreamerConfig, seed: int, out_dir: str | Path, total_steps: int | None = None,
                  log=print) -> dict:
    """Entraîne un Dreamer-lite (une seed) ; écrit la courbe, le journal d'entraînement et le point de reprise. Renvoie un résumé."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = total_steps or cfg.total_steps
    torch.set_num_threads(cfg.torch_threads)
    seed_everything(seed)
    seeds = SeedManager(seed)
    device = get_device(cfg.device)
    model = WorldModel(cfg.wm).to(device)
    world = ModelWorld(model, world_cfg.sim.max_steps)
    ac = ActorCritic(world.feat_dim, cfg.ac, device)
    agent = DreamerAgent(model, ac, max_steps=world_cfg.sim.max_steps)
    updater = WorldModelUpdater(model, cfg, device)
    buffer = ReplayBuffer(cfg.buffer_episodes, world_cfg.world.size, world_cfg.sim.max_steps, cfg.ac.gamma)
    env = AntWorldEnv(world_cfg)
    pool = train_seeds(cfg.train_pool_size)
    val_worlds = val_seeds(cfg.eval_worlds)
    rng_world, rng_wm, rng_start = (seeds.rng(f"dreamer/{n}") for n in ("worlds", "wm_batches", "starts"))
    gen = torch.Generator(device=device).manual_seed(seeds.int_seed("dreamer/imagine"))
    curve_path, train_log = out_dir / f"dreamer_seed{seed}_curve.jsonl", out_dir / f"dreamer_seed{seed}_train.jsonl"
    curve_path.write_text("", encoding="utf-8"); train_log.write_text("", encoding="utf-8")
    log(f"[dreamer seed {seed}] modèle {count_parameters(model):,} params, acteur+critique "
        f"{count_parameters(ac.actor) + count_parameters(ac.critic):,}, device {device}")

    st = {"steps": 0, "episodes": 0, "wall": 0.0, "next_eval": cfg.warmup_steps, "next_refit": cfg.refit_norm_every,
          "acc_wm": 0.0, "acc_ac": 0.0}

    def clock(fn):
        t0 = time.perf_counter()
        out = fn()
        st["wall"] += time.perf_counter() - t0
        return out

    def play(policy, tag: str) -> float:
        ws = int(rng_world.choice(pool))
        ep = play_episode(policy, env, ws, seeds.int_seed(f"train_episode/{st['episodes']}"))
        buffer.add(ep)
        st["steps"] += len(ep["actions"]); st["episodes"] += 1
        agent.train_env_steps = st["steps"]
        return float(ep["rewards"].sum())

    def train_ac_once() -> dict:
        belief, real_ret = sample_start_beliefs(model, buffer, cfg.ac_batch, cfg.ac_context, rng_start, device)
        belief = world.start(belief)
        return ac.update(ac.imagine(world, belief, cfg.ac.horizon, gen), (world.features(belief), real_ret))

    def evaluate_now() -> None:
        t0 = time.perf_counter()
        rec = {"agent": "dreamer", "seed": seed, "train_env_steps": st["steps"], "wall_s": st["wall"], "wm_updates": updater.n,
               "ac_updates": ac.n_updates, **evaluate_dreamer(agent, world_cfg, cfg, val_worlds, seed)}
        rec["eval_wall_s"] = time.perf_counter() - t0
        with curve_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"[dreamer seed {seed}] {st['steps']:>8d} pas  retour val {rec['mean_return_stochastic']:7.1f}  pop {rec['mean_pop']:5.1f}  "
            f"écart rêve−réel {cfg.ac.horizon} pas : politique {rec['gap_policy']:+.2f} / aléatoire {rec['gap_random']:+.2f}  "
            f"biais critique {rec['value_bias']:+.2f}  ({rec['wall_s']:.0f} s)")
        save_dreamer(out_dir / f"dreamer_seed{seed}.pt", model, ac, {"env_steps": st["steps"]})

    # 1) échauffement : épisodes aléatoires (moitié uniformes, moitié « par blocs », comme l'étape 5)
    warm = (RandomAgent(), BlockRandomAgent())
    while st["steps"] < cfg.warmup_steps:
        clock(lambda: play(warm[st["episodes"] % 2], "warmup"))
    log(f"[dreamer seed {seed}] échauffement : {st['steps']} pas, {st['episodes']} épisodes")
    clock(lambda: fit_norm(model, buffer.data, cfg.wm_weight_floor))
    for i in range(cfg.wm_pretrain_updates):
        info = clock(lambda: updater.update(buffer, rng_wm))
        if (i + 1) % 100 == 0:
            log(f"[dreamer seed {seed}] pré-entraînement du modèle {i + 1}/{cfg.wm_pretrain_updates} : perte {info['loss']:.3f} "
                f"(carte {info['map']:.2f} état {info['state']:.2f} récompense {info['reward']:.2f})")
    evaluate_now()                                         # point de départ de la courbe (politique non entraînée, modèle pré-entraîné)
    st["next_eval"] = st["steps"] + cfg.eval_every

    # 2) boucle principale : jouer -> modèle -> rêver
    while st["steps"] < total:
        t_before = st["steps"]
        ret = clock(lambda: play(agent, "train"))
        n = st["steps"] - t_before
        st["acc_wm"] += cfg.wm_updates_per_1k * n / 1000; st["acc_ac"] += cfg.ac_updates_per_1k * n / 1000
        wm_info, ac_info = {}, {}
        while st["acc_wm"] >= 1:
            wm_info = clock(lambda: updater.update(buffer, rng_wm)); st["acc_wm"] -= 1
        while st["acc_ac"] >= 1:
            ac_info = clock(train_ac_once); st["acc_ac"] -= 1
        if st["steps"] >= st["next_refit"]:
            clock(lambda: refit_norm(model, buffer, cfg.wm_weight_floor)); st["next_refit"] += cfg.refit_norm_every
        with train_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"env_steps": st["steps"], "episode": st["episodes"], "train_return": ret, "wall_s": st["wall"],
                                 **{f"wm_{k}": v for k, v in wm_info.items()}, **{f"ac_{k}": v for k, v in ac_info.items()}}) + "\n")
        if st["steps"] >= st["next_eval"]:
            evaluate_now(); st["next_eval"] += cfg.eval_every
    if st["next_eval"] - cfg.eval_every != st["steps"]:
        evaluate_now()
    save_dreamer(out_dir / f"dreamer_seed{seed}.pt", model, ac, {"env_steps": st["steps"]})
    return {"seed": seed, "train_env_steps": st["steps"], "train_wall_s": st["wall"], "curve": str(curve_path)}
