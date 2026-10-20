"""Planificateur dans le world model APPRIS (étape 6) : CEM/MPC sans apprentissage de politique.

C'est le MÊME planificateur que l'oracle (`agents.planning.cem_plan`) ; seule la fonction qui évalue les séquences d'actions change :
l'oracle les simule dans le vrai simulateur (et voit tout le monde), ici elles sont « imaginées » par le modèle à partir de la croyance
de la colonie, construite UNIQUEMENT avec les observations partielles réelles. L'agent n'a donc aucun accès privilégié.

Boucle de décision (MPC) : à chaque pas, la vraie observation met à jour la croyance (`WorldModel.advance`) ; tous les `replan_every`
pas, on planifie : CEM tire des séquences, le modèle les imagine en lot, on garde les meilleures selon la RÉCOMPENSE PRÉDITE cumulée, on
exécute le début de la meilleure séquence puis on replanifie.

Objectif : Σ_t  P(vivante avant le pas t) · r̂_t  sur l'horizon (la récompense d'une colonie morte est nulle ; la probabilité de mort
vient de la tête « fin d'épisode » du modèle). Pas de valeur terminale : l'horizon est court (≤ 25 pas, limite de fiabilité du modèle),
donc l'agent est myope par construction — c'est le compromis assumé de cette étape (l'étape 7 apprend une valeur).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from agents.planning import CEMConfig, cem_plan, shift_probs
from common.device import get_device
from common.seeding import SeedManager
from models.train import load_checkpoint
from models.worldmodel import WorldModel
from antworld import observation as ob


@dataclass
class ModelPlannerConfig:
    model_path: str = "runs/step5/wm_residual/model.pt"
    cem: CEMConfig = field(default_factory=lambda: CEMConfig(horizon=25, block=5, n_samples=64, n_iters=3, n_elites=8,
                                                             smoothing=0.5, replan_every=5, min_prob=0.02))
    use_done: bool = True              # pondérer les récompenses par la probabilité de survie prédite
    device: str = "cpu"                # planifier avec de petits lots est souvent plus rapide sur CPU ; "auto" pour GPU
    torch_threads: int = 1
    max_steps: int = 400               # longueur de l'épisode (la fraction de temps de l'observation est normalisée par elle)
    record_plans: bool = False         # garde le dernier plan et sa récompense prédite pas à pas (diagnostics)


class ModelRollout:
    """`rollout_fn` de CEM : évalue un lot de séquences d'actions (N, H, 5) dans le modèle depuis une croyance donnée."""

    def __init__(self, model: WorldModel, belief: dict[str, torch.Tensor], use_done: bool = True, device="cpu") -> None:
        self.model, self.belief, self.use_done, self.device = model, belief, use_done, device
        self.n_imagined = 0

    def __call__(self, seqs: np.ndarray) -> np.ndarray:
        N, H = seqs.shape[:2]
        belief = {k: v.expand(N, *v.shape[1:]) for k, v in self.belief.items()}
        out = self.model.imagine(belief, torch.as_tensor(seqs, dtype=torch.long, device=self.device))
        reward = out["reward"]
        if self.use_done:
            alive_before = torch.cumprod(1.0 - torch.sigmoid(out["done_logit"]), dim=1)
            alive_before = torch.cat([torch.ones(N, 1, device=reward.device), alive_before[:, :-1]], dim=1)
            reward = reward * alive_before
        self.n_imagined += N * H
        return reward.sum(dim=1).double().cpu().numpy()


class ModelPlannerAgent:
    """Agent planificateur dans le modèle appris. Utilise seulement l'observation partielle (non privilégié)."""

    name = "planner_model"
    privileged = False

    def __init__(self, cfg: ModelPlannerConfig | None = None, model: WorldModel | None = None) -> None:
        self.cfg = cfg or ModelPlannerConfig()
        torch.set_num_threads(self.cfg.torch_threads)
        self.device = get_device(self.cfg.device)
        self.model = (model or load_checkpoint(self.cfg.model_path)).to(self.device).eval()
        self.sim_steps = 0            # pas IMAGINÉS par le modèle (coût de la planification ; le « simulateur » est ici le modèle)
        self.train_env_steps = 0
        self.plan_count = 0
        self.last_plan: np.ndarray | None = None
        self.last_plan_pred: np.ndarray | None = None      # récompense prédite pas à pas du dernier plan (si record_plans)
        self.belief: dict[str, torch.Tensor] | None = None
        self.just_planned = False
        self.reset(0)

    def reset(self, agent_seed: int) -> None:
        self._seeds = SeedManager(agent_seed)
        self._rng = self._seeds.rng("agent/model_planner/cem")
        self.belief, self._prev_action, self._queue, self._probs = None, None, [], None

    @torch.no_grad()
    def act(self, obs, env=None) -> np.ndarray:
        m = torch.as_tensor(obs["map"], device=self.device)[None]
        s = torch.as_tensor(obs["state"], device=self.device)[None]
        if self.belief is None:
            self.belief = self.model.init_belief(m, s)
        else:
            self.belief = self.model.advance(self.belief, self._prev_action, m, s)    # la vraie observation corrige la croyance
        self.just_planned = False
        if not self._queue:
            self._plan(obs)
            self.just_planned = True
        a = self._queue.pop(0)
        self._prev_action = torch.as_tensor(a, dtype=torch.long, device=self.device)[None]
        return a

    def _plan(self, obs) -> None:
        cfg = self.cfg.cem
        # Pas restants, lus dans l'OBSERVATION (fraction de temps écoulé) : on ne planifie pas au-delà de la fin de l'épisode.
        remaining = max(int(round((1.0 - float(obs["state"][ob.S_TIME])) * self.cfg.max_steps)), 1)
        horizon = min(cfg.horizon, remaining)
        rollout = ModelRollout(self.model, self.belief, self.cfg.use_done, self.device)
        nvec = np.array([4, 4, 4, 3, 5])
        init = None
        if self._probs is not None:                       # réutilise la distribution de la planification précédente (décalée)
            n_blocks = -(-horizon // cfg.block)
            if self._probs.shape[0] >= n_blocks:
                valid = np.arange(int(nvec.max()))[None, :] < nvec[:, None]
                init = shift_probs(self._probs, max(cfg.replan_every // cfg.block, 1), valid / valid.sum(1, keepdims=True))[:n_blocks]
        seq, probs, _ = cem_plan(rollout, nvec, cfg, self._rng, horizon, init)
        self._probs = probs
        self.sim_steps += rollout.n_imagined
        self.plan_count += 1
        self.last_plan = seq
        if self.cfg.record_plans:
            with torch.no_grad():
                out = self.model.imagine({k: v for k, v in self.belief.items()}, torch.as_tensor(seq[None], dtype=torch.long, device=self.device))
            self.last_plan_pred = out["reward"][0].double().cpu().numpy()
        k = max(min(cfg.replan_every, horizon), 1)
        self._queue = [x.copy() for x in seq[:k]]
