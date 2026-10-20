"""Diagnostics du planificateur dans le modèle : le modèle se trompe-t-il d'une façon que le planificateur exploite ?

Un planificateur qui maximise la récompense PRÉDITE cherche, entre autres, les séquences où le modèle surestime la récompense : « malédiction
de l'optimiseur ». On le mesure : pour un échantillon de plans choisis, on compare la récompense cumulée PRÉDITE par le modèle à la récompense
RÉELLE obtenue en rejouant exactement ces actions dans le vrai simulateur (plusieurs aléas, depuis l'état réel de la partie) ; et on fait la
même mesure pour des séquences ALÉATOIRES de contrôle. Si l'écart (prédit − réel) est plus grand pour les plans choisis que pour les séquences
aléatoires, le planificateur exploite les erreurs du modèle.

Instrumentation : l'enveloppe lit l'état réel (`env.world`) mais ne le transmet JAMAIS à l'agent ; elle ne sert qu'à mesurer.
"""

from __future__ import annotations

import numpy as np
import torch

from antworld.reward import potential_terms, step_reward
from antworld.world import AntWorld
from common.seeding import SeedManager

NVEC = np.array([4, 4, 4, 3, 5])


def true_step_rewards(world: AntWorld, seq: np.ndarray, rc, rng) -> np.ndarray:
    """Récompenses réelles pas à pas de la séquence `seq`, jouée depuis l'état de `world` (copie ; aléa fourni)."""
    work = world.clone()
    work._rng = rng
    out = []
    for a in seq:
        before = potential_terms(work, rc)
        work.step(a)
        out.append(step_reward(before, work, rc)[0])
    return np.array(out)


class PlanDiagnostics:
    """Enveloppe d'un `ModelPlannerAgent` : mesure prédit-contre-réel sur un plan de temps en temps (non privilégiée pour l'agent)."""

    privileged = True            # reçoit l'env du harnais, uniquement pour MESURER ; l'agent interne ne le reçoit pas
    train_env_steps = 0

    def __init__(self, inner, every: int = 8, n_random: int = 8, n_seeds: int = 3, seed: int = 0) -> None:
        self.inner, self.every, self.n_random, self.n_seeds = inner, every, n_random, n_seeds
        self.name = getattr(inner, "name", "planner")
        self.records: list[dict] = []
        self._seeds = SeedManager(seed)

    @property
    def sim_steps(self) -> int:
        return self.inner.sim_steps

    def reset(self, agent_seed: int) -> None:
        self.inner.reset(agent_seed)
        self.records = []

    def act(self, obs, env=None):
        a = self.inner.act(obs, None)
        if self.inner.just_planned and self.inner.plan_count % self.every == 0:
            self._measure(env)
        return a

    @torch.no_grad()
    def _measure(self, env) -> None:
        inner, world, rc = self.inner, env.world, env.cfg.reward
        rng = self._seeds.rng(f"measure/{inner.plan_count}")
        H = len(inner.last_plan)
        random_seqs = rng.integers(0, NVEC, size=(self.n_random, H, 5))
        rollout_seed = self._seeds.int_seed(f"rollout/{inner.plan_count}")

        def real(seq):
            return np.mean([true_step_rewards(world, seq, rc, SeedManager(rollout_seed + s).rng("diag")) for s in range(self.n_seeds)], axis=0)

        belief = {k: v.expand(self.n_random, *v.shape[1:]) for k, v in inner.belief.items()}
        pred_random = inner.model.imagine(belief, torch.as_tensor(random_seqs, dtype=torch.long, device=inner.device))["reward"].double().cpu().numpy()
        self.records.append({"step": world.macro_step, "pred_chosen": inner.last_plan_pred.copy(), "true_chosen": real(inner.last_plan),
                             "pred_random": pred_random, "true_random": np.stack([real(s) for s in random_seqs])})

    def summary(self, horizons=(5, 10, 25)) -> dict[str, dict]:
        """Écart moyen (prédit − réel) de la récompense cumulée, plans choisis contre séquences aléatoires."""
        out = {}
        for k in horizons:
            ch = [r["pred_chosen"][:k].sum() - r["true_chosen"][:k].sum() for r in self.records if len(r["pred_chosen"]) >= k]
            rn = [(r["pred_random"][:, :k].sum(1) - r["true_random"][:, :k].sum(1)).mean() for r in self.records if r["pred_random"].shape[1] >= k]
            out[str(k)] = {"n_plans": len(ch), "gap_chosen": float(np.mean(ch)) if ch else float("nan"),
                           "gap_random": float(np.mean(rn)) if rn else float("nan")}
        return out
