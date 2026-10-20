"""Noyau de « Dreamer-lite » (étape 7) : acteur-critique entraîné DANS L'IMAGINATION d'un world model.

Principe (version minimale de Dreamer) :
  1. on part de croyances réelles (états latents construits en rejouant des fenêtres de trajectoires réelles) ;
  2. depuis chacune, la politique et le modèle « rêvent » `horizon` pas : l'acteur choisit une action, le modèle prédit la
     récompense, la probabilité de mort et la croyance suivante (boucle ouverte, aucune observation) ;
  3. on calcule des rendements λ avec le critique (il remplace tout ce qui dépasse l'horizon : la politique n'est donc PAS myope,
     contrairement au planificateur de l'étape 6) ;
  4. critique : régression vers ces rendements ; acteur : REINFORCE avec avantage (rendement − valeur) et bonus d'entropie.

Les actions sont discrètes : on ne rétropropage PAS à travers le modèle (REINFORCE), donc aucun gradient ne « pousse » le modèle ;
mais l'acteur peut toujours exploiter ses erreurs. C'est mesuré (écart rêve-réalité) et limité par le ré-entraînement du modèle sur
les données de la politique courante (boucle jouer → mettre à jour le modèle → rêver → mettre à jour la politique).

Tout dépend du modèle à travers un petit adaptateur (`features`, `step`) : les tests valident l'apprentissage sur un monde jouet
dont la solution est connue (`tests/test_dreamer.py`), indépendamment de la qualité du vrai modèle.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.data import MAX_STEPS, TrajectoryDataset
from antworld.observation import N_MAP_CHANNELS, N_STATE, S_TIME

NVEC = (4, 4, 4, 3, 5)


# ----------------------------------------------------------------------------------------------------------------------
# Tampon de rejeu
# ----------------------------------------------------------------------------------------------------------------------

class ReplayBuffer:
    """Épisodes réels joués par l'agent (tampon circulaire : les plus anciens sont écrasés).

    Même format que les jeux de l'étape 5 (cartes en uint8) pour réutiliser `TrajectoryDataset` : les emplacements inutilisés ont une
    longueur 0 et ne sont jamais tirés.
    """

    def __init__(self, capacity_episodes: int, size: int = 32, max_steps: int = MAX_STEPS, gamma: float = 0.99) -> None:
        n = capacity_episodes
        self.gamma = gamma
        self.arrays = {"maps": np.zeros((n, max_steps + 1, N_MAP_CHANNELS, size, size), np.uint8),
                       "states": np.zeros((n, max_steps + 1, N_STATE), np.float32),
                       "actions": np.zeros((n, max_steps, 5), np.int8), "rewards": np.zeros((n, max_steps), np.float32),
                       "dones": np.zeros((n, max_steps), bool), "lengths": np.zeros(n, np.int32),
                       "returns": np.zeros((n, max_steps + 1), np.float32)}       # retour actualisé RÉEL jusqu'à la fin de l'épisode, par observation
        self.data = TrajectoryDataset(self.arrays)      # partage les tableaux (et `lengths`) : les ajouts sont visibles immédiatement
        self.capacity, self.n_added, self.env_steps = n, 0, 0

    def add(self, episode: dict[str, np.ndarray]) -> None:
        i, T = self.n_added % self.capacity, len(episode["actions"])
        a = self.arrays
        a["maps"][i, : T + 1], a["states"][i, : T + 1] = episode["maps"], episode["states"]
        a["actions"][i, :T], a["rewards"][i, :T], a["dones"][i, :T] = episode["actions"], episode["rewards"], episode["dones"]
        a["lengths"][i] = T
        g, acc = np.zeros(T + 1, np.float32), 0.0
        for k in reversed(range(T)):                       # G_k = r_k + γ G_{k+1}, G_T = 0 : fin de partie = fin de l'avenir (comme les rêves)
            acc = float(episode["rewards"][k]) + self.gamma * acc
            g[k] = acc
        a["returns"][i, : T + 1] = g
        self.n_added += 1
        self.env_steps += T

    def __len__(self) -> int:
        return int((self.arrays["lengths"] > 0).sum())

    def sample_batch(self, batch: int, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return self.data.sample_batch(batch, length, rng)

    def sample_with_returns(self, batch: int, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Comme `sample_batch`, plus `ret` (batch,) : retour actualisé RÉEL à partir de la dernière observation de chaque fenêtre."""
        lengths = self.arrays["lengths"]
        ok = np.flatnonzero(lengths >= length)
        weights = (lengths[ok] - length + 1).astype(float)
        eps = rng.choice(ok, size=batch, p=weights / weights.sum())
        starts = [int(rng.integers(0, lengths[e] - length + 1)) for e in eps]
        wins = [self.data.window(int(e), s, length) for e, s in zip(eps, starts)]
        out = {k: np.stack([w[k] for w in wins]) for k in wins[0]}
        out["ret"] = np.array([self.arrays["returns"][e, s + length] for e, s in zip(eps, starts)], np.float32)
        return out

    def max_length(self) -> int:
        return int(self.arrays["lengths"].max())


# ----------------------------------------------------------------------------------------------------------------------
# Réseaux
# ----------------------------------------------------------------------------------------------------------------------

def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    """Politique multi-discrète : une loi catégorielle par dimension de l'action (têtes indépendantes).

    `unimix` mélange un peu d'uniforme dans chaque loi (comme DreamerV3) : aucune action n'a jamais une probabilité nulle, la politique
    ne peut pas devenir déterministe trop tôt et l'exploration ne s'éteint pas.
    """

    def __init__(self, in_dim: int, hidden: int = 256, unimix: float = 0.01, nvec=NVEC) -> None:
        super().__init__()
        self.nvec, self.unimix = tuple(nvec), unimix
        self.net = _mlp(in_dim, hidden, sum(self.nvec))
        nn.init.normal_(self.net[-1].weight, std=0.01)       # politique initiale ≈ uniforme
        nn.init.zeros_(self.net[-1].bias)

    def log_probs(self, f: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for logits in torch.split(self.net(f), list(self.nvec), dim=1):
            p = F.softmax(logits, dim=1)
            out.append(torch.log((1 - self.unimix) * p + self.unimix / p.shape[1]))
        return out

    @torch.no_grad()
    def sample(self, f: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.stack([torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1) for lp in self.log_probs(f)], dim=1)

    def log_prob_entropy(self, f: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lps = self.log_probs(f)
        logp = sum(lp.gather(1, actions[:, d: d + 1]).squeeze(1) for d, lp in enumerate(lps))
        ent = sum(-(lp.exp() * lp).sum(1) for lp in lps)
        return logp, ent


class Critic(nn.Module):
    """Valeur d'une croyance, en unités de récompense. Le réseau prédit v / `scale` (cibles O(1) quelle que soit l'échelle des rendements)."""

    def __init__(self, in_dim: int, hidden: int = 256, scale: float = 10.0) -> None:
        super().__init__()
        self.net, self.scale = _mlp(in_dim, hidden, 1), scale
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def normalized(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f).squeeze(-1)

    def value(self, f: torch.Tensor) -> torch.Tensor:
        return self.normalized(f) * self.scale


# ----------------------------------------------------------------------------------------------------------------------
# Rendements et mise à jour
# ----------------------------------------------------------------------------------------------------------------------

def lambda_returns(rewards: torch.Tensor, cont: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Rendements λ. rewards, cont : (B, H) ; values : (B, H+1) (valeur de la croyance AVANT chaque pas, plus la valeur finale).

    R_t = r_t + γ c_t [ (1 − λ) v_{t+1} + λ R_{t+1} ],  R_H = v_H (amorçage sur la valeur finale),
    c_t = probabilité que la colonie survive au pas t. λ = 0 : TD(0) ; λ = 1 : somme actualisée puis valeur finale.
    """
    H = rewards.shape[1]
    nxt, out = values[:, H], []
    for t in reversed(range(H)):
        nxt = rewards[:, t] + gamma * cont[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        out.append(nxt)
    return torch.stack(out[::-1], dim=1)


@dataclass
class ACConfig:
    hidden: int = 256
    horizon: int = 15                 # pas imaginés par départ (limite de fiabilité du modèle : H ≤ 25, voir l'étape 6)
    gamma: float = 0.99
    lam: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    ent_coef: float = 0.01            # bonus d'entropie (sur l'avantage standardisé, comme PPO)
    adv_floor: float = 0.1            # on divise l'avantage par max(écart-type, adv_floor) : pas d'amplification du bruit si tous les rêves se valent
    value_scale: float = 10.0
    target_tau: float = 0.02          # critique « lent » (moyenne mobile) pour les cibles : stabilise les rendements
    grad_clip: float = 1.0
    unimix: float = 0.01
    real_value_weight: float = 1.0    # poids de l'ancrage du critique sur les retours RÉELS du tampon (0 : Dreamer pur, critique appris dans les rêves seulement)


class ModelWorld:
    """Adaptateur du vrai world model (interface `start` / `features` / `step` utilisée par `ActorCritic`).

    L'épisode réel se termine au pas `max_steps` (troncature) : le modèle ne le sait pas (sa tête « fin » ne prédit que la mort), alors on
    suit le temps nous-mêmes (`belief["t"]`, lu dans la VRAIE observation de départ) et un rêve s'arrête exactement là où l'épisode réel
    s'arrête. Sans cela, le critique croirait à un avenir inexistant après le pas 400 et surévaluerait la fin de partie.
    """

    def __init__(self, model, max_steps: int = MAX_STEPS) -> None:
        self.model, self.max_steps = model, max_steps

    @property
    def feat_dim(self) -> int:
        return self.model.cfg.hidden + self.model.cfg.embed

    def start(self, belief: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ajoute le compteur de pas à une croyance construite avec de vraies observations."""
        t = torch.round(belief["state"][:, S_TIME] * self.max_steps)
        return {**belief, "t": t}

    @torch.no_grad()
    def features(self, belief: dict[str, torch.Tensor]) -> torch.Tensor:
        """Ce que voient acteur et critique : l'état latent h et l'encodage de la carte/état de base (identiques en réel et en rêve)."""
        return torch.cat([belief["h"], self.model.encoder(belief["map"], belief["state"])], dim=1)

    @torch.no_grad()
    def step(self, belief: dict[str, torch.Tensor], actions: torch.Tensor):
        """(récompense prédite, probabilité de FIN de rêve, croyance suivante). Fin = mort prédite ou fin de l'épisode (pas 400)."""
        pred, h1 = self.model.predict(belief, actions)
        t = belief["t"] + 1
        end = 1.0 - (1.0 - torch.sigmoid(pred["done_logit"])) * (t < self.max_steps).float()
        return pred["reward"], end, {"h": h1, "map": pred["map"], "state": pred["state"], "t": t}


class ActorCritic:
    """Acteur, critique (et critique lent) + leur mise à jour sur des rêves."""

    def __init__(self, feat_dim: int, cfg: ACConfig | None = None, device="cpu") -> None:
        self.cfg, self.device = cfg or ACConfig(), device
        c = self.cfg
        self.actor = Actor(feat_dim, c.hidden, c.unimix).to(device)
        self.critic = Critic(feat_dim, c.hidden, c.value_scale).to(device)
        self.critic_slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr)
        self.n_updates = 0

    @torch.no_grad()
    def imagine(self, world, belief: dict[str, torch.Tensor], horizon: int | None = None,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Rêve `horizon` pas depuis `belief` en suivant l'acteur. Renvoie des tenseurs (B, H, ...) ; `feats` a H+1 entrées."""
        H = horizon or self.cfg.horizon
        feats, acts, rews, conts = [], [], [], []
        for _ in range(H):
            f = world.features(belief)
            a = self.actor.sample(f, generator)
            r, done_p, belief = world.step(belief, a)
            feats.append(f); acts.append(a); rews.append(r); conts.append(1.0 - done_p)
        feats.append(world.features(belief))
        return {"feats": torch.stack(feats, 1), "actions": torch.stack(acts, 1), "rewards": torch.stack(rews, 1), "cont": torch.stack(conts, 1)}

    def update(self, roll: dict[str, torch.Tensor], real: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, float]:
        """`real` = (caractéristiques de croyances RÉELLES, retours réels depuis ces croyances) : terme de régression supplémentaire du critique."""
        c = self.cfg
        f, actions, r, cont = roll["feats"], roll["actions"], roll["rewards"], roll["cont"]
        B, H = actions.shape[:2]
        with torch.no_grad():
            returns = lambda_returns(r, cont, self.critic_slow.value(f), c.gamma, c.lam)               # (B, H)
            # poids d'un pas = probabilité d'être encore en vie, actualisée : les rêves où la colonie est déjà morte comptent peu
            w = torch.cat([torch.ones(B, 1, device=f.device), torch.cumprod(c.gamma * cont, dim=1)[:, :-1]], dim=1)
        v_norm = self.critic.normalized(f[:, :H])
        critic_loss = (w * (v_norm - returns / c.value_scale) ** 2).mean()
        if real is not None and c.real_value_weight > 0:
            critic_loss = critic_loss + c.real_value_weight * ((self.critic.normalized(real[0]) - real[1] / c.value_scale) ** 2).mean()
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip); self.critic_opt.step()

        adv = returns - v_norm.detach() * c.value_scale
        adv_std = float(adv.std())
        adv = (adv - adv.mean()) / max(adv_std, c.adv_floor)
        logp, ent = self.actor.log_prob_entropy(f[:, :H].reshape(B * H, -1), actions.reshape(B * H, -1))
        logp, ent = logp.view(B, H), ent.view(B, H)
        actor_loss = -(w * adv * logp).mean() - c.ent_coef * (w * ent).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip); self.actor_opt.step()

        with torch.no_grad():                                                                            # critique lent : moyenne mobile
            for ps, p in zip(self.critic_slow.parameters(), self.critic.parameters()):
                ps.lerp_(p, c.target_tau)
        self.n_updates += 1
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()), "entropy": float(ent.detach().mean()),
                "dream_return": float(returns[:, 0].mean()), "dream_reward": float(r.mean()), "adv_std": adv_std, "dream_alive": float(w[:, -1].mean())}


# ----------------------------------------------------------------------------------------------------------------------
# Outils côté world model
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def sample_start_beliefs(model, buffer: ReplayBuffer, batch: int, context: int, rng: np.random.Generator, device
                         ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Croyances de départ des rêves : fenêtres réelles de `context` pas rejouées dans le modèle (mode « observé »).

    Renvoie (croyances, retour actualisé RÉEL à partir de chacune) : ce retour sert à ancrer l'échelle du critique sur la réalité."""
    context = min(context, buffer.max_length())
    w = buffer.sample_with_returns(batch, context, rng)
    maps, states, actions = (torch.as_tensor(w[k], device=device) for k in ("maps", "states", "actions"))
    belief = model.init_belief(maps[:, 0], states[:, 0])
    for t in range(context):
        belief = model.advance(belief, actions[:, t], maps[:, t + 1], states[:, t + 1])
    return belief, torch.as_tensor(w["ret"], device=device)


def rescale_reward_head(model, new_mean: float, new_std: float) -> None:
    """Change la normalisation de la récompense SANS changer ses prédictions (on absorbe le changement dans la dernière couche)."""
    old_mean, old_std = float(model.reward_mean), float(model.reward_std)
    new_std = max(float(new_std), 1e-6)
    last = model.reward_head[-1]
    with torch.no_grad():
        last.weight.mul_(old_std / new_std)
        last.bias.copy_((last.bias * old_std + old_mean - new_mean) / new_std)
        model.reward_mean.fill_(float(new_mean))
        model.reward_std.fill_(new_std)


def refit_norm(model, buffer: ReplayBuffer, floor: float = 5e-6) -> None:
    """Recalcule les poids de perte (variance de changement à 1 pas) et la normalisation de la récompense sur les données COURANTES.

    Les statistiques dépendent de la politique (une colonie qui grandit change plus qu'une colonie aléatoire) : on les ré-estime
    périodiquement. Les prédictions de la tête de récompense sont préservées (`rescale_reward_head`).
    """
    d = buffer.data
    st = d.persistence_stats()
    r = np.concatenate([d.a["rewards"][e, : d.lengths[e]] for e in range(len(d)) if d.lengths[e] > 0])
    w = lambda var: 1.0 / (torch.as_tensor(var, dtype=torch.float32) + floor)
    model.map_w.copy_(w(st["map_var"]))
    model.state_w.copy_(w(st["state_var"]))
    rescale_reward_head(model, float(r.mean()), float(r.std()))
