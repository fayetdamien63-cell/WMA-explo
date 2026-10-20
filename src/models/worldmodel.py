"""World model supervisé : encodeur + état latent récurrent (GRU) + décodeurs (carte, état, récompense, fin d'épisode).

Deux façons d'avancer d'un pas macro (c'est ce qui permet à la fois de s'entraîner à 1 pas et d'« imaginer ») :

  OBSERVER   : le modèle voit la vraie observation o_{t+1} après avoir prédit : h ← GRU_obs(encodeur(o_{t+1}), h).
  BOUCLE OUVERTE (imagination) : pas d'observation ; le modèle n'a que ses actions et sa propre prédiction.
               C'est ce que fera un planificateur (étape 6) ou un agent dans l'imagination (étape 7).

Un pas de prédiction (dans les deux modes) :   h' = GRU_dyn(h, [action, état de base])  puis décodage de (o_{t+1}, r_t, fin_t).

Deux variantes (cfg.residual) :
  residual=False : « latent vectoriel seul » — la carte et l'état sont décodés UNIQUEMENT depuis h' (256 valeurs). Spécification
                   minimale de l'énoncé. Limite attendue : un vecteur de 256 valeurs ne peut pas mémoriser une carte 8×32×32 exacte,
                   donc il ne peut pas battre « rien ne change » sur les parties statiques (roche, nid, nourriture déjà vue).
  residual=True  : le décodeur prédit un RÉSIDU par rapport à une carte de base (la dernière observation, ou, en boucle ouverte, sa propre
                   dernière prédiction) : p = borne[0,1](base + δ(h', base)). Delta nul = « rien ne change » : le modèle n'a à apprendre
                   que ce qui change. La carte de base fait partie de l'état du modèle ; h' garde le contexte (rôles, charges, caps...).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class WMConfig:
    hidden: int = 256               # taille de l'état latent récurrent h
    embed: int = 192                # taille de l'encodage d'une observation
    action_embed: int = 64
    enc_channels: tuple[int, ...] = (16, 32, 32)
    head_channels: int = 16         # canaux du décodeur résiduel de carte
    residual: bool = True           # False : latent vectoriel seul (voir l'en-tête)
    map_channels: int = 8
    map_size: int = 32
    state_dim: int = 24
    n_action_dims: tuple[int, ...] = (4, 4, 4, 3, 5)


class ObsEncoder(nn.Module):
    """Observation (carte + état) -> vecteur : 3 convolutions de pas 2 + MLP d'état (même esprit que l'extracteur de PPO)."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        layers, c_in = [], cfg.map_channels
        for c in cfg.enc_channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.ReLU()]
            c_in = c
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, cfg.map_channels, cfg.map_size, cfg.map_size)).shape[1]
        self.map_fc = nn.Sequential(nn.Linear(n_flat, cfg.embed - 64), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(cfg.state_dim, 64), nn.ReLU())

    def forward(self, maps: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.map_fc(self.cnn(maps)), self.state_fc(states)], dim=1)


class ResidualMapHead(nn.Module):
    """Convolutions (dilatées : champ de vision 15×15) modulées par h' (FiLM) -> résidu additif, par canal et par case.

    La couche de sortie est initialisée à zéro : au départ le modèle prédit exactement « rien ne change ».
    """

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        c, k = cfg.head_channels, cfg.map_channels
        self.conv1 = nn.Conv2d(k, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(c, c, 3, padding=4, dilation=4)
        self.out = nn.Conv2d(c, k, 3, padding=1)
        self.film = nn.Linear(cfg.hidden, 6 * c)       # (échelle, décalage) pour chacune des 3 couches
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.c = c

    def forward(self, h: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        g = self.film(h).view(h.shape[0], 3, 2, self.c, 1, 1)
        x = F.relu(self.conv1(base) * (1 + g[:, 0, 0]) + g[:, 0, 1])
        x = F.relu(self.conv2(x) * (1 + g[:, 1, 0]) + g[:, 1, 1])
        x = F.relu(self.conv3(x) * (1 + g[:, 2, 0]) + g[:, 2, 1])
        return self.out(x)


class VectorMapDecoder(nn.Module):
    """Variante « latent seul » : h' -> carte complète par convolutions transposées (4×4 -> 8 -> 16 -> 32)."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.hidden, 64 * 4 * 4)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, cfg.map_channels, 4, stride=2, padding=1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(F.relu(self.fc(h)).view(-1, 64, 4, 4))


def _mlp(i: int, h: int, o: int, zero_last: bool = False) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WMConfig()
        c = self.cfg
        self.encoder = ObsEncoder(c)
        self.obs_cell = nn.GRUCell(c.embed, c.hidden)                           # mise à jour par observation
        self.act_embed = nn.Linear(sum(c.n_action_dims), c.action_embed)
        self.dyn_cell = nn.GRUCell(c.action_embed + c.state_dim, c.hidden)      # avancée d'un pas (action + état de base)
        if c.residual:
            self.map_head = ResidualMapHead(c)
            self.state_head = _mlp(c.hidden, 128, c.state_dim, zero_last=True)
            self.reward_head = _mlp(c.hidden + 3 * c.state_dim, 128, 1)
            self.done_head = _mlp(c.hidden + c.state_dim, 64, 1)
        else:
            self.map_head = VectorMapDecoder(c)
            self.state_head = _mlp(c.hidden, 128, c.state_dim)
            self.reward_head = _mlp(c.hidden, 128, 1)
            self.done_head = _mlp(c.hidden, 64, 1)
        nn.init.constant_(self.done_head[-1].bias, -4.0)                        # la mort est rare
        # Normalisations (fixées à partir des données, voir set_norm) : tampons, donc sauvegardés avec le modèle.
        self.register_buffer("map_w", torch.ones(c.map_channels))
        self.register_buffer("state_w", torch.ones(c.state_dim))
        self.register_buffer("reward_mean", torch.zeros(()))
        self.register_buffer("reward_std", torch.ones(()))

    # ------------------------------------------------------------------ outils
    def set_norm(self, map_var, state_var, reward_mean: float, reward_std: float) -> None:
        """Poids de perte w = 1 / (variance de changement à 1 pas + ε).

        Pour un canal dynamique, w · MSE ≈ MSE / MSE_{« rien ne change »} = 1 − compétence : une prédiction qui ne change rien vaut ≈ 1,
        une prédiction parfaite 0, et tous les canaux dynamiques pèsent pareil, quelle que soit leur rareté. Le terme de récompense
        (normalisé par son écart-type) vaut 1 pour un prédicteur constant : les trois termes sont ainsi comparables. On ne renormalise PAS
        les poids (une moyenne à 1 écraserait carte et état d'un facteur ~1000 devant la récompense : défaut corrigé pendant l'étape 5).
        ε = 5 % de la variance moyenne évite qu'un canal quasi immobile (nid : variance 0) reçoive un poids infini.
        """
        def weights(var):
            var = torch.as_tensor(var, dtype=torch.float32)
            return 1.0 / (var + 0.05 * var.mean())
        self.map_w.copy_(weights(map_var))
        self.state_w.copy_(weights(state_var))
        self.reward_mean.fill_(float(reward_mean))
        self.reward_std.fill_(max(float(reward_std), 1e-6))

    def onehot(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, 5) entiers -> (B, 20) concaténation de one-hot."""
        return torch.cat([F.one_hot(actions[:, d], n) for d, n in enumerate(self.cfg.n_action_dims)], dim=1).float()

    # ------------------------------------------------------------------ un pas
    def init_belief(self, maps0: torch.Tensor, states0: torch.Tensor) -> dict[str, torch.Tensor]:
        """Croyance initiale après avoir observé o_0."""
        h = self.obs_cell(self.encoder(maps0, states0), maps0.new_zeros(maps0.shape[0], self.cfg.hidden))
        return {"h": h, "map": maps0, "state": states0}

    def predict(self, belief: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Prédit (o_{t+1}, r_t, fin_t) depuis la croyance et l'action a_t. Renvoie (prédictions, h')."""
        h, base_map, base_state = belief["h"], belief["map"], belief["state"]
        h1 = self.dyn_cell(torch.cat([self.act_embed(self.onehot(actions)), base_state], dim=1), h)
        if self.cfg.residual:
            # p = base + δ, borné à [0, 1] par un « straight-through » : en sortie la valeur est bornée, mais le gradient passe toujours.
            # δ = 0 donne EXACTEMENT « rien ne change ». (Défaut corrigé pendant l'étape 5 : une version σ(logit(base) + δ) atténuait le
            # gradient d'un facteur ~1000 sur les cases valant exactement 0 ou 1, et les canaux binaires — visible, connu, roche — ne
            # pouvaient apprendre aucun changement, même en sur-apprentissage.)
            raw = base_map + self.map_head(h1, base_map)
            map_prob = raw + (raw.clamp(0.0, 1.0) - raw).detach()
            state = base_state + self.state_head(h1)
            r_in = torch.cat([h1, state, base_state, state - base_state], dim=1)
            d_in = torch.cat([h1, state], dim=1)
        else:
            map_prob = torch.sigmoid(self.map_head(h1))
            state = torch.sigmoid(self.state_head(h1))
            r_in, d_in = h1, h1
        reward = self.reward_head(r_in).squeeze(1) * self.reward_std + self.reward_mean
        return {"map": map_prob, "state": state, "reward": reward,
                "done_logit": self.done_head(d_in).squeeze(1)}, h1

    # ------------------------------------------------------------------ séquences
    def rollout(self, maps: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, n_forced: int,
                detach_base: bool = True) -> dict[str, torch.Tensor]:
        """Fait avancer le modèle sur une fenêtre de L pas.

        maps (B, L+1, C, H, W), states (B, L+1, S), actions (B, L, 5) : fenêtre RÉELLE.
        Les `n_forced` premiers pas sont « observés » (les vraies o_1..o_{n_forced} sont injectées) : prédictions à 1 pas.
        Les pas suivants sont en BOUCLE OUVERTE : le modèle n'utilise plus aucune observation après o_{n_forced} ; la prédiction
        du pas t ≥ n_forced est une prédiction à k = t − n_forced + 1 pas.
        Renvoie des tenseurs (B, L, ...) : pred_maps (prédiction de o_{t+1}), pred_states, rewards, done_logits.

        detach_base=True (défaut) : en boucle ouverte, la carte/l'état prédits réinjectés comme base du pas suivant sont COUPÉS du graphe
        de calcul (l'état latent h, lui, est rétropropagé dans le temps). Mesuré pendant l'étape 5 : en rétropropageant aussi à travers
        la chaîne de cartes prédites, l'entraînement est instable (perte 302 et divergence avec un pas de 1e-3) ; avec la coupure, le
        modèle sur-apprend proprement un petit lot en imagination.
        """
        B, L = actions.shape[:2]
        belief = self.init_belief(maps[:, 0], states[:, 0])
        outs = {"map": [], "state": [], "reward": [], "done_logit": []}
        for t in range(L):
            pred, h1 = self.predict(belief, actions[:, t])
            for k in outs:
                outs[k].append(pred[k])
            if t < n_forced:        # on observe o_{t+1}
                belief = {"h": self.obs_cell(self.encoder(maps[:, t + 1], states[:, t + 1]), h1),
                          "map": maps[:, t + 1], "state": states[:, t + 1]}
            else:                   # imagination : la base devient la prédiction
                nm, ns = pred["map"], pred["state"]
                belief = {"h": h1, "map": nm.detach() if detach_base else nm, "state": ns.detach() if detach_base else ns}
        return {k: torch.stack(v, dim=1) for k, v in outs.items()}

    def loss(self, batch: dict[str, torch.Tensor], n_forced: int, weights=(1.0, 1.0, 1.0, 0.1),
             detach_base: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        """Perte normalisée sur une fenêtre : carte, état, récompense, fin ; séparée entre pas observés et boucle ouverte."""
        maps, states, actions = batch["maps"], batch["states"], batch["actions"]
        rewards, dones = batch["rewards"], batch["dones"].float()
        out = self.rollout(maps, states, actions, n_forced, detach_base)
        w_map, w_state, w_rew, w_done = weights
        tgt_map, tgt_state = maps[:, 1:], states[:, 1:]
        per_map = (((out["map"] - tgt_map) ** 2).mean(dim=(3, 4)) * self.map_w).mean(dim=2)        # (B, L)
        per_state = (((out["state"] - tgt_state) ** 2) * self.state_w).mean(dim=2)
        per_rew = ((out["reward"] - rewards) / self.reward_std) ** 2
        per_done = F.binary_cross_entropy_with_logits(out["done_logit"], dones, reduction="none", pos_weight=torch.tensor(10.0))
        per_step = w_map * per_map + w_state * per_state + w_rew * per_rew + w_done * per_done        # (B, L)
        total = per_step.mean()
        info = {"loss": float(total.detach()), "map": float(per_map.detach().mean()), "state": float(per_state.detach().mean()),
                "reward": float(per_rew.detach().mean()), "done": float(per_done.detach().mean())}
        if n_forced > 0:
            info["loss_forced"] = float(per_step[:, :n_forced].detach().mean())
        if n_forced < actions.shape[1]:
            info["loss_open"] = float(per_step[:, n_forced:].detach().mean())
        return total, info


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
