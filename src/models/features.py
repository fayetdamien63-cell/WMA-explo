"""Extracteur de caractéristiques pour PPO : petit CNN sur la carte + MLP sur l'état interne.

La carte (8, 32, 32) passe par trois convolutions de pas 2 (32 → 16 → 8 → 4), puis une couche dense ; l'état
(24 valeurs) par une couche dense ; les deux sont concaténés. C'est volontairement petit : le CNN par défaut de
stable-baselines3 exige des images d'au moins 36 pixels de côté et est inutilement lourd pour du calcul sur CPU.
Le même type d'encodeur servira au world model (étape 5).
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class AntFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, channels: tuple[int, ...] = (16, 32, 32),
                 map_dim: int = 128, state_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim=map_dim + state_dim)
        map_shape = observation_space["map"].shape            # (C, H, W)
        layers: list[nn.Module] = []
        c_in = map_shape[0]
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            c_in = c_out
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():                                  # taille de sortie du CNN, déduite d'un exemple
            n_flat = self.cnn(torch.zeros(1, *map_shape)).shape[1]
        self.map_head = nn.Sequential(nn.Linear(n_flat, map_dim), nn.ReLU())
        self.state_head = nn.Sequential(nn.Linear(observation_space["state"].shape[0], state_dim), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.map_head(self.cnn(obs["map"])), self.state_head(obs["state"])], dim=1)
