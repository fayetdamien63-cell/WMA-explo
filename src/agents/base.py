"""Interface commune des agents et utilitaires.

Un agent reçoit l'observation (dict `map` + `state`) et renvoie une action macro (5 entiers).
Seuls les agents « privilégiés » (l'oracle) reçoivent aussi l'environnement, pour lire le vrai état et
planifier avec le vrai simulateur ; tous les autres ne voient que l'observation partielle.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    name: str
    privileged: bool           # True : `act` reçoit aussi l'env (accès au vrai simulateur)
    sim_steps: int             # pas de simulateur consommés pour décider (planification)
    train_env_steps: int       # pas d'environnement consommés pour l'entraînement (0 hors apprentissage)

    def reset(self, agent_seed: int) -> None:
        """Début d'épisode. `agent_seed` rend les agents stochastiques reproductibles."""

    def act(self, obs: dict[str, np.ndarray], env=None) -> np.ndarray:
        """Action macro (5 entiers) pour l'observation courante."""
