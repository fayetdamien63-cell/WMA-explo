"""Simulateur du monde de fourmilières."""

from antworld.config import AntWorldConfig
from antworld.world import AntWorld

__all__ = ["AntWorld", "AntWorldConfig"]
# L'environnement Gymnasium se trouve dans antworld.env (nécessite l'extra "env").
