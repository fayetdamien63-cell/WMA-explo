"""Gestion centralisée de l'aléa.

Principe : TOUT l'aléa du projet dérive d'une seule seed maître. On ne crée jamais de
générateur "à la main" (pas de `np.random.default_rng()` sans argument, pas de
`random.random()` global) : on demande un flux nommé au `SeedManager`.

    seeds = SeedManager(42)
    rng_world = seeds.rng("world_gen")     # génération procédurale
    rng_ants  = seeds.rng("ants")          # comportement des fourmis

Pourquoi des flux *nommés* plutôt qu'un seul générateur partagé ? Parce qu'avec un
générateur unique, ajouter un tirage quelque part change tous les tirages suivants
ailleurs : un changement local casse la reproductibilité globale. Avec un flux par
composant, chaque composant a sa propre suite, indépendante des autres.
"""

from __future__ import annotations

import hashlib
import random
import struct

import numpy as np


def _name_to_key(name: str) -> tuple[int, ...]:
    """Transforme un nom en clé entière stable.

    On n'utilise PAS `hash(name)` : Python le randomise d'un processus à l'autre
    (PYTHONHASHSEED), ce qui casserait la reproductibilité entre deux exécutions.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return struct.unpack(">4I", digest[:16])


class SeedManager:
    """Fabrique de générateurs aléatoires déterministes à partir d'une seed maître.

    Les méthodes sont des *fonctions pures* de (seed maître, nom) : appeler deux fois
    `rng("x")` renvoie deux générateurs distincts qui produisent la même suite. Si un
    composant a besoin d'un flux qui avance, il garde le générateur qu'il a obtenu.
    """

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"La seed doit être un entier >= 0, reçu : {seed!r}")
        self.seed = int(seed)

    def _sequence(self, name: str) -> np.random.SeedSequence:
        # SeedSequence garantit que des (seed, nom) différents donnent des flux
        # statistiquement indépendants, même si les seeds se ressemblent (42 vs 43).
        return np.random.SeedSequence(entropy=self.seed, spawn_key=_name_to_key(name))

    def rng(self, name: str) -> np.random.Generator:
        """Générateur NumPy pour le composant `name`."""
        return np.random.default_rng(self._sequence(name))

    def int_seed(self, name: str) -> int:
        """Entier dans [0, 2**31) pour les bibliothèques qui veulent une seed entière
        (gymnasium `reset(seed=...)`, stable-baselines3, torch...)."""
        return int(self._sequence(name).generate_state(1, dtype=np.uint32)[0] >> 1)

    def child(self, name: str) -> "SeedManager":
        """Sous-gestionnaire, p. ex. un par monde ou par expérience."""
        return SeedManager(self.int_seed(name))


def seed_everything(seed: int) -> SeedManager:
    """Fixe les générateurs *globaux* (random, np.random legacy, torch) et renvoie un SeedManager.

    Le code du projet utilise le SeedManager ; les générateurs globaux ne sont fixés que
    pour les bibliothèques tierces qui y piochent (init des poids PyTorch, SB3...).
    À appeler une fois, au début d'un script. Limite connue : PYTHONHASHSEED ne peut pas
    être fixé après le démarrage de l'interpréteur ; on évite donc de dépendre de l'ordre
    des `set`/`hash()` dans le code.
    """
    manager = SeedManager(seed)
    random.seed(manager.int_seed("python.random"))
    np.random.seed(manager.int_seed("numpy.global"))
    try:
        import torch
    except ImportError:  # le simulateur seul n'a pas besoin de torch
        return manager
    torch.manual_seed(manager.int_seed("torch"))  # fixe aussi CUDA/MPS
    return manager
