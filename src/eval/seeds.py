"""Seeds de mondes : plages d'entraînement et de test STRICTEMENT disjointes.

Règle d'or du banc d'évaluation : tout réglage (hyperparamètres, constantes de l'heuristique, choix du
budget de l'oracle...) se fait sur les seeds d'entraînement/validation ; les résultats rapportés viennent
des seeds de test, que personne n'a regardées pendant le développement.
"""

from __future__ import annotations

TRAIN_START = 0              # entraînement : [0, 100_000)
VAL_START = 500_000          # validation / réglages d'hyperparamètres : [500_000, 600_000)
TEST_START = 1_000_000       # test : [1_000_000, ...)
RANGE_SIZE = 100_000


def train_seeds(n: int, start: int = 0) -> list[int]:
    _check(start, n)
    return list(range(TRAIN_START + start, TRAIN_START + start + n))


def val_seeds(n: int, start: int = 0) -> list[int]:
    _check(start, n)
    return list(range(VAL_START + start, VAL_START + start + n))


def test_seeds(n: int, start: int = 0) -> list[int]:
    _check(start, n)
    return list(range(TEST_START + start, TEST_START + start + n))


def _check(start: int, n: int) -> None:
    if start < 0 or n < 0 or start + n > RANGE_SIZE:
        raise ValueError("plage de seeds hors limites")


test_seeds.__test__ = False   # le nom commence par « test_ » : empêche pytest de la collecter quand elle est importée dans un test
